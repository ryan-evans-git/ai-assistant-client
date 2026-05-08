"""Starlette HTTP service that wraps the agent loop in a SSE endpoint.

Routes:
    POST /chat          → SSE stream of agent events.
    GET  /healthz       → liveness probe.
    GET  /tools         → debug dump of the discovered MCP catalog.

Configure upstream MCP servers via ``MCP_SERVERS`` env (JSON):

    MCP_SERVERS='[
      {"name":"oapi","sse_url":"http://localhost:8765/sse"},
      {"name":"local","command":"ai-assistant-server","args":["--tools-dir","./tools"]}
    ]'

Pick the LLM with ``LLM_PROVIDER`` (``anthropic`` | ``openai`` |
``gemini`` | ``bedrock``; default ``anthropic``) and override the
model with ``LLM_MODEL``.  Each provider reads its own credentials
from the SDK's standard env (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``, or the standard AWS chain).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:  # pragma: no cover
    EventSourceResponse = None  # type: ignore[assignment]

from ai_assistant_client.agent import AgentRunConfig, run_agent
from ai_assistant_client.confirmation_store import (
    ConfirmationOutcome,
    PendingConfirmationStore,
    make_confirmation_store,
)
from ai_assistant_client.discovery import (
    ProgressiveToolRegistry,
    RemoteToolDescriptor,
)
from ai_assistant_client.llm import LLMProvider, default_model, make_provider
from ai_assistant_client.mcp_pool import McpPool, McpServerConfig


log = logging.getLogger(__name__)


# Module-level state — reset by ``lifespan`` on startup.
_state: dict[str, Any] = {
    "pool": None,
    "registry": None,
    "provider": None,
    "config": None,
    # Pending HITL confirmations.  Either an in-process map or a
    # Redis-backed pubsub store, picked from ``REDIS_URL`` env.
    "confirmations": None,
}


def _load_server_configs() -> list[McpServerConfig]:
    raw = os.environ.get("MCP_SERVERS")
    if not raw:
        return []
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("MCP_SERVERS must be a JSON array")
    return [
        McpServerConfig(
            name=item["name"],
            command=item.get("command"),
            args=tuple(item.get("args", []) or []),
            env=item.get("env") or {},
            sse_url=item.get("sse_url"),
            forwarded_credentials=item.get("forwarded_credentials") or {},
        )
        for item in parsed
    ]


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    configs = _load_server_configs()
    if configs:
        pool = await McpPool(configs).__aenter__()
        try:
            tools = await pool.list_all_tools()
            descriptors = [
                RemoteToolDescriptor(
                    name=t.name,
                    description=t.description,
                    input_schema=t.input_schema,
                    hitl=getattr(t, "hitl", None),
                )
                for t in tools
            ]
            registry = ProgressiveToolRegistry(descriptors)
            _state["pool"] = pool
            _state["registry"] = registry
        except Exception:
            await pool.__aexit__(None, None, None)
            raise
    else:
        log.warning("No MCP_SERVERS configured — only the meta-tools will work")
        _state["pool"] = None
        _state["registry"] = ProgressiveToolRegistry([])

    # Confirmation store: in-process by default, Redis-backed when
    # REDIS_URL is set so multi-worker deployments can route a
    # POST /chat/confirm to whichever worker owns the SSE stream.
    _state["confirmations"] = make_confirmation_store()

    provider_name = os.environ.get("LLM_PROVIDER", "anthropic")
    model = os.environ.get("LLM_MODEL") or default_model(provider_name)
    if not model:
        raise RuntimeError(
            f"No default model for LLM_PROVIDER={provider_name!r}; "
            "set LLM_MODEL explicitly."
        )
    log.info("Using LLM provider=%s model=%s", provider_name, model)
    _state["config"] = AgentRunConfig(model=model)
    _state["provider"] = make_provider(provider_name)
    try:
        yield
    finally:
        pool = _state.get("pool")
        if pool is not None:
            await pool.__aexit__(None, None, None)
        store: PendingConfirmationStore | None = _state.get("confirmations")
        if store is not None:
            await store.aclose()
        _state["pool"] = None
        _state["registry"] = None
        _state["provider"] = None
        _state["confirmations"] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


async def healthz(_: Request) -> Response:
    return JSONResponse({"ok": True})


async def list_tools(_: Request) -> Response:
    pool: McpPool | None = _state.get("pool")
    if pool is None:
        return JSONResponse({"tools": []})
    tools = await pool.list_all_tools()
    return JSONResponse(
        {
            "tools": [
                {"name": t.name, "description": t.description, "server": t.server_name}
                for t in tools
            ]
        }
    )


async def chat(request: Request) -> Response:
    if EventSourceResponse is None:
        return JSONResponse(
            {"error": "sse-starlette is not installed"}, status_code=500
        )
    body = await request.json()
    message = str(body.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)
    history = list(body.get("history") or [])

    registry: ProgressiveToolRegistry = _state["registry"]
    pool: McpPool | None = _state.get("pool")
    config: AgentRunConfig = _state["config"]
    provider: LLMProvider = _state["provider"]
    store: PendingConfirmationStore | None = _state.get("confirmations")

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        if pool is None:
            return f"Tool '{name}' is unavailable — no MCP servers configured."
        return await pool.call_tool(name, arguments)

    # Track outstanding confirmation request_ids so a stream cancel
    # can drop them from the store (otherwise their futures would
    # leak until the next process restart).
    outstanding: list[str] = []

    async def confirmation_hook(payload: dict[str, Any]) -> ConfirmationOutcome:
        if store is None:
            # No store configured (lifespan never ran, e.g. unit
            # test) — auto-confirm so the chat doesn't deadlock.
            return ConfirmationOutcome(decision="confirm")
        request_id = payload["request_id"]
        outstanding.append(request_id)
        fut = await store.register(request_id)
        try:
            return await fut
        finally:
            if request_id in outstanding:
                outstanding.remove(request_id)

    async def event_iter() -> AsyncIterator[dict[str, str]]:
        try:
            async for event in run_agent(
                user_message=message,
                history=history,
                registry=registry,
                dispatcher=dispatcher,
                provider=provider,
                config=config,
                confirmation_hook=confirmation_hook,
            ):
                yield event.to_sse()
        except asyncio.CancelledError:
            yield {"event": "cancelled", "data": "{}"}
            raise
        finally:
            # Whether the stream finished cleanly or was cancelled,
            # cancel any still-pending confirmations so their futures
            # don't leak.
            if store is not None:
                for rid in outstanding:
                    await store.cancel(rid)

    return EventSourceResponse(event_iter())


async def chat_confirm(request: Request) -> Response:
    """Resolve a pending HITL confirmation.

    Body: ``{request_id, decision: "confirm"|"decline", note?}``.
    Returns 200 on success, 404 if the id is unknown / already
    resolved / expired, 400 on a malformed body.
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    request_id = (body.get("request_id") or "").strip()
    decision = (body.get("decision") or "").strip()
    note = body.get("note")
    if not request_id or decision not in ("confirm", "decline"):
        return JSONResponse(
            {"error": "request_id and decision (confirm|decline) are required"},
            status_code=400,
        )
    store: PendingConfirmationStore | None = _state.get("confirmations")
    if store is None:
        return JSONResponse(
            {"error": "confirmation store not initialized"}, status_code=503
        )
    note_str = str(note) if note else None
    delivered = await store.resolve(
        request_id, ConfirmationOutcome(decision=decision, note=note_str)
    )
    if not delivered:
        return JSONResponse(
            {"error": f"unknown or expired request_id {request_id!r}"},
            status_code=404,
        )
    return JSONResponse({"ok": True})


def build_app() -> Starlette:
    return Starlette(
        debug=False,
        lifespan=lifespan,
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/tools", list_tools, methods=["GET"]),
            Route("/chat", chat, methods=["POST"]),
            Route("/chat/confirm", chat_confirm, methods=["POST"]),
        ],
    )


app = build_app()


def main() -> int:
    import uvicorn

    logging.basicConfig(
        level=os.environ.get("AI_ASSISTANT_CLIENT_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    host = os.environ.get("AI_ASSISTANT_CLIENT_HOST", "127.0.0.1")
    port = int(os.environ.get("AI_ASSISTANT_CLIENT_PORT", "8080"))
    uvicorn.run(
        "ai_assistant_client.app:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
