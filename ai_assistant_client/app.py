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
        _state["pool"] = None
        _state["registry"] = None
        _state["provider"] = None


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

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        if pool is None:
            return f"Tool '{name}' is unavailable — no MCP servers configured."
        return await pool.call_tool(name, arguments)

    async def event_iter() -> AsyncIterator[dict[str, str]]:
        try:
            async for event in run_agent(
                user_message=message,
                history=history,
                registry=registry,
                dispatcher=dispatcher,
                provider=provider,
                config=config,
            ):
                yield event.to_sse()
        except asyncio.CancelledError:
            yield {"event": "cancelled", "data": "{}"}
            raise

    return EventSourceResponse(event_iter())


def build_app() -> Starlette:
    return Starlette(
        debug=False,
        lifespan=lifespan,
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/tools", list_tools, methods=["GET"]),
            Route("/chat", chat, methods=["POST"]),
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
