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
from ai_assistant_client.persistence import (
    TranscriptStore,
    make_transcript_store,
)
from ai_assistant_client.workflows import (
    WorkflowRegistry,
    load_workflows_from_directory,
)


# Env flag that opts every dispatched workflow into transcript
# recording.  Accepts the usual truthy variants ("1", "true",
# "yes", "on") — anything else (or unset) is off.  The store
# itself is picked by :func:`make_transcript_store` from its own
# env vars (AAC_TRANSCRIPT_BACKEND etc.).
RECORDING_ENV = "AAC_WORKFLOW_RECORDING"


def _recording_enabled() -> bool:
    raw = os.environ.get(RECORDING_ENV, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


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
    # Loaded workflows from WORKFLOWS_DIR.  Empty registry by default.
    "workflows": None,
    # Transcript store for workflow recording.  Populated when
    # AAC_WORKFLOW_RECORDING is truthy; otherwise None and the
    # agent loop uses the plain (non-recording) dispatch.
    "transcripts": None,
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
    # Load client-side workflows first so we can register them in
    # the same catalog as MCP-derived tools — the LLM should see
    # both kinds in one progressive-discovery surface.
    workflows_dir = os.environ.get("WORKFLOWS_DIR", "workflows")
    workflows = load_workflows_from_directory(workflows_dir)
    workflow_registry = WorkflowRegistry(workflows)

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
            descriptors.extend(workflow_registry.as_descriptors())
            registry = ProgressiveToolRegistry(descriptors)
            _state["pool"] = pool
            _state["registry"] = registry
        except Exception:
            await pool.__aexit__(None, None, None)
            raise
    else:
        log.warning("No MCP_SERVERS configured — only the meta-tools will work")
        _state["pool"] = None
        _state["registry"] = ProgressiveToolRegistry(
            workflow_registry.as_descriptors()
        )
    _state["workflows"] = workflow_registry

    # Transcript store — only constructed when recording is on so
    # a default deployment doesn't open files / databases it doesn't
    # need.  The backend itself is picked by make_transcript_store()
    # from its own env vars (AAC_TRANSCRIPT_BACKEND / _DIR / _SQLITE_PATH).
    if _recording_enabled():
        _state["transcripts"] = make_transcript_store()
        log.info("Workflow recording enabled (backend: %s)",
                 type(_state["transcripts"]).__name__)
    else:
        _state["transcripts"] = None

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
        _state["workflows"] = None
        _state["transcripts"] = None


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
    workflow_registry: WorkflowRegistry | None = _state.get("workflows")
    transcript_store: TranscriptStore | None = _state.get("transcripts")

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
                workflow_registry=workflow_registry,
                transcript_store=transcript_store,
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


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Three sub-commands:

    * (no args) — serve the HTTP / SSE chat endpoint (the
      original / default behaviour).  Recording is opted in via
      ``AAC_WORKFLOW_RECORDING=on``; the backend is picked by
      ``AAC_TRANSCRIPT_BACKEND`` (memory / file / sqlite).
    * ``replay <run_id>`` — read a recorded workflow run from the
      transcript store and emit each event as a JSON line on
      stdout.  Offline; doesn't bind a port.
    * ``graph <run_id> [--kind flowchart|sequence]`` — render
      the recorded run as a Mermaid diagram on stdout.  Pair with
      ``> diagram.mmd`` or pipe into ``mmdc`` for an SVG.

    Both offline sub-commands honour the same env vars used by
    the server, so a recording captured by the server is
    immediately legible to ``replay`` / ``graph`` without extra
    configuration.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="ai-assistant-client",
        description=(
            "Streaming chat client. Run with no arguments to serve "
            "the SSE chat endpoint; use 'replay' / 'graph' for "
            "offline inspection of recorded workflow runs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    replay_parser = subparsers.add_parser(
        "replay",
        help="Emit recorded workflow events as JSON lines on stdout.",
    )
    replay_parser.add_argument("run_id", help="Run id to replay.")

    graph_parser = subparsers.add_parser(
        "graph",
        help="Render a recorded workflow run as a Mermaid diagram.",
    )
    graph_parser.add_argument("run_id", help="Run id to render.")
    graph_parser.add_argument(
        "--kind",
        choices=("flowchart", "sequence"),
        default="flowchart",
        help="Mermaid diagram kind (default: flowchart).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("AI_ASSISTANT_CLIENT_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.command == "replay":
        return asyncio.run(_cli_replay(args.run_id))
    if args.command == "graph":
        return asyncio.run(_cli_graph(args.run_id, kind=args.kind))

    # Default: serve.
    import uvicorn

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


async def _cli_replay(run_id: str) -> int:
    """Stream recorded events as JSON lines.

    Returns ``0`` on success, ``1`` if the run id is unknown.
    The transcript store is built from the same env vars the
    server uses, so a host can record on one process and replay
    from another against the same backend.

    Defined async (rather than synchronous + an internal
    ``asyncio.run``) so tests can await it directly inside the
    pytest-asyncio event loop without nesting ``asyncio.run``
    calls.  :func:`main` does the ``asyncio.run`` for the CLI.
    """
    import json
    from dataclasses import asdict

    store = make_transcript_store()
    try:
        transcript = await store.read(run_id)
    except KeyError:
        print(f"unknown run id: {run_id!r}", flush=True)
        return 1
    for event in transcript.events:
        print(json.dumps(asdict(event), default=str), flush=True)
    return 0


async def _cli_graph(run_id: str, *, kind: str) -> int:
    """Render a recorded run as a Mermaid diagram on stdout."""
    from ai_assistant_client.workflows.graph import transcript_to_mermaid

    store = make_transcript_store()
    try:
        transcript = await store.read(run_id)
    except KeyError:
        print(f"unknown run id: {run_id!r}", flush=True)
        return 1
    print(transcript_to_mermaid(transcript, kind=kind))  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
