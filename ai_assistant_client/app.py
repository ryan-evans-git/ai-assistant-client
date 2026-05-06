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

Anthropic credentials come from the standard ``ANTHROPIC_API_KEY``.
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
from ai_assistant_client.mcp_pool import McpPool, McpServerConfig


log = logging.getLogger(__name__)


# Module-level state — reset by ``lifespan`` on startup.
_state: dict[str, Any] = {
    "pool": None,
    "registry": None,
    "anthropic_client": None,
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

    _state["config"] = AgentRunConfig()
    _state["anthropic_client"] = _anthropic_client()
    try:
        yield
    finally:
        pool = _state.get("pool")
        if pool is not None:
            await pool.__aexit__(None, None, None)
        _state["pool"] = None
        _state["registry"] = None
        _state["anthropic_client"] = None


def _anthropic_client() -> Any:
    try:
        from anthropic import AsyncAnthropic
    except ImportError as err:
        raise RuntimeError(
            "Install the 'anthropic' package: pip install anthropic"
        ) from err
    return AsyncAnthropic()


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
    client = _state["anthropic_client"]

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
                anthropic_client=client,
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
