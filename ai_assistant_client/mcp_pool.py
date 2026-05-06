"""Pool of upstream MCP servers we can call out to.

Hosts configure one or more upstream MCP servers (over stdio or
SSE).  At startup the pool connects to each and discovers its
tool catalog.  Tool calls are dispatched to whichever server
exposes the named tool.

The pool is intentionally thin: it owns the lifetime of the MCP
sessions and routes calls.  Anything to do with *which* tools to
expose to the model — including progressive discovery — lives in
:mod:`ai_assistant_client.discovery`.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

try:  # The MCP SDK is heavy; keep imports optional for tests.
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError:  # pragma: no cover
    ClientSession = None  # type: ignore[assignment]
    sse_client = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class McpServerConfig:
    """How to reach a single upstream MCP server.

    Either ``command`` (stdio: launch a subprocess) or ``sse_url``
    (HTTP+SSE: connect to a remote MCP server).  Exactly one
    must be set.

    ``forwarded_credentials`` is passed to the upstream server
    as the ``X_AI_ASSISTANT_AUTH_<scheme>`` env vars when using
    stdio, or as ``X-AI-Assistant-Auth-<scheme>`` headers when
    using SSE.  Useful for acting on behalf of an end-user.
    """

    name: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    sse_url: str | None = None
    forwarded_credentials: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if bool(self.command) == bool(self.sse_url):
            raise ValueError(
                f"server config {self.name!r}: exactly one of "
                "'command' or 'sse_url' must be set"
            )


@dataclass(frozen=True)
class RemoteTool:
    """A tool exposed by an upstream MCP server."""

    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any]


class McpPool:
    """Holds live sessions with N upstream MCP servers."""

    def __init__(self, configs: list[McpServerConfig]) -> None:
        if ClientSession is None:
            raise RuntimeError(
                "The 'mcp' package is required.  Install with: pip install mcp"
            )
        if not configs:
            raise ValueError("At least one McpServerConfig is required")
        names = [c.name for c in configs]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate server names in pool: {names}")
        self._configs = configs
        self._sessions: dict[str, ClientSession] = {}
        self._tool_index: dict[str, str] = {}  # tool name → server name
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "McpPool":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        for cfg in self._configs:
            session = await self._open_session(cfg)
            self._sessions[cfg.name] = session
            await self._index_server(cfg.name, session)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(*exc)
        self._sessions.clear()
        self._tool_index.clear()
        self._stack = None

    async def _open_session(self, cfg: McpServerConfig) -> "ClientSession":
        assert self._stack is not None
        if cfg.command:
            env = dict(cfg.env)
            for scheme, secret in cfg.forwarded_credentials.items():
                env[f"X_AI_ASSISTANT_AUTH_{scheme}"] = secret
            params = StdioServerParameters(
                command=cfg.command, args=list(cfg.args), env=env
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        else:
            assert cfg.sse_url is not None
            headers: dict[str, str] = {}
            for scheme, secret in cfg.forwarded_credentials.items():
                headers[f"X-AI-Assistant-Auth-{scheme}"] = secret
            read, write = await self._stack.enter_async_context(
                sse_client(cfg.sse_url, headers=headers or None)
            )
        session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        return session

    async def _index_server(
        self, server_name: str, session: "ClientSession"
    ) -> None:
        listing = await session.list_tools()
        for tool in listing.tools:
            if tool.name in self._tool_index:
                # Disambiguate by prefixing with the server name.
                qualified = f"{server_name}.{tool.name}"
                self._tool_index[qualified] = server_name
            else:
                self._tool_index[tool.name] = server_name

    async def list_all_tools(self) -> list[RemoteTool]:
        out: list[RemoteTool] = []
        for cfg in self._configs:
            session = self._sessions[cfg.name]
            listing = await session.list_tools()
            for tool in listing.tools:
                out.append(
                    RemoteTool(
                        server_name=cfg.name,
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema,
                    )
                )
        return out

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        server_name = self._tool_index.get(name)
        if server_name is None:
            raise KeyError(f"Tool not found in pool: {name}")
        session = self._sessions[server_name]
        # Strip the server-prefix if we added one for disambiguation.
        upstream_name = name.split(".", 1)[1] if "." in name and name.startswith(
            f"{server_name}."
        ) else name
        result = await session.call_tool(upstream_name, arguments=arguments)
        # MCP returns a list of content items; flatten to text.
        return _result_to_text(result)


def _result_to_text(result: Any) -> str:
    """Render an MCP CallToolResult into a single text string.

    The MCP SDK returns content blocks (text / image / resource).
    For the chat use-case we collapse to text — non-text blocks
    pass through as a brief placeholder so the agent knows
    something was returned.
    """
    parts: list[str] = []
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue
        block_type = getattr(block, "type", "<unknown>")
        parts.append(f"[non-text block: {block_type}]")
    if getattr(result, "isError", False):
        return "[upstream error] " + "\n".join(parts)
    return "\n".join(parts) if parts else ""


# Convenience for tests / scripts that don't want to manage the
# full pool — not part of the public API.
async def list_tools_once(configs: list[McpServerConfig]) -> list[RemoteTool]:
    async with McpPool(configs) as pool:
        return await pool.list_all_tools()


__all__ = [
    "McpPool",
    "McpServerConfig",
    "RemoteTool",
    "list_tools_once",
]


# Re-export asyncio for convenience scripts.
_ = asyncio  # silence linters if unused at import time
