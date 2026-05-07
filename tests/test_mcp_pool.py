"""McpPool tests with mocked MCP sessions.

We patch ``stdio_client`` / ``sse_client`` / ``ClientSession`` at the
module level so :class:`McpPool` exercises its routing + auth-forwarding
+ tool-result flattening logic against fakes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_assistant_client.mcp_pool import (
    McpPool,
    McpServerConfig,
    RemoteTool,
    _result_to_text,
    list_tools_once,
)


# ---------------------------------------------------------------------------
# McpServerConfig validation
# ---------------------------------------------------------------------------


def test_server_config_requires_exactly_one_transport() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        McpServerConfig(name="x")  # neither command nor sse_url
    with pytest.raises(ValueError, match="exactly one"):
        McpServerConfig(name="x", command="c", sse_url="http://x")


def test_server_config_accepts_either_transport() -> None:
    McpServerConfig(name="a", command="c")
    McpServerConfig(name="b", sse_url="http://x")


# ---------------------------------------------------------------------------
# McpPool construction-time validation
# ---------------------------------------------------------------------------


def test_pool_requires_at_least_one_config() -> None:
    with pytest.raises(ValueError, match="At least one"):
        McpPool([])


def test_pool_rejects_duplicate_names() -> None:
    cfgs = [
        McpServerConfig(name="dup", sse_url="http://1"),
        McpServerConfig(name="dup", sse_url="http://2"),
    ]
    with pytest.raises(ValueError, match="Duplicate server names"):
        McpPool(cfgs)


def test_pool_raises_when_mcp_sdk_unavailable() -> None:
    with patch("ai_assistant_client.mcp_pool.ClientSession", None):
        with pytest.raises(RuntimeError, match="mcp"):
            McpPool([McpServerConfig(name="x", sse_url="http://x")])


# ---------------------------------------------------------------------------
# Lifecycle: __aenter__ wires sessions + indexes tools; call_tool routes;
# __aexit__ unwinds.
# ---------------------------------------------------------------------------


def _make_fake_session(tool_names: list[str]) -> MagicMock:
    session = MagicMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(
        return_value=SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name=n, description=f"d-{n}", inputSchema={"type": "object"}
                )
                for n in tool_names
            ]
        )
    )
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"called")],
            isError=False,
        )
    )
    return session


@asynccontextmanager
async def _fake_transport(*_a: Any, **_kw: Any):
    yield (object(), object())  # (read_stream, write_stream)


@pytest.mark.asyncio
async def test_pool_lifecycle_with_stdio_transport() -> None:
    """Open one stdio server, list its tools, dispatch a call, close."""
    cfg = McpServerConfig(
        name="local",
        command="ai-assistant-server",
        args=("--tools-dir", "tools"),
        env={"FOO": "1"},
        forwarded_credentials={"BEARER": "tok"},
    )
    session = _make_fake_session(["alpha", "beta"])

    @asynccontextmanager
    async def fake_session_factory(*_a: Any, **_kw: Any):
        yield session

    @asynccontextmanager
    async def fake_stdio_client(params: Any):
        # Verify env-forwarded credentials made it into the params.
        assert params.env["X_AI_ASSISTANT_AUTH_BEARER"] == "tok"
        assert params.env["FOO"] == "1"
        assert params.command == "ai-assistant-server"
        yield (object(), object())

    captured_params: dict[str, Any] = {}

    def stdio_params_ctor(**kwargs: Any) -> Any:
        captured_params.update(kwargs)
        return SimpleNamespace(**kwargs)

    with (
        patch("ai_assistant_client.mcp_pool.stdio_client", fake_stdio_client),
        patch(
            "ai_assistant_client.mcp_pool.StdioServerParameters", stdio_params_ctor
        ),
        patch(
            "ai_assistant_client.mcp_pool.ClientSession",
            lambda r, w: _SessionContextManager(session),
        ),
    ):
        async with McpPool([cfg]) as pool:
            tools = await pool.list_all_tools()
            names = [t.name for t in tools]
            assert names == ["alpha", "beta"]

            result_text = await pool.call_tool("alpha", {"x": 1})
            assert result_text == "called"


class _SessionContextManager:
    """Wraps a pre-built session as an async context manager."""

    def __init__(self, session: MagicMock) -> None:
        self._session = session

    async def __aenter__(self) -> MagicMock:
        return self._session

    async def __aexit__(self, *_a: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_pool_with_sse_transport_passes_forwarded_headers() -> None:
    cfg = McpServerConfig(
        name="remote",
        sse_url="http://x/sse",
        forwarded_credentials={"BEARER": "tok"},
    )
    session = _make_fake_session(["search"])

    captured_headers: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_sse_client(url: str, *, headers: Any = None):
        captured_headers["url"] = url
        captured_headers["headers"] = headers
        yield (object(), object())

    with (
        patch("ai_assistant_client.mcp_pool.sse_client", fake_sse_client),
        patch(
            "ai_assistant_client.mcp_pool.ClientSession",
            lambda r, w: _SessionContextManager(session),
        ),
    ):
        async with McpPool([cfg]) as pool:
            await pool.list_all_tools()

    assert captured_headers["url"] == "http://x/sse"
    assert captured_headers["headers"]["X-AI-Assistant-Auth-BEARER"] == "tok"


@pytest.mark.asyncio
async def test_pool_call_tool_unknown_raises() -> None:
    cfg = McpServerConfig(name="r", sse_url="http://x")
    session = _make_fake_session(["a"])

    @asynccontextmanager
    async def fake_sse_client(*_a: Any, **_kw: Any):
        yield (object(), object())

    with (
        patch("ai_assistant_client.mcp_pool.sse_client", fake_sse_client),
        patch(
            "ai_assistant_client.mcp_pool.ClientSession",
            lambda r, w: _SessionContextManager(session),
        ),
    ):
        async with McpPool([cfg]) as pool:
            with pytest.raises(KeyError, match="not found"):
                await pool.call_tool("ghost", {})


@pytest.mark.asyncio
async def test_pool_disambiguates_duplicate_tool_names() -> None:
    """When two servers expose the same tool, the second one is
    indexed under ``server.tool``."""
    cfg_a = McpServerConfig(name="a", sse_url="http://a")
    cfg_b = McpServerConfig(name="b", sse_url="http://b")
    session_a = _make_fake_session(["dup"])
    session_b = _make_fake_session(["dup"])
    sessions = iter([session_a, session_b])

    @asynccontextmanager
    async def fake_sse_client(*_a: Any, **_kw: Any):
        yield (object(), object())

    def session_factory(*_a: Any, **_kw: Any) -> Any:
        return _SessionContextManager(next(sessions))

    with (
        patch("ai_assistant_client.mcp_pool.sse_client", fake_sse_client),
        patch("ai_assistant_client.mcp_pool.ClientSession", session_factory),
    ):
        async with McpPool([cfg_a, cfg_b]) as pool:
            # First server occupies the unprefixed name.
            assert "dup" in pool._tool_index  # type: ignore[attr-defined]
            # Second one falls under the qualified key.
            assert "b.dup" in pool._tool_index  # type: ignore[attr-defined]
            await pool.call_tool("b.dup", {})
            session_b.call_tool.assert_awaited_with("dup", arguments={})


@pytest.mark.asyncio
async def test_list_tools_once_helper() -> None:
    cfg = McpServerConfig(name="r", sse_url="http://x")
    session = _make_fake_session(["one"])

    @asynccontextmanager
    async def fake_sse_client(*_a: Any, **_kw: Any):
        yield (object(), object())

    with (
        patch("ai_assistant_client.mcp_pool.sse_client", fake_sse_client),
        patch(
            "ai_assistant_client.mcp_pool.ClientSession",
            lambda r, w: _SessionContextManager(session),
        ),
    ):
        tools = await list_tools_once([cfg])
    assert [t.name for t in tools] == ["one"]


# ---------------------------------------------------------------------------
# _result_to_text
# ---------------------------------------------------------------------------


def test_result_to_text_concats_text_blocks() -> None:
    res = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(type="text", text="world"),
        ],
        isError=False,
    )
    assert _result_to_text(res) == "hello\nworld"


def test_result_to_text_replaces_non_text_blocks() -> None:
    res = SimpleNamespace(
        content=[SimpleNamespace(type="image", text=None)],
        isError=False,
    )
    assert "non-text block" in _result_to_text(res)


def test_result_to_text_marks_errors() -> None:
    res = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="boom")],
        isError=True,
    )
    assert _result_to_text(res).startswith("[upstream error]")


def test_result_to_text_empty_content() -> None:
    res = SimpleNamespace(content=[], isError=False)
    assert _result_to_text(res) == ""
