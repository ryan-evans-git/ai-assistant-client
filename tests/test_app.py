"""Starlette app surface tests with mocked MCP pool + provider.

We don't spin up real MCP sessions or LLM streams.  Instead we patch
:func:`build_app`'s collaborators so the routes exercise their
business logic against fakes.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from ai_assistant_client import app as app_module
from ai_assistant_client.app import (
    _load_server_configs,
    build_app,
)


# ---------------------------------------------------------------------------
# _load_server_configs
# ---------------------------------------------------------------------------


def test_load_server_configs_empty_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MCP_SERVERS", raising=False)
    assert _load_server_configs() == []


def test_load_server_configs_parses_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps(
            [
                {"name": "remote", "sse_url": "http://x/sse"},
                {
                    "name": "local",
                    "command": "ai-assistant-server",
                    "args": ["--tools-dir", "tools"],
                    "env": {"FOO": "1"},
                    "forwarded_credentials": {"BEARER": "tok"},
                },
            ]
        ),
    )
    cfgs = _load_server_configs()
    assert [c.name for c in cfgs] == ["remote", "local"]
    assert cfgs[0].sse_url == "http://x/sse"
    assert cfgs[1].command == "ai-assistant-server"
    assert cfgs[1].forwarded_credentials == {"BEARER": "tok"}


def test_load_server_configs_rejects_non_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_SERVERS", json.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="JSON array"):
        _load_server_configs()


# ---------------------------------------------------------------------------
# Routes: /healthz, /tools, /chat
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_state(
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """Build the app with the lifespan side-stepped — we manually
    populate the module-level state so routes have all collaborators."""
    monkeypatch.delenv("MCP_SERVERS", raising=False)

    class _StubRegistry:
        def __init__(self) -> None:
            self.calls: list[Any] = []

    class _StubPool:
        async def list_all_tools(self) -> list[Any]:
            from types import SimpleNamespace

            return [
                SimpleNamespace(name="t1", description="d1", server_name="s1"),
                SimpleNamespace(name="t2", description="d2", server_name="s2"),
            ]

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            return f"called {name}"

    class _StubProvider:
        async def stream_turn(self, **_kw: Any):  # pragma: no cover - not used
            if False:
                yield  # pragma: no cover

    app_module._state["pool"] = _StubPool()
    app_module._state["registry"] = _StubRegistry()
    app_module._state["provider"] = _StubProvider()
    from ai_assistant_client.agent import AgentRunConfig

    app_module._state["config"] = AgentRunConfig(model="x")

    app = build_app()
    yield TestClient(app)
    # Reset state.
    for k in app_module._state:
        app_module._state[k] = None


def test_healthz(client_with_state: TestClient) -> None:
    res = client_with_state.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"ok": True}


def test_tools_returns_pool_listing(client_with_state: TestClient) -> None:
    res = client_with_state.get("/tools")
    assert res.status_code == 200
    payload = res.json()
    names = {t["name"] for t in payload["tools"]}
    assert names == {"t1", "t2"}


def test_tools_returns_empty_when_pool_unavailable() -> None:
    """Pool is None → empty list, not 500."""
    app_module._state["pool"] = None
    app = build_app()
    res = TestClient(app).get("/tools")
    assert res.status_code == 200
    assert res.json() == {"tools": []}


def test_chat_rejects_empty_message(client_with_state: TestClient) -> None:
    res = client_with_state.post("/chat", json={"message": "  "})
    assert res.status_code == 400
    assert "required" in res.json()["error"]


def test_chat_returns_500_when_sse_starlette_missing(
    client_with_state: TestClient,
) -> None:
    with patch("ai_assistant_client.app.EventSourceResponse", None):
        res = client_with_state.post("/chat", json={"message": "hi"})
    assert res.status_code == 500
    assert "sse-starlette" in res.json()["error"]


def test_chat_routes_through_run_agent(
    client_with_state: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``message`` is supplied, ``/chat`` calls run_agent and
    streams its events."""

    async def fake_agent(**_kw: Any):
        from ai_assistant_client.agent import AgentEvent
        yield AgentEvent("user_message", {"content": "hi"})
        yield AgentEvent("turn_complete", {"stop_reason": "end_turn"})

    monkeypatch.setattr("ai_assistant_client.app.run_agent", fake_agent)
    res = client_with_state.post("/chat", json={"message": "hi"})
    assert res.status_code == 200
    body = res.text
    assert "user_message" in body
    assert "turn_complete" in body


def test_chat_dispatcher_returns_no_pool_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the pool is None, the dispatcher returned by /chat reports
    that no MCP servers are configured."""
    app_module._state["pool"] = None

    class _StubRegistry:
        pass

    class _StubProvider:
        pass

    app_module._state["registry"] = _StubRegistry()
    app_module._state["provider"] = _StubProvider()
    from ai_assistant_client.agent import AgentRunConfig
    app_module._state["config"] = AgentRunConfig(model="x")

    captured: dict[str, Any] = {}

    async def fake_agent(**kw: Any):
        captured["dispatcher"] = kw["dispatcher"]
        from ai_assistant_client.agent import AgentEvent
        yield AgentEvent("turn_complete", {"stop_reason": "end_turn"})

    monkeypatch.setattr("ai_assistant_client.app.run_agent", fake_agent)
    app = build_app()
    res = TestClient(app).post("/chat", json={"message": "hi"})
    assert res.status_code == 200
    dispatcher = captured["dispatcher"]
    msg = pytest_run(dispatcher("foo", {}))
    assert "unavailable" in msg


def pytest_run(coro: Any) -> Any:
    """Helper: run a coroutine to completion synchronously."""
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# build_app sanity
# ---------------------------------------------------------------------------


def test_build_app_routes_registered() -> None:
    app = build_app()
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/healthz" in paths
    assert "/tools" in paths
    assert "/chat" in paths


# ---------------------------------------------------------------------------
# lifespan — exercise the success + recovery paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_with_no_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No MCP_SERVERS → registry is empty, pool is None, provider built."""
    monkeypatch.delenv("MCP_SERVERS", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-x")

    fake_provider = object()
    with patch("ai_assistant_client.app.make_provider", return_value=fake_provider):
        app = build_app()
        async with app.router.lifespan_context(app):
            assert app_module._state["pool"] is None
            assert app_module._state["registry"] is not None
            assert app_module._state["provider"] is fake_provider


@pytest.mark.asyncio
async def test_lifespan_raises_when_no_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MCP_SERVERS", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "unknown_provider_zz")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    app = build_app()
    with pytest.raises(RuntimeError, match="No default model"):
        async with app.router.lifespan_context(app):
            pass


@pytest.mark.asyncio
async def test_lifespan_with_pool_setup_and_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful pool init → state is populated; cleanup unwinds it."""
    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps([{"name": "stub", "sse_url": "http://x"}]),
    )
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-x")

    class _FakePool:
        def __init__(self, _configs: Any) -> None:
            self.entered = False
            self.exited = False

        async def __aenter__(self) -> "_FakePool":
            self.entered = True
            return self

        async def __aexit__(self, *_a: Any) -> None:
            self.exited = True

        async def list_all_tools(self) -> list[Any]:
            from types import SimpleNamespace
            return [SimpleNamespace(name="t", description="d", input_schema={})]

    pool_holder: dict[str, _FakePool] = {}

    def make_pool(configs: Any) -> _FakePool:
        p = _FakePool(configs)
        pool_holder["p"] = p
        return p

    with (
        patch("ai_assistant_client.app.McpPool", make_pool),
        patch("ai_assistant_client.app.make_provider", return_value=object()),
    ):
        app = build_app()
        async with app.router.lifespan_context(app):
            assert app_module._state["pool"] is pool_holder["p"]
        assert pool_holder["p"].exited is True


@pytest.mark.asyncio
async def test_lifespan_pool_setup_failure_unwinds_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If listing tools after pool entry fails, the pool is closed."""
    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps([{"name": "stub", "sse_url": "http://x"}]),
    )
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-x")

    class _FakePool:
        def __init__(self, _configs: Any) -> None:
            self.exited = False

        async def __aenter__(self) -> "_FakePool":
            return self

        async def __aexit__(self, *_a: Any) -> None:
            self.exited = True

        async def list_all_tools(self) -> list[Any]:
            raise RuntimeError("boom")

    pool_holder: dict[str, _FakePool] = {}

    def make_pool(configs: Any) -> _FakePool:
        p = _FakePool(configs)
        pool_holder["p"] = p
        return p

    with (
        patch("ai_assistant_client.app.McpPool", make_pool),
        patch("ai_assistant_client.app.make_provider", return_value=object()),
    ):
        app = build_app()
        with pytest.raises(RuntimeError, match="boom"):
            async with app.router.lifespan_context(app):
                pass
        assert pool_holder["p"].exited is True
