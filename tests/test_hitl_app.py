"""HTTP-surface tests for the HITL confirmation flow.

Exercises the ``POST /chat/confirm`` route end-to-end against the
in-memory store plus the ``mcp_pool`` → ``RemoteToolDescriptor.hitl``
propagation through ``lifespan``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from ai_assistant_client import app as app_module
from ai_assistant_client.app import build_app
from ai_assistant_client.confirmation_store import (
    InMemoryStore,
)


@pytest.fixture
def client_with_store() -> TestClient:
    """Build the app with module state populated manually so the
    routes can run without a real MCP pool / LLM provider."""
    app_module._state["pool"] = None
    app_module._state["registry"] = SimpleNamespace()
    app_module._state["provider"] = SimpleNamespace()
    from ai_assistant_client.agent import AgentRunConfig

    app_module._state["config"] = AgentRunConfig(model="x")
    app_module._state["confirmations"] = InMemoryStore()
    yield TestClient(build_app())
    for k in list(app_module._state):
        app_module._state[k] = None


def test_chat_confirm_rejects_missing_body_fields(
    client_with_store: TestClient,
) -> None:
    res = client_with_store.post("/chat/confirm", json={})
    assert res.status_code == 400


def test_chat_confirm_rejects_invalid_decision(
    client_with_store: TestClient,
) -> None:
    res = client_with_store.post(
        "/chat/confirm", json={"request_id": "x", "decision": "maybe"}
    )
    assert res.status_code == 400


def test_chat_confirm_404_on_unknown_request(
    client_with_store: TestClient,
) -> None:
    res = client_with_store.post(
        "/chat/confirm", json={"request_id": "ghost", "decision": "confirm"}
    )
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_chat_confirm_resolves_pending_future() -> None:
    """End-to-end on the resolution path:  agent registers a pending
    confirmation; HTTP route resolves it; the awaited future fires."""
    store = InMemoryStore()
    app_module._state["confirmations"] = store
    app_module._state["pool"] = None
    app_module._state["registry"] = SimpleNamespace()
    app_module._state["provider"] = SimpleNamespace()
    from ai_assistant_client.agent import AgentRunConfig

    app_module._state["config"] = AgentRunConfig(model="x")

    fut = await store.register("rid-100")

    client = TestClient(build_app())
    res = client.post(
        "/chat/confirm",
        json={"request_id": "rid-100", "decision": "confirm", "note": "approved"},
    )
    assert res.status_code == 200
    outcome = await asyncio.wait_for(fut, timeout=1.0)
    assert outcome.decision == "confirm"
    assert outcome.note == "approved"

    for k in list(app_module._state):
        app_module._state[k] = None


def test_chat_confirm_503_when_store_uninitialized() -> None:
    """Without ``lifespan`` running the confirmations key stays None;
    the route reports 503 instead of crashing."""
    for k in list(app_module._state):
        app_module._state[k] = None
    app_module._state["pool"] = None
    app_module._state["registry"] = SimpleNamespace()
    app_module._state["provider"] = SimpleNamespace()
    from ai_assistant_client.agent import AgentRunConfig

    app_module._state["config"] = AgentRunConfig(model="x")
    app_module._state["confirmations"] = None  # explicit

    client = TestClient(build_app())
    res = client.post(
        "/chat/confirm", json={"request_id": "x", "decision": "confirm"}
    )
    assert res.status_code == 503


def test_chat_confirm_400_on_invalid_json() -> None:
    app_module._state["confirmations"] = InMemoryStore()
    app_module._state["pool"] = None
    app_module._state["registry"] = SimpleNamespace()
    app_module._state["provider"] = SimpleNamespace()
    from ai_assistant_client.agent import AgentRunConfig
    app_module._state["config"] = AgentRunConfig(model="x")

    client = TestClient(build_app())
    res = client.post(
        "/chat/confirm",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert res.status_code == 400
    for k in list(app_module._state):
        app_module._state[k] = None


# ---------------------------------------------------------------------------
# Lifespan: HITL propagation from RemoteTool → RemoteToolDescriptor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_propagates_hitl_metadata_into_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RemoteTool with .hitl set should land on the descriptor."""
    import json

    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps([{"name": "stub", "sse_url": "http://x"}]),
    )
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-x")
    monkeypatch.delenv("REDIS_URL", raising=False)

    class _FakePool:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        async def __aenter__(self) -> "_FakePool":
            return self

        async def __aexit__(self, *_a: Any) -> None:
            return None

        async def list_all_tools(self) -> list[Any]:
            return [
                SimpleNamespace(
                    name="send_email",
                    description="send",
                    input_schema={"type": "object"},
                    hitl={"requires_confirmation": True, "message": "Send?"},
                ),
                SimpleNamespace(
                    name="ping",
                    description="ping",
                    input_schema={"type": "object"},
                    hitl=None,
                ),
            ]

    with (
        patch("ai_assistant_client.app.McpPool", _FakePool),
        patch("ai_assistant_client.app.make_provider", return_value=object()),
    ):
        app = build_app()
        async with app.router.lifespan_context(app):
            registry = app_module._state["registry"]
            d_send = registry.get_descriptor("send_email")
            d_ping = registry.get_descriptor("ping")
            assert d_send is not None
            assert d_send.hitl == {"requires_confirmation": True, "message": "Send?"}
            assert d_ping is not None
            assert d_ping.hitl is None


# ---------------------------------------------------------------------------
# mcp_pool annotation extraction
# ---------------------------------------------------------------------------


def test_extract_hitl_annotation_picks_up_aai_key() -> None:
    from ai_assistant_client.mcp_pool import _extract_hitl_annotation

    fake_tool = SimpleNamespace(
        annotations=SimpleNamespace(aai={"requires_confirmation": True})
    )
    assert _extract_hitl_annotation(fake_tool) == {"requires_confirmation": True}


def test_extract_hitl_annotation_returns_none_when_not_required() -> None:
    from ai_assistant_client.mcp_pool import _extract_hitl_annotation

    fake_tool = SimpleNamespace(
        annotations=SimpleNamespace(aai={"requires_confirmation": False})
    )
    assert _extract_hitl_annotation(fake_tool) is None


def test_extract_hitl_annotation_handles_missing_attribute() -> None:
    from ai_assistant_client.mcp_pool import _extract_hitl_annotation

    fake_tool = SimpleNamespace()  # no annotations at all
    assert _extract_hitl_annotation(fake_tool) is None


def test_extract_hitl_annotation_handles_dict_annotations() -> None:
    """Hand-rolled wrappers in tests sometimes pass a dict instead of a
    pydantic model.  The extractor should accept both."""
    from ai_assistant_client.mcp_pool import _extract_hitl_annotation

    fake_tool = SimpleNamespace(
        annotations={"aai": {"requires_confirmation": True}}
    )
    assert _extract_hitl_annotation(fake_tool) == {"requires_confirmation": True}
