"""Agent loop HITL gating tests.

Drive the agent through a tool_use for a HITL-flagged tool and
verify it pauses, emits the expected events, and routes the
decision into the eventual tool_result.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from ai_assistant_client.agent import AgentEvent, AgentRunConfig, run_agent
from ai_assistant_client.confirmation_store import ConfirmationOutcome
from ai_assistant_client.discovery import (
    ProgressiveToolRegistry,
    RemoteToolDescriptor,
)
from ai_assistant_client.llm import (
    CacheHint,
    LLMProvider,
    MessageStop,
    NormalizedEvent,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    ToolUseStop,
)


# ---------------------------------------------------------------------------
# Stub provider that emits a tool_use for a single tool, then end_turn
# on the second turn so the loop terminates after the result is returned.
# ---------------------------------------------------------------------------


def _tool_use_turn(*, tool_id: str, tool_name: str, args_json: str) -> list[NormalizedEvent]:
    return [
        ToolUseStart(index=0, id=tool_id, name=tool_name),
        ToolInputDelta(index=0, partial_json=args_json),
        ToolUseStop(index=0),
        MessageStop(stop_reason="tool_use"),
    ]


def _end_turn() -> list[NormalizedEvent]:
    return [TextDelta(text="ok"), MessageStop(stop_reason="end_turn")]


class _StubProvider(LLMProvider):
    def __init__(self, scripted: list[list[NormalizedEvent]]) -> None:
        self._scripted = list(scripted)

    async def stream_turn(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        cache_hint: CacheHint | None = None,
    ) -> AsyncIterator[NormalizedEvent]:
        events = self._scripted.pop(0) if self._scripted else _end_turn()
        for ev in events:
            yield ev


def _registry_with_hitl_tool(name: str) -> ProgressiveToolRegistry:
    return ProgressiveToolRegistry(
        [
            RemoteToolDescriptor(
                name=name,
                description=f"{name} description",
                input_schema={"type": "object", "properties": {}},
                hitl={"requires_confirmation": True, "message": "Do it?"},
            )
        ]
    )


def _registry_with_plain_tool(name: str) -> ProgressiveToolRegistry:
    return ProgressiveToolRegistry(
        [
            RemoteToolDescriptor(
                name=name,
                description=f"{name} description",
                input_schema={"type": "object", "properties": {}},
            )
        ]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_emits_confirmation_event_and_dispatches_on_confirm() -> None:
    dispatched: list[tuple[str, dict[str, Any]]] = []

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        dispatched.append((name, arguments))
        return "tool ran"

    async def hook(payload: dict[str, Any]) -> ConfirmationOutcome:
        # Capture the request shape to assert on.
        hook.last_payload = payload  # type: ignore[attr-defined]
        return ConfirmationOutcome(decision="confirm")

    provider = _StubProvider([_tool_use_turn(
        tool_id="tu_1", tool_name="send_email", args_json='{"to":"a"}'
    )])
    history: list[dict[str, Any]] = []
    events: list[AgentEvent] = []
    async for ev in run_agent(
        user_message="please",
        history=history,
        registry=_registry_with_hitl_tool("send_email"),
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(model="x"),
        confirmation_hook=hook,
    ):
        events.append(ev)

    types = [e.type for e in events]
    assert "tool_confirmation_request" in types
    assert "tool_confirmation_resolved" in types
    assert "tool_result" in types
    # Tool was dispatched after the confirm.
    assert dispatched == [("send_email", {"to": "a"})]
    # Confirmation request payload matches the UI's shape.
    payload = hook.last_payload  # type: ignore[attr-defined]
    assert payload["tool_use_id"] == "tu_1"
    assert payload["tool_name"] == "send_email"
    assert payload["tool_input"] == {"to": "a"}
    assert payload["message"] == "Do it?"
    assert "request_id" in payload


@pytest.mark.asyncio
async def test_agent_decline_skips_dispatch() -> None:
    dispatched: list[Any] = []

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        dispatched.append(name)
        return "should not happen"

    async def hook(payload: dict[str, Any]) -> ConfirmationOutcome:
        return ConfirmationOutcome(decision="decline", note="not now")

    provider = _StubProvider([_tool_use_turn(
        tool_id="tu_2", tool_name="delete_records", args_json='{}'
    )])
    events: list[AgentEvent] = []
    async for ev in run_agent(
        user_message="delete",
        history=[],
        registry=_registry_with_hitl_tool("delete_records"),
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(model="x"),
        confirmation_hook=hook,
    ):
        events.append(ev)

    assert dispatched == []  # never called
    result_evs = [e for e in events if e.type == "tool_result"]
    assert len(result_evs) == 1
    assert result_evs[0].data["is_error"] is True
    assert "not now" in result_evs[0].data["content"]


@pytest.mark.asyncio
async def test_agent_timeout_treated_as_decline_by_default() -> None:
    dispatched: list[Any] = []

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        dispatched.append(name)
        return "x"

    async def hook(payload: dict[str, Any]) -> ConfirmationOutcome:
        # Block forever — timeout fires.
        await asyncio.sleep(10)
        return ConfirmationOutcome(decision="confirm")  # pragma: no cover

    provider = _StubProvider([_tool_use_turn(
        tool_id="tu_3", tool_name="x", args_json='{}'
    )])
    registry = ProgressiveToolRegistry(
        [
            RemoteToolDescriptor(
                name="x",
                description="d",
                input_schema={"type": "object", "properties": {}},
                hitl={"requires_confirmation": True, "timeout_seconds": 0.05},
            )
        ]
    )
    events: list[AgentEvent] = []
    async for ev in run_agent(
        user_message="m",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(model="x"),
        confirmation_hook=hook,
    ):
        events.append(ev)

    assert dispatched == []
    resolved = next(e for e in events if e.type == "tool_confirmation_resolved")
    assert resolved.data["decision"] == "decline"
    assert "timed out" in resolved.data["note"]


@pytest.mark.asyncio
async def test_agent_no_hook_auto_confirms_with_warning() -> None:
    """A host that hasn't wired a confirmation hook shouldn't deadlock.
    The agent emits the request event (so a host that *did* wire the
    UI sees the modal) and falls back to confirm."""
    dispatched: list[Any] = []

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        dispatched.append(name)
        return "ok"

    provider = _StubProvider([_tool_use_turn(
        tool_id="tu_4", tool_name="x", args_json='{}'
    )])
    events: list[AgentEvent] = []
    async for ev in run_agent(
        user_message="m",
        history=[],
        registry=_registry_with_hitl_tool("x"),
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(model="x"),
        confirmation_hook=None,
    ):
        events.append(ev)

    assert dispatched == ["x"]
    types = [e.type for e in events]
    assert "tool_confirmation_request" in types  # event still fires


@pytest.mark.asyncio
async def test_agent_plain_tool_skips_gate_entirely() -> None:
    """Tools without HITL metadata should dispatch immediately; no
    confirmation events should appear."""
    dispatched: list[Any] = []

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        dispatched.append(name)
        return "ok"

    async def hook(payload: dict[str, Any]) -> ConfirmationOutcome:
        raise AssertionError("hook should not fire for plain tools")

    provider = _StubProvider([_tool_use_turn(
        tool_id="tu_5", tool_name="echo", args_json='{}'
    )])
    events: list[AgentEvent] = []
    async for ev in run_agent(
        user_message="m",
        history=[],
        registry=_registry_with_plain_tool("echo"),
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(model="x"),
        confirmation_hook=hook,
    ):
        events.append(ev)

    assert dispatched == ["echo"]
    types = [e.type for e in events]
    assert "tool_confirmation_request" not in types


@pytest.mark.asyncio
async def test_agent_confirm_with_note_prepends_to_tool_result() -> None:
    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        return "the result"

    async def hook(payload: dict[str, Any]) -> ConfirmationOutcome:
        return ConfirmationOutcome(decision="confirm", note="approved by alice")

    provider = _StubProvider([_tool_use_turn(
        tool_id="tu_6", tool_name="send_email", args_json='{}'
    )])
    events: list[AgentEvent] = []
    async for ev in run_agent(
        user_message="m",
        history=[],
        registry=_registry_with_hitl_tool("send_email"),
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(model="x"),
        confirmation_hook=hook,
    ):
        events.append(ev)

    result = next(e for e in events if e.type == "tool_result")
    assert "approved by alice" in result.data["content"]
    assert "the result" in result.data["content"]
