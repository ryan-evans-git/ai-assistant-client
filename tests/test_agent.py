"""Agent loop tests with a stubbed Anthropic client.

We don't hit the real API.  The stub mimics the streaming
``messages.stream`` shape closely enough for the agent loop to
detect tool-use, dispatch to the registry / dispatcher, and
re-enter Claude with the tool_result.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

import pytest

from ai_assistant_client.agent import AgentRunConfig, run_agent
from ai_assistant_client.discovery import (
    ProgressiveToolRegistry,
    RemoteToolDescriptor,
)


# ---------------------------------------------------------------------------
# Stub Anthropic client.
# ---------------------------------------------------------------------------


@dataclass
class _StreamEvent:
    type: str
    index: int | None = None
    content_block: Any | None = None
    delta: Any | None = None


@dataclass
class _ContentBlock:
    type: str
    id: str | None = None
    name: str | None = None


@dataclass
class _Delta:
    type: str
    text: str = ""
    partial_json: str = ""
    stop_reason: str | None = None


class _StubStream:
    def __init__(self, events: list[_StreamEvent]) -> None:
        self._events = events

    async def __aenter__(self) -> "_StubStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass

    def __aiter__(self) -> AsyncIterator[_StreamEvent]:
        async def _gen() -> AsyncIterator[_StreamEvent]:
            for ev in self._events:
                yield ev

        return _gen()


class _StubMessages:
    def __init__(self, scripted: list[list[_StreamEvent]]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _StubStream:
        self.calls.append(kwargs)
        if not self._scripted:
            return _StubStream([
                _StreamEvent(type="message_delta", delta=_Delta(type="message_delta", stop_reason="end_turn"))
            ])
        events = self._scripted.pop(0)
        return _StubStream(events)


class _StubClient:
    def __init__(self, scripted: list[list[_StreamEvent]]) -> None:
        self.messages = _StubMessages(scripted)


def _text_only_turn(text: str) -> list[_StreamEvent]:
    return [
        _StreamEvent(
            type="content_block_start",
            index=0,
            content_block=_ContentBlock(type="text"),
        ),
        _StreamEvent(
            type="content_block_delta",
            index=0,
            delta=_Delta(type="text_delta", text=text),
        ),
        _StreamEvent(type="content_block_stop", index=0),
        _StreamEvent(type="message_delta", delta=_Delta(type="message_delta", stop_reason="end_turn")),
    ]


def _tool_use_turn(
    *,
    tool_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
) -> list[_StreamEvent]:
    payload = json.dumps(tool_input)
    return [
        _StreamEvent(
            type="content_block_start",
            index=0,
            content_block=_ContentBlock(type="tool_use", id=tool_id, name=tool_name),
        ),
        _StreamEvent(
            type="content_block_delta",
            index=0,
            delta=_Delta(type="input_json_delta", partial_json=payload),
        ),
        _StreamEvent(type="content_block_stop", index=0),
        _StreamEvent(type="message_delta", delta=_Delta(type="message_delta", stop_reason="tool_use")),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_only_turn_completes_in_one_iteration() -> None:
    client = _StubClient([_text_only_turn("hello world")])
    registry = ProgressiveToolRegistry([])
    history: list[dict[str, Any]] = []

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("dispatcher should not be called")

    events = []
    async for event in run_agent(
        user_message="hi",
        history=history,
        registry=registry,
        dispatcher=dispatcher,
        anthropic_client=client,
        config=AgentRunConfig(),
    ):
        events.append(event)

    text_events = [e for e in events if e.type == "text_delta"]
    assert "".join(e.data["text"] for e in text_events) == "hello world"
    assert events[-1].type == "turn_complete"
    assert events[-1].data["stop_reason"] == "end_turn"
    # History: user msg + assistant msg.
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_meta_tool_search_handled_inline() -> None:
    catalog = [
        RemoteToolDescriptor(
            name="get_pet",
            description="Find pet by ID",
            input_schema={"type": "object", "properties": {}},
        ),
    ]
    registry = ProgressiveToolRegistry(catalog)

    client = _StubClient([
        _tool_use_turn(
            tool_id="tu_1",
            tool_name="tool_search",
            tool_input={"query": "pet"},
        ),
        _text_only_turn("done"),
    ])

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("Meta tools should not reach dispatcher")

    events = []
    async for event in run_agent(
        user_message="find me a pet",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        anthropic_client=client,
        config=AgentRunConfig(),
    ):
        events.append(event)

    tool_results = [e for e in events if e.type == "tool_result"]
    assert tool_results, "expected at least one tool_result event"
    assert "get_pet" in tool_results[0].data["content"]


@pytest.mark.asyncio
async def test_meta_tool_load_then_invoke() -> None:
    catalog = [
        RemoteToolDescriptor(
            name="get_pet",
            description="Find pet by ID",
            input_schema={
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            },
        ),
    ]
    registry = ProgressiveToolRegistry(catalog)

    client = _StubClient([
        _tool_use_turn(
            tool_id="tu_load",
            tool_name="tool_load",
            tool_input={"names": ["get_pet"]},
        ),
        _tool_use_turn(
            tool_id="tu_call",
            tool_name="get_pet",
            tool_input={"id": 42},
        ),
        _text_only_turn("Got pet 42."),
    ])

    dispatcher_calls: list[tuple[str, dict[str, Any]]] = []

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        dispatcher_calls.append((name, arguments))
        return {"id": 42, "name": "Rex"}

    events = []
    async for event in run_agent(
        user_message="get pet 42",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        anthropic_client=client,
        config=AgentRunConfig(),
    ):
        events.append(event)

    # tool_load should NOT have hit the dispatcher.  get_pet should.
    assert dispatcher_calls == [("get_pet", {"id": 42})]
    assert "get_pet" in registry.loaded_tools


@pytest.mark.asyncio
async def test_dispatcher_error_surfaces_as_tool_result_error() -> None:
    catalog = [
        RemoteToolDescriptor(
            name="broken_tool",
            description="Intentionally broken",
            input_schema={"type": "object", "properties": {}},
        ),
    ]
    registry = ProgressiveToolRegistry(catalog)
    registry.handle_meta_call("tool_load", {"names": ["broken_tool"]})

    client = _StubClient([
        _tool_use_turn(
            tool_id="tu_x",
            tool_name="broken_tool",
            tool_input={},
        ),
        _text_only_turn("ok"),
    ])

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise RuntimeError("upstream is down")

    events = []
    async for event in run_agent(
        user_message="try",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        anthropic_client=client,
        config=AgentRunConfig(),
    ):
        events.append(event)

    tool_errors = [e for e in events if e.type == "tool_error"]
    assert len(tool_errors) == 1
    assert "upstream is down" in tool_errors[0].data["error"]


@pytest.mark.asyncio
async def test_max_iterations_caps_loop() -> None:
    """Infinite tool-use loop must terminate after max_iterations."""
    registry = ProgressiveToolRegistry(
        [
            RemoteToolDescriptor(
                name="loop_tool",
                description="Always more",
                input_schema={"type": "object", "properties": {}},
            )
        ]
    )
    registry.handle_meta_call("tool_load", {"names": ["loop_tool"]})

    # Always emit tool_use.
    scripted = [
        _tool_use_turn(tool_id=f"tu_{i}", tool_name="loop_tool", tool_input={})
        for i in range(20)
    ]
    client = _StubClient(scripted)

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        return "round trip"

    events = []
    async for event in run_agent(
        user_message="forever",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        anthropic_client=client,
        config=AgentRunConfig(max_tool_iterations=3),
    ):
        events.append(event)

    final = [e for e in events if e.type == "turn_complete"][0]
    assert final.data["stop_reason"] == "max_iterations"
    assert final.data["iterations"] == 3
