"""Agent loop tests with a stubbed LLMProvider.

The provider stub yields normalized events directly, so these tests
exercise the agent loop without touching any vendor SDK.  Provider
adapters (Anthropic / OpenAI / Gemini / Bedrock) are tested
separately for their translation logic.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from ai_assistant_client.agent import AgentRunConfig, run_agent
from ai_assistant_client.discovery import (
    ProgressiveToolRegistry,
    RemoteToolDescriptor,
)
from ai_assistant_client.llm import (
    LLMProvider,
    MessageStop,
    NormalizedEvent,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    ToolUseStop,
)


# ---------------------------------------------------------------------------
# Stub provider.
# ---------------------------------------------------------------------------


class _StubProvider(LLMProvider):
    """Replays a scripted sequence of normalized event lists, one per
    ``stream_turn`` call."""

    def __init__(self, scripted: list[list[NormalizedEvent]]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    async def stream_turn(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[NormalizedEvent]:
        self.calls.append(
            {
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "tools": tools,
                "messages": [m for m in messages],
            }
        )
        events = (
            self._scripted.pop(0)
            if self._scripted
            else [MessageStop(stop_reason="end_turn")]
        )
        for ev in events:
            yield ev


def _text_only_turn(text: str) -> list[NormalizedEvent]:
    return [TextDelta(text=text), MessageStop(stop_reason="end_turn")]


def _tool_use_turn(
    *, tool_id: str, tool_name: str, tool_input_json: str
) -> list[NormalizedEvent]:
    return [
        ToolUseStart(index=0, id=tool_id, name=tool_name),
        ToolInputDelta(index=0, partial_json=tool_input_json),
        ToolUseStop(index=0),
        MessageStop(stop_reason="tool_use"),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_only_turn_completes_in_one_iteration() -> None:
    provider = _StubProvider([_text_only_turn("hello world")])
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
        provider=provider,
        config=AgentRunConfig(),
    ):
        events.append(event)

    text_events = [e for e in events if e.type == "text_delta"]
    assert "".join(e.data["text"] for e in text_events) == "hello world"
    assert events[-1].type == "turn_complete"
    assert events[-1].data["stop_reason"] == "end_turn"
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

    provider = _StubProvider(
        [
            _tool_use_turn(
                tool_id="tu_1",
                tool_name="tool_search",
                tool_input_json='{"query": "pet"}',
            ),
            _text_only_turn("done"),
        ]
    )

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("Meta tools should not reach dispatcher")

    events = []
    async for event in run_agent(
        user_message="find me a pet",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
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

    provider = _StubProvider(
        [
            _tool_use_turn(
                tool_id="tu_load",
                tool_name="tool_load",
                tool_input_json='{"names": ["get_pet"]}',
            ),
            _tool_use_turn(
                tool_id="tu_call",
                tool_name="get_pet",
                tool_input_json='{"id": 42}',
            ),
            _text_only_turn("Got pet 42."),
        ]
    )

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
        provider=provider,
        config=AgentRunConfig(),
    ):
        events.append(event)

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

    provider = _StubProvider(
        [
            _tool_use_turn(
                tool_id="tu_x", tool_name="broken_tool", tool_input_json="{}"
            ),
            _text_only_turn("ok"),
        ]
    )

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise RuntimeError("upstream is down")

    events = []
    async for event in run_agent(
        user_message="try",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
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

    scripted = [
        _tool_use_turn(
            tool_id=f"tu_{i}", tool_name="loop_tool", tool_input_json="{}"
        )
        for i in range(20)
    ]
    provider = _StubProvider(scripted)

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        return "round trip"

    events = []
    async for event in run_agent(
        user_message="forever",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(max_tool_iterations=3),
    ):
        events.append(event)

    final = [e for e in events if e.type == "turn_complete"][0]
    assert final.data["stop_reason"] == "max_iterations"
    assert final.data["iterations"] == 3
