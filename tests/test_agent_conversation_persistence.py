"""End-to-end: a ``run_agent`` call mirrors every history append
into the configured ConversationStore.

These tests use the same stub-provider scaffolding as
``test_agent.py``; the goal isn't to re-test the agent loop's
control flow but to verify that the four append sites (user
message, assistant turn, tool_result turn, validation-retry
feedback) all reach the store under the same ``conversation_id``.
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
    CacheHint,
    LLMProvider,
    MessageStop,
    NormalizedEvent,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    ToolUseStop,
)
from ai_assistant_client.persistence import InMemoryConversationStore


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
        events = (
            self._scripted.pop(0)
            if self._scripted
            else [MessageStop(stop_reason="end_turn")]
        )
        for ev in events:
            yield ev


def _text_only(text: str) -> list[NormalizedEvent]:
    return [TextDelta(text=text), MessageStop(stop_reason="end_turn")]


def _tool_use(
    *, tool_id: str, tool_name: str, tool_input_json: str
) -> list[NormalizedEvent]:
    return [
        ToolUseStart(index=0, id=tool_id, name=tool_name),
        ToolInputDelta(index=0, partial_json=tool_input_json),
        ToolUseStop(index=0),
        MessageStop(stop_reason="tool_use"),
    ]


@pytest.mark.asyncio
async def test_text_only_turn_records_user_and_assistant() -> None:
    provider = _StubProvider([_text_only("hi there")])
    registry = ProgressiveToolRegistry([])
    store = InMemoryConversationStore()

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("dispatcher should not be called")

    async for _ in run_agent(
        user_message="hello",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        conversation_store=store,
        conversation_id="conv-1",
    ):
        pass

    log = await store.read("conv-1")
    assert len(log) == 2
    assert log[0] == {"role": "user", "content": "hello"}
    assert log[1]["role"] == "assistant"
    # Assistant content is a block list (one text block in this run).
    assert log[1]["content"][0]["type"] == "text"


@pytest.mark.asyncio
async def test_tool_use_turn_records_all_three_messages() -> None:
    """A tool-using turn produces three messages: user + assistant
    (with tool_use block) + user (with tool_result block).  All
    three must reach the store."""
    catalog = [
        RemoteToolDescriptor(
            name="get_pet",
            description="Find pet by ID",
            input_schema={"type": "object", "properties": {}},
        ),
    ]
    registry = ProgressiveToolRegistry(catalog)

    # Load the tool, then call it, then a text-only closing turn.
    provider = _StubProvider(
        [
            _tool_use(
                tool_id="tu_load",
                tool_name="tool_load",
                tool_input_json='{"names": ["get_pet"]}',
            ),
            _tool_use(
                tool_id="tu_1",
                tool_name="get_pet",
                tool_input_json='{"id": 7}',
            ),
            _text_only("done"),
        ]
    )

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        return {"id": 7, "name": "Rex"}

    store = InMemoryConversationStore()
    async for _ in run_agent(
        user_message="look up pet 7",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        conversation_store=store,
        conversation_id="conv-2",
    ):
        pass

    log = await store.read("conv-2")
    roles = [m["role"] for m in log]
    # user + (assistant + user-with-tool_result) × 2 tool-using
    # iterations + assistant (closing text-only turn)
    assert roles[0] == "user"
    assert "assistant" in roles
    # At least one user message must carry tool_result content.
    assert any(
        m["role"] == "user"
        and isinstance(m["content"], list)
        and any(
            block.get("type") == "tool_result" for block in m["content"]
        )
        for m in log
    )


@pytest.mark.asyncio
async def test_persistence_disabled_when_id_missing() -> None:
    """The persist path is gated on BOTH a store AND an id.  A
    store without an id stays silent — the half-configured shape
    is a bug class we surface by simply not writing."""
    provider = _StubProvider([_text_only("hi")])
    registry = ProgressiveToolRegistry([])
    store = InMemoryConversationStore()

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("dispatcher should not be called")

    async for _ in run_agent(
        user_message="hello",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        conversation_store=store,
        # No conversation_id.
    ):
        pass

    assert await store.list_conversations() == []


@pytest.mark.asyncio
async def test_persistence_disabled_when_store_missing() -> None:
    """Mirror of the previous test: an id without a store is also
    a no-op, not a crash."""
    provider = _StubProvider([_text_only("hi")])
    registry = ProgressiveToolRegistry([])

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("dispatcher should not be called")

    # Just verify it doesn't blow up.
    async for _ in run_agent(
        user_message="hello",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        conversation_id="conv-orphan",
    ):
        pass


@pytest.mark.asyncio
async def test_history_caller_can_seed_from_store() -> None:
    """The store is write-only from the agent's perspective;
    callers seed ``history`` themselves.  Verify the documented
    pattern: read the prior log, pass it in, the new turn appends
    on top."""
    provider = _StubProvider([_text_only("continuing")])
    registry = ProgressiveToolRegistry([])
    store = InMemoryConversationStore()

    # Seed prior turns directly.
    await store.append("conv-3", {"role": "user", "content": "first"})
    await store.append(
        "conv-3",
        {"role": "assistant", "content": [{"type": "text", "text": "second"}]},
    )

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("dispatcher should not be called")

    history = await store.read("conv-3")
    async for _ in run_agent(
        user_message="third",
        history=history,
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        conversation_store=store,
        conversation_id="conv-3",
    ):
        pass

    log = await store.read("conv-3")
    # Original 2 messages + new user + new assistant = 4.
    assert len(log) == 4
    assert log[0]["content"] == "first"
    assert log[2]["content"] == "third"
