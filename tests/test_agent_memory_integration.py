"""End-to-end: the agent loop surfaces memory meta-tools to the
LLM and dispatches them inline with the agent's ``user_id``
context.  Covers the user-visible payoff of slice #7 (storage)
+ this PR (agent wiring).
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest

from ai_assistant_client.agent import AgentRunConfig, run_agent
from ai_assistant_client.discovery import ProgressiveToolRegistry
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
from ai_assistant_client.memory_meta import (
    MEMORY_FORGET,
    MEMORY_RECALL,
    MEMORY_REMEMBER,
    MEMORY_UPDATE,
)
from ai_assistant_client.persistence import LocalMemoryStore


class _StubProvider(LLMProvider):
    """Records the tool-catalog it was called with on each turn so
    tests can assert what the LLM actually saw."""

    def __init__(self, scripted: list[list[NormalizedEvent]]) -> None:
        self._scripted = list(scripted)
        self.tool_calls_seen: list[list[dict[str, Any]]] = []

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
        self.tool_calls_seen.append(tools)
        events = (
            self._scripted.pop(0)
            if self._scripted
            else [MessageStop(stop_reason="end_turn")]
        )
        for ev in events:
            yield ev


def _tool_use(
    *, tool_id: str, tool_name: str, tool_input_json: str = "{}"
) -> list[NormalizedEvent]:
    return [
        ToolUseStart(index=0, id=tool_id, name=tool_name),
        ToolInputDelta(index=0, partial_json=tool_input_json),
        ToolUseStop(index=0),
        MessageStop(stop_reason="tool_use"),
    ]


def _text(t: str) -> list[NormalizedEvent]:
    return [TextDelta(text=t), MessageStop(stop_reason="end_turn")]


# ---------------------------------------------------------------------------
# Tool-catalog surfacing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_tools_surfaced_only_when_both_store_and_user_id_set() -> None:
    """The catalog the LLM sees must include the three memory
    meta-tools when (and only when) both ``memory_store`` and
    ``user_id`` are wired."""
    provider = _StubProvider([_text("ok")])
    registry = ProgressiveToolRegistry([])
    store = LocalMemoryStore()

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("not expected")

    async for _ in run_agent(
        user_message="hi",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        memory_store=store,
        user_id="alice",
    ):
        pass

    tools_seen = provider.tool_calls_seen[-1]
    names = {t["name"] for t in tools_seen}
    assert {
        MEMORY_RECALL,
        MEMORY_REMEMBER,
        MEMORY_UPDATE,
        MEMORY_FORGET,
    }.issubset(names)


@pytest.mark.asyncio
async def test_memory_tools_hidden_when_user_id_missing() -> None:
    """A store without a user_id is a no-op — the tools stay
    invisible so a misconfig can't expose the recall surface
    without isolation."""
    provider = _StubProvider([_text("ok")])
    registry = ProgressiveToolRegistry([])
    store = LocalMemoryStore()

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("not expected")

    async for _ in run_agent(
        user_message="hi",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        memory_store=store,
        # No user_id.
    ):
        pass

    names = {t["name"] for t in provider.tool_calls_seen[-1]}
    assert MEMORY_RECALL not in names
    assert MEMORY_REMEMBER not in names


# ---------------------------------------------------------------------------
# End-to-end dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_can_remember_then_recall() -> None:
    """The agent dispatches LLM-issued memory_remember / memory_recall
    calls inline and the data round-trips through the store."""
    provider = _StubProvider(
        [
            _tool_use(
                tool_id="tu_1",
                tool_name=MEMORY_REMEMBER,
                tool_input_json='{"key": "role", "value": "data scientist"}',
            ),
            _tool_use(
                tool_id="tu_2",
                tool_name=MEMORY_RECALL,
                tool_input_json="{}",
            ),
            _text("done"),
        ]
    )
    registry = ProgressiveToolRegistry([])
    store = LocalMemoryStore()

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("memory tools should not reach the dispatcher")

    tool_results: list[dict[str, Any]] = []
    async for event in run_agent(
        user_message="remember my role and recall it",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        memory_store=store,
        user_id="alice",
    ):
        if event.type == "tool_result":
            tool_results.append(event.data)

    # First result is from memory_remember; second from memory_recall.
    assert len(tool_results) == 2
    remember_payload = json.loads(tool_results[0]["content"])
    assert "memory_id" in remember_payload
    recalled = json.loads(tool_results[1]["content"])
    assert recalled[0]["key"] == "role"
    assert recalled[0]["value"] == "data scientist"


@pytest.mark.asyncio
async def test_llm_supplied_user_id_is_ignored() -> None:
    """Security: an LLM tool-use that includes ``user_id`` in
    arguments must NOT escape the agent's user-id context.
    A jailbreak attempt to write as 'admin' lands in the
    caller's bucket instead."""
    provider = _StubProvider(
        [
            _tool_use(
                tool_id="tu_1",
                tool_name=MEMORY_REMEMBER,
                tool_input_json=(
                    '{"key": "evil", "value": "x", "user_id": "admin"}'
                ),
            ),
            _text("done"),
        ]
    )
    registry = ProgressiveToolRegistry([])
    store = LocalMemoryStore()

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("not expected")

    async for _ in run_agent(
        user_message="x",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        memory_store=store,
        user_id="alice",
    ):
        pass

    # Stored under alice, NOT admin.
    assert await store.list(user_id="admin") == []
    alice = await store.list(user_id="alice")
    assert len(alice) == 1


@pytest.mark.asyncio
async def test_memory_recall_only_sees_caller_user() -> None:
    """End-to-end isolation: alice's agent run can only see
    alice's memories, even if bob has memories in the same
    store."""
    store = LocalMemoryStore()
    await store.add(user_id="alice", key="a", value=1)
    await store.add(user_id="bob", key="b", value=2)

    provider = _StubProvider(
        [
            _tool_use(
                tool_id="tu_1",
                tool_name=MEMORY_RECALL,
                tool_input_json="{}",
            ),
            _text("done"),
        ]
    )
    registry = ProgressiveToolRegistry([])

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("not expected")

    tool_results: list[str] = []
    async for event in run_agent(
        user_message="recall",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        memory_store=store,
        user_id="alice",
    ):
        if event.type == "tool_result":
            tool_results.append(event.data["content"])

    recalled = json.loads(tool_results[0])
    keys = {r["key"] for r in recalled}
    assert keys == {"a"}  # bob's memory not visible


@pytest.mark.asyncio
async def test_llm_can_update_existing_memory() -> None:
    """An LLM-issued ``memory_update`` lands in the right user's
    store and the tool_result contains the updated record."""
    store = LocalMemoryStore()
    rec = await store.add(
        user_id="alice", key="role", value="data scientist"
    )

    provider = _StubProvider(
        [
            _tool_use(
                tool_id="tu_1",
                tool_name=MEMORY_UPDATE,
                tool_input_json=(
                    '{"memory_id": "' + rec.memory_id + '", '
                    '"value": "ML engineer"}'
                ),
            ),
            _text("done"),
        ]
    )
    registry = ProgressiveToolRegistry([])

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("not expected")

    tool_results: list[str] = []
    async for event in run_agent(
        user_message="update my role",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        memory_store=store,
        user_id="alice",
    ):
        if event.type == "tool_result":
            tool_results.append(event.data["content"])

    payload = json.loads(tool_results[0])
    assert payload["value"] == "ML engineer"
    assert payload["key"] == "role"  # preserved

    # Persisted in the store too.
    again = await store.get(user_id="alice", memory_id=rec.memory_id)
    assert again.value == "ML engineer"


@pytest.mark.asyncio
async def test_memory_update_cannot_cross_users_via_agent() -> None:
    """End-to-end: even if Bob's agent run somehow knows Alice's
    memory_id, the update is rejected with 'not found'."""
    store = LocalMemoryStore()
    alice_rec = await store.add(user_id="alice", key="role", value="dev")

    provider = _StubProvider(
        [
            _tool_use(
                tool_id="tu_1",
                tool_name=MEMORY_UPDATE,
                tool_input_json=(
                    '{"memory_id": "' + alice_rec.memory_id + '", '
                    '"value": "hacked"}'
                ),
            ),
            _text("done"),
        ]
    )
    registry = ProgressiveToolRegistry([])

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("not expected")

    tool_results: list[str] = []
    async for event in run_agent(
        user_message="try to update someone else's memory",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        memory_store=store,
        user_id="bob",
    ):
        if event.type == "tool_result":
            tool_results.append(event.data["content"])

    assert "not found" in tool_results[0].lower()
    # Alice's memory unchanged.
    again = await store.get(user_id="alice", memory_id=alice_rec.memory_id)
    assert again.value == "dev"
