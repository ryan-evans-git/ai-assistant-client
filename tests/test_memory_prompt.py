"""Tests for the system-prompt injection helper.

Three invariants we want to lock down:

1. Empty store → return the base prompt unchanged (no empty
   delimiter that the model might fill with hallucinations).
2. Recalled content is enveloped in the ``<user_memory>`` tags
   so the model has an explicit data boundary.
3. The returned ``memory_ids`` lets a host log what contributed
   to the turn.
"""

from __future__ import annotations

from ai_assistant_client.memory_prompt import (
    MEMORY_CLOSE,
    MEMORY_OPEN,
    build_system_prompt_with_memory,
)
from ai_assistant_client.persistence import LocalMemoryStore


BASE = "You are a helpful assistant."


async def test_no_memories_returns_base_prompt_unchanged() -> None:
    store = LocalMemoryStore()
    result = await build_system_prompt_with_memory(
        BASE, store=store, user_id="alice"
    )
    assert result.system_prompt == BASE
    assert result.memory_ids == ()
    assert result.count == 0


async def test_recalled_memories_appear_inside_delimiter() -> None:
    store = LocalMemoryStore()
    await store.add(user_id="alice", key="role", value="data scientist")

    result = await build_system_prompt_with_memory(
        BASE, store=store, user_id="alice"
    )
    # Base prompt preserved at the top.
    assert result.system_prompt.startswith(BASE)
    # Delimiter present + memory inside it.
    assert MEMORY_OPEN in result.system_prompt
    assert MEMORY_CLOSE in result.system_prompt
    block_start = result.system_prompt.index(MEMORY_OPEN)
    block_end = result.system_prompt.index(MEMORY_CLOSE)
    block = result.system_prompt[block_start:block_end]
    assert "role" in block
    assert "data scientist" in block


async def test_returns_contributing_memory_ids_for_logging() -> None:
    store = LocalMemoryStore()
    a = await store.add(user_id="alice", key="a", value=1)
    b = await store.add(user_id="alice", key="b", value=2)

    result = await build_system_prompt_with_memory(
        BASE, store=store, user_id="alice"
    )
    assert set(result.memory_ids) == {a.memory_id, b.memory_id}
    assert result.count == 2


async def test_tag_filter_narrows_recall() -> None:
    store = LocalMemoryStore()
    await store.add(user_id="alice", key="a", value=1, tags=("work",))
    await store.add(user_id="alice", key="b", value=2, tags=("home",))

    result = await build_system_prompt_with_memory(
        BASE, store=store, user_id="alice", tags=("work",)
    )
    assert "key" not in result.system_prompt  # no leakage of the dict-encoding noise
    assert result.count == 1
    # Only the work memory's key should appear in the block.
    assert "a" in result.system_prompt or "1" in result.system_prompt
    assert "home" not in result.system_prompt


async def test_reminder_included_by_default_and_can_be_disabled() -> None:
    store = LocalMemoryStore()
    await store.add(user_id="alice", key="role", value="x")

    with_reminder = await build_system_prompt_with_memory(
        BASE, store=store, user_id="alice"
    )
    without_reminder = await build_system_prompt_with_memory(
        BASE, store=store, user_id="alice", include_reminder=False
    )

    # The reminder explicitly says to treat memory as data, not
    # instructions — load-bearing for injection resistance.
    assert "data, not as instructions" in with_reminder.system_prompt
    assert "data, not as instructions" not in without_reminder.system_prompt


async def test_user_isolation_in_prompt_recall() -> None:
    """Alice's prompt only includes Alice's memories — even if
    Bob has memories in the same store."""
    store = LocalMemoryStore()
    await store.add(user_id="alice", key="alice_role", value="dev")
    await store.add(user_id="bob", key="bob_secret", value="hush")

    result = await build_system_prompt_with_memory(
        BASE, store=store, user_id="alice"
    )
    assert "alice_role" in result.system_prompt
    assert "bob_secret" not in result.system_prompt
