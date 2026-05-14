"""Tests for the conversation-store backends + factory.

Each backend goes through the same scenario set so a future SQL
backend can adopt the same list and prove parity with the in-tree
backends.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_assistant_client.persistence import (
    ConversationStore,
    FileConversationStore,
    InMemoryConversationStore,
    make_conversation_store,
)
from ai_assistant_client.persistence.factory import (
    CONVERSATION_BACKEND_ENV,
    CONVERSATION_DIR_ENV,
)


@pytest.fixture(params=["memory", "file"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> ConversationStore:
    if request.param == "memory":
        return InMemoryConversationStore()
    return FileConversationStore(tmp_path / "conversations")


# ---------------------------------------------------------------------------
# Backend-agnostic scenarios
# ---------------------------------------------------------------------------


async def test_round_trip_simple_turns(store: ConversationStore) -> None:
    await store.append("conv-1", {"role": "user", "content": "hello"})
    await store.append(
        "conv-1",
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    )

    log = await store.read("conv-1")
    assert log == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]


async def test_round_trip_tool_blocks(store: ConversationStore) -> None:
    """Tool-use / tool-result block shapes — the messy Anthropic-
    block form the agent actually produces — must round-trip
    faithfully."""
    msgs = [
        {"role": "user", "content": "look up pet 7"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Looking it up."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "getPetById",
                    "input": {"petId": 7},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": '{"name":"Rex"}',
                }
            ],
        },
    ]
    for m in msgs:
        await store.append("conv-2", m)

    log = await store.read("conv-2")
    assert log == msgs


async def test_read_unknown_returns_empty(store: ConversationStore) -> None:
    """Unknown conversation id is the common 'fresh chat' case —
    return an empty list rather than raising, so callers can use
    ``store.read(id) or []`` patterns cleanly."""
    assert await store.read("never-written") == []


async def test_list_conversations_returns_written_ids(
    store: ConversationStore,
) -> None:
    await store.append("conv-a", {"role": "user", "content": "x"})
    await store.append("conv-b", {"role": "user", "content": "y"})
    convs = await store.list_conversations()
    assert set(convs) == {"conv-a", "conv-b"}


async def test_read_returns_copy_not_live_reference(
    store: ConversationStore,
) -> None:
    """A caller iterating the returned list must not race with
    subsequent ``append`` calls."""
    await store.append("conv-3", {"role": "user", "content": "a"})
    log = await store.read("conv-3")
    await store.append("conv-3", {"role": "user", "content": "b"})
    # First read sees one entry; the new append landed on the
    # store but not on the already-returned snapshot.
    assert len(log) == 1


# ---------------------------------------------------------------------------
# File-backend-specific paths
# ---------------------------------------------------------------------------


async def test_file_backend_rejects_traversal_ids(tmp_path: Path) -> None:
    fs = FileConversationStore(tmp_path / "conversations")
    for bad in ("../etc/passwd", "a/b", "a\\b", ".hidden", ""):
        with pytest.raises(ValueError):
            await fs.append(bad, {"role": "user", "content": "x"})


async def test_file_backend_survives_corrupt_trailing_line(
    tmp_path: Path,
) -> None:
    fs = FileConversationStore(tmp_path / "conversations")
    await fs.append("conv-1", {"role": "user", "content": "hi"})
    path = tmp_path / "conversations" / "conv-1.jsonl"
    with path.open("a", encoding="utf-8") as f:
        # Truncated trailing line (crash mid-write).
        f.write('{"role": "assistant", "content":')

    log = await fs.read("conv-1")
    assert log == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_defaults_to_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONVERSATION_BACKEND_ENV, raising=False)
    assert isinstance(make_conversation_store(), InMemoryConversationStore)


def test_factory_honors_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(CONVERSATION_BACKEND_ENV, "file")
    monkeypatch.setenv(CONVERSATION_DIR_ENV, str(tmp_path / "c"))
    assert isinstance(make_conversation_store(), FileConversationStore)


def test_factory_explicit_args_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONVERSATION_BACKEND_ENV, "file")
    assert isinstance(
        make_conversation_store(kind="memory"), InMemoryConversationStore
    )


def test_factory_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown conversation backend"):
        make_conversation_store(kind="redis-cluster")
