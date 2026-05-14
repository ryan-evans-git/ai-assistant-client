"""Unit tests for the memory meta-tool dispatch layer.

End-to-end agent integration (LLM → tool_use → tool_result) is
covered by ``test_agent_memory_integration.py``; this file
exercises ``handle_memory_meta_call`` directly so the protocol
contract is locked in independently of the agent loop.
"""

from __future__ import annotations

import json

from ai_assistant_client.memory_meta import (
    MEMORY_FORGET,
    MEMORY_RECALL,
    MEMORY_REMEMBER,
    MEMORY_UPDATE,
    MEMORY_META_TOOL_NAMES,
    handle_memory_meta_call,
    is_memory_meta_tool,
    memory_meta_tool_schemas,
)
from ai_assistant_client.persistence import LocalMemoryStore


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def test_schemas_cover_the_four_meta_tools() -> None:
    schemas = memory_meta_tool_schemas()
    names = {s["name"] for s in schemas}
    assert names == MEMORY_META_TOOL_NAMES == {
        MEMORY_RECALL,
        MEMORY_REMEMBER,
        MEMORY_UPDATE,
        MEMORY_FORGET,
    }


def test_update_schema_requires_memory_id_and_value() -> None:
    schema = next(
        s for s in memory_meta_tool_schemas() if s["name"] == MEMORY_UPDATE
    )
    assert set(schema["input_schema"]["required"]) == {"memory_id", "value"}


def test_remember_schema_requires_key_and_value() -> None:
    schema = next(
        s for s in memory_meta_tool_schemas() if s["name"] == MEMORY_REMEMBER
    )
    assert "key" in schema["input_schema"]["required"]
    assert "value" in schema["input_schema"]["required"]


def test_recall_schema_has_optional_tags() -> None:
    schema = next(
        s for s in memory_meta_tool_schemas() if s["name"] == MEMORY_RECALL
    )
    assert "required" not in schema["input_schema"] or (
        "tags" not in schema["input_schema"]["required"]
    )


def test_is_memory_meta_tool() -> None:
    assert is_memory_meta_tool("memory_recall")
    assert is_memory_meta_tool("memory_remember")
    assert is_memory_meta_tool("memory_update")
    assert is_memory_meta_tool("memory_forget")
    assert not is_memory_meta_tool("tool_search")
    assert not is_memory_meta_tool("some_remote_tool")


# ---------------------------------------------------------------------------
# Dispatch — happy paths
# ---------------------------------------------------------------------------


async def test_remember_then_recall_round_trip() -> None:
    store = LocalMemoryStore()
    remember_out = await handle_memory_meta_call(
        MEMORY_REMEMBER,
        {"key": "role", "value": "data scientist", "tags": ["work"]},
        store=store,
        user_id="alice",
    )
    parsed = json.loads(remember_out)
    assert "memory_id" in parsed

    recall_out = await handle_memory_meta_call(
        MEMORY_RECALL,
        {},
        store=store,
        user_id="alice",
    )
    recalled = json.loads(recall_out)
    assert len(recalled) == 1
    assert recalled[0]["key"] == "role"
    assert recalled[0]["value"] == "data scientist"
    assert recalled[0]["tags"] == ["work"]


async def test_recall_tag_filter() -> None:
    store = LocalMemoryStore()
    await store.add(user_id="alice", key="a", value=1, tags=("work",))
    await store.add(user_id="alice", key="b", value=2, tags=("home",))
    out = await handle_memory_meta_call(
        MEMORY_RECALL,
        {"tags": ["work"]},
        store=store,
        user_id="alice",
    )
    recalled = json.loads(out)
    assert {r["key"] for r in recalled} == {"a"}


async def test_forget_round_trip() -> None:
    store = LocalMemoryStore()
    rec = await store.add(user_id="alice", key="x", value=1)
    out = await handle_memory_meta_call(
        MEMORY_FORGET,
        {"memory_id": rec.memory_id},
        store=store,
        user_id="alice",
    )
    assert "Forgot" in out
    # Memory is gone.
    rest = await store.list(user_id="alice")
    assert rest == []


# ---------------------------------------------------------------------------
# Dispatch — security / error paths
# ---------------------------------------------------------------------------


async def test_remember_user_id_in_arguments_is_ignored() -> None:
    """Security: the LLM can pass a 'user_id' field in arguments
    but the dispatcher closes over user_id from the context.
    A jailbreak attempt that tries to write as 'admin' must
    land in the *caller's* bucket."""
    store = LocalMemoryStore()
    await handle_memory_meta_call(
        MEMORY_REMEMBER,
        {"key": "evil", "value": "x", "user_id": "admin"},
        store=store,
        user_id="alice",
    )
    # The record landed under 'alice', not 'admin'.
    assert await store.list(user_id="admin") == []
    alice_recs = await store.list(user_id="alice")
    assert len(alice_recs) == 1


async def test_forget_cross_user_returns_not_found_string() -> None:
    """Bob tries to forget Alice's memory — gets the same
    'not found' string as a non-existent id, no leak."""
    store = LocalMemoryStore()
    rec = await store.add(user_id="alice", key="x", value=1)

    out = await handle_memory_meta_call(
        MEMORY_FORGET,
        {"memory_id": rec.memory_id},
        store=store,
        user_id="bob",
    )
    assert "not found" in out.lower()
    # Alice's memory is intact.
    alice = await store.list(user_id="alice")
    assert len(alice) == 1


async def test_recall_isolates_users() -> None:
    store = LocalMemoryStore()
    await store.add(user_id="alice", key="a", value=1)
    await store.add(user_id="bob", key="b", value=2)

    alice_out = await handle_memory_meta_call(
        MEMORY_RECALL, {}, store=store, user_id="alice"
    )
    bob_out = await handle_memory_meta_call(
        MEMORY_RECALL, {}, store=store, user_id="bob"
    )

    alice_data = json.loads(alice_out)
    bob_data = json.loads(bob_out)
    assert {r["key"] for r in alice_data} == {"a"}
    assert {r["key"] for r in bob_data} == {"b"}


async def test_remember_missing_key_returns_error_string() -> None:
    store = LocalMemoryStore()
    out = await handle_memory_meta_call(
        MEMORY_REMEMBER, {"value": "x"}, store=store, user_id="alice"
    )
    assert "key" in out.lower()


async def test_remember_missing_value_returns_error_string() -> None:
    store = LocalMemoryStore()
    out = await handle_memory_meta_call(
        MEMORY_REMEMBER, {"key": "x"}, store=store, user_id="alice"
    )
    assert "value" in out.lower()


async def test_forget_missing_id_returns_error_string() -> None:
    store = LocalMemoryStore()
    out = await handle_memory_meta_call(
        MEMORY_FORGET, {}, store=store, user_id="alice"
    )
    assert "memory_id" in out.lower()


async def test_unknown_tool_name_returns_error_string() -> None:
    store = LocalMemoryStore()
    out = await handle_memory_meta_call(
        "memory_explode", {}, store=store, user_id="alice"
    )
    assert "unknown" in out.lower()


# ---------------------------------------------------------------------------
# memory_update — dispatch
# ---------------------------------------------------------------------------


async def test_update_replaces_value_and_preserves_metadata() -> None:
    """The key, tags, and created_at of the original record must
    survive — only value + updated_at change.  This is what
    makes update preferable to forget+remember."""
    store = LocalMemoryStore()
    original = await store.add(
        user_id="alice",
        key="role",
        value="data scientist",
        tags=("work",),
    )

    out = await handle_memory_meta_call(
        MEMORY_UPDATE,
        {"memory_id": original.memory_id, "value": "ML engineer"},
        store=store,
        user_id="alice",
    )
    payload = json.loads(out)
    assert payload["memory_id"] == original.memory_id
    assert payload["key"] == "role"           # preserved
    assert payload["tags"] == ["work"]        # preserved
    assert payload["value"] == "ML engineer"  # changed
    assert payload["created_at"] == original.created_at  # preserved
    assert payload["updated_at"] >= original.updated_at  # bumped


async def test_update_unknown_id_returns_not_found_string() -> None:
    store = LocalMemoryStore()
    out = await handle_memory_meta_call(
        MEMORY_UPDATE,
        {"memory_id": "mem_does_not_exist", "value": 1},
        store=store,
        user_id="alice",
    )
    assert "not found" in out.lower()


async def test_update_cross_user_returns_not_found_string() -> None:
    """Bob can't update Alice's memory — same 'not found' string
    as a non-existent id, preserving the anti-enumeration
    contract."""
    store = LocalMemoryStore()
    rec = await store.add(user_id="alice", key="x", value=1)

    out = await handle_memory_meta_call(
        MEMORY_UPDATE,
        {"memory_id": rec.memory_id, "value": 999},
        store=store,
        user_id="bob",
    )
    assert "not found" in out.lower()
    # Alice's memory unchanged.
    again = await store.get(user_id="alice", memory_id=rec.memory_id)
    assert again.value == 1


async def test_update_user_id_in_arguments_is_ignored() -> None:
    """Mirror of the remember-side jailbreak test: an LLM-supplied
    user_id in the args must not escape the agent's user_id
    context.  The update lands in Alice's record, not admin's."""
    store = LocalMemoryStore()
    rec = await store.add(user_id="alice", key="x", value=1)

    await handle_memory_meta_call(
        MEMORY_UPDATE,
        {
            "memory_id": rec.memory_id,
            "value": 999,
            "user_id": "admin",
        },
        store=store,
        user_id="alice",
    )
    again = await store.get(user_id="alice", memory_id=rec.memory_id)
    assert again.value == 999
    # admin has nothing.
    assert await store.list(user_id="admin") == []


async def test_update_missing_memory_id_returns_error_string() -> None:
    store = LocalMemoryStore()
    out = await handle_memory_meta_call(
        MEMORY_UPDATE, {"value": 1}, store=store, user_id="alice"
    )
    assert "memory_id" in out.lower()


async def test_update_missing_value_returns_error_string() -> None:
    """Missing ``value`` is a different failure from missing
    ``memory_id`` — we want both to surface clearly so the
    LLM can fix its call."""
    store = LocalMemoryStore()
    out = await handle_memory_meta_call(
        MEMORY_UPDATE,
        {"memory_id": "mem_x"},
        store=store,
        user_id="alice",
    )
    assert "value" in out.lower()


async def test_update_value_can_be_falsy_zero_or_empty() -> None:
    """``value: 0`` and ``value: ""`` are legitimate values, not
    "missing".  Make sure the validator distinguishes "absent
    key" from "falsy value"."""
    store = LocalMemoryStore()
    rec = await store.add(user_id="alice", key="count", value=10)
    out = await handle_memory_meta_call(
        MEMORY_UPDATE,
        {"memory_id": rec.memory_id, "value": 0},
        store=store,
        user_id="alice",
    )
    payload = json.loads(out)
    assert payload["value"] == 0
