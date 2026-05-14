"""Tests for the per-user memory backends + factory.

Each backend goes through the same scenario set so a future SQL /
async backend (planned follow-up PRs) can adopt the same list and
prove protocol parity.

The protocol-level invariants under test:

1. **Add → get → list** is a clean round trip.
2. **Cross-user isolation.**  User A can't see, update, or delete
   user B's memories — ``get`` / ``update`` / ``remove`` all
   raise the same ``KeyError`` when the user_id mismatches, so
   a probing caller can't distinguish "wrong owner" from
   "doesn't exist."
3. **Tag filtering.**  ``list(tags=...)`` returns records matching
   any of the requested tags (union semantics).
4. **forget_all.**  Erases every record for one user; returns the
   count; doesn't touch other users.  This is the GDPR-style
   "delete everything about me" path — it MUST not leave a
   tombstone, just gone.
5. **Update bumps ``updated_at`` but preserves ``created_at``.**
6. **Memory ids are server-assigned and opaque** — callers can't
   inject custom ids that could escape the filesystem layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_assistant_client.persistence import (
    FileMemoryStore,
    LocalMemoryStore,
    Memory,
    MemoryStore,
    make_memory_store,
)
from ai_assistant_client.persistence.factory import (
    MEMORY_BACKEND_ENV,
    MEMORY_DIR_ENV,
)


@pytest.fixture(params=["local", "file"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> MemoryStore:
    if request.param == "local":
        return LocalMemoryStore()
    return FileMemoryStore(tmp_path / "memories")


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


async def test_add_then_get_round_trips(store: MemoryStore) -> None:
    record = await store.add(
        user_id="alice", key="role", value="data scientist", tags=("work",)
    )
    fetched = await store.get(user_id="alice", memory_id=record.memory_id)
    assert fetched.user_id == "alice"
    assert fetched.key == "role"
    assert fetched.value == "data scientist"
    assert fetched.tags == ("work",)
    assert fetched.created_at != ""
    assert fetched.updated_at == fetched.created_at


async def test_add_assigns_unique_memory_ids(store: MemoryStore) -> None:
    a = await store.add(user_id="alice", key="x", value=1)
    b = await store.add(user_id="alice", key="x", value=2)
    assert a.memory_id != b.memory_id


async def test_value_can_be_structured_dict(store: MemoryStore) -> None:
    """``value`` is opaque JSON-serializable — structured payloads
    round-trip without flattening."""
    payload = {"kind": "preference", "topic": "verbosity", "level": 3}
    rec = await store.add(user_id="bob", key="prefs", value=payload)
    again = await store.get(user_id="bob", memory_id=rec.memory_id)
    assert again.value == payload


# ---------------------------------------------------------------------------
# List + tag filtering
# ---------------------------------------------------------------------------


async def test_list_returns_user_memories_in_insertion_order(
    store: MemoryStore,
) -> None:
    await store.add(user_id="alice", key="first", value=1)
    await store.add(user_id="alice", key="second", value=2)
    await store.add(user_id="alice", key="third", value=3)

    listed = await store.list(user_id="alice")
    assert [r.key for r in listed] == ["first", "second", "third"]


async def test_list_with_tags_returns_union(store: MemoryStore) -> None:
    await store.add(user_id="alice", key="a", value=1, tags=("work",))
    await store.add(user_id="alice", key="b", value=2, tags=("home",))
    await store.add(
        user_id="alice", key="c", value=3, tags=("work", "urgent")
    )

    work = await store.list(user_id="alice", tags=("work",))
    assert {r.key for r in work} == {"a", "c"}

    work_or_home = await store.list(user_id="alice", tags=("work", "home"))
    assert {r.key for r in work_or_home} == {"a", "b", "c"}


async def test_list_unknown_user_returns_empty(store: MemoryStore) -> None:
    """Reading an unseen user is the common 'first turn' case —
    empty list, not an error."""
    assert await store.list(user_id="never-seen") == []


# ---------------------------------------------------------------------------
# Cross-user isolation
# ---------------------------------------------------------------------------


async def test_get_wrong_user_raises_keyerror(store: MemoryStore) -> None:
    rec = await store.add(user_id="alice", key="secret", value="42")
    # Bob tries to read Alice's memory — must look identical to
    # "the id doesn't exist" so Bob can't probe Alice's catalog.
    with pytest.raises(KeyError):
        await store.get(user_id="bob", memory_id=rec.memory_id)


async def test_update_wrong_user_raises_keyerror(store: MemoryStore) -> None:
    rec = await store.add(user_id="alice", key="x", value=1)
    with pytest.raises(KeyError):
        await store.update(user_id="bob", memory_id=rec.memory_id, value=2)


async def test_remove_wrong_user_raises_keyerror(store: MemoryStore) -> None:
    rec = await store.add(user_id="alice", key="x", value=1)
    with pytest.raises(KeyError):
        await store.remove(user_id="bob", memory_id=rec.memory_id)
    # Alice's record is still there.
    again = await store.get(user_id="alice", memory_id=rec.memory_id)
    assert again.value == 1


async def test_list_does_not_leak_across_users(store: MemoryStore) -> None:
    await store.add(user_id="alice", key="a", value=1)
    await store.add(user_id="bob", key="b", value=2)
    assert {r.key for r in await store.list(user_id="alice")} == {"a"}
    assert {r.key for r in await store.list(user_id="bob")} == {"b"}


# ---------------------------------------------------------------------------
# Update / remove
# ---------------------------------------------------------------------------


async def test_update_changes_value_and_bumps_updated_at(
    store: MemoryStore,
) -> None:
    rec = await store.add(user_id="alice", key="x", value=1)
    updated = await store.update(
        user_id="alice", memory_id=rec.memory_id, value=99
    )
    assert updated.value == 99
    # created_at preserved, updated_at moved forward (lexical
    # compare works for ISO-8601 UTC).
    assert updated.created_at == rec.created_at
    assert updated.updated_at >= rec.updated_at


async def test_update_unknown_memory_raises_keyerror(
    store: MemoryStore,
) -> None:
    with pytest.raises(KeyError):
        await store.update(
            user_id="alice", memory_id="mem_does_not_exist", value=1
        )


async def test_remove_makes_get_raise(store: MemoryStore) -> None:
    rec = await store.add(user_id="alice", key="x", value=1)
    await store.remove(user_id="alice", memory_id=rec.memory_id)
    with pytest.raises(KeyError):
        await store.get(user_id="alice", memory_id=rec.memory_id)


async def test_remove_twice_raises_on_second_call(store: MemoryStore) -> None:
    """Removing the same id twice surfaces the second call as
    KeyError so a caller can detect accidental double-deletes."""
    rec = await store.add(user_id="alice", key="x", value=1)
    await store.remove(user_id="alice", memory_id=rec.memory_id)
    with pytest.raises(KeyError):
        await store.remove(user_id="alice", memory_id=rec.memory_id)


# ---------------------------------------------------------------------------
# forget_all
# ---------------------------------------------------------------------------


async def test_forget_all_returns_count_and_clears_user(
    store: MemoryStore,
) -> None:
    await store.add(user_id="alice", key="a", value=1)
    await store.add(user_id="alice", key="b", value=2)
    await store.add(user_id="alice", key="c", value=3)

    count = await store.forget_all(user_id="alice")
    assert count == 3
    assert await store.list(user_id="alice") == []


async def test_forget_all_does_not_touch_other_users(
    store: MemoryStore,
) -> None:
    await store.add(user_id="alice", key="a", value=1)
    bob_rec = await store.add(user_id="bob", key="b", value=2)

    await store.forget_all(user_id="alice")

    bob_fetched = await store.get(user_id="bob", memory_id=bob_rec.memory_id)
    assert bob_fetched.value == 2


async def test_forget_all_unknown_user_returns_zero(
    store: MemoryStore,
) -> None:
    """Erasing a user we've never seen is a no-op success, not
    an error — keeps the GDPR-style 'delete me' path idempotent
    from the host's side."""
    assert await store.forget_all(user_id="never-seen") == 0


# ---------------------------------------------------------------------------
# list_users
# ---------------------------------------------------------------------------


async def test_list_users(store: MemoryStore) -> None:
    await store.add(user_id="alice", key="x", value=1)
    await store.add(user_id="bob", key="y", value=2)
    users = await store.list_users()
    assert set(users) == {"alice", "bob"}


# ---------------------------------------------------------------------------
# File-backend specific paths
# ---------------------------------------------------------------------------


async def test_file_backend_rejects_traversal_user_ids(
    tmp_path: Path,
) -> None:
    fs = FileMemoryStore(tmp_path / "m")
    for bad in ("../etc/passwd", "a/b", "a\\b", ".hidden", ""):
        with pytest.raises(ValueError):
            await fs.add(user_id=bad, key="x", value=1)


async def test_file_backend_survives_corrupt_trailing_line(
    tmp_path: Path,
) -> None:
    fs = FileMemoryStore(tmp_path / "m")
    await fs.add(user_id="alice", key="x", value=1)
    path = tmp_path / "m" / "alice.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write('{"op": "add", "memory_id":')

    # Recovery: the live record is intact, the partial line dropped.
    records = await fs.list(user_id="alice")
    assert len(records) == 1
    assert records[0].value == 1


async def test_file_backend_replay_skips_unknown_ops(tmp_path: Path) -> None:
    """Forward-compat: a future op kind shouldn't break readers
    written against today's wire format."""
    fs = FileMemoryStore(tmp_path / "m")
    rec = await fs.add(user_id="alice", key="x", value=1)
    path = tmp_path / "m" / "alice.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write('{"op": "future-thing", "memory_id": "anything"}\n')

    again = await fs.get(user_id="alice", memory_id=rec.memory_id)
    assert again.value == 1


async def test_file_backend_forget_all_deletes_the_file(
    tmp_path: Path,
) -> None:
    """forget_all wants the bytes gone, not a tombstone."""
    fs = FileMemoryStore(tmp_path / "m")
    await fs.add(user_id="alice", key="x", value=1)
    path = tmp_path / "m" / "alice.jsonl"
    assert path.exists()
    await fs.forget_all(user_id="alice")
    assert not path.exists()


async def test_file_backend_survives_reopen(tmp_path: Path) -> None:
    """Write through one store instance, read through a fresh one
    rooted at the same dir — exercises the on-disk durability."""
    fs1 = FileMemoryStore(tmp_path / "m")
    rec = await fs1.add(user_id="alice", key="x", value="hello")

    fs2 = FileMemoryStore(tmp_path / "m")
    again = await fs2.get(user_id="alice", memory_id=rec.memory_id)
    assert again.value == "hello"


# ---------------------------------------------------------------------------
# Memory dataclass
# ---------------------------------------------------------------------------


def test_memory_is_immutable() -> None:
    """``Memory`` is a frozen dataclass — callers can't mutate the
    record returned from the store and have changes silently
    not-persist."""
    rec = Memory(
        memory_id="m1",
        user_id="alice",
        key="x",
        value=1,
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        rec.value = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_defaults_to_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(MEMORY_BACKEND_ENV, raising=False)
    assert isinstance(make_memory_store(), LocalMemoryStore)


def test_factory_honors_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(MEMORY_BACKEND_ENV, "file")
    monkeypatch.setenv(MEMORY_DIR_ENV, str(tmp_path / "m"))
    assert isinstance(make_memory_store(), FileMemoryStore)


def test_factory_explicit_args_win(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MEMORY_BACKEND_ENV, "file")
    assert isinstance(make_memory_store(kind="local"), LocalMemoryStore)


def test_factory_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown memory backend"):
        make_memory_store(kind="vector-db")
