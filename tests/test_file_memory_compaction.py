"""Tests for ``FileMemoryStore.compact``.

Invariants we want to lock down:

1. Live state survives compaction byte-for-byte (memory_id, key,
   value, tags, created_at, updated_at all preserved).
2. Log size shrinks when there have been many updates / removes.
3. Compacting an already-compact log is a no-op (no spurious
   rewrites).
4. Missing user (no log file) returns zero-zero stats.
5. Tempfile cleanup on crash mid-write — the original file
   stays intact.
6. Cross-user isolation: compacting Alice's log doesn't touch
   Bob's file.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ai_assistant_client.persistence import (
    CompactionStats,
    FileMemoryStore,
)


# ---------------------------------------------------------------------------
# Live-state preservation
# ---------------------------------------------------------------------------


async def test_compaction_preserves_live_state(tmp_path: Path) -> None:
    """After many updates, the live state read back through the
    same store must match what it would have been on a fresh
    backend instance."""
    store = FileMemoryStore(tmp_path)
    rec = await store.add(
        user_id="alice", key="role", value="dev", tags=("work",)
    )
    # Many updates → log grows.
    for value in ("v1", "v2", "v3", "final"):
        await store.update(
            user_id="alice", memory_id=rec.memory_id, value=value
        )

    await store.compact(user_id="alice")

    # Reopen via a fresh store to prove on-disk durability.
    store2 = FileMemoryStore(tmp_path)
    again = await store2.get(user_id="alice", memory_id=rec.memory_id)
    assert again.value == "final"
    assert again.key == "role"
    assert again.tags == ("work",)
    assert again.created_at == rec.created_at  # preserved
    assert again.updated_at >= rec.updated_at  # bumped by updates


async def test_compaction_drops_removed_memories(tmp_path: Path) -> None:
    """Removed memories must not reappear after compaction —
    compaction is the moment ``remove`` ops actually free up
    storage."""
    store = FileMemoryStore(tmp_path)
    keep = await store.add(user_id="alice", key="keep", value=1)
    gone = await store.add(user_id="alice", key="gone", value=2)
    await store.remove(user_id="alice", memory_id=gone.memory_id)

    await store.compact(user_id="alice")

    records = await store.list(user_id="alice")
    assert {r.memory_id for r in records} == {keep.memory_id}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


async def test_compaction_reports_shrunk_log(tmp_path: Path) -> None:
    """Updates+removes produce more lines than live records;
    after compaction lines == live count."""
    store = FileMemoryStore(tmp_path)
    rec = await store.add(user_id="alice", key="x", value=1)
    for v in range(20):
        await store.update(
            user_id="alice", memory_id=rec.memory_id, value=v
        )

    stats = await store.compact(user_id="alice")
    assert isinstance(stats, CompactionStats)
    assert stats.before_lines == 21  # 1 add + 20 updates
    assert stats.after_lines == 1     # one live record
    assert stats.bytes_saved > 0


async def test_compaction_is_noop_on_clean_log(tmp_path: Path) -> None:
    """A log with N adds and zero updates/removes is already
    compact — compacting again should produce identical stats
    (before == after) and not waste a rewrite."""
    store = FileMemoryStore(tmp_path)
    await store.add(user_id="alice", key="a", value=1)
    await store.add(user_id="alice", key="b", value=2)

    stats = await store.compact(user_id="alice")
    assert stats.before_lines == stats.after_lines == 2
    assert stats.bytes_saved == 0


async def test_compaction_on_missing_user_returns_zero_stats(
    tmp_path: Path,
) -> None:
    """Compacting a user with no log file is a no-op success —
    scheduler-friendly so a host can call compact on every
    user without first checking existence."""
    store = FileMemoryStore(tmp_path)
    stats = await store.compact(user_id="never-seen")
    assert stats.before_lines == 0
    assert stats.after_lines == 0
    assert stats.bytes_saved == 0


# ---------------------------------------------------------------------------
# Atomicity / crash safety
# ---------------------------------------------------------------------------


async def test_compaction_crash_mid_write_leaves_original_intact(
    tmp_path: Path,
) -> None:
    """Simulate a crash mid-compaction by patching ``os.replace``
    to raise.  The original log file must be unchanged and the
    tempfile must be cleaned up.  ``forget_all`` shouldn't be
    needed to recover."""
    import os

    store = FileMemoryStore(tmp_path)
    rec = await store.add(user_id="alice", key="x", value=1)
    await store.update(user_id="alice", memory_id=rec.memory_id, value=2)

    path = tmp_path / "alice.jsonl"
    original_content = path.read_bytes()

    with patch.object(os, "replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            await store.compact(user_id="alice")

    # Original log untouched.
    assert path.read_bytes() == original_content
    # No leftover *.tmp files in the dir.
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == []

    # Live state still readable through a fresh store.
    store2 = FileMemoryStore(tmp_path)
    again = await store2.get(user_id="alice", memory_id=rec.memory_id)
    assert again.value == 2


# ---------------------------------------------------------------------------
# Cross-user isolation
# ---------------------------------------------------------------------------


async def test_compaction_only_touches_target_user(tmp_path: Path) -> None:
    """Compacting alice must not modify bob's log."""
    store = FileMemoryStore(tmp_path)
    await store.add(user_id="alice", key="a", value=1)
    bob_rec = await store.add(user_id="bob", key="b", value=2)
    bob_path = tmp_path / "bob.jsonl"
    bob_before = bob_path.read_bytes()

    await store.compact(user_id="alice")

    assert bob_path.read_bytes() == bob_before
    bob_again = await store.get(user_id="bob", memory_id=bob_rec.memory_id)
    assert bob_again.value == 2


# ---------------------------------------------------------------------------
# Post-compaction writes still work
# ---------------------------------------------------------------------------


async def test_writes_after_compaction_round_trip(tmp_path: Path) -> None:
    """A compacted log is still appendable — the next ``add`` /
    ``update`` / ``remove`` lands on the new file and replays
    correctly."""
    store = FileMemoryStore(tmp_path)
    rec = await store.add(user_id="alice", key="x", value=1)
    await store.compact(user_id="alice")

    # Add another after compaction.
    rec2 = await store.add(user_id="alice", key="y", value=2)
    await store.update(user_id="alice", memory_id=rec.memory_id, value=99)

    records = await store.list(user_id="alice")
    by_key = {r.key: r for r in records}
    assert by_key["x"].value == 99
    assert by_key["y"].value == 2
    assert by_key["y"].memory_id == rec2.memory_id
