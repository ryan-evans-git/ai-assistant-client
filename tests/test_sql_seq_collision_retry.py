"""Tests for the seq-collision retry pattern.

Cross-process writers can both compute the same ``MAX(seq)+1``
between SELECT and INSERT, producing a PK violation
(IntegrityError) on the second INSERT.  The write paths catch
that specific error and retry — up to
``SEQ_COLLISION_MAX_ATTEMPTS`` times — with a fresh SELECT each
attempt.

These tests use a fault-injecting mock cursor that raises an
``IntegrityError`` on the first INSERT(s), then succeeds, to
verify:

1. The retry actually happens (write eventually succeeds).
2. Non-IntegrityError exceptions propagate immediately (no
   silent retry of real bugs).
3. The retry budget is finite (exhausted retries surface the
   last error).
"""

from __future__ import annotations

from typing import Any

import pytest

from ai_assistant_client.persistence import (
    Dialect,
    RunHeader,
    SqlConversationStore,
    SqlTranscriptStore,
)
from ai_assistant_client.persistence.sql_common import (
    SEQ_COLLISION_MAX_ATTEMPTS,
)
from ai_assistant_client.workflows.runtime import WorkflowEvent


class IntegrityError(Exception):
    """Stand-in for the real DB-API IntegrityError.

    ``is_integrity_error`` walks the MRO looking for a class with
    this literal name, so we don't need to import a real
    driver's class to exercise the retry path."""


def _header() -> RunHeader:
    return RunHeader(
        run_id="run-1",
        workflow_name="x",
        tool_use_id="tu",
        args={},
        started_at="t",
    )


# ---------------------------------------------------------------------------
# Mock cursor / connection — same shape as the sql_persistence tests'
# CaptureCursor, plus a fault-injection hook on INSERT.
# ---------------------------------------------------------------------------


class _Cursor:
    def __init__(self, parent: "_Conn") -> None:
        self._parent = parent

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        self._parent.executes.append((sql, tuple(params or ())))
        sql_head = sql.lstrip().lower().split(None, 1)[0]
        if sql_head == "insert":
            # First N INSERTs (for the events / messages table)
            # fail with IntegrityError to simulate a cross-process
            # seq collision.  After ``fail_inserts`` failures the
            # next INSERT succeeds.
            if "events" in sql or "messages" in sql:
                if self._parent.fail_inserts > 0:
                    self._parent.fail_inserts -= 1
                    self._parent.insert_failures += 1
                    raise IntegrityError("UNIQUE constraint failed")

    def fetchone(self) -> Any:
        # For the seq-check / run-exists queries the test seeds
        # ``next_fetchone`` with concrete tuples.  Default to
        # ``(1,)`` so the SELECT MAX(seq)+1 returns 1.
        if self._parent.next_fetchone:
            return self._parent.next_fetchone.pop(0)
        return (1,)

    def fetchall(self) -> list[Any]:
        return []

    def close(self) -> None:
        pass


class _Conn:
    def __init__(self, fail_inserts: int = 0) -> None:
        self.executes: list[tuple[str, tuple[Any, ...]]] = []
        self.next_fetchone: list[Any] = []
        self.fail_inserts = fail_inserts
        self.insert_failures = 0
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


# ---------------------------------------------------------------------------
# Transcript store
# ---------------------------------------------------------------------------


async def test_transcript_append_retries_on_seq_collision() -> None:
    """First two INSERTs raise IntegrityError; the third succeeds.
    The append should complete without raising."""
    conn = _Conn(fail_inserts=2)
    # Pre-seed fetchone for: 1) run-exists check (truthy), 2) MAX(seq)+1.
    # Repeated for each retry.  We seed enough for 3 attempts.
    for _ in range(SEQ_COLLISION_MAX_ATTEMPTS):
        conn.next_fetchone.extend([(1,), (1,)])  # exists, MAX+1

    store = SqlTranscriptStore(conn, dialect=Dialect.SQLITE)
    # Bypass schema bootstrap which would consume DDL executes.
    store._initialized = True  # type: ignore[attr-defined]

    await store.append_event(
        "run-1", WorkflowEvent(type="status", payload={"x": 1})
    )
    # Two failures retried, one final success.
    assert conn.insert_failures == 2
    assert conn.commits == 1


async def test_transcript_append_exhausts_retries_and_raises() -> None:
    """If every INSERT fails (e.g. high contention or a stuck
    leader), the retry loop gives up after ``SEQ_COLLISION_MAX_ATTEMPTS``
    and propagates the last IntegrityError."""
    conn = _Conn(fail_inserts=SEQ_COLLISION_MAX_ATTEMPTS + 5)
    for _ in range(SEQ_COLLISION_MAX_ATTEMPTS):
        conn.next_fetchone.extend([(1,), (1,)])

    store = SqlTranscriptStore(conn, dialect=Dialect.SQLITE)
    store._initialized = True  # type: ignore[attr-defined]

    with pytest.raises(IntegrityError):
        await store.append_event(
            "run-1", WorkflowEvent(type="status", payload={"x": 1})
        )
    assert conn.insert_failures == SEQ_COLLISION_MAX_ATTEMPTS


async def test_transcript_append_does_not_retry_non_integrity_errors() -> None:
    """A non-Integrity error (e.g. real schema constraint
    violation, NOT NULL, type mismatch) must propagate
    immediately so a real bug isn't masked by silent retries."""

    class WeirdError(Exception):
        pass

    class FailingConn(_Conn):
        def cursor(self) -> "_FailingCursor":  # type: ignore[override]
            return _FailingCursor(self)

    class _FailingCursor(_Cursor):
        def execute(
            self, sql: str, params: tuple[Any, ...] | None = None
        ) -> None:
            self._parent.executes.append((sql, tuple(params or ())))
            if "INSERT" in sql:
                raise WeirdError("not an integrity error")

    conn = FailingConn()
    conn.next_fetchone.extend([(1,), (1,)])
    store = SqlTranscriptStore(conn, dialect=Dialect.SQLITE)
    store._initialized = True  # type: ignore[attr-defined]

    with pytest.raises(WeirdError):
        await store.append_event(
            "run-1", WorkflowEvent(type="status", payload={"x": 1})
        )
    # Only one INSERT attempted — no retry for the unrelated error.
    inserts = [
        sql for sql, _ in conn.executes if sql.lstrip().startswith("INSERT")
    ]
    assert len(inserts) == 1


async def test_transcript_append_keyerror_does_not_retry() -> None:
    """Missing run (the up-front existence check) is a permanent
    failure — no point retrying the SELECT MAX(seq)+1 against a
    run that doesn't exist."""
    conn = _Conn()
    # First fetchone returns None → run doesn't exist.
    conn.next_fetchone.append(None)
    store = SqlTranscriptStore(conn, dialect=Dialect.SQLITE)
    store._initialized = True  # type: ignore[attr-defined]

    with pytest.raises(KeyError):
        await store.append_event(
            "missing", WorkflowEvent(type="status", payload={"x": 1})
        )
    # Only one cursor "round" happened — no retry storm on a
    # missing run.
    assert len(conn.executes) == 1


# ---------------------------------------------------------------------------
# Conversation store
# ---------------------------------------------------------------------------


async def test_conversation_append_retries_on_seq_collision() -> None:
    conn = _Conn(fail_inserts=2)
    for _ in range(SEQ_COLLISION_MAX_ATTEMPTS):
        conn.next_fetchone.append((1,))  # MAX(seq)+1

    store = SqlConversationStore(conn, dialect=Dialect.SQLITE)
    store._initialized = True  # type: ignore[attr-defined]

    await store.append("conv-1", {"role": "user", "content": "hi"})
    assert conn.insert_failures == 2
    assert conn.commits == 1


async def test_conversation_append_exhausts_retries_and_raises() -> None:
    conn = _Conn(fail_inserts=SEQ_COLLISION_MAX_ATTEMPTS + 1)
    for _ in range(SEQ_COLLISION_MAX_ATTEMPTS):
        conn.next_fetchone.append((1,))

    store = SqlConversationStore(conn, dialect=Dialect.SQLITE)
    store._initialized = True  # type: ignore[attr-defined]

    with pytest.raises(IntegrityError):
        await store.append("conv-1", {"role": "user", "content": "hi"})
    assert conn.insert_failures == SEQ_COLLISION_MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# is_integrity_error helper
# ---------------------------------------------------------------------------


def test_is_integrity_error_matches_by_name() -> None:
    """Catches the literal-name classes from every DB-API driver
    without importing them at module load."""
    from ai_assistant_client.persistence.sql_common import (
        is_integrity_error,
    )

    class IntegrityError(Exception):
        pass

    class UniqueViolationError(Exception):
        pass

    class OperationalError(Exception):
        pass

    assert is_integrity_error(IntegrityError("x"))
    assert is_integrity_error(UniqueViolationError("x"))
    assert not is_integrity_error(OperationalError("x"))
    assert not is_integrity_error(ValueError("x"))


# ---------------------------------------------------------------------------
# Memory store list ORDER BY tiebreak
# ---------------------------------------------------------------------------


async def test_memory_list_uses_capture_cursor_tiebreak() -> None:
    """The list SQL must include ``memory_id ASC`` as a
    deterministic tiebreak in its ORDER BY clause — verified by
    capturing every SQL statement the store sends to a mock
    cursor."""
    from ai_assistant_client.persistence import SqlMemoryStore

    conn = _Conn()
    # No rows to return — list() just runs its SELECT and gets
    # an empty result.
    store = SqlMemoryStore(conn, dialect=Dialect.SQLITE)
    store._initialized = True  # type: ignore[attr-defined]
    await store.list(user_id="alice")

    list_sqls = [
        sql for sql, _ in conn.executes
        if "aac_user_memories" in sql and "ORDER BY" in sql
    ]
    assert list_sqls
    assert "memory_id ASC" in list_sqls[0]
