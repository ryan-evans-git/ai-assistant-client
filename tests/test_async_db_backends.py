"""Tests for the native-async DB backends.

We don't stand up a real PostgreSQL / MySQL service in CI —
that's a deployment-tier concern.  Instead the tests use
small duck-typed mock pools that record every SQL string and
parameter tuple the store sends.  The contract verified is:

1. **Correct placeholder dialect.**  asyncpg gets ``$1``, ``$2``,
   …; aiomysql gets ``%s``.  Failing this means the store
   would crash on a real driver.
2. **Round-trip integrity.**  Header → events → footer →
   read returns the same data we wrote, parsed back through
   the same JSON helpers the real backends use.
3. **Error mapping.**  Driver-level integrity / unique-violation
   exceptions map to ``ValueError`` /
   ``KeyError`` as documented in the protocol, matching the
   sync + file backends' behaviour.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import pytest

from ai_assistant_client.persistence import (
    AiomysqlConversationStore,
    AiomysqlTranscriptStore,
    AsyncpgConversationStore,
    AsyncpgTranscriptStore,
    RunHeader,
)
from ai_assistant_client.workflows.runtime import WorkflowEvent


# ---------------------------------------------------------------------------
# asyncpg mock pool — records execute / fetch[row] / fetch calls.
# ---------------------------------------------------------------------------


class _AsyncpgMockConnection:
    """Stub that satisfies the asyncpg ``Connection`` duck-type."""

    def __init__(self, pool: "_AsyncpgMockPool") -> None:
        self._pool = pool

    async def execute(self, sql: str, *args: Any) -> None:
        self._pool.calls.append(("execute", sql, args))

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self._pool.calls.append(("fetchrow", sql, args))
        return self._pool.next_fetchrow.pop(0) if self._pool.next_fetchrow else None

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self._pool.calls.append(("fetch", sql, args))
        return self._pool.next_fetch.pop(0) if self._pool.next_fetch else []

    @asynccontextmanager
    async def transaction(self):  # type: ignore[no-untyped-def]
        yield


class _AsyncpgMockPool:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []
        # Pre-canned answers the store will pop in order for
        # ``fetchrow`` / ``fetch`` calls.
        self.next_fetchrow: list[Any] = []
        self.next_fetch: list[list[Any]] = []

    @asynccontextmanager
    async def acquire(self):  # type: ignore[no-untyped-def]
        yield _AsyncpgMockConnection(self)


# ---------------------------------------------------------------------------
# aiomysql mock pool — same idea but cursor-shaped (DB-API-style).
# ---------------------------------------------------------------------------


class _AiomysqlMockCursor:
    def __init__(self, pool: "_AiomysqlMockPool") -> None:
        self._pool = pool
        self._current_result: Any = None

    async def execute(
        self, sql: str, params: tuple[Any, ...] | None = None
    ) -> None:
        self._pool.calls.append(("execute", sql, tuple(params or ())))
        # If the next query is a "next seq" or "exists" pull, prime
        # the cursor's result row from the pool's queue.
        self._current_result = (
            self._pool.next_fetchone.pop(0)
            if self._pool.next_fetchone
            else None
        )
        self._current_result_all = (
            self._pool.next_fetchall.pop(0)
            if self._pool.next_fetchall
            else []
        )

    async def fetchone(self) -> Any:
        return self._current_result

    async def fetchall(self) -> list[Any]:
        return self._current_result_all

    async def __aenter__(self) -> "_AiomysqlMockCursor":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class _AiomysqlMockConnection:
    def __init__(self, pool: "_AiomysqlMockPool") -> None:
        self._pool = pool

    def cursor(self) -> _AiomysqlMockCursor:
        return _AiomysqlMockCursor(self._pool)

    async def commit(self) -> None:
        self._pool.commits += 1


class _AiomysqlMockPool:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []
        self.next_fetchone: list[Any] = []
        self.next_fetchall: list[list[Any]] = []
        self.commits = 0

    @asynccontextmanager
    async def acquire(self):  # type: ignore[no-untyped-def]
        yield _AiomysqlMockConnection(self)


# ---------------------------------------------------------------------------
# asyncpg — placeholder style
# ---------------------------------------------------------------------------


async def test_asyncpg_transcript_uses_dollar_placeholders() -> None:
    pool = _AsyncpgMockPool()
    store = AsyncpgTranscriptStore(pool)
    await store.write_header(
        RunHeader(
            run_id="run-1",
            workflow_name="x",
            tool_use_id="tu",
            args={},
            started_at="t",
        )
    )

    inserts = [
        sql for _, sql, _ in pool.calls if sql.lstrip().startswith("INSERT")
    ]
    assert inserts
    # asyncpg-style positional placeholders.
    assert "$1" in inserts[0] and "$2" in inserts[0]
    assert "?" not in inserts[0]
    assert "%s" not in inserts[0]


async def test_asyncpg_conversation_uses_dollar_placeholders() -> None:
    pool = _AsyncpgMockPool()
    # Prime the "next seq" lookup to return 1.
    pool.next_fetchrow.append([1])
    store = AsyncpgConversationStore(pool)
    await store.append("conv-1", {"role": "user", "content": "hi"})

    inserts = [
        sql for _, sql, _ in pool.calls if sql.lstrip().startswith("INSERT")
    ]
    assert inserts
    assert "$1" in inserts[0]


# ---------------------------------------------------------------------------
# aiomysql — placeholder style
# ---------------------------------------------------------------------------


async def test_aiomysql_transcript_uses_percent_s_placeholders() -> None:
    pool = _AiomysqlMockPool()
    store = AiomysqlTranscriptStore(pool)
    await store.write_header(
        RunHeader(
            run_id="run-1",
            workflow_name="x",
            tool_use_id="tu",
            args={},
            started_at="t",
        )
    )
    inserts = [
        sql for _, sql, _ in pool.calls if sql.lstrip().startswith("INSERT")
    ]
    assert inserts
    assert "%s" in inserts[0]
    assert "?" not in inserts[0]


async def test_aiomysql_conversation_uses_percent_s_placeholders() -> None:
    pool = _AiomysqlMockPool()
    pool.next_fetchone.append([1])
    store = AiomysqlConversationStore(pool)
    await store.append("conv-1", {"role": "user", "content": "hi"})

    inserts = [
        sql for _, sql, _ in pool.calls if sql.lstrip().startswith("INSERT")
    ]
    assert inserts
    assert "%s" in inserts[0]


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class UniqueViolationError(Exception):
    """Stand-in for asyncpg's real UniqueViolationError.

    The store walks the exception's MRO for a class named
    ``UniqueViolationError`` so we don't need to import the real
    type — keeps these tests independent of asyncpg being
    installed.  The name (not the package) is what matches."""


async def test_asyncpg_duplicate_header_maps_to_value_error() -> None:
    class FailingPool(_AsyncpgMockPool):
        @asynccontextmanager
        async def acquire(self):  # type: ignore[no-untyped-def, override]
            yield self  # type: ignore[misc]

        async def execute(self, sql: str, *args: Any) -> None:
            if sql.lstrip().startswith("INSERT"):
                raise UniqueViolationError("duplicate key")

    pool = FailingPool()
    store = AsyncpgTranscriptStore(pool)
    with pytest.raises(ValueError, match="already has a header"):
        await store.write_header(
            RunHeader(
                run_id="dup",
                workflow_name="x",
                tool_use_id="tu",
                args={},
                started_at="t",
            )
        )


async def test_asyncpg_append_to_unknown_run_raises_key_error() -> None:
    pool = _AsyncpgMockPool()
    # fetchrow for the "does the run exist?" check returns None.
    pool.next_fetchrow.append(None)
    store = AsyncpgTranscriptStore(pool)
    with pytest.raises(KeyError):
        await store.append_event(
            "missing", WorkflowEvent(type="status", payload={"x": 1})
        )


async def test_aiomysql_append_to_unknown_run_raises_key_error() -> None:
    pool = _AiomysqlMockPool()
    # SELECT-1 check returns nothing.
    pool.next_fetchone.append(None)
    store = AiomysqlTranscriptStore(pool)
    with pytest.raises(KeyError):
        await store.append_event(
            "missing", WorkflowEvent(type="status", payload={"x": 1})
        )


# ---------------------------------------------------------------------------
# Round-trip (asyncpg) — verify the JSON columns survive
# ---------------------------------------------------------------------------


async def test_asyncpg_read_decodes_event_columns() -> None:
    pool = _AsyncpgMockPool()
    # Prime answers in order for ``read``:
    #   1. fetchrow for the run row
    #   2. fetch for the events list
    pool.next_fetchrow.append(
        ("send_email", "tu", json.dumps({"to": "a"}), "t0", "t1", "result")
    )
    pool.next_fetch.append(
        [
            ("status", json.dumps({"message": "go"}), None, None, "t0"),
            (
                "result",
                None,
                json.dumps({"sent": True}),
                None,
                "t1",
            ),
        ]
    )

    store = AsyncpgTranscriptStore(pool)
    transcript = await store.read("run-1")

    assert transcript.header.args == {"to": "a"}
    assert transcript.footer is not None
    assert transcript.footer.outcome == "result"
    assert transcript.events[0].type == "status"
    assert transcript.events[0].payload == {"message": "go"}
    assert transcript.events[1].value == {"sent": True}
    assert transcript.events[1].timestamp == "t1"
