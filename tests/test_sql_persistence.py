"""Tests for the DB-API-2.0-backed stores.

Two layers of coverage:

1. **Functional / round-trip** — runs against a real ``sqlite3``
   connection (stdlib, no external DB needed in CI).  Exercises
   every protocol method end-to-end.
2. **Dialect correctness** — runs against a captured-cursor mock
   to assert that the SQL we send for ``Dialect.POSTGRESQL`` /
   ``Dialect.MYSQL`` uses ``%s`` placeholders.  This is the
   bit a sqlite-only test can't catch.

The mock also documents the contract we expect a real driver
to satisfy — a DB-API 2.0 connection / cursor pair with
``execute``, ``fetchone``, ``fetchall``, ``commit``, ``rollback``,
``close``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from ai_assistant_client.persistence import (
    Dialect,
    RunFooter,
    RunHeader,
    SqlConversationStore,
    SqlTranscriptStore,
    TranscriptStore,
    make_conversation_store,
    make_transcript_store,
)
from ai_assistant_client.persistence.factory import (
    CONVERSATION_BACKEND_ENV,
    CONVERSATION_SQLITE_PATH_ENV,
    TRANSCRIPT_BACKEND_ENV,
    TRANSCRIPT_SQLITE_PATH_ENV,
)
from ai_assistant_client.persistence.sql_common import adapt_sql
from ai_assistant_client.workflows.runtime import WorkflowEvent


# ---------------------------------------------------------------------------
# Helpers — open a real sqlite connection the way the factory does.
# ---------------------------------------------------------------------------


def _sqlite_conn(tmp_path: Path, name: str = "test.sqlite3") -> sqlite3.Connection:
    return sqlite3.connect(tmp_path / name, check_same_thread=False)


def _header(run_id: str = "run-1") -> RunHeader:
    return RunHeader(
        run_id=run_id,
        workflow_name="send_email",
        tool_use_id="tu-1",
        args={"to": "a@b.com"},
        started_at="2026-05-14T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Transcript store — round-trip against sqlite
# ---------------------------------------------------------------------------


async def test_transcript_header_event_footer_round_trip(tmp_path: Path) -> None:
    conn = _sqlite_conn(tmp_path)
    store = SqlTranscriptStore(conn, dialect=Dialect.SQLITE)

    h = _header()
    await store.write_header(h)
    await store.append_event(
        "run-1", WorkflowEvent(type="status", payload={"msg": "go"})
    )
    await store.append_event(
        "run-1", WorkflowEvent(type="result", value={"ok": True})
    )
    await store.write_footer(
        "run-1", RunFooter(ended_at="2026-05-14T00:00:01+00:00", outcome="result")
    )

    transcript = await store.read("run-1")
    assert transcript.header == h
    assert [e.type for e in transcript.events] == ["status", "result"]
    assert transcript.events[1].value == {"ok": True}
    assert transcript.footer is not None
    assert transcript.footer.outcome == "result"


async def test_transcript_partial_run_has_no_footer(tmp_path: Path) -> None:
    conn = _sqlite_conn(tmp_path)
    store = SqlTranscriptStore(conn, dialect=Dialect.SQLITE)

    await store.write_header(_header())
    await store.append_event(
        "run-1", WorkflowEvent(type="status", payload={"msg": "x"})
    )

    transcript = await store.read("run-1")
    assert transcript.footer is None
    assert len(transcript.events) == 1


async def test_transcript_duplicate_header_raises_value_error(
    tmp_path: Path,
) -> None:
    conn = _sqlite_conn(tmp_path)
    store = SqlTranscriptStore(conn, dialect=Dialect.SQLITE)

    await store.write_header(_header())
    with pytest.raises(ValueError):
        await store.write_header(_header())


async def test_transcript_append_to_unknown_run_raises_key_error(
    tmp_path: Path,
) -> None:
    conn = _sqlite_conn(tmp_path)
    store = SqlTranscriptStore(conn, dialect=Dialect.SQLITE)
    with pytest.raises(KeyError):
        await store.append_event(
            "never-written", WorkflowEvent(type="status", payload={"x": 1})
        )


async def test_transcript_footer_for_unknown_run_raises_key_error(
    tmp_path: Path,
) -> None:
    conn = _sqlite_conn(tmp_path)
    store = SqlTranscriptStore(conn, dialect=Dialect.SQLITE)
    with pytest.raises(KeyError):
        await store.write_footer(
            "never-written",
            RunFooter(ended_at="2026-05-14T00:00:00+00:00", outcome="error"),
        )


async def test_transcript_read_unknown_raises_key_error(tmp_path: Path) -> None:
    conn = _sqlite_conn(tmp_path)
    store = SqlTranscriptStore(conn, dialect=Dialect.SQLITE)
    with pytest.raises(KeyError):
        await store.read("nope")


async def test_transcript_list_runs(tmp_path: Path) -> None:
    conn = _sqlite_conn(tmp_path)
    store = SqlTranscriptStore(conn, dialect=Dialect.SQLITE)
    await store.write_header(_header("run-a"))
    await store.write_header(_header("run-b"))
    runs = await store.list_runs()
    assert set(runs) == {"run-a", "run-b"}


async def test_transcript_survives_reopen(tmp_path: Path) -> None:
    """Data written via one connection is visible through a
    fresh connection to the same file — the durability win
    over the file backend is the whole point of using SQL."""
    conn1 = _sqlite_conn(tmp_path)
    store1 = SqlTranscriptStore(conn1, dialect=Dialect.SQLITE)
    await store1.write_header(_header("run-x"))
    await store1.append_event(
        "run-x", WorkflowEvent(type="result", value=42)
    )
    conn1.close()

    conn2 = _sqlite_conn(tmp_path)
    store2 = SqlTranscriptStore(conn2, dialect=Dialect.SQLITE)
    transcript = await store2.read("run-x")
    assert transcript.events[0].value == 42


# ---------------------------------------------------------------------------
# Conversation store — round-trip against sqlite
# ---------------------------------------------------------------------------


async def test_conversation_round_trip(tmp_path: Path) -> None:
    conn = _sqlite_conn(tmp_path)
    store = SqlConversationStore(conn, dialect=Dialect.SQLITE)

    await store.append("conv-1", {"role": "user", "content": "hi"})
    await store.append(
        "conv-1",
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    )
    log = await store.read("conv-1")
    assert log == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]


async def test_conversation_read_unknown_returns_empty(tmp_path: Path) -> None:
    conn = _sqlite_conn(tmp_path)
    store = SqlConversationStore(conn, dialect=Dialect.SQLITE)
    assert await store.read("never-written") == []


async def test_conversation_list_conversations(tmp_path: Path) -> None:
    conn = _sqlite_conn(tmp_path)
    store = SqlConversationStore(conn, dialect=Dialect.SQLITE)
    await store.append("conv-a", {"role": "user", "content": "x"})
    await store.append("conv-b", {"role": "user", "content": "y"})
    convs = await store.list_conversations()
    assert set(convs) == {"conv-a", "conv-b"}


async def test_conversation_preserves_insert_order(tmp_path: Path) -> None:
    """The seq column drives the read ORDER BY — ten messages
    inserted in order must come back in order even if sqlite
    physical-storage order is permuted."""
    conn = _sqlite_conn(tmp_path)
    store = SqlConversationStore(conn, dialect=Dialect.SQLITE)
    for i in range(10):
        await store.append("conv-1", {"role": "user", "content": f"msg-{i}"})
    log = await store.read("conv-1")
    assert [m["content"] for m in log] == [f"msg-{i}" for i in range(10)]


# ---------------------------------------------------------------------------
# Dialect correctness — placeholder rewrite
# ---------------------------------------------------------------------------


def test_adapt_sql_sqlite_passthrough() -> None:
    assert adapt_sql("SELECT * FROM t WHERE id = ?", Dialect.SQLITE) == (
        "SELECT * FROM t WHERE id = ?"
    )


def test_adapt_sql_postgresql_rewrites_to_format_style() -> None:
    assert adapt_sql(
        "INSERT INTO t (a, b) VALUES (?, ?)", Dialect.POSTGRESQL
    ) == "INSERT INTO t (a, b) VALUES (%s, %s)"


def test_adapt_sql_mysql_rewrites_to_format_style() -> None:
    assert adapt_sql(
        "INSERT INTO t (a, b) VALUES (?, ?)", Dialect.MYSQL
    ) == "INSERT INTO t (a, b) VALUES (%s, %s)"


# ---------------------------------------------------------------------------
# Dialect correctness — capture the SQL sent to a mock cursor.
# ---------------------------------------------------------------------------


class _CapturingCursor:
    """Minimal DB-API 2.0 cursor stand-in.

    Records every ``execute`` call for inspection.  Returns
    canned answers for the lookup queries the stores issue
    before writes (so the write path doesn't bail on a missing
    row check).
    """

    def __init__(self, parent: "_CapturingConnection") -> None:
        self._parent = parent
        self._next_fetchone: Any = (1,)  # generic non-None
        self._next_fetchall: list[tuple[Any, ...]] = []

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        self._parent.executes.append((sql, tuple(params or ())))
        # Heuristic for the "next seq" SELECT — return a counter.
        if "COALESCE(MAX(seq)" in sql:
            self._next_fetchone = (1,)
        elif "SELECT 1" in sql:
            self._next_fetchone = (1,)
        else:
            self._next_fetchone = (1,)

    def fetchone(self) -> Any:
        return self._next_fetchone

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._next_fetchall

    def close(self) -> None:
        pass


class _CapturingConnection:
    def __init__(self) -> None:
        self.executes: list[tuple[str, tuple[Any, ...]]] = []

    def cursor(self) -> _CapturingCursor:
        return _CapturingCursor(self)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


@pytest.mark.parametrize(
    "dialect,expected_marker",
    [(Dialect.POSTGRESQL, "%s"), (Dialect.MYSQL, "%s"), (Dialect.SQLITE, "?")],
)
async def test_transcript_write_uses_dialect_placeholders(
    dialect: Dialect, expected_marker: str
) -> None:
    conn = _CapturingConnection()
    store = SqlTranscriptStore(conn, dialect=dialect)
    await store.write_header(_header())

    # Find the INSERT we issued; assert the placeholders match.
    inserts = [
        sql for sql, _ in conn.executes if sql.startswith("INSERT")
    ]
    assert inserts, "expected at least one INSERT"
    assert all(expected_marker in sql for sql in inserts)
    if dialect is Dialect.SQLITE:
        assert all("%s" not in sql for sql in inserts)
    else:
        assert all("?" not in sql for sql in inserts)


@pytest.mark.parametrize(
    "dialect", [Dialect.POSTGRESQL, Dialect.MYSQL, Dialect.SQLITE]
)
async def test_conversation_write_uses_dialect_placeholders(
    dialect: Dialect,
) -> None:
    expected = "?" if dialect is Dialect.SQLITE else "%s"
    conn = _CapturingConnection()
    store = SqlConversationStore(conn, dialect=dialect)
    await store.append("conv-1", {"role": "user", "content": "hi"})

    inserts = [
        sql for sql, _ in conn.executes if sql.startswith("INSERT")
    ]
    assert inserts
    assert all(expected in sql for sql in inserts)


# ---------------------------------------------------------------------------
# Factory — sqlite path
# ---------------------------------------------------------------------------


def test_make_transcript_store_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TRANSCRIPT_BACKEND_ENV, "sqlite")
    monkeypatch.setenv(TRANSCRIPT_SQLITE_PATH_ENV, str(tmp_path / "t.sqlite3"))
    store = make_transcript_store()
    assert isinstance(store, SqlTranscriptStore)


def test_make_conversation_store_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(CONVERSATION_BACKEND_ENV, "sqlite")
    monkeypatch.setenv(
        CONVERSATION_SQLITE_PATH_ENV, str(tmp_path / "c.sqlite3")
    )
    store = make_conversation_store()
    assert isinstance(store, SqlConversationStore)


async def test_factory_sqlite_actually_writes(tmp_path: Path) -> None:
    """End-to-end through the factory: build a sqlite store, write,
    re-open, read.  Catches any breakage in the factory's
    ``check_same_thread=False`` wiring."""
    path = tmp_path / "f.sqlite3"
    store: TranscriptStore = make_transcript_store(
        kind="sqlite", sqlite_path=str(path)
    )
    await store.write_header(_header("run-factory"))
    await store.append_event(
        "run-factory", WorkflowEvent(type="result", value="ok")
    )

    store2 = make_transcript_store(kind="sqlite", sqlite_path=str(path))
    transcript = await store2.read("run-factory")
    assert transcript.events[0].value == "ok"
