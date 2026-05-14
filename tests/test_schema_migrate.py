"""Tests for the schema migration helpers.

Strategy:

* **Real sqlite** for the end-to-end idempotency contract —
  add the column, add it again, verify the second call is a
  no-op.  Exercise the upgrade path of an "old" table created
  without the new column.
* **Mocked DB-API connection** for the PostgreSQL / MySQL
  dispatch — we don't need a real server to verify the SQL
  the helper emits.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

import pytest

from ai_assistant_client.persistence import (
    Dialect,
    add_column_if_missing,
    ensure_transcript_events_ts_column,
)
from ai_assistant_client.persistence.migrate import (
    _has_column,
    _is_duplicate_column_error,
    _quote_ident,
)


SqliteConnFactory = Callable[..., sqlite3.Connection]


# ---------------------------------------------------------------------------
# sqlite — end-to-end against a real connection
# ---------------------------------------------------------------------------


def _make_legacy_events_table(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Stamp ``conn`` with the events table in its pre-timestamp shape.

    Returns the same connection so tests can use ``conn = make_legacy(...)``
    fluently.  The caller is responsible for opening / closing — this
    helper just runs the DDL.
    """
    conn.execute(
        """
        CREATE TABLE aac_transcript_events (
            run_id       TEXT NOT NULL,
            seq          INTEGER NOT NULL,
            event_type   TEXT NOT NULL,
            payload_json TEXT,
            value_json   TEXT,
            error        TEXT,
            PRIMARY KEY (run_id, seq)
        )
        """
    )
    conn.commit()
    return conn


def test_sqlite_adds_ts_to_legacy_table(
    make_sqlite_conn: SqliteConnFactory,
) -> None:
    conn = _make_legacy_events_table(make_sqlite_conn("old.sqlite3"))
    assert not _has_column(
        conn, dialect=Dialect.SQLITE, table="aac_transcript_events",
        column="ts",
    )
    added = ensure_transcript_events_ts_column(conn, dialect=Dialect.SQLITE)
    assert added is True
    assert _has_column(
        conn, dialect=Dialect.SQLITE, table="aac_transcript_events",
        column="ts",
    )


def test_sqlite_idempotent_on_second_call(
    make_sqlite_conn: SqliteConnFactory,
) -> None:
    conn = _make_legacy_events_table(make_sqlite_conn("old.sqlite3"))
    ensure_transcript_events_ts_column(conn, dialect=Dialect.SQLITE)
    added_again = ensure_transcript_events_ts_column(
        conn, dialect=Dialect.SQLITE
    )
    assert added_again is False


def test_sqlite_existing_rows_get_default_empty_ts(
    make_sqlite_conn: SqliteConnFactory,
) -> None:
    """An old row inserted before the migration must read back
    with ts='' after — matches the WorkflowEvent.timestamp
    default for "unknown" timestamps."""
    conn = _make_legacy_events_table(make_sqlite_conn("old.sqlite3"))
    conn.execute(
        "INSERT INTO aac_transcript_events "
        "(run_id, seq, event_type, payload_json, value_json, error) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("r1", 1, "status", None, None, None),
    )
    conn.commit()

    ensure_transcript_events_ts_column(conn, dialect=Dialect.SQLITE)

    cur = conn.execute(
        "SELECT ts FROM aac_transcript_events WHERE run_id = 'r1'"
    )
    rows = cur.fetchall()
    assert rows == [("",)]


def test_add_column_if_missing_generic_round_trip(
    make_sqlite_conn: SqliteConnFactory,
) -> None:
    """Sanity-check the generic helper on a non-canonical column."""
    conn = make_sqlite_conn("g.sqlite3")
    conn.execute("CREATE TABLE widgets (id INTEGER PRIMARY KEY)")
    conn.commit()

    added = add_column_if_missing(
        conn,
        dialect=Dialect.SQLITE,
        table="widgets",
        column="name",
        column_type_sql="TEXT",
        default_sql="''",
    )
    assert added is True
    again = add_column_if_missing(
        conn,
        dialect=Dialect.SQLITE,
        table="widgets",
        column="name",
        column_type_sql="TEXT",
        default_sql="''",
    )
    assert again is False


# ---------------------------------------------------------------------------
# Identifier quoting / safety
# ---------------------------------------------------------------------------


def test_quote_ident_rejects_dangerous_chars() -> None:
    for bad in ('foo"bar', "foo'bar", "foo;bar", "foo(", "foo\nbar"):
        with pytest.raises(ValueError):
            _quote_ident(bad)


def test_quote_ident_accepts_plain_table_names() -> None:
    assert _quote_ident("aac_transcript_events") == '"aac_transcript_events"'


# ---------------------------------------------------------------------------
# MySQL duplicate-column detection
# ---------------------------------------------------------------------------


def test_is_duplicate_column_error_matches_message() -> None:
    """MySQL drivers raise OperationalError with various message
    formats; match on substring rather than class so any driver
    works."""

    class MockErr(Exception):
        pass

    assert _is_duplicate_column_error(MockErr("Duplicate column name 'ts'"))
    assert _is_duplicate_column_error(MockErr("(1060, 'Duplicate column')"))
    assert not _is_duplicate_column_error(MockErr("some other error"))


# ---------------------------------------------------------------------------
# PostgreSQL / MySQL dispatch (mocked)
# ---------------------------------------------------------------------------


class _CaptureCursor:
    def __init__(self, parent: "_CaptureConn") -> None:
        self._parent = parent

    def execute(
        self, sql: str, params: tuple[Any, ...] | None = None
    ) -> None:
        self._parent.executes.append((sql, tuple(params or ())))

    def fetchone(self) -> Any:
        return self._parent.next_fetchone.pop(0) if self._parent.next_fetchone else None

    def fetchall(self) -> list[Any]:
        return self._parent.next_fetchall.pop(0) if self._parent.next_fetchall else []

    def close(self) -> None:
        pass


class _CaptureConn:
    def __init__(self) -> None:
        self.executes: list[tuple[str, tuple[Any, ...]]] = []
        self.next_fetchone: list[Any] = []
        self.next_fetchall: list[list[Any]] = []

    def cursor(self) -> _CaptureCursor:
        return _CaptureCursor(self)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


def test_postgres_uses_information_schema_introspection() -> None:
    """Postgres path uses information_schema and %s placeholders."""
    conn = _CaptureConn()
    # Pre-canned: no row found → column missing → ALTER fires.
    conn.next_fetchone.append(None)
    added = ensure_transcript_events_ts_column(
        conn, dialect=Dialect.POSTGRESQL
    )
    assert added is True
    # First execute is the introspection query against
    # information_schema; second is the ALTER.
    assert "information_schema.columns" in conn.executes[0][0]
    alter_sql = conn.executes[1][0]
    assert "ALTER TABLE" in alter_sql
    assert "ADD COLUMN" in alter_sql
    assert "ts" in alter_sql


def test_postgres_skips_alter_when_column_present() -> None:
    conn = _CaptureConn()
    conn.next_fetchone.append((1,))  # column exists
    added = ensure_transcript_events_ts_column(
        conn, dialect=Dialect.POSTGRESQL
    )
    assert added is False
    # Only the introspection query ran; no ALTER.
    assert len(conn.executes) == 1


def test_mysql_uses_information_schema_introspection() -> None:
    conn = _CaptureConn()
    conn.next_fetchone.append(None)
    added = ensure_transcript_events_ts_column(conn, dialect=Dialect.MYSQL)
    assert added is True
    assert "information_schema.columns" in conn.executes[0][0]
    assert "table_schema = DATABASE()" in conn.executes[0][0]


def test_mysql_swallows_duplicate_column_error_on_concurrent_add() -> None:
    """If two processes race to add the same column, one of them
    will hit MySQL's duplicate-column error.  The helper must
    treat that as a no-op (the column ended up where it needed
    to be) rather than crashing."""

    class FailingConn(_CaptureConn):
        def cursor(self) -> "_FailingCursor":  # type: ignore[override]
            return _FailingCursor(self)

    class _FailingCursor(_CaptureCursor):
        def execute(
            self, sql: str, params: tuple[Any, ...] | None = None
        ) -> None:
            self._parent.executes.append((sql, tuple(params or ())))
            if "ALTER TABLE" in sql:
                raise Exception("Duplicate column name 'ts'")

    conn = FailingConn()
    conn.next_fetchone.append(None)  # introspection says missing
    added = ensure_transcript_events_ts_column(conn, dialect=Dialect.MYSQL)
    assert added is False
