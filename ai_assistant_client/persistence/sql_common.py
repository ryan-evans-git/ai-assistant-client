"""Shared SQL helpers for the DB-API-2.0-backed stores.

The :class:`Dialect` enum + the helpers in this module are how the
SQL store classes stay portable across the three engines we
explicitly target — **sqlite**, **PostgreSQL** (including AWS
Aurora PostgreSQL + RDS), and **MySQL** (including Aurora MySQL).

Why DB-API-2.0 (PEP 249) and not an async-native driver:

* Any compliant driver works — ``sqlite3`` from the stdlib,
  ``psycopg`` / ``psycopg2`` / ``pg8000`` for PostgreSQL,
  ``PyMySQL`` / ``mysqlclient`` for MySQL.  Users install
  whatever they're already using.
* Aurora Serverless v2 + RDS Proxy use the standard wire
  protocol; the same drivers work transparently.  (Aurora
  Serverless v1's HTTP Data API is the one path that *doesn't*
  work via DB-API — but v1 is the legacy variant.)
* IAM database auth is orthogonal: the driver still works the
  same way; callers mint a short-lived token externally and
  pass it as the password.  No store-side concern.
* Zero new dependencies in this package — moves the credential
  surface entirely to first-party DB drivers the caller has
  already chosen.

DB-API is synchronous; every operation runs through
:func:`asyncio.to_thread` so the agent's event loop never blocks
on the database.

Cross-DB portability wrinkles handled here:

1. **Placeholder style.**  Store SQL is written with ``?``
   placeholders (sqlite-native) and rewritten to ``%s`` at
   execute time for PostgreSQL / MySQL drivers (whose default
   ``paramstyle`` is ``format`` / ``pyformat``).
2. **DDL differences.**  Per-dialect ``CREATE TABLE`` bodies
   live in :func:`transcript_runs_ddl` / etc. below.
3. **JSON storage.**  Stored as ``TEXT`` everywhere — we never
   query *into* the JSON, so the portability win beats native
   ``JSONB``.

Known limitation (v1): the "next per-id seq number" computation
uses ``SELECT COALESCE(MAX(seq), 0) + 1`` inside a transaction.
Within one process the store's own :class:`asyncio.Lock`
serializes writes; **cross-process concurrent writes to the
same id are unsafe** and may collide on the ``(id, seq)`` unique
constraint.  Hosts that need cross-process write concurrency
should run a single recorder process or migrate to a
sequence-per-id scheme.  Workflow runs (each one gets its own
``run_id``) and per-user conversations rarely hit this in
practice.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any


class Dialect(str, Enum):
    """Supported DB-API engines.

    The caller declares the dialect at store construction —
    explicit beats trying to sniff ``module.paramstyle`` from
    the connection's class, which is fiddly and surprising.
    """

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


# ---------------------------------------------------------------------------
# Placeholder rewrite
# ---------------------------------------------------------------------------


def adapt_sql(sql: str, dialect: Dialect) -> str:
    """Rewrite ``?`` placeholders for non-sqlite dialects.

    PostgreSQL drivers (``psycopg``, ``psycopg2``, ``pg8000``)
    and MySQL drivers (``PyMySQL``, ``mysqlclient``,
    ``mysql-connector-python``) use ``%s`` as the positional
    placeholder.  sqlite uses ``?``.

    We intentionally don't try to be clever about strings that
    contain literal ``?`` characters — store SQL is in-tree
    constant text under our control, not user input.
    """
    if dialect is Dialect.SQLITE:
        return sql
    return sql.replace("?", "%s")


# ---------------------------------------------------------------------------
# DDL — one ``CREATE TABLE IF NOT EXISTS`` per dialect.
# ---------------------------------------------------------------------------
# Tables (logical):
#
#   aac_transcript_runs(run_id PK, workflow_name, tool_use_id,
#                       args_json, started_at, ended_at NULL,
#                       outcome NULL)
#   aac_transcript_events(run_id, seq, event_type, payload_json,
#                         value_json, error, PK(run_id, seq))
#   aac_conversation_messages(conversation_id, seq, content_json,
#                             PK(conversation_id, seq))
#
# All JSON-bearing columns are TEXT.  ``seq`` is computed by the
# store as MAX(seq)+1 inside a transaction.


_TRANSCRIPT_RUNS_DDL: dict[Dialect, str] = {
    Dialect.SQLITE: """
        CREATE TABLE IF NOT EXISTS aac_transcript_runs (
            run_id        TEXT PRIMARY KEY,
            workflow_name TEXT NOT NULL,
            tool_use_id   TEXT NOT NULL,
            args_json     TEXT NOT NULL,
            started_at    TEXT NOT NULL,
            ended_at      TEXT,
            outcome       TEXT
        )
    """,
    Dialect.POSTGRESQL: """
        CREATE TABLE IF NOT EXISTS aac_transcript_runs (
            run_id        TEXT PRIMARY KEY,
            workflow_name TEXT NOT NULL,
            tool_use_id   TEXT NOT NULL,
            args_json     TEXT NOT NULL,
            started_at    TEXT NOT NULL,
            ended_at      TEXT,
            outcome       TEXT
        )
    """,
    Dialect.MYSQL: """
        CREATE TABLE IF NOT EXISTS aac_transcript_runs (
            run_id        VARCHAR(255) PRIMARY KEY,
            workflow_name TEXT NOT NULL,
            tool_use_id   TEXT NOT NULL,
            args_json     MEDIUMTEXT NOT NULL,
            started_at    VARCHAR(64) NOT NULL,
            ended_at      VARCHAR(64),
            outcome       VARCHAR(16)
        )
    """,
}


_TRANSCRIPT_EVENTS_DDL: dict[Dialect, str] = {
    Dialect.SQLITE: """
        CREATE TABLE IF NOT EXISTS aac_transcript_events (
            run_id       TEXT NOT NULL,
            seq          INTEGER NOT NULL,
            event_type   TEXT NOT NULL,
            payload_json TEXT,
            value_json   TEXT,
            error        TEXT,
            PRIMARY KEY (run_id, seq)
        )
    """,
    Dialect.POSTGRESQL: """
        CREATE TABLE IF NOT EXISTS aac_transcript_events (
            run_id       TEXT NOT NULL,
            seq          BIGINT NOT NULL,
            event_type   TEXT NOT NULL,
            payload_json TEXT,
            value_json   TEXT,
            error        TEXT,
            PRIMARY KEY (run_id, seq)
        )
    """,
    Dialect.MYSQL: """
        CREATE TABLE IF NOT EXISTS aac_transcript_events (
            run_id       VARCHAR(255) NOT NULL,
            seq          BIGINT NOT NULL,
            event_type   VARCHAR(64) NOT NULL,
            payload_json MEDIUMTEXT,
            value_json   MEDIUMTEXT,
            error        TEXT,
            PRIMARY KEY (run_id, seq)
        )
    """,
}


_CONVERSATION_MESSAGES_DDL: dict[Dialect, str] = {
    Dialect.SQLITE: """
        CREATE TABLE IF NOT EXISTS aac_conversation_messages (
            conversation_id TEXT NOT NULL,
            seq             INTEGER NOT NULL,
            content_json    TEXT NOT NULL,
            PRIMARY KEY (conversation_id, seq)
        )
    """,
    Dialect.POSTGRESQL: """
        CREATE TABLE IF NOT EXISTS aac_conversation_messages (
            conversation_id TEXT NOT NULL,
            seq             BIGINT NOT NULL,
            content_json    TEXT NOT NULL,
            PRIMARY KEY (conversation_id, seq)
        )
    """,
    Dialect.MYSQL: """
        CREATE TABLE IF NOT EXISTS aac_conversation_messages (
            conversation_id VARCHAR(255) NOT NULL,
            seq             BIGINT NOT NULL,
            content_json    MEDIUMTEXT NOT NULL,
            PRIMARY KEY (conversation_id, seq)
        )
    """,
}


def transcript_runs_ddl(dialect: Dialect) -> str:
    return _TRANSCRIPT_RUNS_DDL[dialect]


def transcript_events_ddl(dialect: Dialect) -> str:
    return _TRANSCRIPT_EVENTS_DDL[dialect]


def conversation_messages_ddl(dialect: Dialect) -> str:
    return _CONVERSATION_MESSAGES_DDL[dialect]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def to_json(value: Any) -> str:
    """JSON-encode for a TEXT column.

    ``default=str`` matches the file-backend choice — non-JSON-
    native values (datetime, UUID, set) survive as strings
    instead of crashing the write.  Round-trip fidelity is good
    enough for transcript/conversation use cases where we never
    query *into* the JSON.
    """
    return json.dumps(value, default=str, ensure_ascii=False)


def from_json(text: str | None) -> Any:
    """JSON-decode a TEXT column, treating NULL / empty as ``None``."""
    if text is None or text == "":
        return None
    return json.loads(text)
