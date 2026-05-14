"""Portable ``ADD COLUMN IF NOT EXISTS`` for the SQL stores.

``CREATE TABLE IF NOT EXISTS`` (used by the stores' ``_ensure_schema``)
doesn't *migrate* existing tables when a new column is added to
the canonical DDL — it only protects fresh tables.  When a column
ships in a later version of this library (e.g. the ``ts`` column
introduced for per-event timestamps), operators with existing
data need a way to add it idempotently without dropping the
table.

The three target dialects diverge on the syntax:

* **sqlite** — pre-3.35 has no ``IF NOT EXISTS``; we check
  ``PRAGMA table_info`` and skip if the column is present.
* **PostgreSQL** — ``ALTER TABLE … ADD COLUMN IF NOT EXISTS …``
  works in 9.6+ (covers Aurora PG too).
* **MySQL** — no ``IF NOT EXISTS`` for ``ADD COLUMN``; we try
  the ALTER and swallow the ``Duplicate column`` error.

Public surface:

* :func:`add_column_if_missing` — generic helper.
* :func:`ensure_transcript_events_ts_column` — convenience for
  the specific column upgrade shipped in the per-event-timestamps
  PR.  Hosts upgrading across that version call this once at
  startup; new deployments don't need it (the column is in the
  ``CREATE TABLE`` DDL).
"""

from __future__ import annotations

from typing import Any

from ai_assistant_client.persistence.sql_common import Dialect


def add_column_if_missing(
    conn: Any,
    *,
    dialect: Dialect,
    table: str,
    column: str,
    column_type_sql: str,
    default_sql: str | None = None,
) -> bool:
    """Add ``column`` to ``table`` if not already present.

    Returns ``True`` if the column was added on this call,
    ``False`` if it already existed.  Idempotent — safe to call
    every startup; the cost is one introspection query if the
    column is present.

    ``column_type_sql`` is the engine-specific type clause (e.g.
    ``"TEXT"``, ``"BIGINT"``, ``"VARCHAR(64)"``).  Callers pass
    the right value for the dialect; we don't try to map a
    cross-DB abstract type.  ``default_sql`` is a literal SQL
    expression (e.g. ``"''"``, ``"0"``); set when the column is
    NOT NULL so existing rows have a valid value after the
    upgrade.

    DB-API 2.0 connection expected — runs the work synchronously
    (callers should wrap in ``asyncio.to_thread`` if they're on
    the agent loop's event loop).
    """
    if _has_column(conn, dialect=dialect, table=table, column=column):
        return False
    parts = ["ALTER TABLE", table, "ADD COLUMN", column, column_type_sql]
    if default_sql is not None:
        parts += ["NOT NULL", "DEFAULT", default_sql]
    sql = " ".join(parts)
    cur = conn.cursor()
    try:
        try:
            cur.execute(sql)
            conn.commit()
            return True
        except Exception as err:
            # MySQL pre-`IF NOT EXISTS` raises on duplicate adds;
            # ignore so the operation stays idempotent.
            if _is_duplicate_column_error(err):
                conn.rollback()
                return False
            conn.rollback()
            raise
    finally:
        cur.close()


def ensure_transcript_events_ts_column(
    conn: Any, *, dialect: Dialect
) -> bool:
    """Idempotently add the ``ts`` column to ``aac_transcript_events``.

    The column ships in the canonical DDL since the per-event-
    timestamps PR (#18 / the per-event-timestamps slice in this
    stack); operators who created the table on an earlier
    version need this migration on first boot of the upgraded
    deployment.  Returns ``True`` when the column was added.
    """
    # NOT NULL DEFAULT '' so old rows read back as "timestamp
    # unknown" (matches the WorkflowEvent.timestamp default for
    # records reconstructed without a ts).
    column_type_sql = (
        "VARCHAR(64)" if dialect is Dialect.MYSQL else "TEXT"
    )
    return add_column_if_missing(
        conn,
        dialect=dialect,
        table="aac_transcript_events",
        column="ts",
        column_type_sql=column_type_sql,
        default_sql="''",
    )


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def _has_column(
    conn: Any, *, dialect: Dialect, table: str, column: str
) -> bool:
    """Return whether ``table`` has ``column``.

    Each dialect has its own catalog query — none of the three
    expose ``information_schema`` identically (sqlite doesn't
    have it at all), so we dispatch.
    """
    cur = conn.cursor()
    try:
        if dialect is Dialect.SQLITE:
            # PRAGMA returns one row per column: (cid, name, type, ...).
            cur.execute(f"PRAGMA table_info({_quote_ident(table)})")
            for row in cur.fetchall():
                if row[1] == column:
                    return True
            return False
        if dialect is Dialect.POSTGRESQL:
            cur.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = %s AND column_name = %s",
                (table, column),
            )
            return cur.fetchone() is not None
        # MySQL
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = DATABASE() "
            "AND table_name = %s AND column_name = %s",
            (table, column),
        )
        return cur.fetchone() is not None
    finally:
        cur.close()


def _quote_ident(name: str) -> str:
    """Defensive identifier quote for ``PRAGMA table_info``.

    sqlite's PRAGMA accepts a bare name; this just prevents
    accidental SQL-injection if a future caller passes a
    table name containing a paren or quote.  We reject the
    pathological characters outright rather than try to
    escape — table names in this codebase are constants under
    our control.
    """
    if any(c in name for c in '"()\\\'\n\r;'):
        raise ValueError(f"invalid table name {name!r}")
    return f'"{name}"'


def _is_duplicate_column_error(err: Exception) -> bool:
    """Best-effort detection of MySQL's duplicate-column error
    without importing the driver at module load.

    MySQL drivers raise ``OperationalError`` with error 1060
    ("Duplicate column name") on a re-add.  We match on the
    error string rather than the class so any driver works.
    """
    s = str(err).lower()
    return "duplicate column" in s or "1060" in s
