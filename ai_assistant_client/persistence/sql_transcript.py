"""DB-API-2.0-backed :class:`TranscriptStore`.

Works against any compliant driver — sqlite (stdlib), PostgreSQL
(``psycopg``/``psycopg2``/``pg8000``), or MySQL (``PyMySQL``/
``mysqlclient``/``mysql-connector-python``) — including AWS
Aurora PG/MySQL and RDS Proxy.  See
:mod:`ai_assistant_client.persistence.sql_common` for the
why-DB-API write-up.

Connection management is the caller's responsibility (IAM token
minting, TLS, pooling all stay upstream).  The store accepts a
single live connection and serializes all access through an
:class:`asyncio.Lock` — appropriate for a recorder embedded in a
single agent process.  Hosts that need cross-process write
fan-out should run a dedicated recorder or extend this with
their own pooling layer.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ai_assistant_client.persistence.sql_common import (
    SEQ_COLLISION_MAX_ATTEMPTS,
    Dialect,
    adapt_sql,
    from_json,
    is_integrity_error,
    to_json,
    transcript_events_ddl,
    transcript_runs_ddl,
)
from ai_assistant_client.persistence.transcript import (
    RunFooter,
    RunHeader,
    RunTranscript,
)
from ai_assistant_client.workflows.runtime import WorkflowEvent


class SqlTranscriptStore:
    """DB-API-2.0 implementation of
    :class:`~ai_assistant_client.persistence.transcript.TranscriptStore`.

    Tables ``aac_transcript_runs`` and ``aac_transcript_events`` are
    created on first use (``CREATE TABLE IF NOT EXISTS`` so a
    re-run against an existing schema is a no-op).
    """

    def __init__(self, conn: Any, *, dialect: Dialect) -> None:
        self._conn = conn
        self._dialect = dialect
        # One lock per store instance — serializes every DB op on
        # the underlying connection.  DB-API is sync; we offload
        # the actual execute call via asyncio.to_thread so the
        # agent's event loop stays responsive even when the DB is
        # slow.
        self._lock = asyncio.Lock()
        self._initialized = False

    # -- schema bootstrap ------------------------------------------

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return

        def _run() -> None:
            cur = self._conn.cursor()
            try:
                cur.execute(transcript_runs_ddl(self._dialect))
                cur.execute(transcript_events_ddl(self._dialect))
                self._conn.commit()
            finally:
                cur.close()

        await asyncio.to_thread(_run)
        self._initialized = True

    # -- write path ------------------------------------------------

    async def write_header(self, header: RunHeader) -> None:
        await self._ensure_schema()
        sql = adapt_sql(
            "INSERT INTO aac_transcript_runs "
            "(run_id, workflow_name, tool_use_id, args_json, started_at) "
            "VALUES (?, ?, ?, ?, ?)",
            self._dialect,
        )

        def _run() -> None:
            cur = self._conn.cursor()
            try:
                cur.execute(
                    sql,
                    (
                        header.run_id,
                        header.workflow_name,
                        header.tool_use_id,
                        to_json(header.args),
                        header.started_at,
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

        async with self._lock:
            try:
                await asyncio.to_thread(_run)
            except Exception as err:
                # PK violation on duplicate run_id — surface as
                # ValueError to match the in-memory + file
                # backends' contract.
                if _is_integrity_error(err):
                    raise ValueError(
                        f"run id {header.run_id!r} already has a header — "
                        "transcripts are append-only and not resumable"
                    ) from err
                raise

    async def append_event(self, run_id: str, event: WorkflowEvent) -> None:
        await self._ensure_schema()
        # Guard against append-before-header by checking the runs
        # table inside the same transaction as the insert.
        select_run_sql = adapt_sql(
            "SELECT 1 FROM aac_transcript_runs WHERE run_id = ?",
            self._dialect,
        )
        next_seq_sql = adapt_sql(
            "SELECT COALESCE(MAX(seq), 0) + 1 "
            "FROM aac_transcript_events WHERE run_id = ?",
            self._dialect,
        )
        insert_sql = adapt_sql(
            "INSERT INTO aac_transcript_events "
            "(run_id, seq, event_type, payload_json, value_json, error, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            self._dialect,
        )

        def _run() -> None:
            # Retry loop around the SELECT-MAX(seq)+INSERT pattern.
            # Within a single process the asyncio.Lock prevents
            # collisions; this loop handles the cross-process case
            # where another writer commits a row with the same
            # ``seq`` between our SELECT and INSERT, producing an
            # IntegrityError on the ``(run_id, seq)`` PK.
            last_err: Exception | None = None
            for _attempt in range(SEQ_COLLISION_MAX_ATTEMPTS):
                cur = self._conn.cursor()
                try:
                    cur.execute(select_run_sql, (run_id,))
                    if cur.fetchone() is None:
                        raise KeyError(run_id)
                    cur.execute(next_seq_sql, (run_id,))
                    row = cur.fetchone()
                    next_seq = int(row[0]) if row and row[0] is not None else 1
                    cur.execute(
                        insert_sql,
                        (
                            run_id,
                            next_seq,
                            event.type,
                            to_json(event.payload) if event.payload is not None else None,
                            to_json(event.value) if event.value is not None else None,
                            event.error,
                            event.timestamp,
                        ),
                    )
                    self._conn.commit()
                    return
                except KeyError:
                    self._conn.rollback()
                    raise
                except Exception as err:
                    self._conn.rollback()
                    if is_integrity_error(err):
                        last_err = err
                        continue  # retry with fresh MAX(seq)+1
                    raise
                finally:
                    cur.close()
            # Exhausted retries — propagate the last seq-collision
            # error so the caller knows to back off (or run a
            # single-leader recorder).
            assert last_err is not None
            raise last_err

        async with self._lock:
            await asyncio.to_thread(_run)

    async def write_footer(self, run_id: str, footer: RunFooter) -> None:
        await self._ensure_schema()
        select_run_sql = adapt_sql(
            "SELECT 1 FROM aac_transcript_runs WHERE run_id = ?",
            self._dialect,
        )
        update_sql = adapt_sql(
            "UPDATE aac_transcript_runs "
            "SET ended_at = ?, outcome = ? WHERE run_id = ?",
            self._dialect,
        )

        def _run() -> None:
            cur = self._conn.cursor()
            try:
                cur.execute(select_run_sql, (run_id,))
                if cur.fetchone() is None:
                    raise KeyError(run_id)
                cur.execute(
                    update_sql,
                    (footer.ended_at, footer.outcome, run_id),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

        async with self._lock:
            await asyncio.to_thread(_run)

    # -- read path -------------------------------------------------

    async def read(self, run_id: str) -> RunTranscript:
        await self._ensure_schema()
        run_sql = adapt_sql(
            "SELECT workflow_name, tool_use_id, args_json, started_at, "
            "ended_at, outcome FROM aac_transcript_runs WHERE run_id = ?",
            self._dialect,
        )
        events_sql = adapt_sql(
            "SELECT event_type, payload_json, value_json, error, ts "
            "FROM aac_transcript_events WHERE run_id = ? ORDER BY seq ASC",
            self._dialect,
        )

        def _run() -> RunTranscript:
            cur = self._conn.cursor()
            try:
                cur.execute(run_sql, (run_id,))
                row = cur.fetchone()
                if row is None:
                    raise KeyError(run_id)
                header = RunHeader(
                    run_id=run_id,
                    workflow_name=row[0],
                    tool_use_id=row[1],
                    args=from_json(row[2]) or {},
                    started_at=row[3],
                )
                ended_at, outcome = row[4], row[5]
                footer: RunFooter | None = None
                if ended_at is not None and outcome is not None:
                    footer = RunFooter(ended_at=ended_at, outcome=outcome)

                cur.execute(events_sql, (run_id,))
                events: list[WorkflowEvent] = []
                for ev_row in cur.fetchall():
                    events.append(
                        WorkflowEvent(
                            type=ev_row[0],
                            payload=from_json(ev_row[1]),
                            value=from_json(ev_row[2]),
                            error=ev_row[3],
                            timestamp=ev_row[4] or "",
                        )
                    )
                return RunTranscript(
                    header=header, events=events, footer=footer
                )
            finally:
                cur.close()

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def list_runs(self) -> list[str]:
        await self._ensure_schema()
        sql = "SELECT run_id FROM aac_transcript_runs"

        def _run() -> list[str]:
            cur = self._conn.cursor()
            try:
                cur.execute(sql)
                return [r[0] for r in cur.fetchall()]
            finally:
                cur.close()

        async with self._lock:
            return await asyncio.to_thread(_run)


def _is_integrity_error(err: Exception) -> bool:
    """Best-effort detection of unique/PK violations across drivers.

    DB-API 2.0 defines an ``IntegrityError`` exception class on
    every compliant module, but the actual class lives on the
    driver module — not on the connection or the exception itself
    in a uniform way.  Walk the MRO looking for a class literally
    named ``IntegrityError``; falls back to ``False`` if not
    found, in which case the caller re-raises the original.
    """
    for cls in type(err).__mro__:
        if cls.__name__ == "IntegrityError":
            return True
    return False
