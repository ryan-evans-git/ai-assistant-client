"""asyncpg-backed :class:`TranscriptStore` for PostgreSQL.

Native-async equivalent of
:class:`~ai_assistant_client.persistence.sql_transcript.SqlTranscriptStore`
for hosts that want to avoid the ``asyncio.to_thread`` hop the
DB-API path takes.  Performance win shows up at high recording
throughput (hundreds of events/sec); for a single-user
deployment the difference is unmeasurable and the DB-API path
is the simpler choice.

Connection management:

* Accepts an ``asyncpg.Pool`` (the canonical asyncpg primitive).
* Every operation does ``async with pool.acquire() as conn`` so
  short transactions don't block the pool.
* The caller is responsible for the Pool lifecycle (creation,
  IAM-token refresh, RDS Proxy endpoints, TLS config) — same
  upstream-owns-credentials philosophy as the DB-API stores.

Schema reuses the DDL from
:mod:`ai_assistant_client.persistence.sql_common` (PostgreSQL
dialect), so a database initialised by the DB-API store and one
initialised by this store are byte-equivalent — you can switch
between the two without migrations.

This is an *optional* extra: ``pip install ai-assistant-client[asyncpg]``
installs the driver.  Import-failing here surfaces as a clear
ImportError pointing at the extra rather than the cryptic
``ModuleNotFoundError`` you'd otherwise hit.
"""

from __future__ import annotations

import json
from typing import Any

from ai_assistant_client.persistence.sql_common import (
    SEQ_COLLISION_MAX_ATTEMPTS,
    Dialect,
    adapt_sql_asyncpg,
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


class AsyncpgTranscriptStore:
    """asyncpg implementation of :class:`TranscriptStore`."""

    def __init__(self, pool: Any) -> None:
        # Duck-typed: we only need ``pool.acquire()`` returning an
        # async-context-manager-yielding-Connection with
        # ``execute``, ``fetchrow``, ``fetch`` async methods.  This
        # keeps the test mocks tiny and decouples the store from
        # any specific asyncpg version.
        self._pool = pool
        self._initialized = False

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        runs_ddl = transcript_runs_ddl(Dialect.POSTGRESQL)
        events_ddl = transcript_events_ddl(Dialect.POSTGRESQL)
        async with self._pool.acquire() as conn:
            await conn.execute(runs_ddl)
            await conn.execute(events_ddl)
        self._initialized = True

    async def write_header(self, header: RunHeader) -> None:
        await self._ensure_schema()
        sql = adapt_sql_asyncpg(
            "INSERT INTO aac_transcript_runs "
            "(run_id, workflow_name, tool_use_id, args_json, started_at) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    sql,
                    header.run_id,
                    header.workflow_name,
                    header.tool_use_id,
                    to_json(header.args),
                    header.started_at,
                )
        except Exception as err:
            # asyncpg raises ``UniqueViolationError`` (subclass of
            # ``PostgresError``) on PK collisions.  Surface as
            # ValueError to match the contract of the other stores.
            if _is_unique_violation(err):
                raise ValueError(
                    f"run id {header.run_id!r} already has a header — "
                    "transcripts are append-only and not resumable"
                ) from err
            raise

    async def append_event(self, run_id: str, event: WorkflowEvent) -> None:
        await self._ensure_schema()
        check_sql = adapt_sql_asyncpg(
            "SELECT 1 FROM aac_transcript_runs WHERE run_id = ?"
        )
        next_seq_sql = adapt_sql_asyncpg(
            "SELECT COALESCE(MAX(seq), 0) + 1 "
            "FROM aac_transcript_events WHERE run_id = ?"
        )
        insert_sql = adapt_sql_asyncpg(
            "INSERT INTO aac_transcript_events "
            "(run_id, seq, event_type, payload_json, value_json, error, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        # Retry on seq-collision IntegrityError from cross-process
        # races — same shape as SqlTranscriptStore.append_event,
        # using asyncpg's UniqueViolationError (subclass of
        # IntegrityError) as the trigger.  Each attempt opens a
        # fresh transaction so the prior aborted one is cleaned up.
        last_err: Exception | None = None
        for _attempt in range(SEQ_COLLISION_MAX_ATTEMPTS):
            try:
                async with self._pool.acquire() as conn:
                    async with conn.transaction():
                        row = await conn.fetchrow(check_sql, run_id)
                        if row is None:
                            raise KeyError(run_id)
                        seq_row = await conn.fetchrow(next_seq_sql, run_id)
                        next_seq = (
                            int(seq_row[0])
                            if seq_row and seq_row[0] is not None
                            else 1
                        )
                        await conn.execute(
                            insert_sql,
                            run_id,
                            next_seq,
                            event.type,
                            to_json(event.payload) if event.payload is not None else None,
                            to_json(event.value) if event.value is not None else None,
                            event.error,
                            event.timestamp,
                        )
                return
            except KeyError:
                raise
            except Exception as err:
                if is_integrity_error(err):
                    last_err = err
                    continue
                raise
        assert last_err is not None
        raise last_err

    async def write_footer(self, run_id: str, footer: RunFooter) -> None:
        await self._ensure_schema()
        check_sql = adapt_sql_asyncpg(
            "SELECT 1 FROM aac_transcript_runs WHERE run_id = ?"
        )
        update_sql = adapt_sql_asyncpg(
            "UPDATE aac_transcript_runs "
            "SET ended_at = ?, outcome = ? WHERE run_id = ?"
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(check_sql, run_id)
            if row is None:
                raise KeyError(run_id)
            await conn.execute(
                update_sql, footer.ended_at, footer.outcome, run_id
            )

    async def read(self, run_id: str) -> RunTranscript:
        await self._ensure_schema()
        run_sql = adapt_sql_asyncpg(
            "SELECT workflow_name, tool_use_id, args_json, started_at, "
            "ended_at, outcome FROM aac_transcript_runs WHERE run_id = ?"
        )
        events_sql = adapt_sql_asyncpg(
            "SELECT event_type, payload_json, value_json, error, ts "
            "FROM aac_transcript_events WHERE run_id = ? ORDER BY seq ASC"
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(run_sql, run_id)
            if row is None:
                raise KeyError(run_id)
            args_raw = row[2]
            header = RunHeader(
                run_id=run_id,
                workflow_name=row[0],
                tool_use_id=row[1],
                args=_decode_json_arg(args_raw),
                started_at=row[3],
            )
            ended_at, outcome = row[4], row[5]
            footer: RunFooter | None = None
            if ended_at is not None and outcome is not None:
                footer = RunFooter(ended_at=ended_at, outcome=outcome)

            events: list[WorkflowEvent] = []
            for ev_row in await conn.fetch(events_sql, run_id):
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

    async def list_runs(self) -> list[str]:
        await self._ensure_schema()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT run_id FROM aac_transcript_runs")
            return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_unique_violation(err: Exception) -> bool:
    """Detect asyncpg unique-constraint violations without importing
    asyncpg at module load (so the optional-dep import error stays
    localised to the constructor / first call)."""
    for cls in type(err).__mro__:
        # asyncpg uses ``UniqueViolationError`` for both PK and
        # unique-constraint failures.
        if cls.__name__ == "UniqueViolationError":
            return True
    return False


def _decode_json_arg(raw: Any) -> dict[str, Any]:
    """Decode a JSON-encoded args column.

    asyncpg's TEXT columns come back as ``str``; if a host wires
    a ``JSONB`` column instead, asyncpg will deserialize for them
    and the value arrives as a dict already.  Handle both.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw) if raw else {}
    return {}
