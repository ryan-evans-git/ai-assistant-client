"""aiomysql-backed :class:`TranscriptStore` for MySQL / Aurora MySQL.

Native-async equivalent of the DB-API path for hosts that want
to avoid the ``asyncio.to_thread`` hop.  Same upstream-owns-the-
pool philosophy as the asyncpg store.

aiomysql's cursor API is DB-API-shaped but async (``await
cur.execute(...)``, ``await cur.fetchone()``).  Placeholder
style is ``%s`` — same as the sync DB-API stores' MySQL path —
so we reuse ``adapt_sql(..., Dialect.MYSQL)`` from sql_common.

Schema reuses the MySQL DDL from
:mod:`ai_assistant_client.persistence.sql_common`, so a database
initialised by the DB-API store and one initialised by this
store are byte-equivalent.

Optional extra: ``pip install ai-assistant-client[aiomysql]``.
"""

from __future__ import annotations

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


class AiomysqlTranscriptStore:
    """aiomysql implementation of :class:`TranscriptStore`."""

    def __init__(self, pool: Any) -> None:
        # ``pool`` must expose ``acquire()`` returning an async
        # context manager that yields a Connection with a
        # ``cursor()`` async context manager.  That's the
        # standard aiomysql.Pool / aiomysql.connect shape; any
        # compatible duck-type works (which is how the test
        # mocks stay small).
        self._pool = pool
        self._initialized = False

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        runs_ddl = transcript_runs_ddl(Dialect.MYSQL)
        events_ddl = transcript_events_ddl(Dialect.MYSQL)
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(runs_ddl)
                await cur.execute(events_ddl)
            await conn.commit()
        self._initialized = True

    async def write_header(self, header: RunHeader) -> None:
        await self._ensure_schema()
        sql = adapt_sql(
            "INSERT INTO aac_transcript_runs "
            "(run_id, workflow_name, tool_use_id, args_json, started_at) "
            "VALUES (?, ?, ?, ?, ?)",
            Dialect.MYSQL,
        )
        try:
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        sql,
                        (
                            header.run_id,
                            header.workflow_name,
                            header.tool_use_id,
                            to_json(header.args),
                            header.started_at,
                        ),
                    )
                await conn.commit()
        except Exception as err:
            if _is_integrity_error(err):
                raise ValueError(
                    f"run id {header.run_id!r} already has a header — "
                    "transcripts are append-only and not resumable"
                ) from err
            raise

    async def append_event(self, run_id: str, event: WorkflowEvent) -> None:
        await self._ensure_schema()
        check_sql = adapt_sql(
            "SELECT 1 FROM aac_transcript_runs WHERE run_id = ?",
            Dialect.MYSQL,
        )
        next_seq_sql = adapt_sql(
            "SELECT COALESCE(MAX(seq), 0) + 1 "
            "FROM aac_transcript_events WHERE run_id = ?",
            Dialect.MYSQL,
        )
        insert_sql = adapt_sql(
            "INSERT INTO aac_transcript_events "
            "(run_id, seq, event_type, payload_json, value_json, error, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            Dialect.MYSQL,
        )
        # Cross-process seq-collision retry — see
        # SqlTranscriptStore.append_event for the full rationale.
        last_err: Exception | None = None
        for _attempt in range(SEQ_COLLISION_MAX_ATTEMPTS):
            try:
                async with self._pool.acquire() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(check_sql, (run_id,))
                        if await cur.fetchone() is None:
                            raise KeyError(run_id)
                        await cur.execute(next_seq_sql, (run_id,))
                        row = await cur.fetchone()
                        next_seq = (
                            int(row[0])
                            if row and row[0] is not None
                            else 1
                        )
                        await cur.execute(
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
                    await conn.commit()
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
        check_sql = adapt_sql(
            "SELECT 1 FROM aac_transcript_runs WHERE run_id = ?",
            Dialect.MYSQL,
        )
        update_sql = adapt_sql(
            "UPDATE aac_transcript_runs "
            "SET ended_at = ?, outcome = ? WHERE run_id = ?",
            Dialect.MYSQL,
        )
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(check_sql, (run_id,))
                if await cur.fetchone() is None:
                    raise KeyError(run_id)
                await cur.execute(
                    update_sql, (footer.ended_at, footer.outcome, run_id)
                )
            await conn.commit()

    async def read(self, run_id: str) -> RunTranscript:
        await self._ensure_schema()
        run_sql = adapt_sql(
            "SELECT workflow_name, tool_use_id, args_json, started_at, "
            "ended_at, outcome FROM aac_transcript_runs WHERE run_id = ?",
            Dialect.MYSQL,
        )
        events_sql = adapt_sql(
            "SELECT event_type, payload_json, value_json, error, ts "
            "FROM aac_transcript_events WHERE run_id = ? ORDER BY seq ASC",
            Dialect.MYSQL,
        )
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(run_sql, (run_id,))
                row = await cur.fetchone()
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

                await cur.execute(events_sql, (run_id,))
                events: list[WorkflowEvent] = []
                for ev_row in await cur.fetchall():
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
            async with conn.cursor() as cur:
                await cur.execute("SELECT run_id FROM aac_transcript_runs")
                rows = await cur.fetchall()
                return [r[0] for r in rows]


def _is_integrity_error(err: Exception) -> bool:
    """Detect aiomysql / PyMySQL integrity errors without importing
    them at module load.  Same pattern as the sync SQL store."""
    for cls in type(err).__mro__:
        if cls.__name__ == "IntegrityError":
            return True
    return False
