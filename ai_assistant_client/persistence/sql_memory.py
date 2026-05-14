"""DB-API-2.0-backed :class:`MemoryStore`.

Same engine matrix as the transcript / conversation SQL stores
— sqlite (stdlib), PostgreSQL (incl. AWS Aurora PG + RDS),
MySQL (incl. Aurora MySQL).  See
:mod:`ai_assistant_client.persistence.sql_common` for the
why-DB-API write-up that applies to all three SQL families.

Per-user isolation invariants from the protocol are enforced
at the SQL level: every ``get`` / ``update`` / ``remove`` filters
on ``WHERE memory_id = ? AND user_id = ?`` so a probing caller
that knows another user's memory id can't read or mutate it.
Cross-owner lookups produce the same ``KeyError`` as missing
records — same anti-enumeration property as the in-process and
file backends.

``seq`` is a per-row monotonic integer assigned at insert time so
``list(user_id=...)`` can ``ORDER BY seq ASC`` and return
insertion-order without depending on physical-storage order.
Computed as ``MAX(seq)+1`` inside a transaction; same
cross-process race the other SQL stores have — single-process
recording is safe, multi-process needs a leader.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable

from ai_assistant_client.persistence.sql_common import (
    Dialect,
    adapt_sql,
    from_json,
    memory_ddl,
    memory_index_ddl,
    to_json,
)
from ai_assistant_client.persistence.user_memory import Memory, MemoryStore
from ai_assistant_client.persistence.user_memory_local import (
    _new_memory_id,
    _utc_iso,
)


class SqlMemoryStore(MemoryStore):
    """DB-API 2.0 implementation of :class:`MemoryStore`."""

    def __init__(self, conn: Any, *, dialect: Dialect) -> None:
        self._conn = conn
        self._dialect = dialect
        self._lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return

        def _run() -> None:
            cur = self._conn.cursor()
            try:
                cur.execute(memory_ddl(self._dialect))
                try:
                    cur.execute(memory_index_ddl(self._dialect))
                except Exception:
                    # MySQL < 8 doesn't support CREATE INDEX IF NOT
                    # EXISTS; on a re-run the duplicate-key error
                    # is the expected outcome.  Swallow it so the
                    # schema bootstrap stays idempotent.
                    pass
                self._conn.commit()
            finally:
                cur.close()

        await asyncio.to_thread(_run)
        self._initialized = True

    # -- write -----------------------------------------------------

    async def add(
        self,
        *,
        user_id: str,
        key: str,
        value: Any,
        tags: Iterable[str] = (),
    ) -> Memory:
        await self._ensure_schema()
        ts = _utc_iso()
        tag_tuple = tuple(tags)
        record = Memory(
            memory_id=_new_memory_id(),
            user_id=user_id,
            key=key,
            value=value,
            tags=tag_tuple,
            created_at=ts,
            updated_at=ts,
        )
        next_seq_sql = adapt_sql(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM aac_user_memories "
            "WHERE user_id = ?",
            self._dialect,
        )
        insert_sql = adapt_sql(
            "INSERT INTO aac_user_memories "
            "(memory_id, user_id, mkey, value_json, tags_json, "
            "created_at, updated_at, seq) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            self._dialect,
        )

        def _run() -> None:
            cur = self._conn.cursor()
            try:
                cur.execute(next_seq_sql, (user_id,))
                row = cur.fetchone()
                next_seq = int(row[0]) if row and row[0] is not None else 1
                cur.execute(
                    insert_sql,
                    (
                        record.memory_id,
                        record.user_id,
                        record.key,
                        to_json(record.value),
                        to_json(list(record.tags)),
                        record.created_at,
                        record.updated_at,
                        next_seq,
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

        async with self._lock:
            await asyncio.to_thread(_run)
        return record

    # -- read ------------------------------------------------------

    async def get(self, *, user_id: str, memory_id: str) -> Memory:
        await self._ensure_schema()
        sql = adapt_sql(
            "SELECT mkey, value_json, tags_json, created_at, updated_at "
            "FROM aac_user_memories WHERE memory_id = ? AND user_id = ?",
            self._dialect,
        )

        def _run() -> Memory:
            cur = self._conn.cursor()
            try:
                cur.execute(sql, (memory_id, user_id))
                row = cur.fetchone()
                if row is None:
                    raise KeyError(memory_id)
                return Memory(
                    memory_id=memory_id,
                    user_id=user_id,
                    key=row[0],
                    value=from_json(row[1]),
                    tags=tuple(from_json(row[2]) or []),
                    created_at=row[3],
                    updated_at=row[4],
                )
            finally:
                cur.close()

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def list(
        self,
        *,
        user_id: str,
        tags: Iterable[str] | None = None,
    ) -> list[Memory]:
        await self._ensure_schema()
        # ``memory_id`` as a deterministic tiebreak: if two
        # processes computed the same ``seq`` (cross-process
        # MAX+1 race; within a process the asyncio.Lock prevents
        # it), reads still return them in a stable order rather
        # than the DB engine's arbitrary tie-break.  memory_id is
        # the PK so the tiebreak is guaranteed unique.
        sql = adapt_sql(
            "SELECT memory_id, mkey, value_json, tags_json, "
            "created_at, updated_at "
            "FROM aac_user_memories WHERE user_id = ? "
            "ORDER BY seq ASC, memory_id ASC",
            self._dialect,
        )
        wanted: set[str] | None = set(tags) if tags is not None else None

        def _run() -> list[Memory]:
            cur = self._conn.cursor()
            try:
                cur.execute(sql, (user_id,))
                out: list[Memory] = []
                for row in cur.fetchall():
                    row_tags = tuple(from_json(row[3]) or [])
                    if wanted is not None and not wanted.intersection(
                        row_tags
                    ):
                        continue
                    out.append(
                        Memory(
                            memory_id=row[0],
                            user_id=user_id,
                            key=row[1],
                            value=from_json(row[2]),
                            tags=row_tags,
                            created_at=row[4],
                            updated_at=row[5],
                        )
                    )
                return out
            finally:
                cur.close()

        async with self._lock:
            return await asyncio.to_thread(_run)

    # -- update / remove ------------------------------------------

    async def update(
        self,
        *,
        user_id: str,
        memory_id: str,
        value: Any,
    ) -> Memory:
        await self._ensure_schema()
        select_sql = adapt_sql(
            "SELECT mkey, tags_json, created_at FROM aac_user_memories "
            "WHERE memory_id = ? AND user_id = ?",
            self._dialect,
        )
        update_sql = adapt_sql(
            "UPDATE aac_user_memories SET value_json = ?, updated_at = ? "
            "WHERE memory_id = ? AND user_id = ?",
            self._dialect,
        )
        ts = _utc_iso()
        encoded = to_json(value)

        def _run() -> Memory:
            cur = self._conn.cursor()
            try:
                cur.execute(select_sql, (memory_id, user_id))
                row = cur.fetchone()
                if row is None:
                    raise KeyError(memory_id)
                cur.execute(update_sql, (encoded, ts, memory_id, user_id))
                self._conn.commit()
                return Memory(
                    memory_id=memory_id,
                    user_id=user_id,
                    key=row[0],
                    value=value,
                    tags=tuple(from_json(row[1]) or []),
                    created_at=row[2],
                    updated_at=ts,
                )
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def remove(self, *, user_id: str, memory_id: str) -> None:
        await self._ensure_schema()
        select_sql = adapt_sql(
            "SELECT 1 FROM aac_user_memories "
            "WHERE memory_id = ? AND user_id = ?",
            self._dialect,
        )
        delete_sql = adapt_sql(
            "DELETE FROM aac_user_memories "
            "WHERE memory_id = ? AND user_id = ?",
            self._dialect,
        )

        def _run() -> None:
            cur = self._conn.cursor()
            try:
                cur.execute(select_sql, (memory_id, user_id))
                if cur.fetchone() is None:
                    raise KeyError(memory_id)
                cur.execute(delete_sql, (memory_id, user_id))
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

        async with self._lock:
            await asyncio.to_thread(_run)

    async def forget_all(self, *, user_id: str) -> int:
        await self._ensure_schema()
        count_sql = adapt_sql(
            "SELECT COUNT(*) FROM aac_user_memories WHERE user_id = ?",
            self._dialect,
        )
        delete_sql = adapt_sql(
            "DELETE FROM aac_user_memories WHERE user_id = ?",
            self._dialect,
        )

        def _run() -> int:
            cur = self._conn.cursor()
            try:
                cur.execute(count_sql, (user_id,))
                row = cur.fetchone()
                count = int(row[0]) if row else 0
                cur.execute(delete_sql, (user_id,))
                self._conn.commit()
                return count
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def list_users(self) -> list[str]:
        await self._ensure_schema()
        sql = "SELECT DISTINCT user_id FROM aac_user_memories"

        def _run() -> list[str]:
            cur = self._conn.cursor()
            try:
                cur.execute(sql)
                return [r[0] for r in cur.fetchall()]
            finally:
                cur.close()

        async with self._lock:
            return await asyncio.to_thread(_run)
