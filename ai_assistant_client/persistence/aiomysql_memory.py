"""aiomysql-backed :class:`MemoryStore` for MySQL / Aurora MySQL.

See :mod:`ai_assistant_client.persistence.aiomysql_transcript` for
the rationale that applies to all three native-async stores.
Schema reuses the MySQL memory DDL from
:mod:`ai_assistant_client.persistence.sql_common`.

Per-user isolation enforced at the SQL level with the same
``WHERE memory_id = %s AND user_id = %s`` pattern as the DB-API
store; placeholders use the aiomysql-native ``%s`` style via
:func:`adapt_sql`.
"""

from __future__ import annotations

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


class AiomysqlMemoryStore(MemoryStore):
    """aiomysql implementation of :class:`MemoryStore`."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._initialized = False

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(memory_ddl(Dialect.MYSQL))
                try:
                    await cur.execute(memory_index_ddl(Dialect.MYSQL))
                except Exception:
                    # MySQL pre-8 lacks CREATE INDEX IF NOT EXISTS;
                    # on rerun the duplicate-key error is expected.
                    pass
            await conn.commit()
        self._initialized = True

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
            Dialect.MYSQL,
        )
        insert_sql = adapt_sql(
            "INSERT INTO aac_user_memories "
            "(memory_id, user_id, mkey, value_json, tags_json, "
            "created_at, updated_at, seq) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            Dialect.MYSQL,
        )
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(next_seq_sql, (user_id,))
                row = await cur.fetchone()
                next_seq = int(row[0]) if row and row[0] is not None else 1
                await cur.execute(
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
            await conn.commit()
        return record

    async def get(self, *, user_id: str, memory_id: str) -> Memory:
        await self._ensure_schema()
        sql = adapt_sql(
            "SELECT mkey, value_json, tags_json, created_at, updated_at "
            "FROM aac_user_memories WHERE memory_id = ? AND user_id = ?",
            Dialect.MYSQL,
        )
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (memory_id, user_id))
                row = await cur.fetchone()
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

    async def list(
        self,
        *,
        user_id: str,
        tags: Iterable[str] | None = None,
    ) -> list[Memory]:
        await self._ensure_schema()
        # memory_id tiebreak — see SqlMemoryStore.list for the
        # full rationale (deterministic order even under the
        # cross-process MAX(seq)+1 race).
        sql = adapt_sql(
            "SELECT memory_id, mkey, value_json, tags_json, "
            "created_at, updated_at "
            "FROM aac_user_memories WHERE user_id = ? "
            "ORDER BY seq ASC, memory_id ASC",
            Dialect.MYSQL,
        )
        wanted: set[str] | None = set(tags) if tags is not None else None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (user_id,))
                rows = await cur.fetchall()
                out: list[Memory] = []
                for row in rows:
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
            Dialect.MYSQL,
        )
        update_sql = adapt_sql(
            "UPDATE aac_user_memories SET value_json = ?, updated_at = ? "
            "WHERE memory_id = ? AND user_id = ?",
            Dialect.MYSQL,
        )
        ts = _utc_iso()
        encoded = to_json(value)
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(select_sql, (memory_id, user_id))
                row = await cur.fetchone()
                if row is None:
                    raise KeyError(memory_id)
                await cur.execute(
                    update_sql, (encoded, ts, memory_id, user_id)
                )
            await conn.commit()
            return Memory(
                memory_id=memory_id,
                user_id=user_id,
                key=row[0],
                value=value,
                tags=tuple(from_json(row[1]) or []),
                created_at=row[2],
                updated_at=ts,
            )

    async def remove(self, *, user_id: str, memory_id: str) -> None:
        await self._ensure_schema()
        select_sql = adapt_sql(
            "SELECT 1 FROM aac_user_memories "
            "WHERE memory_id = ? AND user_id = ?",
            Dialect.MYSQL,
        )
        delete_sql = adapt_sql(
            "DELETE FROM aac_user_memories "
            "WHERE memory_id = ? AND user_id = ?",
            Dialect.MYSQL,
        )
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(select_sql, (memory_id, user_id))
                if await cur.fetchone() is None:
                    raise KeyError(memory_id)
                await cur.execute(delete_sql, (memory_id, user_id))
            await conn.commit()

    async def forget_all(self, *, user_id: str) -> int:
        await self._ensure_schema()
        count_sql = adapt_sql(
            "SELECT COUNT(*) FROM aac_user_memories WHERE user_id = ?",
            Dialect.MYSQL,
        )
        delete_sql = adapt_sql(
            "DELETE FROM aac_user_memories WHERE user_id = ?",
            Dialect.MYSQL,
        )
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(count_sql, (user_id,))
                row = await cur.fetchone()
                count = int(row[0]) if row else 0
                await cur.execute(delete_sql, (user_id,))
            await conn.commit()
            return count

    async def list_users(self) -> list[str]:
        await self._ensure_schema()
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT DISTINCT user_id FROM aac_user_memories"
                )
                rows = await cur.fetchall()
                return [r[0] for r in rows]
