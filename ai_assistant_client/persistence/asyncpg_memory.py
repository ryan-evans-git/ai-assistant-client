r"""asyncpg-backed :class:`MemoryStore` for PostgreSQL.

See :mod:`ai_assistant_client.persistence.asyncpg_transcript` for
the rationale that applies to all three native-async stores.
Schema reuses the PostgreSQL memory DDL from
:mod:`ai_assistant_client.persistence.sql_common` so a database
initialised by this store is byte-equivalent to one initialised
by :class:`SqlMemoryStore` — switch sync/async without migrating.

Per-user isolation invariants from the protocol are enforced at
the SQL level on every ``get`` / ``update`` / ``remove`` — same
``WHERE memory_id = $1 AND user_id = $2`` pattern as the
DB-API store, with asyncpg-style ``$N`` placeholders.
"""

from __future__ import annotations

from typing import Any, Iterable

from ai_assistant_client.persistence.sql_common import (
    Dialect,
    adapt_sql_asyncpg,
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


class AsyncpgMemoryStore(MemoryStore):
    """asyncpg implementation of :class:`MemoryStore`."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._initialized = False

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(memory_ddl(Dialect.POSTGRESQL))
            try:
                await conn.execute(memory_index_ddl(Dialect.POSTGRESQL))
            except Exception:
                pass
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
        next_seq_sql = adapt_sql_asyncpg(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM aac_user_memories "
            "WHERE user_id = ?"
        )
        insert_sql = adapt_sql_asyncpg(
            "INSERT INTO aac_user_memories "
            "(memory_id, user_id, mkey, value_json, tags_json, "
            "created_at, updated_at, seq) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                seq_row = await conn.fetchrow(next_seq_sql, user_id)
                next_seq = int(seq_row[0]) if seq_row and seq_row[0] is not None else 1
                await conn.execute(
                    insert_sql,
                    record.memory_id,
                    record.user_id,
                    record.key,
                    to_json(record.value),
                    to_json(list(record.tags)),
                    record.created_at,
                    record.updated_at,
                    next_seq,
                )
        return record

    async def get(self, *, user_id: str, memory_id: str) -> Memory:
        await self._ensure_schema()
        sql = adapt_sql_asyncpg(
            "SELECT mkey, value_json, tags_json, created_at, updated_at "
            "FROM aac_user_memories WHERE memory_id = ? AND user_id = ?"
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, memory_id, user_id)
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
        sql = adapt_sql_asyncpg(
            "SELECT memory_id, mkey, value_json, tags_json, "
            "created_at, updated_at "
            "FROM aac_user_memories WHERE user_id = ? ORDER BY seq ASC"
        )
        wanted: set[str] | None = set(tags) if tags is not None else None
        async with self._pool.acquire() as conn:
            out: list[Memory] = []
            for row in await conn.fetch(sql, user_id):
                row_tags = tuple(from_json(row[3]) or [])
                if wanted is not None and not wanted.intersection(row_tags):
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
        select_sql = adapt_sql_asyncpg(
            "SELECT mkey, tags_json, created_at FROM aac_user_memories "
            "WHERE memory_id = ? AND user_id = ?"
        )
        update_sql = adapt_sql_asyncpg(
            "UPDATE aac_user_memories SET value_json = ?, updated_at = ? "
            "WHERE memory_id = ? AND user_id = ?"
        )
        ts = _utc_iso()
        encoded = to_json(value)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(select_sql, memory_id, user_id)
                if row is None:
                    raise KeyError(memory_id)
                await conn.execute(
                    update_sql, encoded, ts, memory_id, user_id
                )
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
        select_sql = adapt_sql_asyncpg(
            "SELECT 1 FROM aac_user_memories "
            "WHERE memory_id = ? AND user_id = ?"
        )
        delete_sql = adapt_sql_asyncpg(
            "DELETE FROM aac_user_memories "
            "WHERE memory_id = ? AND user_id = ?"
        )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(select_sql, memory_id, user_id)
                if row is None:
                    raise KeyError(memory_id)
                await conn.execute(delete_sql, memory_id, user_id)

    async def forget_all(self, *, user_id: str) -> int:
        await self._ensure_schema()
        count_sql = adapt_sql_asyncpg(
            "SELECT COUNT(*) FROM aac_user_memories WHERE user_id = ?"
        )
        delete_sql = adapt_sql_asyncpg(
            "DELETE FROM aac_user_memories WHERE user_id = ?"
        )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(count_sql, user_id)
                count = int(row[0]) if row else 0
                await conn.execute(delete_sql, user_id)
                return count

    async def list_users(self) -> list[str]:
        await self._ensure_schema()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT user_id FROM aac_user_memories"
            )
            return [r[0] for r in rows]
