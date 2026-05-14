"""asyncpg-backed :class:`ConversationStore` for PostgreSQL.

See :mod:`ai_assistant_client.persistence.asyncpg_transcript` for the
why-and-how write-up that applies to both async stores.  Schema
reuses the PostgreSQL DDL from
:mod:`ai_assistant_client.persistence.sql_common` — a database
initialised by this store is byte-equivalent to one initialised
by :class:`SqlConversationStore`.
"""

from __future__ import annotations

from typing import Any

from ai_assistant_client.persistence.conversation import (
    ConversationStore,
    Message,
)
from ai_assistant_client.persistence.sql_common import (
    SEQ_COLLISION_MAX_ATTEMPTS,
    Dialect,
    adapt_sql_asyncpg,
    conversation_messages_ddl,
    from_json,
    is_integrity_error,
    to_json,
)


class AsyncpgConversationStore(ConversationStore):
    """asyncpg implementation of :class:`ConversationStore`."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._initialized = False

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        ddl = conversation_messages_ddl(Dialect.POSTGRESQL)
        async with self._pool.acquire() as conn:
            await conn.execute(ddl)
        self._initialized = True

    async def append(self, conversation_id: str, message: Message) -> None:
        await self._ensure_schema()
        next_seq_sql = adapt_sql_asyncpg(
            "SELECT COALESCE(MAX(seq), 0) + 1 "
            "FROM aac_conversation_messages WHERE conversation_id = ?"
        )
        insert_sql = adapt_sql_asyncpg(
            "INSERT INTO aac_conversation_messages "
            "(conversation_id, seq, content_json) VALUES (?, ?, ?)"
        )
        encoded = to_json(message)
        # Cross-process seq-collision retry — see
        # SqlConversationStore.append for the full rationale.
        last_err: Exception | None = None
        for _attempt in range(SEQ_COLLISION_MAX_ATTEMPTS):
            try:
                async with self._pool.acquire() as conn:
                    async with conn.transaction():
                        row = await conn.fetchrow(
                            next_seq_sql, conversation_id
                        )
                        next_seq = (
                            int(row[0])
                            if row and row[0] is not None
                            else 1
                        )
                        await conn.execute(
                            insert_sql, conversation_id, next_seq, encoded
                        )
                return
            except Exception as err:
                if is_integrity_error(err):
                    last_err = err
                    continue
                raise
        assert last_err is not None
        raise last_err

    async def read(self, conversation_id: str) -> list[Message]:
        await self._ensure_schema()
        sql = adapt_sql_asyncpg(
            "SELECT content_json FROM aac_conversation_messages "
            "WHERE conversation_id = ? ORDER BY seq ASC"
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, conversation_id)
            return [from_json(r[0]) for r in rows]

    async def list_conversations(self) -> list[str]:
        await self._ensure_schema()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT conversation_id FROM aac_conversation_messages"
            )
            return [r[0] for r in rows]
