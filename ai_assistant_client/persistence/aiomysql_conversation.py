"""aiomysql-backed :class:`ConversationStore` for MySQL / Aurora MySQL.

See :mod:`ai_assistant_client.persistence.aiomysql_transcript` for
the rationale that applies to both async MySQL stores.  Schema
reuses the MySQL DDL from
:mod:`ai_assistant_client.persistence.sql_common` so a database
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
    Dialect,
    adapt_sql,
    conversation_messages_ddl,
    from_json,
    to_json,
)


class AiomysqlConversationStore(ConversationStore):
    """aiomysql implementation of :class:`ConversationStore`."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._initialized = False

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        ddl = conversation_messages_ddl(Dialect.MYSQL)
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(ddl)
            await conn.commit()
        self._initialized = True

    async def append(self, conversation_id: str, message: Message) -> None:
        await self._ensure_schema()
        next_seq_sql = adapt_sql(
            "SELECT COALESCE(MAX(seq), 0) + 1 "
            "FROM aac_conversation_messages WHERE conversation_id = ?",
            Dialect.MYSQL,
        )
        insert_sql = adapt_sql(
            "INSERT INTO aac_conversation_messages "
            "(conversation_id, seq, content_json) VALUES (?, ?, ?)",
            Dialect.MYSQL,
        )
        encoded = to_json(message)
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(next_seq_sql, (conversation_id,))
                row = await cur.fetchone()
                next_seq = int(row[0]) if row and row[0] is not None else 1
                await cur.execute(
                    insert_sql, (conversation_id, next_seq, encoded)
                )
            await conn.commit()

    async def read(self, conversation_id: str) -> list[Message]:
        await self._ensure_schema()
        sql = adapt_sql(
            "SELECT content_json FROM aac_conversation_messages "
            "WHERE conversation_id = ? ORDER BY seq ASC",
            Dialect.MYSQL,
        )
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (conversation_id,))
                rows = await cur.fetchall()
                return [from_json(r[0]) for r in rows]

    async def list_conversations(self) -> list[str]:
        await self._ensure_schema()
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT DISTINCT conversation_id "
                    "FROM aac_conversation_messages"
                )
                rows = await cur.fetchall()
                return [r[0] for r in rows]
