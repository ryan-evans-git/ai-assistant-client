"""DB-API-2.0-backed :class:`ConversationStore`.

Same engine matrix as
:class:`~ai_assistant_client.persistence.sql_transcript.SqlTranscriptStore`
— sqlite, PostgreSQL (incl. Aurora PG + RDS), MySQL (incl. Aurora
MySQL) — using only the standard Python DB-API.  See
:mod:`ai_assistant_client.persistence.sql_common` for the why-and-
how write-up that applies to both stores.

Messages are stored as JSON in a ``content_json`` TEXT column;
``seq`` is computed inside the same transaction as the insert
so the per-conversation ordering is correct on single-process
recorders.  Cross-process recorders writing to the same
conversation should run a leader-only recorder or migrate to a
sequence-per-id scheme (the SELECT MAX+1 pattern races across
processes).
"""

from __future__ import annotations

import asyncio
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


class SqlConversationStore(ConversationStore):
    """DB-API-2.0 implementation of :class:`ConversationStore`."""

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
                cur.execute(conversation_messages_ddl(self._dialect))
                self._conn.commit()
            finally:
                cur.close()

        await asyncio.to_thread(_run)
        self._initialized = True

    async def append(self, conversation_id: str, message: Message) -> None:
        await self._ensure_schema()
        next_seq_sql = adapt_sql(
            "SELECT COALESCE(MAX(seq), 0) + 1 "
            "FROM aac_conversation_messages WHERE conversation_id = ?",
            self._dialect,
        )
        insert_sql = adapt_sql(
            "INSERT INTO aac_conversation_messages "
            "(conversation_id, seq, content_json) VALUES (?, ?, ?)",
            self._dialect,
        )
        encoded = to_json(message)

        def _run() -> None:
            cur = self._conn.cursor()
            try:
                cur.execute(next_seq_sql, (conversation_id,))
                row = cur.fetchone()
                next_seq = int(row[0]) if row and row[0] is not None else 1
                cur.execute(
                    insert_sql, (conversation_id, next_seq, encoded)
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

        async with self._lock:
            await asyncio.to_thread(_run)

    async def read(self, conversation_id: str) -> list[Message]:
        await self._ensure_schema()
        sql = adapt_sql(
            "SELECT content_json FROM aac_conversation_messages "
            "WHERE conversation_id = ? ORDER BY seq ASC",
            self._dialect,
        )

        def _run() -> list[Message]:
            cur = self._conn.cursor()
            try:
                cur.execute(sql, (conversation_id,))
                return [from_json(r[0]) for r in cur.fetchall()]
            finally:
                cur.close()

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def list_conversations(self) -> list[str]:
        await self._ensure_schema()
        sql = "SELECT DISTINCT conversation_id FROM aac_conversation_messages"

        def _run() -> list[str]:
            cur = self._conn.cursor()
            try:
                cur.execute(sql)
                return [r[0] for r in cur.fetchall()]
            finally:
                cur.close()

        async with self._lock:
            return await asyncio.to_thread(_run)
