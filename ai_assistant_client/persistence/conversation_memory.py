"""In-memory :class:`ConversationStore` for dev and tests.

Dict-of-lists keyed by conversation id.  Data is lost when the
process exits — callers that need cross-restart durability pick
the file or (future) SQL backends.
"""

from __future__ import annotations

import asyncio

from ai_assistant_client.persistence.conversation import (
    ConversationStore,
    Message,
)


class InMemoryConversationStore(ConversationStore):
    """Dict-backed conversation log."""

    def __init__(self) -> None:
        self._logs: dict[str, list[Message]] = {}
        self._lock = asyncio.Lock()

    async def append(self, conversation_id: str, message: Message) -> None:
        async with self._lock:
            self._logs.setdefault(conversation_id, []).append(message)

    async def read(self, conversation_id: str) -> list[Message]:
        async with self._lock:
            # Return a shallow copy so a caller iterating doesn't
            # race with concurrent appends.
            return list(self._logs.get(conversation_id, []))

    async def list_conversations(self) -> list[str]:
        async with self._lock:
            return list(self._logs.keys())
