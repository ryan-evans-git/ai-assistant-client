"""Pluggable persistence for workflow transcripts and conversation
history.

Two parallel protocols, each with two backends to start:

* :class:`TranscriptStore` — bracketed (header + events + footer)
  records of a single workflow run.
  Implementations: :class:`InMemoryTranscriptStore`,
  :class:`FileTranscriptStore` (JSONL on disk).
* :class:`ConversationStore` — open-ended append-only log of
  Anthropic-block-style messages keyed by conversation id.
  Implementations: :class:`InMemoryConversationStore`,
  :class:`FileConversationStore` (JSONL on disk).

Why two protocols instead of one: workflow runs have a clear
terminal state (the footer marks ``result`` or ``error``);
conversations don't.  Unifying the two shapes would require
weakening the contract enough that neither caller could trust
it.

Future backends (SQL on a local DB, SQL on a managed cloud DB,
etc.) plug in by implementing the relevant protocol — adding one
backend doesn't force changes to application code.  Selection at
runtime is driven by env vars (see :func:`make_transcript_store`
and :func:`make_conversation_store`).
"""

from __future__ import annotations

from ai_assistant_client.persistence.aiomysql_conversation import (
    AiomysqlConversationStore,
)
from ai_assistant_client.persistence.aiomysql_memory import (
    AiomysqlMemoryStore,
)
from ai_assistant_client.persistence.aiomysql_transcript import (
    AiomysqlTranscriptStore,
)
from ai_assistant_client.persistence.asyncpg_conversation import (
    AsyncpgConversationStore,
)
from ai_assistant_client.persistence.asyncpg_memory import (
    AsyncpgMemoryStore,
)
from ai_assistant_client.persistence.asyncpg_transcript import (
    AsyncpgTranscriptStore,
)
from ai_assistant_client.persistence.conversation import (
    ConversationStore,
    Message,
)
from ai_assistant_client.persistence.conversation_file import (
    FileConversationStore,
)
from ai_assistant_client.persistence.conversation_memory import (
    InMemoryConversationStore,
)
from ai_assistant_client.persistence.factory import (
    CONVERSATION_BACKEND_ENV,
    CONVERSATION_DIR_ENV,
    CONVERSATION_SQLITE_PATH_ENV,
    MEMORY_BACKEND_ENV,
    MEMORY_DIR_ENV,
    MEMORY_SQLITE_PATH_ENV,
    TRANSCRIPT_BACKEND_ENV,
    TRANSCRIPT_DIR_ENV,
    TRANSCRIPT_SQLITE_PATH_ENV,
    make_conversation_store,
    make_memory_store,
    make_transcript_store,
)
from ai_assistant_client.persistence.file import FileTranscriptStore
from ai_assistant_client.persistence.memory import InMemoryTranscriptStore
from ai_assistant_client.persistence.migrate import (
    add_column_if_missing,
    ensure_transcript_events_ts_column,
)
from ai_assistant_client.persistence.sql_common import Dialect
from ai_assistant_client.persistence.sql_conversation import (
    SqlConversationStore,
)
from ai_assistant_client.persistence.sql_memory import SqlMemoryStore
from ai_assistant_client.persistence.sql_transcript import SqlTranscriptStore
from ai_assistant_client.persistence.transcript import (
    RunFooter,
    RunHeader,
    RunTranscript,
    TranscriptStore,
)
from ai_assistant_client.persistence.user_memory import Memory, MemoryStore
from ai_assistant_client.persistence.user_memory_file import (
    CompactionStats,
    FileMemoryStore,
)
from ai_assistant_client.persistence.user_memory_local import LocalMemoryStore


__all__ = [
    "AiomysqlConversationStore",
    "AiomysqlMemoryStore",
    "AiomysqlTranscriptStore",
    "AsyncpgConversationStore",
    "AsyncpgMemoryStore",
    "AsyncpgTranscriptStore",
    "CONVERSATION_BACKEND_ENV",
    "CONVERSATION_DIR_ENV",
    "CONVERSATION_SQLITE_PATH_ENV",
    "CompactionStats",
    "ConversationStore",
    "Dialect",
    "FileConversationStore",
    "FileMemoryStore",
    "FileTranscriptStore",
    "InMemoryConversationStore",
    "InMemoryTranscriptStore",
    "LocalMemoryStore",
    "MEMORY_BACKEND_ENV",
    "MEMORY_DIR_ENV",
    "MEMORY_SQLITE_PATH_ENV",
    "Memory",
    "MemoryStore",
    "Message",
    "RunFooter",
    "RunHeader",
    "RunTranscript",
    "SqlConversationStore",
    "SqlMemoryStore",
    "SqlTranscriptStore",
    "TRANSCRIPT_BACKEND_ENV",
    "TRANSCRIPT_DIR_ENV",
    "TRANSCRIPT_SQLITE_PATH_ENV",
    "TranscriptStore",
    "add_column_if_missing",
    "ensure_transcript_events_ts_column",
    "make_conversation_store",
    "make_memory_store",
    "make_transcript_store",
]
