"""Backend selection for the persistence stores.

Each store gets its own pair of env vars so a host can pick
backends independently — e.g. transcripts in memory (cheap, lossy)
while conversations go to disk (durable across restarts):

* ``AAC_TRANSCRIPT_BACKEND`` / ``AAC_TRANSCRIPT_DIR`` /
  ``AAC_TRANSCRIPT_SQLITE_PATH`` — workflow run transcripts
  (see :class:`~ai_assistant_client.persistence.transcript.TranscriptStore`).
* ``AAC_CONVERSATION_BACKEND`` / ``AAC_CONVERSATION_DIR`` /
  ``AAC_CONVERSATION_SQLITE_PATH`` — conversation message logs
  (see :class:`~ai_assistant_client.persistence.conversation.ConversationStore`).

The factory covers the ``memory`` / ``file`` / ``sqlite``
backends — the three that need no extra dependency and no
caller-supplied connection.  The PostgreSQL / MySQL backends
(including Aurora) accept a caller-managed DB-API connection
and live outside the env-var-driven factory by design: things
like IAM-token minting, RDS Proxy endpoints, pooling, and TLS
config are upstream concerns the factory shouldn't try to own.
Hosts construct those directly:

.. code-block:: python

    from ai_assistant_client.persistence import SqlTranscriptStore
    from ai_assistant_client.persistence.sql_common import Dialect

    conn = psycopg.connect(...)
    store = SqlTranscriptStore(conn, dialect=Dialect.POSTGRESQL)
"""

from __future__ import annotations

import os
from typing import Any

from ai_assistant_client.persistence.conversation import ConversationStore
from ai_assistant_client.persistence.conversation_file import (
    FileConversationStore,
)
from ai_assistant_client.persistence.conversation_memory import (
    InMemoryConversationStore,
)
from ai_assistant_client.persistence.file import FileTranscriptStore
from ai_assistant_client.persistence.memory import InMemoryTranscriptStore
from ai_assistant_client.persistence.sql_common import Dialect
from ai_assistant_client.persistence.sql_conversation import (
    SqlConversationStore,
)
from ai_assistant_client.persistence.sql_transcript import SqlTranscriptStore
from ai_assistant_client.persistence.transcript import TranscriptStore


TRANSCRIPT_BACKEND_ENV = "AAC_TRANSCRIPT_BACKEND"
TRANSCRIPT_DIR_ENV = "AAC_TRANSCRIPT_DIR"
TRANSCRIPT_SQLITE_PATH_ENV = "AAC_TRANSCRIPT_SQLITE_PATH"

CONVERSATION_BACKEND_ENV = "AAC_CONVERSATION_BACKEND"
CONVERSATION_DIR_ENV = "AAC_CONVERSATION_DIR"
CONVERSATION_SQLITE_PATH_ENV = "AAC_CONVERSATION_SQLITE_PATH"

_DEFAULT_TRANSCRIPT_DIR = "./transcripts"
_DEFAULT_CONVERSATION_DIR = "./conversations"
_DEFAULT_TRANSCRIPT_SQLITE_PATH = "./transcripts.sqlite3"
_DEFAULT_CONVERSATION_SQLITE_PATH = "./conversations.sqlite3"


def make_transcript_store(
    *,
    kind: str | None = None,
    base_dir: str | None = None,
    sqlite_path: str | None = None,
) -> TranscriptStore:
    """Construct a transcript store from env vars (or explicit args).

    ``kind`` overrides ``AAC_TRANSCRIPT_BACKEND``; ``base_dir``
    overrides ``AAC_TRANSCRIPT_DIR`` (file backend); ``sqlite_path``
    overrides ``AAC_TRANSCRIPT_SQLITE_PATH`` (sqlite backend).
    Unknown backends raise :class:`ValueError` instead of falling
    through to a default — a misspelled env var should fail
    loudly rather than silently dropping records.
    """
    resolved = (kind or os.environ.get(TRANSCRIPT_BACKEND_ENV) or "memory").lower()
    if resolved == "memory":
        return InMemoryTranscriptStore()
    if resolved == "file":
        directory = base_dir or os.environ.get(
            TRANSCRIPT_DIR_ENV, _DEFAULT_TRANSCRIPT_DIR
        )
        return FileTranscriptStore(directory)
    if resolved == "sqlite":
        path = sqlite_path or os.environ.get(
            TRANSCRIPT_SQLITE_PATH_ENV, _DEFAULT_TRANSCRIPT_SQLITE_PATH
        )
        return SqlTranscriptStore(_open_sqlite(path), dialect=Dialect.SQLITE)
    raise ValueError(
        f"unknown transcript backend {resolved!r} — expected one of: "
        "memory, file, sqlite"
    )


def make_conversation_store(
    *,
    kind: str | None = None,
    base_dir: str | None = None,
    sqlite_path: str | None = None,
) -> ConversationStore:
    """Construct a conversation store from env vars (or explicit args).

    Mirror of :func:`make_transcript_store` for the parallel
    :class:`ConversationStore` protocol.  Same fail-loud rule for
    unknown backends.
    """
    resolved = (
        kind or os.environ.get(CONVERSATION_BACKEND_ENV) or "memory"
    ).lower()
    if resolved == "memory":
        return InMemoryConversationStore()
    if resolved == "file":
        directory = base_dir or os.environ.get(
            CONVERSATION_DIR_ENV, _DEFAULT_CONVERSATION_DIR
        )
        return FileConversationStore(directory)
    if resolved == "sqlite":
        path = sqlite_path or os.environ.get(
            CONVERSATION_SQLITE_PATH_ENV, _DEFAULT_CONVERSATION_SQLITE_PATH
        )
        return SqlConversationStore(_open_sqlite(path), dialect=Dialect.SQLITE)
    raise ValueError(
        f"unknown conversation backend {resolved!r} — expected one of: "
        "memory, file, sqlite"
    )


def _open_sqlite(path: str) -> Any:
    """Open a sqlite connection suitable for the SQL stores.

    ``check_same_thread=False`` is required because the stores run
    every DB call through :func:`asyncio.to_thread`, which dispatches
    to a thread pool — without this flag, stdlib ``sqlite3`` raises on
    the first cross-thread call.  Concurrent access stays safe because
    the store serializes every operation through its own
    :class:`asyncio.Lock`.
    """
    import sqlite3

    return sqlite3.connect(path, check_same_thread=False)
