"""Backend selection for the persistence stores.

Each store gets its own pair of env vars so a host can pick
backends independently — e.g. transcripts in memory (cheap, lossy)
while conversations go to disk (durable across restarts):

* ``AAC_TRANSCRIPT_BACKEND`` / ``AAC_TRANSCRIPT_DIR`` — workflow
  run transcripts (see
  :class:`~ai_assistant_client.persistence.transcript.TranscriptStore`).
* ``AAC_CONVERSATION_BACKEND`` / ``AAC_CONVERSATION_DIR`` —
  conversation message logs (see
  :class:`~ai_assistant_client.persistence.conversation.ConversationStore`).

When new backends land (SQL on a local DB, SQL on a managed cloud
DB, etc.) they slot in here.  Each backend constructor accepts
plain kwargs so a host wiring code from outside this package can
build one directly without going through env vars.
"""

from __future__ import annotations

import os

from ai_assistant_client.persistence.conversation import ConversationStore
from ai_assistant_client.persistence.conversation_file import (
    FileConversationStore,
)
from ai_assistant_client.persistence.conversation_memory import (
    InMemoryConversationStore,
)
from ai_assistant_client.persistence.file import FileTranscriptStore
from ai_assistant_client.persistence.memory import InMemoryTranscriptStore
from ai_assistant_client.persistence.transcript import TranscriptStore


TRANSCRIPT_BACKEND_ENV = "AAC_TRANSCRIPT_BACKEND"
TRANSCRIPT_DIR_ENV = "AAC_TRANSCRIPT_DIR"

CONVERSATION_BACKEND_ENV = "AAC_CONVERSATION_BACKEND"
CONVERSATION_DIR_ENV = "AAC_CONVERSATION_DIR"

_DEFAULT_TRANSCRIPT_DIR = "./transcripts"
_DEFAULT_CONVERSATION_DIR = "./conversations"


def make_transcript_store(
    *, kind: str | None = None, base_dir: str | None = None
) -> TranscriptStore:
    """Construct a transcript store from env vars (or explicit args).

    ``kind`` overrides ``AAC_TRANSCRIPT_BACKEND``; ``base_dir``
    overrides ``AAC_TRANSCRIPT_DIR``.  Unknown backends raise
    :class:`ValueError` instead of falling through to a default —
    a misspelled env var should fail loudly rather than silently
    dropping records.
    """
    resolved = (kind or os.environ.get(TRANSCRIPT_BACKEND_ENV) or "memory").lower()
    if resolved == "memory":
        return InMemoryTranscriptStore()
    if resolved == "file":
        directory = base_dir or os.environ.get(
            TRANSCRIPT_DIR_ENV, _DEFAULT_TRANSCRIPT_DIR
        )
        return FileTranscriptStore(directory)
    raise ValueError(
        f"unknown transcript backend {resolved!r} — expected one of: "
        "memory, file"
    )


def make_conversation_store(
    *, kind: str | None = None, base_dir: str | None = None
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
    raise ValueError(
        f"unknown conversation backend {resolved!r} — expected one of: "
        "memory, file"
    )
