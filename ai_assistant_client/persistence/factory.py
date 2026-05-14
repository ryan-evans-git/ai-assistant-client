"""Backend selection for :class:`TranscriptStore`.

The host picks a backend by env var so application code stays
agnostic to the storage choice:

* ``AAC_TRANSCRIPT_BACKEND`` — ``memory`` (default) or ``file``.
* ``AAC_TRANSCRIPT_DIR`` — base directory for the ``file``
  backend; defaults to ``./transcripts``.

When new backends land (SQL on a local DB, SQL on a managed cloud
DB, etc.) they slot in here.  Each backend constructor accepts
plain kwargs so a host wiring code from outside this package can
build one directly without going through env vars.
"""

from __future__ import annotations

import os

from ai_assistant_client.persistence.file import FileTranscriptStore
from ai_assistant_client.persistence.memory import InMemoryTranscriptStore
from ai_assistant_client.persistence.transcript import TranscriptStore


TRANSCRIPT_BACKEND_ENV = "AAC_TRANSCRIPT_BACKEND"
TRANSCRIPT_DIR_ENV = "AAC_TRANSCRIPT_DIR"

_DEFAULT_FILE_DIR = "./transcripts"


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
            TRANSCRIPT_DIR_ENV, _DEFAULT_FILE_DIR
        )
        return FileTranscriptStore(directory)
    raise ValueError(
        f"unknown transcript backend {resolved!r} — expected one of: "
        "memory, file"
    )
