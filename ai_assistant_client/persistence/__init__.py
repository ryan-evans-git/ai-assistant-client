"""Pluggable persistence for workflow transcripts and (future)
conversation history / per-user memory.

The :class:`TranscriptStore` protocol is the single abstraction the
rest of the client talks to.  Concrete backends in this package:

* :class:`InMemoryTranscriptStore` — dict-backed, default for dev
  and tests.  Loses data on process exit.
* :class:`FileTranscriptStore` — JSON-lines on local disk, one
  file per run id.  Survives restarts; suitable for single-host
  deployments and CI captures.

Future backends (SQL/managed cloud) are out of scope for this
package's first cut: a backend with one real consumer is cleaner
than a backend with three placeholder ones.  When the SQL story
lands it implements the same protocol.

Selection at runtime is driven by env vars (see
:func:`make_transcript_store`) so a host can switch backends
without touching application code.
"""

from __future__ import annotations

from ai_assistant_client.persistence.factory import (
    TRANSCRIPT_BACKEND_ENV,
    TRANSCRIPT_DIR_ENV,
    make_transcript_store,
)
from ai_assistant_client.persistence.file import FileTranscriptStore
from ai_assistant_client.persistence.memory import InMemoryTranscriptStore
from ai_assistant_client.persistence.transcript import (
    RunFooter,
    RunHeader,
    RunTranscript,
    TranscriptStore,
)


__all__ = [
    "FileTranscriptStore",
    "InMemoryTranscriptStore",
    "RunFooter",
    "RunHeader",
    "RunTranscript",
    "TRANSCRIPT_BACKEND_ENV",
    "TRANSCRIPT_DIR_ENV",
    "TranscriptStore",
    "make_transcript_store",
]
