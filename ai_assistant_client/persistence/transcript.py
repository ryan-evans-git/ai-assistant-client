"""Wire shapes + the :class:`TranscriptStore` protocol.

A *transcript* is the record of a single workflow run: a header
(workflow identity + arguments), a sequence of
:class:`~ai_assistant_client.workflows.runtime.WorkflowEvent`
emissions, and a footer (outcome).  Replay re-emits the recorded
events without invoking the original handler — sufficient for
regression tests and post-hoc inspection, and intentionally
decoupled from any tool calls the handler made internally
(those stay inside the handler's blackbox).

Header / footer are split from the event stream so a partial
transcript (one whose process died mid-run) is still legible —
the footer's absence flags the run as incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from ai_assistant_client.workflows.runtime import WorkflowEvent


@dataclass(frozen=True)
class RunHeader:
    """Workflow identity + input args + start timestamp.

    Written exactly once at the start of a recorded run.  The
    ``started_at`` value is an ISO-8601 string (UTC) for backend
    portability — different stores serialize timestamps differently
    and a string survives every round trip.
    """

    run_id: str
    workflow_name: str
    tool_use_id: str
    args: dict[str, Any]
    started_at: str


@dataclass(frozen=True)
class RunFooter:
    """End-of-run marker.

    ``outcome`` mirrors the final :class:`WorkflowEvent.type`
    (``"result"`` or ``"error"``).  ``ended_at`` is ISO-8601 UTC.
    A transcript missing its footer is treated as incomplete —
    replay still works; callers can use the absence as a signal.
    """

    ended_at: str
    outcome: Literal["result", "error"]


@dataclass(frozen=True)
class RunTranscript:
    """Reconstructed transcript: header + events + (optional) footer."""

    header: RunHeader
    events: list[WorkflowEvent] = field(default_factory=list)
    footer: RunFooter | None = None


@runtime_checkable
class TranscriptStore(Protocol):
    """Append-only store of workflow run transcripts.

    Implementations must be safe for concurrent use from a single
    process (the workflow runtime drives one task per run, but a
    host typically runs many runs concurrently).  Cross-process
    safety is backend-specific — the file backend uses ``O_APPEND``
    so concurrent appends interleave cleanly inside a single run
    file; a future SQL backend would lean on the database.
    """

    async def write_header(self, header: RunHeader) -> None: ...

    async def append_event(self, run_id: str, event: WorkflowEvent) -> None: ...

    async def write_footer(self, run_id: str, footer: RunFooter) -> None: ...

    async def read(self, run_id: str) -> RunTranscript:
        """Return the full transcript for ``run_id``.

        Raises :class:`KeyError` when no run with that id exists.
        Partial runs (missing footer) are returned with
        ``transcript.footer is None``.
        """
        ...

    async def list_runs(self) -> list[str]:
        """Return every run id the store knows about.

        Ordering is backend-defined — callers that need
        chronological order should sort by ``header.started_at``
        after reading.
        """
        ...
