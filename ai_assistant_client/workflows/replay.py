"""Record + replay for workflow runs.

Two thin wrappers around :func:`run_workflow`:

* :func:`run_workflow_recording` — drives a normal run and tees
  every emitted event into a :class:`TranscriptStore`.  The header
  goes in before the first event (so a partial transcript still
  identifies the workflow); the footer goes in after the terminal
  ``result`` / ``error`` event.
* :func:`replay_workflow` — reads a recorded transcript and
  yields its events in order, without invoking the original
  handler.  Useful for regression tests ("given these inputs,
  the runtime emitted these events") and forensic inspection
  ("what actually happened in run X").

What replay *doesn't* try to do: re-run the workflow handler
against mocked tool outputs.  That's a different (and harder)
problem — handler-internal calls go through arbitrary user code,
not the runtime boundary.  Treating the transcript as authoritative
keeps replay deterministic by construction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, AsyncIterator, Literal

from ai_assistant_client.persistence.transcript import (
    RunFooter,
    RunHeader,
    TranscriptStore,
)
from ai_assistant_client.workflows.decorator import Workflow
from ai_assistant_client.workflows.runtime import (
    ConfirmationHook,
    WorkflowEvent,
    run_workflow,
)


def _utc_iso() -> str:
    """Single source of truth for transcript timestamps.

    ``datetime.now(timezone.utc).isoformat()`` rather than
    ``utcnow()`` because the latter is naive (no tzinfo) and the
    transcript wire format prefers explicit-UTC strings so a
    downstream reader doesn't have to guess.
    """
    return datetime.now(timezone.utc).isoformat()


async def run_workflow_recording(
    wf: Workflow,
    args: dict[str, Any],
    *,
    tool_use_id: str,
    confirmation_hook: ConfirmationHook | None,
    store: TranscriptStore,
    run_id: str,
    default_timeout_seconds: int = 60,
    on_timeout_decision: Literal["confirm", "decline"] = "decline",
) -> AsyncIterator[WorkflowEvent]:
    """Run ``wf`` and tee every event into ``store`` under ``run_id``.

    Events still stream to the caller in real time — this is a
    transparent wrapper, not a buffered capture.  The terminal
    event is written to the store before being yielded so a caller
    that crashes mid-iteration still ends up with a complete
    transcript on disk.
    """
    header = RunHeader(
        run_id=run_id,
        workflow_name=wf.name,
        tool_use_id=tool_use_id,
        args=args,
        started_at=_utc_iso(),
    )
    await store.write_header(header)

    outcome: Literal["result", "error"] = "error"
    try:
        async for event in run_workflow(
            wf,
            args,
            tool_use_id=tool_use_id,
            confirmation_hook=confirmation_hook,
            default_timeout_seconds=default_timeout_seconds,
            on_timeout_decision=on_timeout_decision,
        ):
            await store.append_event(run_id, event)
            if event.type in ("result", "error"):
                outcome = event.type
            yield event
    finally:
        # Even if the consumer breaks early or the run was
        # cancelled, write SOMETHING marking the end-of-record —
        # a footer'd transcript is the integrity signal replay
        # callers look for.
        await store.write_footer(
            run_id, RunFooter(ended_at=_utc_iso(), outcome=outcome)
        )


async def replay_workflow(
    store: TranscriptStore, run_id: str
) -> AsyncIterator[WorkflowEvent]:
    """Yield the recorded events for ``run_id`` in order.

    Raises :class:`KeyError` when no transcript exists for the
    given id.  A partial transcript (no footer) replays whatever
    events made it to disk before the original run was interrupted.
    """
    transcript = await store.read(run_id)
    for event in transcript.events:
        yield event
