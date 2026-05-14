"""Tests for the record + replay wrappers around the workflow runtime.

The record-mode invariant is: yielding events stays semantically
identical to ``run_workflow``, while the store ends up with a
header + every event + a footer.  Replay invariant: feeding the
recorded transcript back yields the same event sequence without
invoking the original handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ai_assistant_client.confirmation_store import ConfirmationOutcome
from ai_assistant_client.persistence import (
    FileTranscriptStore,
    InMemoryTranscriptStore,
    TranscriptStore,
)
from ai_assistant_client.workflows import (
    emit_status,
    get_workflow,
    pause_for_confirmation,
    workflow,
)
from ai_assistant_client.workflows.replay import (
    replay_workflow,
    run_workflow_recording,
)


@pytest.fixture(params=["memory", "file"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> TranscriptStore:
    if request.param == "memory":
        return InMemoryTranscriptStore()
    return FileTranscriptStore(tmp_path / "transcripts")


# ---------------------------------------------------------------------------
# Record mode
# ---------------------------------------------------------------------------


async def test_record_writes_header_and_footer(store: TranscriptStore) -> None:
    @workflow(description="d")
    async def add(*, a: int, b: int) -> int:
        return a + b

    wf = get_workflow(add)
    assert wf is not None

    events = []
    async for ev in run_workflow_recording(
        wf,
        {"a": 2, "b": 3},
        tool_use_id="tu",
        confirmation_hook=None,
        store=store,
        run_id="run-record-1",
    ):
        events.append(ev)

    assert len(events) == 1
    assert events[0].type == "result"
    assert events[0].value == 5

    transcript = await store.read("run-record-1")
    assert transcript.header.workflow_name == add.__name__ or transcript.header.workflow_name == "add"
    assert transcript.header.args == {"a": 2, "b": 3}
    assert len(transcript.events) == 1
    assert transcript.events[0].type == "result"
    assert transcript.events[0].value == 5
    assert transcript.footer is not None
    assert transcript.footer.outcome == "result"


async def test_record_captures_status_and_pauses(store: TranscriptStore) -> None:
    @workflow(description="d")
    async def progressive(*, x: str) -> dict:
        await emit_status("step 1")
        outcome = await pause_for_confirmation(message="?", preview={"x": x})
        await emit_status("step 2")
        return {"decision": outcome.decision}

    async def hook(_payload: dict[str, Any]) -> ConfirmationOutcome:
        return ConfirmationOutcome(decision="confirm")

    wf = get_workflow(progressive)
    assert wf is not None

    async for _ in run_workflow_recording(
        wf,
        {"x": "hi"},
        tool_use_id="tu",
        confirmation_hook=hook,
        store=store,
        run_id="run-record-2",
    ):
        pass

    transcript = await store.read("run-record-2")
    types = [e.type for e in transcript.events]
    assert types == [
        "status",
        "confirmation_request",
        "confirmation_resolved",
        "status",
        "result",
    ]
    assert transcript.footer is not None
    assert transcript.footer.outcome == "result"


async def test_record_marks_error_outcome_in_footer(store: TranscriptStore) -> None:
    @workflow(description="d")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    wf = get_workflow(boom)
    assert wf is not None

    async for _ in run_workflow_recording(
        wf,
        {},
        tool_use_id="tu",
        confirmation_hook=None,
        store=store,
        run_id="run-record-err",
    ):
        pass

    transcript = await store.read("run-record-err")
    assert transcript.events[-1].type == "error"
    assert transcript.footer is not None
    assert transcript.footer.outcome == "error"


async def test_record_footer_written_on_consumer_aclose(
    store: TranscriptStore,
) -> None:
    """If the consumer stops draining early but closes the
    generator (the well-behaved way: ``async with aclosing(...)``
    or an explicit ``aclose()``), the wrapper's ``finally`` runs
    and writes a footer so the transcript stays well-formed."""
    from contextlib import aclosing

    @workflow(description="d")
    async def chatty() -> str:
        await emit_status("one")
        await emit_status("two")
        return "done"

    wf = get_workflow(chatty)
    assert wf is not None

    gen = run_workflow_recording(
        wf,
        {},
        tool_use_id="tu",
        confirmation_hook=None,
        store=store,
        run_id="run-broken",
    )
    async with aclosing(gen):
        async for ev in gen:
            if ev.type == "status":
                break

    transcript = await store.read("run-broken")
    assert transcript.footer is not None
    # Outcome stays ``error`` because we never saw a terminal
    # ``result`` event before the consumer bailed — that's the
    # correct integrity signal.
    assert transcript.footer.outcome == "error"


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


async def test_replay_emits_recorded_events_in_order(
    store: TranscriptStore,
) -> None:
    @workflow(description="d")
    async def progressive() -> str:
        await emit_status("a")
        await emit_status("b")
        return "ok"

    wf = get_workflow(progressive)
    assert wf is not None

    # First, record.
    recorded: list[Any] = []
    async for ev in run_workflow_recording(
        wf,
        {},
        tool_use_id="tu",
        confirmation_hook=None,
        store=store,
        run_id="run-replay-1",
    ):
        recorded.append((ev.type, ev.payload, ev.value, ev.error))

    # Then replay.  Handler is not re-invoked; events come straight
    # off the store.
    replayed: list[Any] = []
    async for ev in replay_workflow(store, "run-replay-1"):
        replayed.append((ev.type, ev.payload, ev.value, ev.error))

    assert replayed == recorded


async def test_replay_unknown_run_raises(store: TranscriptStore) -> None:
    with pytest.raises(KeyError):
        async for _ in replay_workflow(store, "no-such-run"):
            pass


async def test_replay_does_not_invoke_handler(store: TranscriptStore) -> None:
    """The whole point: replay yields the recorded events without
    running the workflow again.  Verify by recording a workflow
    that bumps a module counter, then asserting the counter
    stays put during replay."""
    call_count = {"n": 0}

    @workflow(description="d")
    async def counted() -> int:
        call_count["n"] += 1
        return call_count["n"]

    wf = get_workflow(counted)
    assert wf is not None

    async for _ in run_workflow_recording(
        wf,
        {},
        tool_use_id="tu",
        confirmation_hook=None,
        store=store,
        run_id="run-replay-2",
    ):
        pass

    assert call_count["n"] == 1

    async for _ in replay_workflow(store, "run-replay-2"):
        pass

    # Replay didn't re-enter the handler.
    assert call_count["n"] == 1
