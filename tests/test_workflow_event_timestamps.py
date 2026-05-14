"""Tests for the per-event ``timestamp`` field on WorkflowEvent.

Two invariants:

1. **Live emissions are stamped.**  Every event the runtime
   produces (status, confirmation request/resolved, terminal
   result/error) carries a non-empty ISO-8601 UTC timestamp.
2. **Old transcripts read back as 'unknown'.**  Records written
   before timestamps shipped have no ``timestamp`` field; the
   read path preserves the empty-string default rather than
   fabricating ``now`` (which would silently lie about when
   the event happened).

The ordering invariant ("monotonically non-decreasing across one
run") is exercised end-to-end by recording a multi-event run and
parsing the timestamps back.
"""

from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path
from typing import Any

from ai_assistant_client.confirmation_store import ConfirmationOutcome
from ai_assistant_client.persistence import (
    Dialect,
    FileTranscriptStore,
    InMemoryTranscriptStore,
    SqlTranscriptStore,
)
from ai_assistant_client.workflows import (
    emit_status,
    get_workflow,
    pause_for_confirmation,
    run_workflow,
    workflow,
)
from ai_assistant_client.workflows.replay import run_workflow_recording
from ai_assistant_client.workflows.runtime import WorkflowEvent


# ---------------------------------------------------------------------------
# Field default
# ---------------------------------------------------------------------------


def test_workflow_event_timestamp_defaults_to_empty_string() -> None:
    """Unstamped events read 'unknown' — never a fabricated time."""
    event = WorkflowEvent(type="status", payload={"x": 1})
    assert event.timestamp == ""


def test_workflow_event_explicit_timestamp_preserved() -> None:
    ts = "2026-05-14T12:00:00+00:00"
    event = WorkflowEvent(type="status", payload={"x": 1}, timestamp=ts)
    assert event.timestamp == ts


# ---------------------------------------------------------------------------
# Live emissions are stamped
# ---------------------------------------------------------------------------


async def test_status_event_carries_timestamp() -> None:
    @workflow(description="d")
    async def stamped() -> str:
        await emit_status("step")
        return "ok"

    wf = get_workflow(stamped)
    assert wf is not None
    events = []
    async for ev in run_workflow(
        wf, {}, tool_use_id="tu", confirmation_hook=None
    ):
        events.append(ev)

    status = next(e for e in events if e.type == "status")
    assert status.timestamp != ""
    # Coarse sanity check on shape: ISO 8601 with timezone.
    assert "T" in status.timestamp
    assert status.timestamp.endswith("+00:00")


async def test_result_and_error_events_carry_timestamps() -> None:
    @workflow(description="d")
    async def ok() -> str:
        return "done"

    @workflow(description="d")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    wf_ok = get_workflow(ok)
    wf_err = get_workflow(boom)
    assert wf_ok is not None and wf_err is not None

    ok_events = []
    async for ev in run_workflow(
        wf_ok, {}, tool_use_id="tu", confirmation_hook=None
    ):
        ok_events.append(ev)
    err_events = []
    async for ev in run_workflow(
        wf_err, {}, tool_use_id="tu", confirmation_hook=None
    ):
        err_events.append(ev)

    assert ok_events[-1].type == "result"
    assert ok_events[-1].timestamp != ""
    assert err_events[-1].type == "error"
    assert err_events[-1].timestamp != ""


async def test_confirmation_request_and_resolved_both_stamped() -> None:
    @workflow(description="d")
    async def pause_once() -> str:
        outcome = await pause_for_confirmation(message="?")
        return outcome.decision

    async def hook(_payload: dict[str, Any]) -> ConfirmationOutcome:
        return ConfirmationOutcome(decision="confirm")

    wf = get_workflow(pause_once)
    assert wf is not None
    events = []
    async for ev in run_workflow(
        wf, {}, tool_use_id="tu", confirmation_hook=hook
    ):
        events.append(ev)

    req = next(e for e in events if e.type == "confirmation_request")
    res = next(e for e in events if e.type == "confirmation_resolved")
    assert req.timestamp != ""
    assert res.timestamp != ""
    # Resolved happens strictly after the request — even at the
    # extreme of identical clock readings, lexical compare of
    # ISO-8601 strings is also a valid time compare.
    assert res.timestamp >= req.timestamp


async def test_run_emits_events_in_non_decreasing_timestamp_order() -> None:
    @workflow(description="d")
    async def chatty() -> str:
        await emit_status("one")
        await emit_status("two")
        await emit_status("three")
        return "done"

    wf = get_workflow(chatty)
    assert wf is not None
    timestamps: list[str] = []
    async for ev in run_workflow(
        wf, {}, tool_use_id="tu", confirmation_hook=None
    ):
        timestamps.append(ev.timestamp)

    # Skip events that somehow ended up unstamped — there
    # shouldn't be any, but guard for clarity.
    stamps = [t for t in timestamps if t]
    assert stamps == sorted(stamps)


# ---------------------------------------------------------------------------
# Round-trip through persistence
# ---------------------------------------------------------------------------


async def test_in_memory_store_round_trips_timestamps() -> None:
    @workflow(description="d")
    async def go() -> str:
        await emit_status("hello")
        return "ok"

    wf = get_workflow(go)
    assert wf is not None
    store = InMemoryTranscriptStore()
    async for _ in run_workflow_recording(
        wf, {}, tool_use_id="tu", confirmation_hook=None,
        store=store, run_id="run-mem",
    ):
        pass

    transcript = await store.read("run-mem")
    assert all(e.timestamp != "" for e in transcript.events)


async def test_file_store_round_trips_timestamps(tmp_path: Path) -> None:
    @workflow(description="d")
    async def go() -> str:
        await emit_status("hello")
        return "ok"

    wf = get_workflow(go)
    assert wf is not None
    store = FileTranscriptStore(tmp_path / "t")
    async for _ in run_workflow_recording(
        wf, {}, tool_use_id="tu", confirmation_hook=None,
        store=store, run_id="run-file",
    ):
        pass

    transcript = await store.read("run-file")
    assert all(e.timestamp != "" for e in transcript.events)


async def test_sql_store_round_trips_timestamps(tmp_path: Path) -> None:
    @workflow(description="d")
    async def go() -> str:
        await emit_status("hello")
        return "ok"

    wf = get_workflow(go)
    assert wf is not None
    conn = sqlite3.connect(tmp_path / "t.sqlite3", check_same_thread=False)
    store = SqlTranscriptStore(conn, dialect=Dialect.SQLITE)
    async for _ in run_workflow_recording(
        wf, {}, tool_use_id="tu", confirmation_hook=None,
        store=store, run_id="run-sql",
    ):
        pass

    transcript = await store.read("run-sql")
    assert all(e.timestamp != "" for e in transcript.events)


# ---------------------------------------------------------------------------
# Backward compatibility — reading a pre-timestamp transcript
# ---------------------------------------------------------------------------


async def test_file_store_reads_old_transcript_without_timestamp_field(
    tmp_path: Path,
) -> None:
    """An on-disk JSONL line recorded by a pre-timestamp client
    has no ``timestamp`` key.  Read path must preserve the
    empty-string default rather than crashing or fabricating
    ``now``."""
    base = tmp_path / "t"
    base.mkdir()
    legacy = base / "old-run.jsonl"
    legacy.write_text(
        textwrap.dedent(
            """\
            {"kind": "header", "run_id": "old-run", "workflow_name": "send", "tool_use_id": "tu", "args": {}, "started_at": "2024-01-01T00:00:00+00:00"}
            {"kind": "event", "type": "status", "payload": {"message": "go"}}
            {"kind": "event", "type": "result", "value": "ok"}
            """
        ),
        encoding="utf-8",
    )

    store = FileTranscriptStore(base)
    transcript = await store.read("old-run")
    assert len(transcript.events) == 2
    assert all(e.timestamp == "" for e in transcript.events)
