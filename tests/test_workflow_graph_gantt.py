"""Tests for the Gantt diagram kind.

The Gantt renderer is a pure function over RunTranscript; we
assert on the produced string the same way the flowchart /
sequence tests do.
"""

from __future__ import annotations

from ai_assistant_client.persistence.transcript import (
    RunFooter,
    RunHeader,
    RunTranscript,
)
from ai_assistant_client.workflows.graph import transcript_to_mermaid
from ai_assistant_client.workflows.runtime import WorkflowEvent


def _header() -> RunHeader:
    return RunHeader(
        run_id="run-g",
        workflow_name="send_email",
        tool_use_id="tu",
        args={},
        started_at="2026-05-14T00:00:00+00:00",
    )


def _transcript(
    events: list[WorkflowEvent], *, footer: RunFooter | None = None
) -> RunTranscript:
    return RunTranscript(header=_header(), events=events, footer=footer)


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------


def test_gantt_starts_with_required_headers() -> None:
    out = transcript_to_mermaid(_transcript([]), kind="gantt")
    lines = out.splitlines()
    assert lines[0] == "gantt"
    assert any(line.startswith("    title") for line in lines)
    assert any("dateFormat" in line for line in lines)
    assert any("axisFormat" in line for line in lines)


def test_gantt_no_events_renders_skeleton_only() -> None:
    """A run with no recorded events renders just the title +
    axis lines — useful for the 'recording started, no events
    yet' case."""
    out = transcript_to_mermaid(_transcript([]), kind="gantt")
    # No task lines (no colon-separated id, start, end).
    assert not any(
        ":e" in line and "," in line for line in out.splitlines()
    )


def test_gantt_omits_events_without_timestamps() -> None:
    """Pre-timestamp events (timestamp='') are skipped so the
    chart doesn't render against bogus dates.  Mixed-mode runs
    show only the events that have a real ts."""
    events = [
        WorkflowEvent(type="status", payload={"message": "old"}),  # no ts
        WorkflowEvent(
            type="status",
            payload={"message": "new"},
            timestamp="2026-05-14T00:00:01+00:00",
        ),
    ]
    out = transcript_to_mermaid(_transcript(events), kind="gantt")
    assert "new" in out
    # Only the one stamped event becomes a task (one ``:e`` task id).
    e_id_count = sum(1 for line in out.splitlines() if ":e" in line)
    assert e_id_count == 1


# ---------------------------------------------------------------------------
# Sectioning
# ---------------------------------------------------------------------------


def test_gantt_groups_events_into_sections() -> None:
    events = [
        WorkflowEvent(
            type="status",
            payload={"message": "drafting"},
            timestamp="2026-05-14T00:00:01+00:00",
        ),
        WorkflowEvent(
            type="confirmation_request",
            payload={"message": "send?"},
            timestamp="2026-05-14T00:00:02+00:00",
        ),
        WorkflowEvent(
            type="confirmation_resolved",
            payload={"decision": "confirm"},
            timestamp="2026-05-14T00:00:03+00:00",
        ),
        WorkflowEvent(
            type="result",
            value={"sent": True},
            timestamp="2026-05-14T00:00:04+00:00",
        ),
    ]
    out = transcript_to_mermaid(_transcript(events), kind="gantt")
    assert "section Active" in out
    assert "section Confirmation" in out
    assert "section Outcome" in out


def test_gantt_terminal_result_section_label() -> None:
    """Result events land under 'Outcome' so the terminal step
    stands out."""
    events = [
        WorkflowEvent(
            type="result",
            value="ok",
            timestamp="2026-05-14T00:00:01+00:00",
        )
    ]
    out = transcript_to_mermaid(_transcript(events), kind="gantt")
    assert "section Outcome" in out
    assert "result" in out


def test_gantt_terminal_error_section_label() -> None:
    events = [
        WorkflowEvent(
            type="error",
            error="boom",
            timestamp="2026-05-14T00:00:01+00:00",
        )
    ]
    out = transcript_to_mermaid(_transcript(events), kind="gantt")
    assert "section Outcome" in out
    assert "error" in out


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------


def test_gantt_renders_a_task_line_per_event() -> None:
    events = [
        WorkflowEvent(
            type="status",
            payload={"message": "go"},
            timestamp="2026-05-14T00:00:01+00:00",
        ),
        WorkflowEvent(
            type="result",
            value="ok",
            timestamp="2026-05-14T00:00:02+00:00",
        ),
    ]
    out = transcript_to_mermaid(_transcript(events), kind="gantt")
    # Mermaid task line format: ``    Label :id, start, end``.
    task_lines = [
        line for line in out.splitlines()
        if ":e" in line and "," in line
    ]
    assert len(task_lines) == 2


def test_gantt_uses_footer_to_extend_last_bar() -> None:
    """The last event's bar extends to the footer's ``ended_at``
    so the terminal task is visibly sized even when it's the
    last thing emitted."""
    events = [
        WorkflowEvent(
            type="result",
            value="ok",
            timestamp="2026-05-14T00:00:01+00:00",
        )
    ]
    footer = RunFooter(
        ended_at="2026-05-14T00:00:05+00:00", outcome="result"
    )
    out = transcript_to_mermaid(
        _transcript(events, footer=footer), kind="gantt"
    )
    # The end timestamp should reference the footer time.
    assert "2026-05-14T00:00:05" in out


def test_gantt_labels_carry_event_data() -> None:
    """Status messages, confirmation prompts, and decisions all
    appear in the task label so a viewer can read the chart
    without cross-referencing the transcript."""
    events = [
        WorkflowEvent(
            type="status",
            payload={"message": "Drafting email"},
            timestamp="2026-05-14T00:00:01+00:00",
        ),
        WorkflowEvent(
            type="confirmation_request",
            payload={"message": "Send?"},
            timestamp="2026-05-14T00:00:02+00:00",
        ),
        WorkflowEvent(
            type="confirmation_resolved",
            payload={"decision": "confirm"},
            timestamp="2026-05-14T00:00:03+00:00",
        ),
    ]
    out = transcript_to_mermaid(_transcript(events), kind="gantt")
    assert "Drafting email" in out
    assert "Send?" in out
    assert "confirm" in out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_gantt_is_a_recognized_kind() -> None:
    """Regression: ``kind='gantt'`` no longer raises
    ``ValueError`` (it did before this PR)."""
    transcript_to_mermaid(_transcript([]), kind="gantt")
