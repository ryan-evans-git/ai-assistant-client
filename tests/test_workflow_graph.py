"""Tests for the Mermaid renderer over a recorded workflow run.

The renderer is a pure function — we assert on the produced
string, not on a rendered image.  Goals:

1. Both kinds (``flowchart`` / ``sequence``) emit a valid
   Mermaid header + at least one node / line per event.
2. Confirmation request + resolved pair up into a decision
   branch in the flowchart.
3. Terminal result / error gets a style class.
4. Special characters in labels are escaped enough to keep
   Mermaid parsers happy.
5. Empty transcripts (no events) still render — useful for
   the case where a run was interrupted before the first
   event was emitted.
"""

from __future__ import annotations

import pytest

from ai_assistant_client.persistence.transcript import (
    RunFooter,
    RunHeader,
    RunTranscript,
)
from ai_assistant_client.workflows.graph import transcript_to_mermaid
from ai_assistant_client.workflows.runtime import WorkflowEvent


def _header() -> RunHeader:
    return RunHeader(
        run_id="run-1",
        workflow_name="send_email",
        tool_use_id="tu-1",
        args={"to": "a@b.com"},
        started_at="2026-05-14T00:00:00+00:00",
    )


def _transcript(events: list[WorkflowEvent], *, footer: RunFooter | None = None) -> RunTranscript:
    return RunTranscript(header=_header(), events=events, footer=footer)


# ---------------------------------------------------------------------------
# Flowchart
# ---------------------------------------------------------------------------


def test_flowchart_starts_with_header_line() -> None:
    out = transcript_to_mermaid(_transcript([]), kind="flowchart")
    assert out.splitlines()[0] == "flowchart TD"


def test_flowchart_renders_one_node_per_event() -> None:
    events = [
        WorkflowEvent(type="status", payload={"message": "Drafting"}),
        WorkflowEvent(type="status", payload={"message": "Reviewing"}),
        WorkflowEvent(type="result", value={"ok": True}),
    ]
    out = transcript_to_mermaid(_transcript(events), kind="flowchart")

    # H header + 3 event nodes — each event node id is E0/E1/E2.
    for i in range(3):
        assert f"E{i}" in out
    # Header arrow chain.
    assert "H --> E0" in out
    assert "E0 --> E1" in out
    assert "E1 --> E2" in out


def test_flowchart_terminal_result_gets_ok_class() -> None:
    events = [
        WorkflowEvent(type="status", payload={"message": "go"}),
        WorkflowEvent(type="result", value={"ok": True}),
    ]
    out = transcript_to_mermaid(_transcript(events), kind="flowchart")
    assert "class E1 ok" in out
    assert "class E1 err" not in out


def test_flowchart_terminal_error_gets_err_class() -> None:
    events = [WorkflowEvent(type="error", error="kaboom")]
    out = transcript_to_mermaid(_transcript(events), kind="flowchart")
    assert "class E0 err" in out


def test_flowchart_confirmation_pair_renders_as_decision() -> None:
    """A confirmation_request followed by confirmation_resolved
    must produce a decision-shaped node with the decision on
    the edge label."""
    events = [
        WorkflowEvent(
            type="confirmation_request",
            payload={"message": "Send email?", "request_id": "r1"},
        ),
        WorkflowEvent(
            type="confirmation_resolved",
            payload={"decision": "confirm", "request_id": "r1"},
        ),
    ]
    out = transcript_to_mermaid(_transcript(events), kind="flowchart")

    # The request is rendered with a decision-style shape ``{{ … }}``.
    assert "{{\"Confirmation: Send email?\"}}" in out
    # The resolved-edge label carries the decision.
    assert "|confirm|" in out


def test_flowchart_decline_with_note_in_edge_label() -> None:
    events = [
        WorkflowEvent(
            type="confirmation_request",
            payload={"message": "Send?"},
        ),
        WorkflowEvent(
            type="confirmation_resolved",
            payload={"decision": "decline", "note": "wrong recipient"},
        ),
    ]
    out = transcript_to_mermaid(_transcript(events), kind="flowchart")
    assert "decline: wrong recipient" in out


def test_flowchart_handles_empty_event_list() -> None:
    """A run interrupted before any event still renders — useful
    when surfacing a crashed run's partial transcript."""
    out = transcript_to_mermaid(_transcript([]), kind="flowchart")
    assert "flowchart TD" in out
    assert "Workflow: send_email" in out
    # No event nodes referenced.
    assert "E0" not in out


def test_flowchart_escapes_quotes_in_labels() -> None:
    events = [
        WorkflowEvent(
            type="status", payload={"message": 'He said "hi"'}
        )
    ]
    out = transcript_to_mermaid(_transcript(events), kind="flowchart")
    # The double-quote inside the message is collapsed to single
    # so it doesn't terminate the Mermaid string literal.
    assert '"hi"' not in out
    assert "'hi'" in out


def test_flowchart_escapes_pipe_in_labels() -> None:
    """A literal ``|`` would split a Mermaid edge label — escape it."""
    events = [
        WorkflowEvent(type="status", payload={"message": "a|b"})
    ]
    out = transcript_to_mermaid(_transcript(events), kind="flowchart")
    assert "a|b" not in out
    assert "a/b" in out


def test_flowchart_escapes_newlines_as_br() -> None:
    events = [
        WorkflowEvent(type="status", payload={"message": "line1\nline2"})
    ]
    out = transcript_to_mermaid(_transcript(events), kind="flowchart")
    assert "line1<br/>line2" in out


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------


def test_sequence_starts_with_correct_header() -> None:
    out = transcript_to_mermaid(_transcript([]), kind="sequence")
    lines = out.splitlines()
    assert lines[0] == "sequenceDiagram"
    assert "participant W as send_email" in lines[1]
    assert "participant U as User" in lines[2]


def test_sequence_status_event_goes_workflow_to_user() -> None:
    events = [WorkflowEvent(type="status", payload={"message": "Drafting"})]
    out = transcript_to_mermaid(_transcript(events), kind="sequence")
    assert "W->>U: status — Drafting" in out


def test_sequence_confirmation_round_trip() -> None:
    events = [
        WorkflowEvent(
            type="confirmation_request", payload={"message": "Send?"}
        ),
        WorkflowEvent(
            type="confirmation_resolved", payload={"decision": "confirm"}
        ),
    ]
    out = transcript_to_mermaid(_transcript(events), kind="sequence")
    assert "W->>U: confirm? Send?" in out
    assert "U->>W: confirm" in out


def test_sequence_renders_result_value() -> None:
    events = [WorkflowEvent(type="result", value={"sent": True})]
    out = transcript_to_mermaid(_transcript(events), kind="sequence")
    assert "W->>U: result" in out
    assert "sent" in out


def test_sequence_renders_error_message() -> None:
    events = [WorkflowEvent(type="error", error="boom")]
    out = transcript_to_mermaid(_transcript(events), kind="sequence")
    assert "W->>U: error — boom" in out


def test_sequence_appends_footer_note_when_present() -> None:
    events = [WorkflowEvent(type="result", value="ok")]
    footer = RunFooter(
        ended_at="2026-05-14T00:00:05+00:00", outcome="result"
    )
    out = transcript_to_mermaid(
        _transcript(events, footer=footer), kind="sequence"
    )
    assert "Note over W: ended 2026-05-14T00:00:05+00:00 (result)" in out


def test_sequence_no_footer_no_note() -> None:
    """Partial run (no footer) renders without a trailing note —
    we don't fabricate timestamps."""
    out = transcript_to_mermaid(_transcript([]), kind="sequence")
    assert "Note over" not in out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_unknown_kind_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown diagram kind"):
        transcript_to_mermaid(_transcript([]), kind="gantt")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Edge cases (label repr)
# ---------------------------------------------------------------------------


def test_long_result_value_is_truncated_with_ellipsis() -> None:
    """Result values longer than the label limit get truncated so
    a huge JSON blob doesn't blow up the diagram width."""
    big = {"items": [f"item-{i}" for i in range(50)]}
    events = [WorkflowEvent(type="result", value=big)]
    out = transcript_to_mermaid(_transcript(events), kind="flowchart")
    assert "…" in out


def test_none_payload_renders_as_empty_glyph() -> None:
    """An event with a literal-None payload should still render
    something legible rather than the string ``None``."""
    events = [WorkflowEvent(type="status", payload=None)]
    out = transcript_to_mermaid(_transcript(events), kind="flowchart")
    # _payload_message falls through to _short_repr({}), which
    # renders the empty payload — the key invariant is that the
    # output is still well-formed Mermaid (not ``None``).
    assert "None" not in out.split("classDef")[0]


def test_confirmation_without_message_uses_fallback() -> None:
    """Confirmation event with no message gets the
    ``(no message)`` placeholder — better than rendering a
    truthy-but-empty string into the decision label."""
    events = [
        WorkflowEvent(
            type="confirmation_request", payload={"request_id": "r"}
        ),
        WorkflowEvent(
            type="confirmation_resolved", payload={"decision": "confirm"}
        ),
    ]
    out = transcript_to_mermaid(_transcript(events), kind="flowchart")
    assert "(no message)" in out


def test_unknown_event_type_falls_back_to_payload_repr() -> None:
    """The renderer shouldn't blow up on forward-compatible event
    types it doesn't recognize — fall back to a generic
    payload repr."""
    events = [
        WorkflowEvent(  # type: ignore[arg-type]
            type="future-thing",  # type: ignore[arg-type]
            payload={"x": 1},
        )
    ]
    out = transcript_to_mermaid(_transcript(events), kind="flowchart")
    assert "future-thing" in out
    assert "'x'" in out  # repr({"x": 1}) appears in the label
