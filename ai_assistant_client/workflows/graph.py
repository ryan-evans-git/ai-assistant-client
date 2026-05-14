"""Render a recorded workflow run as a Mermaid diagram.

Two output kinds, both producing a self-contained ``mermaid``
code block string:

* ``flowchart`` (default) — top-down DAG with one node per
  recorded event.  Confirmation pauses become decision diamonds
  branching by decision (confirm / decline).  Terminal nodes
  carry a style hint so renderers colour ``result`` and
  ``error`` differently.
* ``sequence`` — sequence diagram with the workflow on one
  side and the user on the other.  Status / result events
  arrow from workflow → user; confirmation requests arrow
  workflow → user, and the resolved decision arrows back.

Both forms are pure functions of :class:`RunTranscript` — no
side effects, no I/O.  Callers persist the output wherever
they want (a markdown file, a PR comment, a debug page).

Why Mermaid: text-based (round-trips through git), renders
natively in GitHub markdown / GitLab / Notion / many static-
site generators, and has zero runtime dependency on this
package.  Nothing else to install.

This is intentionally a *post-hoc* renderer: it consumes a
recorded transcript rather than instrumenting the runtime
directly.  Decoupling means the graph helpers can produce
diagrams from any transcript source (file, sqlite, SQL DB)
without caring how the run was originally captured.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Literal

from ai_assistant_client.persistence.transcript import RunTranscript
from ai_assistant_client.workflows.runtime import WorkflowEvent


DiagramKind = Literal["flowchart", "sequence", "gantt"]


def transcript_to_mermaid(
    transcript: RunTranscript, *, kind: DiagramKind = "flowchart"
) -> str:
    """Render ``transcript`` as a Mermaid diagram string.

    Returns the full diagram body — no enclosing ``` fences,
    so the caller can embed it however suits their output
    (markdown code block, raw HTML ``<pre>``, an SSE event,
    etc.).
    """
    if kind == "flowchart":
        return _render_flowchart(transcript)
    if kind == "sequence":
        return _render_sequence(transcript)
    if kind == "gantt":
        return _render_gantt(transcript)
    raise ValueError(f"unknown diagram kind {kind!r}")


# ---------------------------------------------------------------------------
# Flowchart
# ---------------------------------------------------------------------------


def _render_flowchart(transcript: RunTranscript) -> str:
    lines: list[str] = ["flowchart TD"]

    # Header node — workflow identity + start args.
    header_label = (
        f"Workflow: {transcript.header.workflow_name}"
        f"<br/>started {transcript.header.started_at}"
    )
    lines.append(f"    H[\"{_escape(header_label)}\"]")

    # Walk events; pair up confirmation_request → confirmation_resolved
    # so the pair renders as a single decision node.
    prev_node = "H"
    confirm_pending: tuple[str, str] | None = None  # (node_id, message)

    for ev_index, event in enumerate(transcript.events):
        node_id = f"E{ev_index}"

        if event.type == "confirmation_request":
            message = _confirmation_message(event)
            confirm_pending = (node_id, message)
            lines.append(
                f"    {node_id}{{{{\"{_escape('Confirmation: ' + message)}\"}}}}"
            )
            lines.append(f"    {prev_node} --> {node_id}")
            prev_node = node_id
            continue

        if event.type == "confirmation_resolved" and confirm_pending is not None:
            decision = (event.payload or {}).get("decision", "?")
            note = (event.payload or {}).get("note") or ""
            # Edge label carries the decision so a reader can
            # see "confirm" vs "decline" without inspecting
            # node bodies.
            label = decision if not note else f"{decision}: {note}"
            # Replace the previous edge target — the resolved
            # event is rendered as the next regular node and
            # the previous arrow gets a decision label.
            res_node = f"E{ev_index}"
            lines.append(
                f"    {res_node}[\"{_escape('Resolved: ' + decision)}\"]"
            )
            # Re-emit an explicit edge with the decision label.
            lines.append(
                f"    {confirm_pending[0]} -->|{_escape(label)}| {res_node}"
            )
            prev_node = res_node
            confirm_pending = None
            continue

        # Default: a rectangle node carrying the event's type +
        # a compact rendition of the relevant payload field.
        body = _event_body(event)
        shape_open, shape_close = _shape_for(event.type)
        lines.append(
            f"    {node_id}{shape_open}\"{_escape(f'{event.type}: {body}')}\"{shape_close}"
        )
        lines.append(f"    {prev_node} --> {node_id}")
        prev_node = node_id

    # Style hints for the terminal outcome so renderers colour
    # success vs failure differently.  Mermaid's ``classDef``
    # / ``class`` works in GitHub's renderer.
    lines.append("    classDef ok fill:#d4edda,stroke:#28a745;")
    lines.append("    classDef err fill:#f8d7da,stroke:#dc3545;")
    if transcript.events:
        terminal = transcript.events[-1]
        terminal_id = f"E{len(transcript.events) - 1}"
        if terminal.type == "result":
            lines.append(f"    class {terminal_id} ok;")
        elif terminal.type == "error":
            lines.append(f"    class {terminal_id} err;")

    return "\n".join(lines)


def _shape_for(event_type: str) -> tuple[str, str]:
    """Pick a Mermaid node shape for a given event type.

    Status / result use a rounded rectangle (``([…])``), errors
    a hex (``[/…/]``), everything else a plain rectangle.  The
    shape gives a reader a quick at-a-glance read on the
    event's kind before they look at the label.
    """
    if event_type == "result":
        return "([", "])"
    if event_type == "error":
        return "[/", "/]"
    return "[", "]"


def _event_body(event: WorkflowEvent) -> str:
    """Compact textual rendition for a node label."""
    if event.type == "status":
        return _payload_message(event)
    if event.type == "result":
        return _short_repr(event.value)
    if event.type == "error":
        return event.error or "(no message)"
    return _short_repr(event.payload)


def _payload_message(event: WorkflowEvent) -> str:
    payload = event.payload or {}
    return str(payload.get("message", _short_repr(payload)))


def _confirmation_message(event: WorkflowEvent) -> str:
    payload = event.payload or {}
    msg = payload.get("message")
    if isinstance(msg, str) and msg.strip():
        return msg.strip()
    return "(no message)"


def _short_repr(value: object, *, limit: int = 80) -> str:
    """Compact, single-line repr for embedding in a diagram label."""
    if value is None:
        return "∅"
    text = repr(value)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


# ---------------------------------------------------------------------------
# Sequence diagram
# ---------------------------------------------------------------------------


def _render_sequence(transcript: RunTranscript) -> str:
    lines: list[str] = [
        "sequenceDiagram",
        f"    participant W as {_escape(transcript.header.workflow_name)}",
        "    participant U as User",
    ]
    for event in transcript.events:
        for line in _sequence_lines(event):
            lines.append(line)
    if transcript.footer is not None:
        lines.append(
            f"    Note over W: ended {_escape(transcript.footer.ended_at)}"
            f" ({_escape(transcript.footer.outcome)})"
        )
    return "\n".join(lines)


def _sequence_lines(event: WorkflowEvent) -> Iterable[str]:
    if event.type == "status":
        yield (
            f"    W->>U: status — {_escape(_payload_message(event))}"
        )
    elif event.type == "confirmation_request":
        yield (
            f"    W->>U: confirm? {_escape(_confirmation_message(event))}"
        )
    elif event.type == "confirmation_resolved":
        decision = (event.payload or {}).get("decision", "?")
        yield f"    U->>W: {_escape(decision)}"
    elif event.type == "result":
        yield f"    W->>U: result — {_escape(_short_repr(event.value))}"
    elif event.type == "error":
        yield f"    W->>U: error — {_escape(event.error or '(no message)')}"


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Gantt timing diagram
# ---------------------------------------------------------------------------


def _render_gantt(transcript: RunTranscript) -> str:
    """Render the run as a Mermaid Gantt chart.

    One task per event, positioned by its ``timestamp`` and
    sized to the gap between consecutive events.  Sections
    separate confirmation pauses from active workflow time so a
    reader can see at a glance how much wall time was spent
    waiting on user input vs. doing work.

    Requires events to have ISO-8601 timestamps (added in the
    per-event-timestamps PR).  Events missing a timestamp are
    skipped — a partial / pre-timestamp transcript renders the
    events it has data for; if no events have timestamps the
    output is just the title + axis line so callers can detect
    the empty case.
    """
    title = (
        f"Workflow {transcript.header.workflow_name}"
        if transcript.header.workflow_name
        else "Workflow run"
    )
    lines: list[str] = [
        "gantt",
        f"    title {_escape(title)}",
        # Use ISO-8601 since that's what WorkflowEvent.timestamp
        # emits; Mermaid's Gantt accepts it directly.
        "    dateFormat YYYY-MM-DDTHH:mm:ss.SSSXXX",
        "    axisFormat %H:%M:%S",
    ]

    # Parse all events with usable timestamps in order.
    stamped: list[tuple[WorkflowEvent, datetime]] = []
    for event in transcript.events:
        ts = _parse_iso(event.timestamp)
        if ts is not None:
            stamped.append((event, ts))

    if not stamped:
        return "\n".join(lines)

    # Use the header's started_at when available so the chart's
    # first bar is anchored at the run start rather than the
    # first event (status events fire after a tick or two).
    run_start = _parse_iso(transcript.header.started_at) or stamped[0][1]
    # Footer ends the chart bound; fall back to the last event.
    run_end_str = transcript.footer.ended_at if transcript.footer else None
    run_end = _parse_iso(run_end_str) if run_end_str else None
    if run_end is None:
        run_end = stamped[-1][1]

    # Each event becomes a task that spans from its timestamp
    # to the next event's timestamp.  The last event runs until
    # the footer / last-known time so the bar is visible rather
    # than zero-width.
    section_seen: set[str] = set()
    for i, (event, ts) in enumerate(stamped):
        section = _section_for(event.type)
        if section not in section_seen:
            lines.append(f"    section {_escape(section)}")
            section_seen.add(section)

        next_ts = stamped[i + 1][1] if i + 1 < len(stamped) else run_end
        # Mermaid Gantt requires a non-zero duration; clamp very
        # short gaps to 1ms so the bar still renders.
        if next_ts <= ts:
            next_ts = ts + (run_end - run_start) / max(1, len(stamped) * 10)

        task_id = f"e{i}"
        label = _gantt_label(event)
        start_iso = _to_mermaid_iso(ts)
        end_iso = _to_mermaid_iso(next_ts)
        lines.append(
            f"    {_escape(label)} :{task_id}, {start_iso}, {end_iso}"
        )

    return "\n".join(lines)


def _section_for(event_type: str) -> str:
    """Group event types into Gantt sections.

    Confirmation request + resolved together so a reader can
    see the pause as one block.  Result / error get their own
    section so the terminal step stands out.
    """
    if event_type in ("confirmation_request", "confirmation_resolved"):
        return "Confirmation"
    if event_type in ("result", "error"):
        return "Outcome"
    return "Active"


def _gantt_label(event: WorkflowEvent) -> str:
    """Compact label for a Gantt task.

    Mermaid Gantt task names appear *to the left of* the bar;
    keep them short so they don't overflow the legend column.
    """
    if event.type == "status":
        return _payload_message(event)[:40]
    if event.type == "confirmation_request":
        return f"confirm? {_confirmation_message(event)[:30]}"
    if event.type == "confirmation_resolved":
        decision = (event.payload or {}).get("decision", "?")
        return f"resolved: {decision}"
    if event.type == "result":
        return "result"
    if event.type == "error":
        return "error"
    return event.type


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string into a tz-aware ``datetime``.

    Returns ``None`` for empty / malformed inputs — used by the
    Gantt renderer to filter events with unknown timestamps
    rather than crashing the whole render.
    """
    if not value:
        return None
    try:
        # ``datetime.fromisoformat`` accepts the ISO-8601 strings
        # the workflow runtime emits (incl. ``+00:00`` offsets).
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _to_mermaid_iso(value: datetime) -> str:
    """Format a datetime back to the ISO-8601 form Mermaid's
    Gantt dateFormat expects (``YYYY-MM-DDTHH:mm:ss.SSSXXX``)."""
    return value.isoformat()


def _escape(text: str) -> str:
    """Make ``text`` safe for inclusion in a Mermaid label.

    Mermaid is whitespace- and quote-sensitive inside node
    bodies.  Replace newlines with the HTML ``<br/>`` tag,
    double-quotes with single (Mermaid 8+ accepts ``#quot;``
    but the GitHub renderer is fussier), and strip semicolons
    + pipes that would split the line on the parser side.
    """
    return (
        text.replace("\n", "<br/>")
        .replace('"', "'")
        .replace(";", ",")
        .replace("|", "/")
    )
