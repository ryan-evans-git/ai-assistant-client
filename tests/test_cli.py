"""Tests for the offline CLI sub-commands.

Three things we want to lock down:

1. ``replay <run_id>`` emits one JSON line per recorded event,
   in order, with all fields preserved.
2. ``graph <run_id>`` writes Mermaid to stdout and exits 0;
   ``--kind sequence`` switches to the sequence diagram.
3. Unknown run ids exit with a non-zero status so a shell
   pipeline can detect the failure.

The tests build a real transcript on disk through a file-
backed store, then invoke ``main()`` with the matching env
vars set so the CLI's ``make_transcript_store()`` picks up
that same store.  That round-trip is the real contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_assistant_client.app import _cli_graph, _cli_replay
from ai_assistant_client.persistence import (
    FileTranscriptStore,
    RunFooter,
    RunHeader,
)
from ai_assistant_client.persistence.factory import (
    TRANSCRIPT_BACKEND_ENV,
    TRANSCRIPT_DIR_ENV,
)
from ai_assistant_client.workflows.runtime import WorkflowEvent


async def _seed_transcript(
    base: Path, run_id: str = "run-cli-1"
) -> None:
    store = FileTranscriptStore(base)
    header = RunHeader(
        run_id=run_id,
        workflow_name="send_email",
        tool_use_id="tu-1",
        args={"to": "a@b.com"},
        started_at="2026-05-14T00:00:00+00:00",
    )
    await store.write_header(header)
    await store.append_event(
        run_id,
        WorkflowEvent(
            type="status",
            payload={"message": "Drafting"},
            timestamp="2026-05-14T00:00:01+00:00",
        ),
    )
    await store.append_event(
        run_id,
        WorkflowEvent(
            type="result",
            value={"sent": True},
            timestamp="2026-05-14T00:00:02+00:00",
        ),
    )
    await store.write_footer(
        run_id,
        RunFooter(ended_at="2026-05-14T00:00:02+00:00", outcome="result"),
    )


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


async def test_replay_emits_one_json_line_per_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = tmp_path / "transcripts"
    await _seed_transcript(base)
    monkeypatch.setenv(TRANSCRIPT_BACKEND_ENV, "file")
    monkeypatch.setenv(TRANSCRIPT_DIR_ENV, str(base))

    rc = await _cli_replay("run-cli-1")
    captured = capsys.readouterr()

    assert rc == 0
    lines = [
        line for line in captured.out.strip().splitlines() if line.strip()
    ]
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["type"] == "status"
    assert parsed[0]["payload"] == {"message": "Drafting"}
    assert parsed[1]["type"] == "result"
    assert parsed[1]["value"] == {"sent": True}
    # Timestamps survive the JSON-line round trip.
    assert parsed[0]["timestamp"] == "2026-05-14T00:00:01+00:00"


async def test_replay_unknown_run_id_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A shell pipeline should be able to detect 'no such run'
    by exit code, not just by parsing stderr."""
    base = tmp_path / "transcripts"
    base.mkdir()
    monkeypatch.setenv(TRANSCRIPT_BACKEND_ENV, "file")
    monkeypatch.setenv(TRANSCRIPT_DIR_ENV, str(base))

    rc = await _cli_replay("does-not-exist")
    captured = capsys.readouterr()

    assert rc == 1
    assert "unknown run id" in captured.out


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------


async def test_graph_default_kind_is_flowchart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = tmp_path / "transcripts"
    await _seed_transcript(base)
    monkeypatch.setenv(TRANSCRIPT_BACKEND_ENV, "file")
    monkeypatch.setenv(TRANSCRIPT_DIR_ENV, str(base))

    rc = await _cli_graph("run-cli-1", kind="flowchart")
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out.startswith("flowchart TD")
    # The status event from the seed run should land in the diagram.
    assert "Drafting" in captured.out


async def test_graph_sequence_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = tmp_path / "transcripts"
    await _seed_transcript(base)
    monkeypatch.setenv(TRANSCRIPT_BACKEND_ENV, "file")
    monkeypatch.setenv(TRANSCRIPT_DIR_ENV, str(base))

    rc = await _cli_graph("run-cli-1", kind="sequence")
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out.startswith("sequenceDiagram")


async def test_graph_gantt_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end CLI dispatch for the Gantt diagram kind — the
    renderer itself is covered by ``test_workflow_graph_gantt.py``;
    this test catches breakage in the CLI's ``--kind`` parsing or
    plumbing into the renderer."""
    base = tmp_path / "transcripts"
    await _seed_transcript(base)
    monkeypatch.setenv(TRANSCRIPT_BACKEND_ENV, "file")
    monkeypatch.setenv(TRANSCRIPT_DIR_ENV, str(base))

    rc = await _cli_graph("run-cli-1", kind="gantt")
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out.startswith("gantt")
    # Mermaid Gantt requires a dateFormat directive.
    assert "dateFormat" in captured.out
    # Seed transcript stamped events with 2026-05-14 timestamps;
    # they should reach the chart.
    assert "2026-05-14" in captured.out


async def test_graph_unknown_run_id_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = tmp_path / "transcripts"
    base.mkdir()
    monkeypatch.setenv(TRANSCRIPT_BACKEND_ENV, "file")
    monkeypatch.setenv(TRANSCRIPT_DIR_ENV, str(base))
    rc = await _cli_graph("does-not-exist", kind="flowchart")
    assert rc == 1


# ---------------------------------------------------------------------------
# argparse plumbing — verify the dispatch table without re-entering
# asyncio.run (which clashes with the pytest-asyncio event loop).
# ---------------------------------------------------------------------------


def test_main_replay_dispatches_to_cli_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch the async helper and make sure ``main`` calls it
    with the parsed run-id.  Avoids running real asyncio.run
    from inside the test's own event loop."""
    from ai_assistant_client import app

    called_with: dict[str, str] = {}

    async def fake_replay(run_id: str) -> int:
        called_with["run_id"] = run_id
        return 0

    monkeypatch.setattr(app, "_cli_replay", fake_replay)
    rc = app.main(["replay", "abc-123"])
    assert rc == 0
    assert called_with == {"run_id": "abc-123"}


def test_main_graph_dispatches_to_cli_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_assistant_client import app

    called_with: dict[str, str] = {}

    async def fake_graph(run_id: str, *, kind: str) -> int:
        called_with["run_id"] = run_id
        called_with["kind"] = kind
        return 0

    monkeypatch.setattr(app, "_cli_graph", fake_graph)
    rc = app.main(["graph", "abc-123", "--kind", "sequence"])
    assert rc == 0
    assert called_with == {"run_id": "abc-123", "kind": "sequence"}


def test_main_graph_accepts_gantt_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """argparse's ``choices`` rejects unknown ``--kind`` values
    with SystemExit; this test pins down that ``gantt`` is on
    the allowed list so the renderer's third kind is actually
    reachable through the CLI."""
    from ai_assistant_client import app

    captured: dict[str, str] = {}

    async def fake_graph(run_id: str, *, kind: str) -> int:
        captured["kind"] = kind
        return 0

    monkeypatch.setattr(app, "_cli_graph", fake_graph)
    rc = app.main(["graph", "abc-123", "--kind", "gantt"])
    assert rc == 0
    assert captured["kind"] == "gantt"
