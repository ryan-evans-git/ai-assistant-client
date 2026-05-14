"""Tests for the transcript-store backends + factory.

Each backend is exercised through the same scenario set so a
future SQL/cloud backend can adopt the same test list and prove
shape parity with the existing ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_assistant_client.persistence import (
    FileTranscriptStore,
    InMemoryTranscriptStore,
    RunFooter,
    RunHeader,
    TranscriptStore,
    make_transcript_store,
)
from ai_assistant_client.persistence.factory import (
    TRANSCRIPT_BACKEND_ENV,
    TRANSCRIPT_DIR_ENV,
)
from ai_assistant_client.workflows.runtime import WorkflowEvent


def _header(run_id: str = "run-1") -> RunHeader:
    return RunHeader(
        run_id=run_id,
        workflow_name="send_email",
        tool_use_id="tu-1",
        args={"to": "a@b.com"},
        started_at="2026-05-14T00:00:00+00:00",
    )


def _event(type_: str = "status") -> WorkflowEvent:
    return WorkflowEvent(type=type_, payload={"msg": "x"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Backend-agnostic scenarios — parametrised across backends so they
# stay symmetric as new ones land.
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_store() -> TranscriptStore:
    return InMemoryTranscriptStore()


@pytest.fixture
def file_store(tmp_path: Path) -> TranscriptStore:
    return FileTranscriptStore(tmp_path / "transcripts")


@pytest.fixture(params=["memory", "file"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> TranscriptStore:
    if request.param == "memory":
        return InMemoryTranscriptStore()
    return FileTranscriptStore(tmp_path / "transcripts")


async def test_header_event_footer_round_trip(store: TranscriptStore) -> None:
    h = _header()
    await store.write_header(h)
    await store.append_event("run-1", _event("status"))
    await store.append_event(
        "run-1", WorkflowEvent(type="result", value={"ok": True})
    )
    await store.write_footer(
        "run-1", RunFooter(ended_at="2026-05-14T00:00:01+00:00", outcome="result")
    )

    transcript = await store.read("run-1")
    assert transcript.header == h
    assert [e.type for e in transcript.events] == ["status", "result"]
    assert transcript.events[1].value == {"ok": True}
    assert transcript.footer is not None
    assert transcript.footer.outcome == "result"


async def test_partial_transcript_missing_footer(store: TranscriptStore) -> None:
    """Crash mid-run → header + some events + no footer."""
    await store.write_header(_header())
    await store.append_event("run-1", _event("status"))

    transcript = await store.read("run-1")
    assert transcript.footer is None
    assert len(transcript.events) == 1


async def test_duplicate_header_rejected(store: TranscriptStore) -> None:
    await store.write_header(_header())
    with pytest.raises(ValueError):
        await store.write_header(_header())


async def test_append_to_unknown_run_raises(store: TranscriptStore) -> None:
    with pytest.raises(KeyError):
        await store.append_event("never-written", _event())


async def test_footer_for_unknown_run_raises(store: TranscriptStore) -> None:
    with pytest.raises(KeyError):
        await store.write_footer(
            "never-written",
            RunFooter(ended_at="2026-05-14T00:00:00+00:00", outcome="error"),
        )


async def test_read_unknown_run_raises(store: TranscriptStore) -> None:
    with pytest.raises(KeyError):
        await store.read("never-written")


async def test_list_runs_returns_written_ids(store: TranscriptStore) -> None:
    await store.write_header(_header("run-a"))
    await store.write_header(_header("run-b"))
    runs = await store.list_runs()
    assert set(runs) == {"run-a", "run-b"}


# ---------------------------------------------------------------------------
# File-backend-specific paths
# ---------------------------------------------------------------------------


async def test_file_backend_rejects_traversal_run_ids(
    file_store: FileTranscriptStore,
) -> None:
    """Run ids should be opaque tokens; reject anything that could
    escape the base directory or otherwise collide with the
    filesystem."""
    for bad in ("../etc/passwd", "a/b", "a\\b", ".hidden", ""):
        with pytest.raises(ValueError):
            await file_store.write_header(_header(bad))


async def test_file_backend_survives_corrupt_trailing_line(
    file_store: FileTranscriptStore, tmp_path: Path
) -> None:
    """A crash mid-write leaves a partial JSON line.  Read should
    return everything before it instead of failing the whole
    transcript."""
    await file_store.write_header(_header())
    await file_store.append_event("run-1", _event("status"))
    # Simulate a partial trailing line.
    path = tmp_path / "transcripts" / "run-1.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write('{"kind": "event", "type": "stat')  # no newline, no close

    transcript = await file_store.read("run-1")
    assert len(transcript.events) == 1
    assert transcript.events[0].type == "status"


async def test_file_backend_skips_unknown_record_kinds(
    file_store: FileTranscriptStore, tmp_path: Path
) -> None:
    """Forward-compat: a future record kind shouldn't break readers
    written against today's wire format."""
    await file_store.write_header(_header())
    path = tmp_path / "transcripts" / "run-1.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write('{"kind": "future-thing", "foo": 1}\n')
    await file_store.append_event("run-1", _event("status"))

    transcript = await file_store.read("run-1")
    assert len(transcript.events) == 1


async def test_file_backend_missing_header_raises_keyerror(
    file_store: FileTranscriptStore, tmp_path: Path
) -> None:
    """A file that exists but has no header line is treated as a
    missing run rather than a half-readable record."""
    path = tmp_path / "transcripts" / "orphan.jsonl"
    path.write_text('{"kind": "event", "type": "status"}\n', encoding="utf-8")
    with pytest.raises(KeyError):
        await file_store.read("orphan")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_defaults_to_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRANSCRIPT_BACKEND_ENV, raising=False)
    store = make_transcript_store()
    assert isinstance(store, InMemoryTranscriptStore)


def test_factory_honors_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TRANSCRIPT_BACKEND_ENV, "file")
    monkeypatch.setenv(TRANSCRIPT_DIR_ENV, str(tmp_path / "t"))
    store = make_transcript_store()
    assert isinstance(store, FileTranscriptStore)


def test_factory_explicit_args_win(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TRANSCRIPT_BACKEND_ENV, "file")
    store = make_transcript_store(kind="memory")
    assert isinstance(store, InMemoryTranscriptStore)


def test_factory_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown transcript backend"):
        make_transcript_store(kind="redis-cluster")
