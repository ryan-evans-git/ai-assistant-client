"""File-backed transcript store (JSON-lines on local disk).

One ``.jsonl`` file per run, named ``<run_id>.jsonl`` under a
configured base directory.  Each line is a single JSON object
with a ``kind`` discriminator:

* ``{"kind": "header", ...}`` — written first, exactly once.
* ``{"kind": "event", "type": ..., ...}`` — one per emitted
  :class:`~ai_assistant_client.workflows.runtime.WorkflowEvent`.
* ``{"kind": "footer", ...}`` — written last; absent on a
  partial / interrupted run.

Wire format choice: JSONL stays legible after a crash, can be
``tail -f``'d during development, and round-trips through
existing tooling (``jq``, ``grep``).  A future SQL backend
implements the same protocol with normalised rows instead.

I/O is offloaded to a thread (via :func:`asyncio.to_thread`) so
record-mode never blocks the workflow's event loop.  Per-run
write serialization comes from a per-id :class:`asyncio.Lock` —
the runtime emits events from a single task so contention is
nominal, but the lock makes interleaving deterministic and
keeps a future multi-writer extension straightforward.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ai_assistant_client.persistence.transcript import (
    RunFooter,
    RunHeader,
    RunTranscript,
)
from ai_assistant_client.workflows.runtime import WorkflowEvent


class FileTranscriptStore:
    """JSONL :class:`TranscriptStore` rooted at a base directory."""

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    # -- file naming ------------------------------------------------

    def _path_for(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
            # Reject anything that could escape the base dir or
            # collide with directory traversal — run ids should be
            # opaque tokens (UUIDs, hashes), never user-supplied
            # paths.
            raise ValueError(f"invalid run id {run_id!r}")
        return self._base / f"{run_id}.jsonl"

    def _lock_for(self, run_id: str) -> asyncio.Lock:
        lock = self._locks.get(run_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[run_id] = lock
        return lock

    # -- write path -------------------------------------------------

    async def write_header(self, header: RunHeader) -> None:
        path = self._path_for(header.run_id)
        async with self._lock_for(header.run_id):
            if path.exists():
                raise ValueError(
                    f"run id {header.run_id!r} already has a transcript file "
                    "— transcripts are append-only and not resumable"
                )
            record = {"kind": "header", **asdict(header)}
            await asyncio.to_thread(_append_line, path, record)

    async def append_event(self, run_id: str, event: WorkflowEvent) -> None:
        path = self._path_for(run_id)
        async with self._lock_for(run_id):
            if not path.exists():
                raise KeyError(run_id)
            record = {"kind": "event", **asdict(event)}
            await asyncio.to_thread(_append_line, path, record)

    async def write_footer(self, run_id: str, footer: RunFooter) -> None:
        path = self._path_for(run_id)
        async with self._lock_for(run_id):
            if not path.exists():
                raise KeyError(run_id)
            record = {"kind": "footer", **asdict(footer)}
            await asyncio.to_thread(_append_line, path, record)

    # -- read path --------------------------------------------------

    async def read(self, run_id: str) -> RunTranscript:
        path = self._path_for(run_id)
        if not path.exists():
            raise KeyError(run_id)
        async with self._lock_for(run_id):
            lines = await asyncio.to_thread(_read_lines, path)

        header: RunHeader | None = None
        events: list[WorkflowEvent] = []
        footer: RunFooter | None = None
        for line in lines:
            kind = line.get("kind")
            if kind == "header":
                header = RunHeader(
                    run_id=line["run_id"],
                    workflow_name=line["workflow_name"],
                    tool_use_id=line["tool_use_id"],
                    args=line.get("args", {}),
                    started_at=line["started_at"],
                )
            elif kind == "event":
                events.append(
                    WorkflowEvent(
                        type=line["type"],
                        payload=line.get("payload"),
                        value=line.get("value"),
                        error=line.get("error"),
                        # Older transcripts lack ``timestamp`` —
                        # preserve "unknown" rather than fabricating
                        # one at read time.
                        timestamp=line.get("timestamp", ""),
                    )
                )
            elif kind == "footer":
                footer = RunFooter(
                    ended_at=line["ended_at"],
                    outcome=line["outcome"],
                )
            # Unknown kinds are skipped — keeps the format
            # forward-compatible with later record types.

        if header is None:
            # A file with no header is a corrupt or partial record
            # from a crash before the very first write.  Surfacing
            # as KeyError keeps the contract uniform with the
            # "missing run" case.
            raise KeyError(run_id)
        return RunTranscript(header=header, events=events, footer=footer)

    async def list_runs(self) -> list[str]:
        return await asyncio.to_thread(_list_run_ids, self._base)


# ---------------------------------------------------------------------------
# Sync helpers (run on a worker thread)
# ---------------------------------------------------------------------------


def _append_line(path: Path, record: dict[str, Any]) -> None:
    # ``default=str`` keeps non-JSON-native values (datetime, UUID,
    # set) survivable as strings instead of crashing the recording
    # — round-trip fidelity is good enough for regression-test
    # replay and the alternative is dropping records on every
    # field a workflow author wasn't strict about.
    line = json.dumps(record, default=str, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def _read_lines(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                # A truncated trailing line from a crash mid-write
                # — preserve everything before it rather than
                # failing the whole read.
                continue
    return out


def _list_run_ids(base: Path) -> list[str]:
    return sorted(p.stem for p in base.glob("*.jsonl"))
