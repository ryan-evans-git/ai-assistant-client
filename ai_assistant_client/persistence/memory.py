"""In-memory transcript store.

Single-process dict-of-records backend.  Used as the default in
unit tests and for ad-hoc local recording where durability isn't
needed.  Data is lost when the process exits — callers that need
persistence across restarts should pick the file or (future)
SQL backends.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ai_assistant_client.persistence.transcript import (
    RunFooter,
    RunHeader,
    RunTranscript,
)
from ai_assistant_client.workflows.runtime import WorkflowEvent


@dataclass
class _Slot:
    header: RunHeader
    events: list[WorkflowEvent] = field(default_factory=list)
    footer: RunFooter | None = None


class InMemoryTranscriptStore:
    """Dict-backed :class:`TranscriptStore` for dev and tests."""

    def __init__(self) -> None:
        self._slots: dict[str, _Slot] = {}
        # One lock for the whole store is fine — operations are
        # short and there's no realistic contention pattern at
        # the scale this backend serves.
        self._lock = asyncio.Lock()

    async def write_header(self, header: RunHeader) -> None:
        async with self._lock:
            if header.run_id in self._slots:
                raise ValueError(
                    f"run id {header.run_id!r} already has a header — "
                    "transcripts are append-only and not resumable"
                )
            self._slots[header.run_id] = _Slot(header=header)

    async def append_event(
        self, run_id: str, event: WorkflowEvent
    ) -> None:
        async with self._lock:
            slot = self._slots.get(run_id)
            if slot is None:
                raise KeyError(run_id)
            slot.events.append(event)

    async def write_footer(self, run_id: str, footer: RunFooter) -> None:
        async with self._lock:
            slot = self._slots.get(run_id)
            if slot is None:
                raise KeyError(run_id)
            slot.footer = footer

    async def read(self, run_id: str) -> RunTranscript:
        async with self._lock:
            slot = self._slots.get(run_id)
            if slot is None:
                raise KeyError(run_id)
            return RunTranscript(
                header=slot.header,
                events=list(slot.events),
                footer=slot.footer,
            )

    async def list_runs(self) -> list[str]:
        async with self._lock:
            return list(self._slots.keys())
