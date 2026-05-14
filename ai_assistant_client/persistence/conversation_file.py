"""JSONL-on-disk :class:`ConversationStore`.

One ``<conversation_id>.jsonl`` file per conversation under a
configured base directory.  Each line is one
:data:`~ai_assistant_client.persistence.conversation.Message`
encoded as JSON.

Same wire-format philosophy as
:class:`~ai_assistant_client.persistence.file.FileTranscriptStore`:

* JSONL stays legible after a crash and survives ``tail -f``.
* Unknown / malformed lines (e.g. a truncated trailing line
  from a crash mid-write) are skipped on read rather than
  failing the whole log — partial durability beats brittleness.

I/O is offloaded to a thread via :func:`asyncio.to_thread` so the
agent loop never blocks on the filesystem.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from ai_assistant_client.persistence.conversation import (
    ConversationStore,
    Message,
)


class FileConversationStore(ConversationStore):
    """JSONL conversation log rooted at ``base_dir``."""

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    def _path_for(self, conversation_id: str) -> Path:
        if (
            not conversation_id
            or "/" in conversation_id
            or "\\" in conversation_id
            or conversation_id.startswith(".")
        ):
            raise ValueError(f"invalid conversation id {conversation_id!r}")
        return self._base / f"{conversation_id}.jsonl"

    def _lock_for(self, conversation_id: str) -> asyncio.Lock:
        lock = self._locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[conversation_id] = lock
        return lock

    async def append(self, conversation_id: str, message: Message) -> None:
        path = self._path_for(conversation_id)
        async with self._lock_for(conversation_id):
            await asyncio.to_thread(_append_line, path, message)

    async def read(self, conversation_id: str) -> list[Message]:
        path = self._path_for(conversation_id)
        if not path.exists():
            return []
        async with self._lock_for(conversation_id):
            return await asyncio.to_thread(_read_lines, path)

    async def list_conversations(self) -> list[str]:
        return await asyncio.to_thread(_list_conversation_ids, self._base)


# ---------------------------------------------------------------------------
# Sync helpers (run on worker threads)
# ---------------------------------------------------------------------------


def _append_line(path: Path, message: Message) -> None:
    line = json.dumps(message, default=str, ensure_ascii=False)
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
                # Truncated trailing line from a crash mid-write —
                # preserve everything before it.
                continue
    return out


def _list_conversation_ids(base: Path) -> list[str]:
    return sorted(p.stem for p in base.glob("*.jsonl"))
