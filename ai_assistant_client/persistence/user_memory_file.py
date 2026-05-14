"""JSONL-on-disk :class:`MemoryStore`.

One ``<user_id>.jsonl`` file per user under a configured base
directory.  Each line is one record:

* ``{"op": "add", ...full Memory body...}``
* ``{"op": "update", "memory_id": ..., "value": ..., "updated_at": ...}``
* ``{"op": "remove", "memory_id": ...}``

Append-only log (event-sourced) so per-line atomicity is enough
to keep the file consistent — there's never an in-place rewrite
that could be torn by a crash.  Read replays the log to
reconstruct the live state.

``forget_all(user_id)`` deletes the user's file outright — the
GDPR-style erasure case wants the bytes gone, not a tombstone
record.

Same filesystem-safety constraints as the other file-backed
stores: user ids must be opaque tokens, never user-supplied
paths.  The validation in :meth:`_path_for` rejects anything
that could escape the base directory.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ai_assistant_client.persistence.user_memory import Memory, MemoryStore


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_memory_id() -> str:
    return f"mem_{uuid.uuid4().hex}"


class FileMemoryStore(MemoryStore):
    """JSONL :class:`MemoryStore` rooted at a base directory."""

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    # -- file naming -----------------------------------------------

    def _path_for(self, user_id: str) -> Path:
        if (
            not user_id
            or "/" in user_id
            or "\\" in user_id
            or user_id.startswith(".")
        ):
            raise ValueError(f"invalid user id {user_id!r}")
        return self._base / f"{user_id}.jsonl"

    def _lock_for(self, user_id: str) -> asyncio.Lock:
        lock = self._locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[user_id] = lock
        return lock

    # -- replay log → live state -----------------------------------

    def _load_records(self, path: Path) -> dict[str, Memory]:
        """Reconstruct live state from the on-disk log.

        Linear in the log size; for a v1 single-user store
        that's fine.  A future PR can snapshot+compact when a
        user's log grows past a threshold.
        """
        if not path.exists():
            return {}
        live: dict[str, Memory] = {}
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    # Truncated trailing line from a crash mid-
                    # write — drop and continue, same crash-
                    # tolerance pattern as the other file stores.
                    continue
                op = record.get("op")
                if op == "add":
                    live[record["memory_id"]] = Memory(
                        memory_id=record["memory_id"],
                        user_id=record["user_id"],
                        key=record["key"],
                        value=record["value"],
                        tags=tuple(record.get("tags", [])),
                        created_at=record.get("created_at", ""),
                        updated_at=record.get("updated_at", ""),
                    )
                elif op == "update":
                    existing = live.get(record["memory_id"])
                    if existing is None:
                        continue
                    live[record["memory_id"]] = Memory(
                        memory_id=existing.memory_id,
                        user_id=existing.user_id,
                        key=existing.key,
                        value=record["value"],
                        tags=existing.tags,
                        created_at=existing.created_at,
                        updated_at=record.get("updated_at", ""),
                    )
                elif op == "remove":
                    live.pop(record["memory_id"], None)
        return live

    # -- write primitives ------------------------------------------

    @staticmethod
    def _append_line(path: Path, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")

    # -- protocol --------------------------------------------------

    async def add(
        self,
        *,
        user_id: str,
        key: str,
        value: Any,
        tags: Iterable[str] = (),
    ) -> Memory:
        path = self._path_for(user_id)
        ts = _utc_iso()
        record = Memory(
            memory_id=_new_memory_id(),
            user_id=user_id,
            key=key,
            value=value,
            tags=tuple(tags),
            created_at=ts,
            updated_at=ts,
        )
        entry = {
            "op": "add",
            "memory_id": record.memory_id,
            "user_id": record.user_id,
            "key": record.key,
            "value": record.value,
            "tags": list(record.tags),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }
        async with self._lock_for(user_id):
            await asyncio.to_thread(self._append_line, path, entry)
        return record

    async def get(self, *, user_id: str, memory_id: str) -> Memory:
        path = self._path_for(user_id)
        async with self._lock_for(user_id):
            live = await asyncio.to_thread(self._load_records, path)
        record = live.get(memory_id)
        if record is None:
            raise KeyError(memory_id)
        return record

    async def list(
        self,
        *,
        user_id: str,
        tags: Iterable[str] | None = None,
    ) -> list[Memory]:
        path = self._path_for(user_id)
        async with self._lock_for(user_id):
            live = await asyncio.to_thread(self._load_records, path)
        wanted: set[str] | None = set(tags) if tags is not None else None
        records = list(live.values())
        # Preserve insertion order — dict iteration is guaranteed
        # insertion-ordered on every CPython since 3.7.
        if wanted is None:
            return records
        return [r for r in records if wanted.intersection(r.tags)]

    async def update(
        self,
        *,
        user_id: str,
        memory_id: str,
        value: Any,
    ) -> Memory:
        path = self._path_for(user_id)
        async with self._lock_for(user_id):
            live = await asyncio.to_thread(self._load_records, path)
            existing = live.get(memory_id)
            if existing is None:
                raise KeyError(memory_id)
            ts = _utc_iso()
            entry = {
                "op": "update",
                "memory_id": memory_id,
                "value": value,
                "updated_at": ts,
            }
            await asyncio.to_thread(self._append_line, path, entry)
            return Memory(
                memory_id=existing.memory_id,
                user_id=existing.user_id,
                key=existing.key,
                value=value,
                tags=existing.tags,
                created_at=existing.created_at,
                updated_at=ts,
            )

    async def remove(self, *, user_id: str, memory_id: str) -> None:
        path = self._path_for(user_id)
        async with self._lock_for(user_id):
            live = await asyncio.to_thread(self._load_records, path)
            if memory_id not in live:
                raise KeyError(memory_id)
            entry = {"op": "remove", "memory_id": memory_id}
            await asyncio.to_thread(self._append_line, path, entry)

    async def forget_all(self, *, user_id: str) -> int:
        path = self._path_for(user_id)
        async with self._lock_for(user_id):
            if not path.exists():
                return 0
            live = await asyncio.to_thread(self._load_records, path)
            count = len(live)
            # GDPR-style erasure: delete the bytes outright, no
            # tombstone.  A subsequent ``list`` returns ``[]`` and
            # ``list_users`` no longer mentions this id.
            await asyncio.to_thread(path.unlink)
            return count

    async def list_users(self) -> list[str]:
        # Stem of every ``*.jsonl`` file under base_dir.  A file
        # that exists but parses to zero live records (every add
        # was followed by a remove) is still listed — the file
        # itself is the existence signal at the filesystem layer.
        return await asyncio.to_thread(_list_user_ids, self._base)


def _list_user_ids(base: Path) -> list[str]:
    return sorted(p.stem for p in base.glob("*.jsonl"))
