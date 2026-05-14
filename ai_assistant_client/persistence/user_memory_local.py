"""In-process :class:`MemoryStore` — dict-of-dicts keyed by user.

Loses data when the process exits.  Use for unit tests and for
dev where durability isn't needed.  Hosts that want
across-restart persistence should pick the file or (future
follow-up) SQL backends.

Implementation note: memory ids are short ``mem_{hex}`` tokens
generated server-side so callers don't choose them.  This keeps
ids opaque (no information leakage about what users have or
when memories were created) and free of host-controlled paths
that could escape a filesystem-backed store.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from ai_assistant_client.persistence.user_memory import Memory, MemoryStore


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_memory_id() -> str:
    return f"mem_{uuid.uuid4().hex}"


class LocalMemoryStore(MemoryStore):
    """Dict-backed :class:`MemoryStore` for dev and tests.

    "Local" rather than "InMemory" to keep ``InMemoryMemoryStore``
    out of the lexicon — the resulting class name would be
    a tongue-twister and read worse than the alternative.
    """

    def __init__(self) -> None:
        # memory_id → Memory.  We also keep a per-user index for
        # cheap ``list`` / ``forget_all`` so a single user with
        # many memories doesn't force a full-table scan.
        self._memories: dict[str, Memory] = {}
        self._by_user: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()

    async def add(
        self,
        *,
        user_id: str,
        key: str,
        value: Any,
        tags: Iterable[str] = (),
    ) -> Memory:
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
        async with self._lock:
            self._memories[record.memory_id] = record
            self._by_user.setdefault(user_id, []).append(record.memory_id)
        return record

    async def get(self, *, user_id: str, memory_id: str) -> Memory:
        async with self._lock:
            record = self._memories.get(memory_id)
            if record is None or record.user_id != user_id:
                # Same KeyError for "missing" and "wrong owner" so
                # a probing caller can't tell which is which.
                raise KeyError(memory_id)
            return record

    async def list(
        self,
        *,
        user_id: str,
        tags: Iterable[str] | None = None,
    ) -> list[Memory]:
        wanted: set[str] | None = set(tags) if tags is not None else None
        async with self._lock:
            ids = list(self._by_user.get(user_id, []))
            out: list[Memory] = []
            for mid in ids:
                record = self._memories.get(mid)
                if record is None:
                    continue
                if wanted is None or wanted.intersection(record.tags):
                    out.append(record)
            return out

    async def update(
        self,
        *,
        user_id: str,
        memory_id: str,
        value: Any,
    ) -> Memory:
        async with self._lock:
            existing = self._memories.get(memory_id)
            if existing is None or existing.user_id != user_id:
                raise KeyError(memory_id)
            updated = Memory(
                memory_id=existing.memory_id,
                user_id=existing.user_id,
                key=existing.key,
                value=value,
                tags=existing.tags,
                created_at=existing.created_at,
                updated_at=_utc_iso(),
            )
            self._memories[memory_id] = updated
            return updated

    async def remove(self, *, user_id: str, memory_id: str) -> None:
        async with self._lock:
            existing = self._memories.get(memory_id)
            if existing is None or existing.user_id != user_id:
                raise KeyError(memory_id)
            del self._memories[memory_id]
            ids = self._by_user.get(user_id, [])
            if memory_id in ids:
                ids.remove(memory_id)

    async def forget_all(self, *, user_id: str) -> int:
        async with self._lock:
            ids = self._by_user.pop(user_id, [])
            for mid in ids:
                self._memories.pop(mid, None)
            return len(ids)

    async def list_users(self) -> list[str]:
        async with self._lock:
            # A user with all memories removed via forget_all still
            # has an empty list entry; filter those out so the
            # report reflects "users with stored data" not "users
            # we've ever seen."
            return [u for u, ids in self._by_user.items() if ids]
