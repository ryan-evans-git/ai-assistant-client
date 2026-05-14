"""Per-user memory: durable notes the host attaches to a user.

This protocol underpins the kind of personalisation Claude Code's
own ``memory/`` system provides — small, typed notes ("user is a
data scientist", "prefers concise replies") that survive across
conversations and inform future turns.

What ships here is **storage only**.  Wiring the store into the
agent loop — how memories are retrieved, when they're injected
into the system prompt, what an LLM-callable
``memory_recall``/``memory_remember`` meta-tool looks like — is
deliberately a separate, later design pass.  Memory crosses two
sensitive surfaces:

1. **Privacy / multi-tenancy.**  Every operation takes a
   ``user_id``; the store can't accidentally cross-pollinate
   data between users.  ``forget_all(user_id)`` covers
   GDPR-style erasure requests.
2. **Prompt-injection resistance.**  Stored memories are
   **untrusted input** when re-injected into a model context —
   a malicious-but-cooperative user can plant text designed to
   override the system prompt on a future turn.  Hosts wiring
   memory into prompts MUST:

   * Wrap recalled content in clear, unambiguous delimiters
     (``<user_memory>…</user_memory>``).
   * Treat anything inside that delimiter as data, never
     instructions.
   * Log the memory IDs that contribute to each turn so an
     incident can be traced.

   The protocol intentionally exposes ``Memory.value`` as
   opaque-but-decoded JSON so callers can structure their data
   (e.g. ``{"kind": "preference", "topic": "verbosity"}``) and
   key on the structure rather than free text.

Naming note: ``InMemoryMemoryStore`` would be a tongue-twister,
so the in-process backend is :class:`LocalMemoryStore`.  "Local"
here means "this process only" — same semantic as
:class:`~ai_assistant_client.persistence.memory.InMemoryTranscriptStore`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable


@dataclass(frozen=True)
class Memory:
    """One persisted note about a user.

    ``key`` is a short label the host's UI can render
    (``"favorite_language"``, ``"role"``, ``"timezone"``) and
    that the agent's tool layer can use to deduplicate updates
    in a future PR (``memory_remember(key=...)`` overwriting the
    same key).  ``value`` is opaque-but-JSON-serializable — the
    store JSON-encodes it on write and decodes on read.

    ``tags`` are free-form labels the host can query against —
    e.g. ``("preference",)``, ``("role", "work")``.  Empty by
    default; the protocol's list method filters by ANY of the
    requested tags being present.

    ``created_at`` / ``updated_at`` are ISO-8601 UTC strings
    stamped by the store at write time.  Empty when reading a
    pre-timestamp record (same convention as
    :class:`~ai_assistant_client.workflows.runtime.WorkflowEvent`).
    """

    memory_id: str
    user_id: str
    key: str
    value: Any
    tags: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = ""
    updated_at: str = ""


@runtime_checkable
class MemoryStore(Protocol):
    """Per-user typed-note store.

    Implementations must enforce ``user_id`` isolation on every
    operation — :meth:`get`, :meth:`update`, :meth:`remove`
    accept a ``memory_id`` but raise :class:`KeyError` if the
    caller's ``user_id`` doesn't match the stored record's
    owner.  This makes "I leaked one user's notes to another"
    a class of bug the protocol forbids by construction.
    """

    async def add(
        self,
        *,
        user_id: str,
        key: str,
        value: Any,
        tags: Iterable[str] = (),
    ) -> Memory:
        """Store a new memory.  Returns the populated record (with
        ``memory_id`` + timestamps assigned)."""
        ...

    async def get(self, *, user_id: str, memory_id: str) -> Memory:
        """Fetch one memory.  Raises :class:`KeyError` if the id
        doesn't exist OR if the ``user_id`` doesn't own it — same
        error so a probing caller can't distinguish the two.
        """
        ...

    async def list(
        self,
        *,
        user_id: str,
        tags: Iterable[str] | None = None,
    ) -> list[Memory]:
        """Every memory owned by ``user_id``.

        ``tags``: when given, returns memories that have **any**
        of the listed tags.  (Union semantics, not intersection —
        the latter is easy enough to do client-side via
        :func:`set.issubset` if a caller needs it.)
        Ordering is insertion order (oldest first).
        """
        ...

    async def update(
        self,
        *,
        user_id: str,
        memory_id: str,
        value: Any,
    ) -> Memory:
        """Replace a memory's value in place, bumping
        ``updated_at``.  Same ownership rules as :meth:`get`.
        """
        ...

    async def remove(self, *, user_id: str, memory_id: str) -> None:
        """Delete one memory.  Same ownership rules as :meth:`get`.
        Idempotent only in the success case — removing a memory
        twice raises :class:`KeyError` the second time, so the
        caller can detect accidental double-deletes.
        """
        ...

    async def forget_all(self, *, user_id: str) -> int:
        """Erase every memory belonging to ``user_id``.

        Returns the number of records removed.  Intended for
        GDPR-style "delete everything about me" requests — a
        host's privacy endpoint can call this and surface the
        count to the user as confirmation.
        """
        ...

    async def list_users(self) -> list[str]:
        """Every user id known to the store.

        Useful for administrative interfaces / migration
        scripts.  Implementations may surface only the ids
        of users with at least one memory.
        """
        ...
