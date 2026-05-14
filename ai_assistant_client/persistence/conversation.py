"""Append-only conversation message log.

A *conversation* is the running list of Anthropic-block-style
messages the agent loop builds up over multiple user turns:

    {"role": "user", "content": "..."}
    {"role": "assistant", "content": [{"type": "text", ...}, {"type": "tool_use", ...}]}
    {"role": "user", "content": [{"type": "tool_result", ...}]}
    ...

Persisting it lets a host resume a conversation across process
restarts, ship it to a different worker mid-session, or render a
historical view in the UI without re-asking the LLM.

This protocol is **deliberately separate** from
:class:`~ai_assistant_client.persistence.transcript.TranscriptStore`:

* Workflow transcripts are bracketed (header + events + footer)
  with a known terminal state — they wrap one run.
* Conversation logs are open-ended — there's no "end of
  conversation" event; turns keep arriving as the user chats.

A single unified store could be made to fit both shapes only by
weakening the contract enough that neither caller would actually
trust it.  Two protocols, parallel backends.

Messages are stored opaquely as ``dict[str, Any]`` — the agent
already produces Anthropic-shaped blocks (provider adapters
translate going out), so the store doesn't need a typed message
class.  Anything JSON-serializable round-trips through the file
backend; the in-memory backend keeps references as-is.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


# Conversation messages are Anthropic-style blocks — the canonical
# in-memory form already used by the agent loop.  Keeping the store
# unaware of provider-specific shapes means a future Gemini-native
# backend (or whatever) wouldn't require a wire-format change here.
Message = dict[str, Any]


@runtime_checkable
class ConversationStore(Protocol):
    """Append-only message log keyed by conversation id."""

    async def append(self, conversation_id: str, message: Message) -> None: ...

    async def read(self, conversation_id: str) -> list[Message]:
        """Return every message in order.

        Returns an empty list when the conversation id has never
        been written — this differs from
        :class:`TranscriptStore.read` (which raises ``KeyError``)
        because the "fresh conversation" case is common and
        expected, whereas reading an unknown run from a transcript
        store usually means a typo.
        """
        ...

    async def list_conversations(self) -> list[str]:
        """Every conversation id the store knows about.

        Ordering is backend-defined.  Callers needing
        chronological order should track timestamps externally
        (or in the message blocks themselves).
        """
        ...
