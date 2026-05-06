"""Provider-neutral LLM streaming interface.

The agent loop talks to LLMs through a small normalized event stream
so we can swap Anthropic / OpenAI / Gemini / Bedrock without touching
the loop, the registry, or the SSE surface.

Internal message + tool-schema format stays in the Anthropic block
shape (it's the most expressive — tool_use blocks carry id+name+input,
tool_result blocks carry tool_use_id).  Each provider adapter is the
*only* code that knows how to translate to/from its native shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Union


@dataclass
class TextDelta:
    """Streaming text from the assistant."""

    text: str


@dataclass
class ToolUseStart:
    """A new tool-use block has begun.  ``index`` is stable for the
    duration of this assistant message and is used to correlate the
    subsequent ``ToolInputDelta`` / ``ToolUseStop`` events."""

    index: int
    id: str
    name: str


@dataclass
class ToolInputDelta:
    """Partial JSON for a tool's input arguments.

    Adapters that can't stream tool inputs (Gemini today) emit one
    delta containing the full serialized JSON in a single chunk.
    """

    index: int
    partial_json: str


@dataclass
class ToolUseStop:
    """The tool-use block at ``index`` is complete."""

    index: int


@dataclass
class MessageStop:
    """The assistant message is complete.  ``stop_reason`` is normalized
    to one of ``"end_turn"``, ``"tool_use"``, ``"max_tokens"``, or
    ``"stop"`` (provider-specific values map onto these)."""

    stop_reason: str


NormalizedEvent = Union[
    TextDelta,
    ToolUseStart,
    ToolInputDelta,
    ToolUseStop,
    MessageStop,
]


class LLMProvider(ABC):
    """A streaming LLM with tool-use, in normalized event form.

    Implementations accept Anthropic-style ``tools`` and ``messages``
    (see :mod:`ai_assistant_client.agent` for the shape) and yield
    :class:`NormalizedEvent` values.  Translation to/from the native
    provider format is the adapter's responsibility.
    """

    @abstractmethod
    def stream_turn(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        cache_hint: "CacheHint | None" = None,
    ) -> AsyncIterator[NormalizedEvent]:
        """Stream one assistant turn.  Returns an async iterator of
        normalized events.  The caller is responsible for re-invoking
        this after appending tool_result messages, until a
        ``MessageStop`` with ``stop_reason != "tool_use"`` is seen.

        ``cache_hint`` is an opt-in request to use the provider's
        prompt-caching feature.  Adapters that support it (Anthropic,
        Bedrock) translate the hint into native cache markers.
        Providers with automatic caching (OpenAI, Gemini 2.5+) ignore
        the hint and rely on the platform default.  Passing ``None``
        disables caching universally.
        """
        raise NotImplementedError


@dataclass
class CacheHint:
    """A request to mark cache breakpoints in the next ``stream_turn``.

    The hint is *advisory* — adapters honor it on a best-effort basis
    using the most cost-effective breakpoints the provider exposes.
    The intent is to cache the conversation prefix that's stable across
    turns (system prompt, tool catalog, completed messages) so each
    follow-up turn pays a fraction of the input cost for re-reading it.

    Attributes
    ----------
    cache_system:
        Cache the system prompt as its own breakpoint.
    cache_tools:
        Cache the serialized tool catalog as its own breakpoint.
    cache_history_through_index:
        Index into ``messages`` (inclusive) marking the last message
        that should be inside the cached prefix.  ``None`` means
        "don't cache the messages array."  ``-1`` is shorthand for
        "everything except the in-flight last message."

        The index is adapter-translated to whatever native marker the
        provider uses (Anthropic ``cache_control`` on the last block of
        that message, Bedrock ``cachePoint`` content block appended
        after it, etc.).
    """

    cache_system: bool = True
    cache_tools: bool = True
    cache_history_through_index: int | None = -1
