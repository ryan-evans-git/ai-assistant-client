"""Anthropic adapter — the reference implementation.

The internal message + tool-schema format is already Anthropic-shaped,
so this adapter is mostly a passthrough that lifts the streaming
event translation out of the agent loop.
"""

from __future__ import annotations

import copy
from typing import Any, AsyncIterator

from ai_assistant_client.llm.base import (
    CacheHint,
    LLMProvider,
    MessageStop,
    NormalizedEvent,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    ToolUseStop,
)


class AnthropicProvider(LLMProvider):
    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as err:  # pragma: no cover
                raise RuntimeError(
                    "Install the 'anthropic' package: "
                    "pip install ai-assistant-client[anthropic]"
                ) from err
            client = AsyncAnthropic()
        self._client = client

    async def stream_turn(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        cache_hint: CacheHint | None = None,
    ) -> AsyncIterator[NormalizedEvent]:
        request_system, request_tools, request_messages = _apply_cache_markers(
            system, tools, messages, cache_hint
        )
        async with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=request_system,
            tools=request_tools,
            messages=request_messages,
        ) as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if getattr(block, "type", None) == "tool_use":
                        yield ToolUseStart(
                            index=getattr(event, "index", 0),
                            id=getattr(block, "id", "") or "",
                            name=getattr(block, "name", "") or "",
                        )

                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None)
                    index = getattr(event, "index", 0)
                    if delta_type == "text_delta":
                        yield TextDelta(text=getattr(delta, "text", "") or "")
                    elif delta_type == "input_json_delta":
                        yield ToolInputDelta(
                            index=index,
                            partial_json=getattr(delta, "partial_json", "") or "",
                        )

                elif event_type == "content_block_stop":
                    yield ToolUseStop(index=getattr(event, "index", 0))

                elif event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    sr = getattr(delta, "stop_reason", None)
                    if sr is not None:
                        yield MessageStop(stop_reason=_normalize_stop(sr))

                elif event_type == "message_stop":
                    # Anthropic emits stop reason on message_delta
                    # already; this is a safety net for SDKs that
                    # only carry it here.
                    pass


def _normalize_stop(reason: str) -> str:
    # Anthropic already uses our canonical names.
    if reason in ("end_turn", "tool_use", "max_tokens", "stop_sequence"):
        return reason
    return reason


# ---------------------------------------------------------------------------
# Prompt caching translation
# ---------------------------------------------------------------------------

# Anthropic allows up to 4 cache_control breakpoints per request, so the
# default hint (system + tools + last completed message) fits comfortably.
# Each marker means "everything from the start of the prompt up to and
# including this block is cached as one prefix." The TTL is 5 minutes by
# default; set ``cache_control.ttl="1h"`` to extend it (more expensive
# write, cheaper reads — only worth it for long-lived sessions).

_EPHEMERAL_MARKER: dict[str, Any] = {"cache_control": {"type": "ephemeral"}}


def _apply_cache_markers(
    system: str,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    hint: CacheHint | None,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (system, tools, messages) with cache_control markers applied.

    The inputs are *not* mutated — this builds shallow-copied request
    objects so the caller's history list (used across many turns) stays
    in its provider-neutral shape.
    """
    if hint is None:
        return system, tools, messages

    out_system: Any = system
    if hint.cache_system and system:
        # Anthropic accepts ``system`` either as a plain string or as
        # an array of typed blocks — caching requires the array form.
        out_system = [
            {"type": "text", "text": system, **_EPHEMERAL_MARKER},
        ]

    out_tools = tools
    if hint.cache_tools and tools:
        # Marking the *last* tool definition caches the entire tools
        # array as one breakpoint (Anthropic caches everything up to
        # and including the marker).
        out_tools = [dict(t) for t in tools]
        out_tools[-1] = {**out_tools[-1], **_EPHEMERAL_MARKER}

    out_messages = messages
    idx = hint.cache_history_through_index
    if idx is not None and messages:
        target = idx if idx >= 0 else len(messages) + idx
        if 0 <= target < len(messages):
            out_messages = [dict(m) for m in messages]
            out_messages[target] = _mark_message_cached(out_messages[target])

    return out_system, out_tools, out_messages


def _mark_message_cached(message: dict[str, Any]) -> dict[str, Any]:
    """Add ``cache_control`` to the last content block in ``message``.

    Anthropic caches up to and including the marked block, so marking
    the *last* block of a message effectively caches the full message
    and everything before it.
    """
    content = message.get("content")
    if isinstance(content, str):
        # Promote string content to a single text block so we can
        # attach the marker.
        return {
            **message,
            "content": [{"type": "text", "text": content, **_EPHEMERAL_MARKER}],
        }
    if isinstance(content, list) and content:
        new_content = [copy.copy(b) if isinstance(b, dict) else b for b in content]
        last = new_content[-1]
        if isinstance(last, dict):
            new_content[-1] = {**last, **_EPHEMERAL_MARKER}
            return {**message, "content": new_content}
    # Empty content: nothing to mark; leave the message as-is.
    return message
