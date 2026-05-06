"""Anthropic adapter — the reference implementation.

The internal message + tool-schema format is already Anthropic-shaped,
so this adapter is mostly a passthrough that lifts the streaming
event translation out of the agent loop.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from ai_assistant_client.llm.base import (
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
    ) -> AsyncIterator[NormalizedEvent]:
        async with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
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
