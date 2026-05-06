"""AWS Bedrock Converse adapter (boto3 ``bedrock-runtime``).

The Converse API normalizes message + tool semantics across all the
foundation models on Bedrock (Anthropic, Meta, Mistral, Cohere, etc).
Its event stream — ``messageStart``, ``contentBlockStart``,
``contentBlockDelta``, ``contentBlockStop``, ``messageStop`` — maps
almost 1:1 to our normalized events.

boto3's streaming response is a synchronous generator; we wrap it
in ``asyncio.to_thread`` calls so we don't block the event loop.
"""

from __future__ import annotations

import asyncio
import json
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


class BedrockProvider(LLMProvider):
    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                import boto3
            except ImportError as err:  # pragma: no cover
                raise RuntimeError(
                    "Install the 'boto3' package: "
                    "pip install ai-assistant-client[bedrock]"
                ) from err
            client = boto3.client("bedrock-runtime")
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
        bedrock_messages = _to_bedrock_messages(messages)
        bedrock_system = [{"text": system}] if system else []
        bedrock_tools = [_to_bedrock_tool(t) for t in tools] if tools else None

        if cache_hint is not None:
            bedrock_system, bedrock_tools, bedrock_messages = _apply_cache_points(
                bedrock_system, bedrock_tools, bedrock_messages, cache_hint
            )

        kwargs: dict[str, Any] = {
            "modelId": model,
            "messages": bedrock_messages,
            "system": bedrock_system,
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if bedrock_tools:
            kwargs["toolConfig"] = {"tools": bedrock_tools}

        response = await asyncio.to_thread(self._client.converse_stream, **kwargs)
        stream = response["stream"]

        # Bedrock's contentBlockStart for a tool_use carries the
        # toolUseId + name.  Subsequent contentBlockDelta events at
        # the same blockIndex carry the input JSON in chunks.
        tool_block_indices: set[int] = set()

        # boto3 EventStream is a sync generator; iterate it in a
        # thread so the event loop stays responsive.
        loop = asyncio.get_running_loop()
        sentinel = object()
        it = iter(stream)

        while True:
            event = await loop.run_in_executor(None, _next_or_sentinel, it, sentinel)
            if event is sentinel:
                break
            assert isinstance(event, dict)

            if "contentBlockStart" in event:
                start = event["contentBlockStart"]
                idx = int(start.get("contentBlockIndex", 0))
                start_data = start.get("start") or {}
                if "toolUse" in start_data:
                    tu = start_data["toolUse"]
                    tool_block_indices.add(idx)
                    yield ToolUseStart(
                        index=idx,
                        id=tu.get("toolUseId", "") or "",
                        name=tu.get("name", "") or "",
                    )

            elif "contentBlockDelta" in event:
                d = event["contentBlockDelta"]
                idx = int(d.get("contentBlockIndex", 0))
                delta = d.get("delta") or {}
                if "text" in delta:
                    yield TextDelta(text=delta["text"] or "")
                elif "toolUse" in delta:
                    yield ToolInputDelta(
                        index=idx,
                        partial_json=delta["toolUse"].get("input", "") or "",
                    )

            elif "contentBlockStop" in event:
                idx = int(event["contentBlockStop"].get("contentBlockIndex", 0))
                if idx in tool_block_indices:
                    yield ToolUseStop(index=idx)

            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason", "")
                yield MessageStop(stop_reason=_normalize_stop(stop_reason))


def _next_or_sentinel(it: Any, sentinel: object) -> Any:
    try:
        return next(it)
    except StopIteration:
        return sentinel


# ---------------------------------------------------------------------------
# Schema + message translation
# ---------------------------------------------------------------------------


def _to_bedrock_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "toolSpec": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "inputSchema": {
                "json": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
    }


def _to_bedrock_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        bedrock_blocks: list[dict[str, Any]] = []
        if isinstance(content, str):
            bedrock_blocks.append({"text": content})
        else:
            for block in content or []:
                btype = block.get("type")
                if btype == "text":
                    bedrock_blocks.append({"text": block.get("text", "")})
                elif btype == "tool_use":
                    bedrock_blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": block.get("id", ""),
                                "name": block.get("name", ""),
                                "input": block.get("input") or {},
                            }
                        }
                    )
                elif btype == "tool_result":
                    bedrock_blocks.append(
                        {
                            "toolResult": {
                                "toolUseId": block.get("tool_use_id", ""),
                                "content": [
                                    {"text": _stringify(block.get("content", ""))}
                                ],
                                **(
                                    {"status": "error"}
                                    if block.get("is_error")
                                    else {}
                                ),
                            }
                        }
                    )
        out.append({"role": role, "content": bedrock_blocks})
    return out


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _normalize_stop(reason: str) -> str:
    if reason == "tool_use":
        return "tool_use"
    if reason == "max_tokens":
        return "max_tokens"
    if reason in ("end_turn", "stop_sequence"):
        return "end_turn"
    return reason or "end_turn"


# ---------------------------------------------------------------------------
# Prompt caching translation
# ---------------------------------------------------------------------------

# Bedrock Converse uses inline ``cachePoint`` content blocks rather than
# block-level markers (the Anthropic style).  A ``cachePoint`` is a
# zero-content separator that means "everything before me in this array
# is part of one cached prefix."  We append them after the system,
# tool, and selected-message blocks the caller asked us to cache.
#
# Caveats: not every Bedrock model supports caching (Claude 3.5 Sonnet
# and friends do; older or non-Anthropic models will fail with a
# ValidationException).  Adapters can't tell which model the caller
# is using safely, so we honor the hint regardless and rely on the
# caller to disable caching when their target model doesn't support it.

_CACHE_POINT_BLOCK: dict[str, Any] = {"cachePoint": {"type": "default"}}


def _apply_cache_points(
    bedrock_system: list[dict[str, Any]],
    bedrock_tools: list[dict[str, Any]] | None,
    bedrock_messages: list[dict[str, Any]],
    hint: CacheHint,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Insert ``cachePoint`` blocks into the request payload."""
    out_system = bedrock_system
    if hint.cache_system and bedrock_system:
        out_system = [*bedrock_system, _CACHE_POINT_BLOCK]

    out_tools = bedrock_tools
    if hint.cache_tools and bedrock_tools:
        # Bedrock's ``toolConfig.tools`` accepts ``cachePoint`` entries
        # alongside ``toolSpec`` entries — appending one after the
        # tool list caches the whole catalog as a single prefix.
        out_tools = [*bedrock_tools, _CACHE_POINT_BLOCK]

    out_messages = bedrock_messages
    idx = hint.cache_history_through_index
    if idx is not None and bedrock_messages:
        target = idx if idx >= 0 else len(bedrock_messages) + idx
        if 0 <= target < len(bedrock_messages):
            out_messages = [dict(m) for m in bedrock_messages]
            msg = out_messages[target]
            content = list(msg.get("content") or [])
            content.append(_CACHE_POINT_BLOCK)
            out_messages[target] = {**msg, "content": content}

    return out_system, out_tools, out_messages
