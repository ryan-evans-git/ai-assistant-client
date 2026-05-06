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
    ) -> AsyncIterator[NormalizedEvent]:
        kwargs: dict[str, Any] = {
            "modelId": model,
            "messages": _to_bedrock_messages(messages),
            "system": [{"text": system}] if system else [],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if tools:
            kwargs["toolConfig"] = {
                "tools": [_to_bedrock_tool(t) for t in tools],
            }

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
