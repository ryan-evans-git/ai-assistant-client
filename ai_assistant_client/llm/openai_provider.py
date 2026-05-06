"""OpenAI Chat Completions adapter.

Translates Anthropic-style block messages → OpenAI's role-based
schema (assistant ``tool_calls``, ``role="tool"`` results) and
maps the streaming ``choices[0].delta`` shape into normalized events.

OpenAI streams text in ``delta.content`` and tool calls in
``delta.tool_calls`` — each tool call carries an ``index`` that's
stable across deltas, an ``id`` and ``function.name`` (typically
on the first chunk for that index), and ``function.arguments`` that
streams as a string in subsequent deltas.
"""

from __future__ import annotations

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


class OpenAIProvider(LLMProvider):
    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as err:  # pragma: no cover
                raise RuntimeError(
                    "Install the 'openai' package: "
                    "pip install ai-assistant-client[openai]"
                ) from err
            client = AsyncOpenAI()
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
        oa_messages = [{"role": "system", "content": system}, *_to_openai_messages(messages)]
        oa_tools = [_to_openai_tool(t) for t in tools] if tools else None

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": oa_messages,
            "stream": True,
        }
        if oa_tools:
            kwargs["tools"] = oa_tools

        # Track which tool-call indices have been announced so we
        # can emit ToolUseStart exactly once per index, plus close
        # them all out at message end.
        seen_indices: set[int] = set()

        stream = await self._client.chat.completions.create(**kwargs)
        finish_reason: str | None = None
        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            delta = choice.delta

            text = getattr(delta, "content", None)
            if text:
                yield TextDelta(text=text)

            tool_calls = getattr(delta, "tool_calls", None) or []
            for tc in tool_calls:
                idx = getattr(tc, "index", 0) or 0
                func = getattr(tc, "function", None)
                tc_id = getattr(tc, "id", None)
                name = getattr(func, "name", None) if func else None
                args = getattr(func, "arguments", None) if func else None

                if idx not in seen_indices and (tc_id or name):
                    seen_indices.add(idx)
                    yield ToolUseStart(
                        index=idx,
                        id=tc_id or "",
                        name=name or "",
                    )
                if args:
                    yield ToolInputDelta(index=idx, partial_json=args)

            fr = getattr(choice, "finish_reason", None)
            if fr is not None:
                finish_reason = fr

        for idx in sorted(seen_indices):
            yield ToolUseStop(index=idx)
        yield MessageStop(stop_reason=_normalize_finish(finish_reason))


# ---------------------------------------------------------------------------
# Schema + message translation
# ---------------------------------------------------------------------------


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        },
    }


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            # User messages are either plain text or a list of
            # blocks that may contain tool_result entries.
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            text_parts: list[str] = []
            for block in content or []:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": _stringify(block.get("content", "")),
                        }
                    )
            if text_parts:
                out.append({"role": "user", "content": "\n".join(text_parts)})
        elif role == "assistant":
            text_parts = []
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content, str):
                out.append({"role": "assistant", "content": content})
                continue
            for block in content or []:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        }
                    )
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            assistant_msg["content"] = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            out.append(assistant_msg)
    return out


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _normalize_finish(reason: str | None) -> str:
    if reason == "tool_calls":
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    if reason == "stop":
        return "end_turn"
    return reason or "end_turn"
