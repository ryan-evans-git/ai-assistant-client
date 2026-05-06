"""Google Gemini adapter (google-genai SDK).

Translates Anthropic-style blocks → Gemini ``contents`` (parts of
``text``, ``function_call``, or ``function_response``) and the
streaming response → normalized events.

**Quirk:** Gemini streams text deltas, but emits ``function_call``
parts as a single complete object — input arguments don't stream.
We synthesize ``ToolUseStart`` + a single ``ToolInputDelta`` carrying
the full serialized JSON + ``ToolUseStop`` so the agent loop sees
the same shape it gets from Anthropic / OpenAI.
"""

from __future__ import annotations

import json
import os
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


class GeminiProvider(LLMProvider):
    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                from google import genai
            except ImportError as err:  # pragma: no cover
                raise RuntimeError(
                    "Install the 'google-genai' package: "
                    "pip install ai-assistant-client[gemini]"
                ) from err
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
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
        from google.genai import types as genai_types  # type: ignore[import-not-found]

        contents = _to_gemini_contents(messages)
        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
        }
        if tools:
            config_kwargs["tools"] = [
                genai_types.Tool(
                    function_declarations=[_to_gemini_function(t) for t in tools]
                )
            ]
        config = genai_types.GenerateContentConfig(**config_kwargs)

        # Synthetic tool-use indices, since Gemini doesn't number them.
        next_index = 0
        synthetic_id_counter = 0
        finish_reason: str | None = None
        any_function_call = False

        stream = await self._client.aio.models.generate_content_stream(
            model=model, contents=contents, config=config
        )
        async for chunk in stream:
            for candidate in getattr(chunk, "candidates", None) or []:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", None) or []:
                    text = getattr(part, "text", None)
                    fc = getattr(part, "function_call", None)
                    if text:
                        yield TextDelta(text=text)
                    elif fc is not None:
                        any_function_call = True
                        # Gemini's function_call.id is optional;
                        # synthesize one when absent so the loop's
                        # tool_use_id correlation still works.
                        call_id = getattr(fc, "id", None) or (
                            f"gemini-call-{synthetic_id_counter}"
                        )
                        synthetic_id_counter += 1
                        name = getattr(fc, "name", "") or ""
                        args = getattr(fc, "args", None) or {}
                        idx = next_index
                        next_index += 1
                        yield ToolUseStart(index=idx, id=call_id, name=name)
                        yield ToolInputDelta(
                            index=idx,
                            partial_json=json.dumps(_coerce_jsonable(args)),
                        )
                        yield ToolUseStop(index=idx)
                fr = getattr(candidate, "finish_reason", None)
                if fr is not None:
                    finish_reason = str(fr)

        yield MessageStop(stop_reason=_normalize_finish(finish_reason, any_function_call))


# ---------------------------------------------------------------------------
# Schema + message translation
# ---------------------------------------------------------------------------


def _to_gemini_function(tool: dict[str, Any]) -> Any:
    from google.genai import types as genai_types  # type: ignore[import-not-found]

    return genai_types.FunctionDeclaration(
        name=tool["name"],
        description=tool.get("description", ""),
        parameters=tool.get("input_schema") or {"type": "object", "properties": {}},
    )


def _to_gemini_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Anthropic-style blocks → Gemini ``contents``.

    Gemini uses ``role: "user"`` for both human messages and tool
    results (carried as ``function_response`` parts), and
    ``role: "model"`` for assistant turns.
    """
    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            parts: list[dict[str, Any]] = []
            if isinstance(content, str):
                parts.append({"text": content})
            else:
                for block in content or []:
                    btype = block.get("type")
                    if btype == "text":
                        parts.append({"text": block.get("text", "")})
                    elif btype == "tool_result":
                        parts.append(
                            {
                                "function_response": {
                                    "name": block.get("tool_use_id", ""),
                                    "response": {
                                        "content": _to_jsonable(
                                            block.get("content", "")
                                        )
                                    },
                                }
                            }
                        )
            contents.append({"role": "user", "parts": parts})
        elif role == "assistant":
            parts = []
            if isinstance(content, str):
                parts.append({"text": content})
            else:
                for block in content or []:
                    btype = block.get("type")
                    if btype == "text":
                        parts.append({"text": block.get("text", "")})
                    elif btype == "tool_use":
                        parts.append(
                            {
                                "function_call": {
                                    "name": block.get("name", ""),
                                    "args": block.get("input") or {},
                                }
                            }
                        )
            contents.append({"role": "model", "parts": parts})
    return contents


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return str(value)


def _coerce_jsonable(value: Any) -> Any:
    """Best-effort coercion of a Gemini ``args`` value (which may be a
    proto-style mapping) into a plain dict."""
    if isinstance(value, dict):
        return {k: _coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "items"):
        return {k: _coerce_jsonable(v) for k, v in value.items()}
    return str(value)


def _normalize_finish(reason: str | None, had_function_call: bool) -> str:
    if had_function_call:
        return "tool_use"
    if not reason:
        return "end_turn"
    r = reason.upper() if isinstance(reason, str) else str(reason).upper()
    if "MAX_TOKENS" in r:
        return "max_tokens"
    return "end_turn"
