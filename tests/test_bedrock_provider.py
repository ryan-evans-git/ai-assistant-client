"""BedrockProvider streaming + translation tests with a stubbed boto3 client."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import Any

import pytest

from ai_assistant_client.llm.base import (
    MessageStop,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    ToolUseStop,
)
from ai_assistant_client.llm.bedrock_provider import (
    BedrockProvider,
    _normalize_stop,
    _stringify,
    _to_bedrock_messages,
    _to_bedrock_tool,
)


class _FakeBedrockClient:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events
        self.last_kwargs: dict[str, Any] | None = None

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        self.last_kwargs = kwargs
        return {"stream": iter(self._events)}


@pytest.mark.asyncio
async def test_bedrock_translates_text_tool_and_stop_events() -> None:
    events: list[dict[str, Any]] = [
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "tu_1", "name": "search"}},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '{"q":"x"}'}},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {
            "contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"text": "ok"},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]
    provider = BedrockProvider(client=_FakeBedrockClient(events))
    out = [
        e
        async for e in provider.stream_turn(
            model="bedrock.x", max_tokens=64, system="sys", tools=[
                {"name": "search", "input_schema": {"type": "object"}}
            ], messages=[]
        )
    ]
    assert any(isinstance(e, ToolUseStart) and e.id == "tu_1" for e in out)
    assert any(
        isinstance(e, ToolInputDelta) and e.partial_json == '{"q":"x"}' for e in out
    )
    assert any(isinstance(e, ToolUseStop) and e.index == 0 for e in out)
    assert any(isinstance(e, TextDelta) and e.text == "ok" for e in out)
    assert any(
        isinstance(e, MessageStop) and e.stop_reason == "tool_use" for e in out
    )


@pytest.mark.asyncio
async def test_bedrock_includes_tool_config_when_tools_present() -> None:
    fake = _FakeBedrockClient([{"messageStop": {"stopReason": "end_turn"}}])
    provider = BedrockProvider(client=fake)
    [_ async for _ in provider.stream_turn(
        model="m",
        max_tokens=10,
        system="sys",
        tools=[{"name": "t", "input_schema": {"type": "object"}}],
        messages=[],
    )]
    assert "toolConfig" in (fake.last_kwargs or {})
    assert fake.last_kwargs["system"] == [{"text": "sys"}]


@pytest.mark.asyncio
async def test_bedrock_omits_system_when_empty() -> None:
    fake = _FakeBedrockClient([{"messageStop": {"stopReason": "end_turn"}}])
    provider = BedrockProvider(client=fake)
    [_ async for _ in provider.stream_turn(
        model="m", max_tokens=10, system="", tools=[], messages=[]
    )]
    assert fake.last_kwargs["system"] == []
    assert "toolConfig" not in fake.last_kwargs


@pytest.mark.asyncio
async def test_bedrock_passes_cache_hint_into_request() -> None:
    from ai_assistant_client.llm.base import CacheHint

    fake = _FakeBedrockClient([{"messageStop": {"stopReason": "end_turn"}}])
    provider = BedrockProvider(client=fake)
    [_ async for _ in provider.stream_turn(
        model="m",
        max_tokens=10,
        system="sys",
        tools=[{"name": "t", "input_schema": {"type": "object"}}],
        messages=[{"role": "user", "content": "hi"}],
        cache_hint=CacheHint(),
    )]
    assert fake.last_kwargs["system"][-1] == {"cachePoint": {"type": "default"}}


def test_to_bedrock_tool_uses_input_schema() -> None:
    out = _to_bedrock_tool({
        "name": "lookup",
        "description": "find",
        "input_schema": {"type": "object", "properties": {}},
    })
    assert out["toolSpec"]["name"] == "lookup"
    assert out["toolSpec"]["inputSchema"]["json"]["type"] == "object"


def test_to_bedrock_tool_falls_back_when_schema_missing() -> None:
    out = _to_bedrock_tool({"name": "noop"})
    assert out["toolSpec"]["inputSchema"]["json"] == {
        "type": "object",
        "properties": {},
    }


def test_to_bedrock_messages_handles_strings_and_block_lists() -> None:
    msgs = _to_bedrock_messages([
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking"},
                {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": 1}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": {"ok": 1}},
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_2",
                    "content": "broken",
                    "is_error": True,
                },
            ],
        },
    ])
    assert msgs[0]["content"] == [{"text": "hi"}]
    assistant_blocks = msgs[1]["content"]
    assert assistant_blocks[0] == {"text": "thinking"}
    assert assistant_blocks[1]["toolUse"]["toolUseId"] == "tu_1"

    user_blocks = msgs[2]["content"]
    assert user_blocks[0]["toolResult"]["toolUseId"] == "tu_1"
    # First tool result: success → no `status`.
    assert "status" not in user_blocks[0]["toolResult"]
    # Second: is_error=True → status: "error".
    assert user_blocks[1]["toolResult"]["status"] == "error"


def test_stringify_handles_str_dict_and_unencodable() -> None:
    assert _stringify("x") == "x"
    assert _stringify({"a": 1}) == '{"a": 1}'
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    assert isinstance(_stringify(cyclic), str)


def test_normalize_stop_paths() -> None:
    assert _normalize_stop("tool_use") == "tool_use"
    assert _normalize_stop("max_tokens") == "max_tokens"
    assert _normalize_stop("end_turn") == "end_turn"
    assert _normalize_stop("stop_sequence") == "end_turn"
    assert _normalize_stop("custom") == "custom"
    assert _normalize_stop("") == "end_turn"


def test_bedrock_provider_constructs_client_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[Any] = []
    fake_module = ModuleType("boto3")
    fake_module.client = lambda *_a, **_kw: instances.append(object()) or instances[-1]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_module)
    BedrockProvider()
    assert len(instances) == 1


def test_apply_cache_points_with_negative_index_out_of_range() -> None:
    from ai_assistant_client.llm.base import CacheHint
    from ai_assistant_client.llm.bedrock_provider import _apply_cache_points

    # Single message, index=-5 → target out of range → no change.
    msgs = [{"role": "user", "content": [{"text": "hi"}]}]
    _, _, out = _apply_cache_points(
        [], None, msgs, CacheHint(cache_history_through_index=-5)
    )
    assert out == msgs
