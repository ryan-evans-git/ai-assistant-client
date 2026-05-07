"""OpenAIProvider streaming + translation tests with a stubbed SDK client."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from ai_assistant_client.llm.base import (
    MessageStop,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    ToolUseStop,
)
from ai_assistant_client.llm.openai_provider import (
    OpenAIProvider,
    _normalize_finish,
    _stringify,
    _to_openai_messages,
    _to_openai_tool,
)


def _delta(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**fields)


def _chunk(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**fields)


class _AsyncIter:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __aiter__(self) -> "_AsyncIter":
        self._iter = iter(self._items)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _FakeCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _AsyncIter:
        self.last_kwargs = kwargs
        return _AsyncIter(self._chunks)


class _FakeOpenAI:
    def __init__(self, chunks: list[Any]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(chunks))


@pytest.mark.asyncio
async def test_openai_translates_text_then_tool_call_stream() -> None:
    """Walk a chat-completions stream that emits text first, then a
    multi-chunk tool call, then a finish_reason."""
    chunks = [
        _chunk(choices=[_chunk(delta=_delta(content="hi "), finish_reason=None)]),
        _chunk(choices=[_chunk(delta=_delta(content="there"), finish_reason=None)]),
        _chunk(
            choices=[
                _chunk(
                    delta=_delta(
                        tool_calls=[
                            _delta(
                                index=0,
                                id="call_1",
                                function=_delta(name="search", arguments=""),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ]
        ),
        _chunk(
            choices=[
                _chunk(
                    delta=_delta(
                        tool_calls=[
                            _delta(
                                index=0,
                                id=None,
                                function=_delta(name=None, arguments='{"q":'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ]
        ),
        _chunk(
            choices=[
                _chunk(
                    delta=_delta(
                        tool_calls=[
                            _delta(
                                index=0,
                                id=None,
                                function=_delta(name=None, arguments='"x"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ]
        ),
    ]
    provider = OpenAIProvider(client=_FakeOpenAI(chunks))
    out = [
        e
        async for e in provider.stream_turn(
            model="gpt-4o", max_tokens=64, system="sys", tools=[
                {"name": "search", "description": "", "input_schema": {}}
            ], messages=[]
        )
    ]
    assert any(isinstance(e, TextDelta) and e.text == "hi " for e in out)
    starts = [e for e in out if isinstance(e, ToolUseStart)]
    assert len(starts) == 1
    assert starts[0].id == "call_1"
    assert starts[0].name == "search"
    deltas = [e for e in out if isinstance(e, ToolInputDelta)]
    assert "".join(d.partial_json for d in deltas) == '{"q":"x"}'
    assert any(isinstance(e, ToolUseStop) and e.index == 0 for e in out)
    stop = next(e for e in out if isinstance(e, MessageStop))
    assert stop.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_openai_handles_chunks_with_no_choices() -> None:
    """Some keep-alive chunks come back with ``choices=[]`` — skip them."""
    chunks = [
        _chunk(choices=[]),
        _chunk(choices=[_chunk(delta=_delta(content="ok"), finish_reason="stop")]),
    ]
    provider = OpenAIProvider(client=_FakeOpenAI(chunks))
    out = [
        e
        async for e in provider.stream_turn(
            model="m", max_tokens=10, system="", tools=[], messages=[]
        )
    ]
    assert any(isinstance(e, TextDelta) and e.text == "ok" for e in out)
    assert any(isinstance(e, MessageStop) and e.stop_reason == "end_turn" for e in out)


@pytest.mark.asyncio
async def test_openai_does_not_set_tools_when_empty() -> None:
    """``tools`` empty → no ``tools`` key in request kwargs."""
    chunks = [_chunk(choices=[_chunk(delta=_delta(), finish_reason="stop")])]
    fake = _FakeOpenAI(chunks)
    provider = OpenAIProvider(client=fake)
    [_ async for _ in provider.stream_turn(
        model="m", max_tokens=1, system="x", tools=[], messages=[]
    )]
    assert "tools" not in (fake.chat.completions.last_kwargs or {})


def test_to_openai_tool_uses_input_schema() -> None:
    out = _to_openai_tool({
        "name": "lookup",
        "description": "Find a thing.",
        "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}},
    })
    assert out["type"] == "function"
    assert out["function"]["name"] == "lookup"
    assert out["function"]["description"] == "Find a thing."
    assert out["function"]["parameters"]["properties"]["id"]["type"] == "string"


def test_to_openai_tool_falls_back_to_empty_schema() -> None:
    out = _to_openai_tool({"name": "noop"})
    assert out["function"]["parameters"] == {"type": "object", "properties": {}}


def test_to_openai_messages_string_user_and_assistant_passthrough() -> None:
    msgs = _to_openai_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_to_openai_messages_assistant_blocks_with_tool_use() -> None:
    msgs = _to_openai_messages([
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "x"}},
            ],
        },
    ])
    assert len(msgs) == 1
    assert msgs[0]["content"] == "calling"
    assert msgs[0]["tool_calls"][0]["id"] == "tu_1"
    args = json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"])
    assert args == {"q": "x"}


def test_to_openai_messages_user_blocks_with_tool_result() -> None:
    msgs = _to_openai_messages([
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": {"ok": True}},
                {"type": "text", "text": "follow up"},
            ],
        },
    ])
    # Tool-result becomes its own role="tool" message; subsequent text
    # becomes the user message.
    roles = [m["role"] for m in msgs]
    assert roles == ["tool", "user"]
    assert msgs[0]["tool_call_id"] == "tu_1"
    assert json.loads(msgs[0]["content"]) == {"ok": True}


def test_stringify_handles_str_dict_and_unencodable() -> None:
    assert _stringify("hello") == "hello"
    assert _stringify({"a": 1}) == '{"a": 1}'
    # Circular reference forces the fallback branch.
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    out = _stringify(cyclic)
    assert isinstance(out, str)


def test_normalize_finish_maps_known_reasons() -> None:
    assert _normalize_finish("tool_calls") == "tool_use"
    assert _normalize_finish("length") == "max_tokens"
    assert _normalize_finish("stop") == "end_turn"
    assert _normalize_finish(None) == "end_turn"
    assert _normalize_finish("custom") == "custom"
