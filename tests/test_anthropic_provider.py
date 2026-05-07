"""AnthropicProvider streaming + native-event translation tests.

We inject a fake Anthropic SDK client into the provider so tests
don't require the ``anthropic`` package or any network calls.  The
fake reproduces the SDK's streaming context-manager + raw event
shape just closely enough to drive the adapter through every event
branch.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_assistant_client.llm.anthropic_provider import (
    AnthropicProvider,
    _normalize_stop,
)
from ai_assistant_client.llm.base import (
    MessageStop,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    ToolUseStop,
)


def _ev(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**fields)


class _FakeStream:
    """Async-context-manager + async-iterator over a scripted event list."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *_a: Any) -> None:
        return None

    def __aiter__(self) -> "_FakeStream":
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _FakeMessages:
    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.last_kwargs: dict[str, Any] | None = None

    def stream(self, **kwargs: Any) -> _FakeStream:
        self.last_kwargs = kwargs
        return _FakeStream(self._events)


class _FakeAnthropicClient:
    def __init__(self, events: list[Any]) -> None:
        self.messages = _FakeMessages(events)


@pytest.mark.asyncio
async def test_anthropic_translates_text_tool_and_stop_events() -> None:
    events = [
        _ev(type="content_block_start", index=0, content_block=_ev(type="text")),
        _ev(
            type="content_block_delta",
            index=0,
            delta=_ev(type="text_delta", text="hello"),
        ),
        _ev(type="content_block_stop", index=0),
        _ev(
            type="content_block_start",
            index=1,
            content_block=_ev(type="tool_use", id="tu_1", name="search"),
        ),
        _ev(
            type="content_block_delta",
            index=1,
            delta=_ev(type="input_json_delta", partial_json='{"q":"x"}'),
        ),
        _ev(type="content_block_stop", index=1),
        _ev(type="message_delta", delta=_ev(stop_reason="tool_use")),
        _ev(type="message_stop"),
    ]
    client = _FakeAnthropicClient(events)
    provider = AnthropicProvider(client=client)

    out: list[Any] = []
    async for ev in provider.stream_turn(
        model="claude-x", max_tokens=64, system="sys", tools=[], messages=[]
    ):
        out.append(ev)

    types = [type(e).__name__ for e in out]
    # text_delta + 2 tool stops + 1 tool start + tool_input_delta + message stop.
    assert "TextDelta" in types
    assert "ToolUseStart" in types
    assert "ToolInputDelta" in types
    assert any(isinstance(e, ToolUseStop) for e in out)
    assert any(isinstance(e, MessageStop) and e.stop_reason == "tool_use" for e in out)


@pytest.mark.asyncio
async def test_anthropic_handles_missing_attributes_gracefully() -> None:
    """Defensive: events missing optional attrs default to safe values."""
    events = [
        _ev(type="content_block_delta", index=0, delta=_ev(type="text_delta")),
        _ev(type="content_block_delta", index=0, delta=_ev(type="input_json_delta")),
        _ev(
            type="content_block_start",
            index=2,
            content_block=_ev(type="tool_use"),
        ),
        _ev(type="message_delta", delta=_ev(stop_reason="end_turn")),
    ]
    provider = AnthropicProvider(client=_FakeAnthropicClient(events))
    out = [e async for e in provider.stream_turn(
        model="m", max_tokens=10, system="", tools=[], messages=[]
    )]
    assert any(isinstance(e, TextDelta) and e.text == "" for e in out)
    assert any(
        isinstance(e, ToolInputDelta) and e.partial_json == "" for e in out
    )
    assert any(
        isinstance(e, ToolUseStart) and e.id == "" and e.name == "" for e in out
    )


@pytest.mark.asyncio
async def test_anthropic_passes_cache_markers_to_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``cache_hint`` is provided, the request payload should
    include ``cache_control`` markers."""
    from ai_assistant_client.llm.base import CacheHint

    events = [_ev(type="message_delta", delta=_ev(stop_reason="end_turn"))]
    client = _FakeAnthropicClient(events)
    provider = AnthropicProvider(client=client)

    [_ async for _ in provider.stream_turn(
        model="m",
        max_tokens=10,
        system="sys",
        tools=[{"name": "t", "description": "", "input_schema": {}}],
        messages=[{"role": "user", "content": "hi"}],
        cache_hint=CacheHint(),
    )]
    sent = client.messages.last_kwargs
    assert sent is not None
    # System promoted to typed-block array with cache_control marker.
    assert isinstance(sent["system"], list)
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_normalize_stop_passthrough() -> None:
    assert _normalize_stop("end_turn") == "end_turn"
    assert _normalize_stop("tool_use") == "tool_use"
    assert _normalize_stop("max_tokens") == "max_tokens"
    assert _normalize_stop("stop_sequence") == "stop_sequence"
    # Unknown → returned as-is (caller decides).
    assert _normalize_stop("custom") == "custom"


def test_anthropic_apply_cache_markers_no_messages() -> None:
    """``messages=[]`` should leave cache_history_through_index a no-op."""
    from ai_assistant_client.llm.anthropic_provider import _apply_cache_markers
    from ai_assistant_client.llm.base import CacheHint

    sys_, tools, msgs = _apply_cache_markers("", [], [], CacheHint())
    assert msgs == []


def test_anthropic_apply_cache_markers_skips_empty_message_content() -> None:
    """A message with no content shouldn't crash the marker code."""
    from ai_assistant_client.llm.anthropic_provider import _apply_cache_markers
    from ai_assistant_client.llm.base import CacheHint

    msgs = [{"role": "user", "content": []}]
    _, _, out_msgs = _apply_cache_markers(
        "", [], msgs, CacheHint(cache_history_through_index=0)
    )
    # Nothing to mark — message returns unchanged.
    assert out_msgs[0]["content"] == []


def test_anthropic_provider_constructs_client_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``client`` isn't injected, the provider lazy-imports the SDK."""
    instances: list[Any] = []

    class _FakeAsync:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            instances.append(self)

    fake_module = type("fake", (), {"AsyncAnthropic": _FakeAsync})
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_module)
    AnthropicProvider()
    assert len(instances) == 1
