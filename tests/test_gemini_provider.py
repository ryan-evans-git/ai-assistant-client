"""GeminiProvider streaming + translation tests with stubbed google-genai."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from ai_assistant_client.llm.base import (
    MessageStop,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    ToolUseStop,
)


def _install_fake_genai_types() -> None:
    """Install a minimal ``google.genai.types`` stub so the provider can
    construct ``Tool`` / ``FunctionDeclaration`` / ``GenerateContentConfig``
    without the real SDK.  Each of these is just a ``SimpleNamespace``-style
    constructor that records its kwargs."""

    def _make(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    types_mod = ModuleType("google.genai.types")
    types_mod.Tool = _make  # type: ignore[attr-defined]
    types_mod.FunctionDeclaration = _make  # type: ignore[attr-defined]
    types_mod.GenerateContentConfig = _make  # type: ignore[attr-defined]

    google_mod = sys.modules.get("google") or ModuleType("google")
    genai_mod = sys.modules.get("google.genai") or ModuleType("google.genai")
    genai_mod.types = types_mod  # type: ignore[attr-defined]
    google_mod.genai = genai_mod  # type: ignore[attr-defined]

    sys.modules["google"] = google_mod
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.types"] = types_mod


_install_fake_genai_types()


# Now import the provider — its top-of-file `from google.genai import
# types` happens inside stream_turn, but module-level imports use the
# stubs we just installed.
from ai_assistant_client.llm.gemini_provider import (  # noqa: E402
    GeminiProvider,
    _coerce_jsonable,
    _normalize_finish,
    _to_gemini_contents,
    _to_gemini_function,
    _to_jsonable,
)


def _ns(**fields: Any) -> SimpleNamespace:
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


class _FakeModels:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.last_kwargs: dict[str, Any] | None = None

    async def generate_content_stream(self, **kwargs: Any) -> _AsyncIter:
        self.last_kwargs = kwargs
        return _AsyncIter(self._chunks)


class _FakeGeminiClient:
    def __init__(self, chunks: list[Any]) -> None:
        self.aio = SimpleNamespace(models=_FakeModels(chunks))


@pytest.mark.asyncio
async def test_gemini_translates_text_and_function_call_stream() -> None:
    chunks = [
        _ns(
            candidates=[
                _ns(
                    content=_ns(parts=[_ns(text="hello", function_call=None)]),
                    finish_reason=None,
                )
            ]
        ),
        _ns(
            candidates=[
                _ns(
                    content=_ns(
                        parts=[
                            _ns(
                                text=None,
                                function_call=_ns(
                                    id=None, name="search", args={"q": "x"}
                                ),
                            )
                        ]
                    ),
                    finish_reason="STOP",
                )
            ]
        ),
    ]
    provider = GeminiProvider(client=_FakeGeminiClient(chunks))

    out = [
        e
        async for e in provider.stream_turn(
            model="gemini-2.0-flash",
            max_tokens=64,
            system="sys",
            tools=[{"name": "search", "input_schema": {"type": "object"}}],
            messages=[],
        )
    ]
    text = next(e for e in out if isinstance(e, TextDelta))
    assert text.text == "hello"
    start = next(e for e in out if isinstance(e, ToolUseStart))
    assert start.id.startswith("gemini-call-")
    assert start.name == "search"
    delta = next(e for e in out if isinstance(e, ToolInputDelta))
    assert json.loads(delta.partial_json) == {"q": "x"}
    assert any(isinstance(e, ToolUseStop) for e in out)
    stop = next(e for e in out if isinstance(e, MessageStop))
    # Had a function_call → stop_reason is "tool_use" regardless of finish.
    assert stop.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_gemini_max_tokens_finish_normalizes() -> None:
    chunks = [
        _ns(
            candidates=[
                _ns(
                    content=_ns(parts=[_ns(text="x", function_call=None)]),
                    finish_reason="MAX_TOKENS",
                )
            ]
        ),
    ]
    provider = GeminiProvider(client=_FakeGeminiClient(chunks))
    out = [
        e
        async for e in provider.stream_turn(
            model="m", max_tokens=1, system="", tools=[], messages=[]
        )
    ]
    stop = next(e for e in out if isinstance(e, MessageStop))
    assert stop.stop_reason == "max_tokens"


@pytest.mark.asyncio
async def test_gemini_no_candidates_yields_only_message_stop() -> None:
    chunks = [_ns(candidates=None)]
    provider = GeminiProvider(client=_FakeGeminiClient(chunks))
    out = [
        e
        async for e in provider.stream_turn(
            model="m", max_tokens=1, system="", tools=[], messages=[]
        )
    ]
    assert any(isinstance(e, MessageStop) for e in out)


def test_to_gemini_function_uses_supplied_schema() -> None:
    out = _to_gemini_function({
        "name": "lookup",
        "description": "Find one.",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    })
    assert out.name == "lookup"
    assert out.description == "Find one."
    assert out.parameters["properties"]["q"]["type"] == "string"


def test_to_gemini_function_default_schema() -> None:
    out = _to_gemini_function({"name": "noop"})
    assert out.parameters == {"type": "object", "properties": {}}


def test_to_gemini_contents_handles_strings_and_blocks() -> None:
    contents = _to_gemini_contents([
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking"},
                {"type": "tool_use", "name": "search", "input": {"q": "x"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "extra"},
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": "ok",
                },
            ],
        },
    ])
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "hi"
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][1]["function_call"]["name"] == "search"
    assert contents[2]["role"] == "user"
    fr = contents[2]["parts"][1]["function_response"]
    assert fr["name"] == "tu_1"
    assert fr["response"]["content"] == "ok"


def test_to_gemini_contents_handles_string_content_for_assistant() -> None:
    contents = _to_gemini_contents([
        {"role": "assistant", "content": "plain"},
    ])
    assert contents[0]["role"] == "model"
    assert contents[0]["parts"][0]["text"] == "plain"


def test_to_jsonable_passthrough_and_fallback() -> None:
    assert _to_jsonable("x") == "x"
    assert _to_jsonable({"a": 1}) == {"a": 1}
    assert _to_jsonable(None) is None

    class _Weird:
        def __str__(self) -> str:
            return "weird-repr"

    assert _to_jsonable(_Weird()) == "weird-repr"


def test_coerce_jsonable_walks_nested_structures() -> None:
    assert _coerce_jsonable({"a": [1, "x", {"b": True}]}) == {
        "a": [1, "x", {"b": True}]
    }
    assert _coerce_jsonable((1, 2)) == [1, 2]
    assert _coerce_jsonable(None) is None

    class _MapLike:
        def items(self) -> list[tuple[str, Any]]:
            return [("k", "v")]

    assert _coerce_jsonable(_MapLike()) == {"k": "v"}

    class _Other:
        def __str__(self) -> str:
            return "fallback"

    assert _coerce_jsonable(_Other()) == "fallback"


def test_normalize_finish_paths() -> None:
    assert _normalize_finish(None, had_function_call=True) == "tool_use"
    assert _normalize_finish(None, had_function_call=False) == "end_turn"
    assert _normalize_finish("MAX_TOKENS", had_function_call=False) == "max_tokens"
    assert _normalize_finish("STOP", had_function_call=False) == "end_turn"


def test_gemini_provider_constructs_client_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an injected client, the provider should call genai.Client()
    via the lazy import path."""
    instances: list[Any] = []

    class _FakeClient:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            instances.append(self)

    fake_genai = ModuleType("google.genai")
    fake_genai.Client = _FakeClient  # type: ignore[attr-defined]
    fake_google = ModuleType("google")
    fake_google.genai = fake_genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    GeminiProvider()
    assert len(instances) == 1


def test_gemini_provider_constructs_client_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When neither GEMINI_API_KEY nor GOOGLE_API_KEY are set, the
    provider passes no api_key kwarg to the SDK."""
    captured: list[dict[str, Any]] = []

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    fake_genai = ModuleType("google.genai")
    fake_genai.Client = _FakeClient  # type: ignore[attr-defined]
    fake_google = ModuleType("google")
    fake_google.genai = fake_genai  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    GeminiProvider()
    assert captured == [{}]
