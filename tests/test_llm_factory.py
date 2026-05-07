"""Tests for ``ai_assistant_client.llm`` package factory + defaults."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_assistant_client.llm import default_model, make_provider


def test_default_model_for_each_provider() -> None:
    assert default_model("anthropic").startswith("claude-")
    assert default_model("openai").startswith("gpt-")
    assert default_model("gemini").startswith("gemini-")
    assert default_model("bedrock").startswith("anthropic.")


def test_default_model_is_case_insensitive_and_strips() -> None:
    assert default_model("  ANTHROPIC ").startswith("claude-")


def test_default_model_unknown_returns_empty_string() -> None:
    assert default_model("nope") == ""
    assert default_model("") == ""


def test_make_provider_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        make_provider("nope")


def test_make_provider_anthropic() -> None:
    """make_provider imports the SDK module — patch it so we don't
    require a real anthropic install in test env."""
    fake_module = type(
        "fake",
        (),
        {"AsyncAnthropic": lambda *a, **kw: object()},
    )
    with patch.dict("sys.modules", {"anthropic": fake_module}):
        from ai_assistant_client.llm.anthropic_provider import AnthropicProvider

        prov = make_provider("anthropic")
        assert isinstance(prov, AnthropicProvider)


def test_make_provider_openai() -> None:
    fake_module = type(
        "fake",
        (),
        {"AsyncOpenAI": lambda *a, **kw: object()},
    )
    with patch.dict("sys.modules", {"openai": fake_module}):
        from ai_assistant_client.llm.openai_provider import OpenAIProvider

        prov = make_provider("openai")
        assert isinstance(prov, OpenAIProvider)


def test_make_provider_gemini() -> None:
    class _FakeClient:
        def __init__(self, *_a, **_kw) -> None:
            pass

    fake_module = type(
        "fake",
        (),
        {"genai": type("g", (), {"Client": _FakeClient})},
    )
    with patch.dict(
        "sys.modules", {"google.genai": fake_module.genai, "google": fake_module}
    ):
        from ai_assistant_client.llm.gemini_provider import GeminiProvider

        prov = make_provider("gemini")
        assert isinstance(prov, GeminiProvider)


def test_make_provider_bedrock() -> None:
    fake_module = type(
        "fake",
        (),
        {"client": lambda *_a, **_kw: object()},
    )
    with patch.dict("sys.modules", {"boto3": fake_module}):
        from ai_assistant_client.llm.bedrock_provider import BedrockProvider

        prov = make_provider("bedrock")
        assert isinstance(prov, BedrockProvider)
