"""LLM provider abstraction.

Pick a provider with :func:`make_provider` (or set the ``LLM_PROVIDER``
env var to ``anthropic``, ``openai``, ``gemini``, or ``bedrock``).
The default model for each provider can be overridden with
``LLM_MODEL`` — see :func:`default_model`.
"""

from __future__ import annotations

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


_PROVIDERS = ("anthropic", "openai", "gemini", "bedrock")

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
    "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
}


def make_provider(name: str) -> LLMProvider:
    """Construct a provider by short name.  SDKs are imported lazily
    so users only need the dependency for the provider they pick."""
    key = (name or "").strip().lower()
    if key == "anthropic":
        from ai_assistant_client.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if key == "openai":
        from ai_assistant_client.llm.openai_provider import OpenAIProvider

        return OpenAIProvider()
    if key == "gemini":
        from ai_assistant_client.llm.gemini_provider import GeminiProvider

        return GeminiProvider()
    if key == "bedrock":
        from ai_assistant_client.llm.bedrock_provider import BedrockProvider

        return BedrockProvider()
    raise ValueError(
        f"Unknown LLM_PROVIDER {name!r}; expected one of {_PROVIDERS}"
    )


def default_model(provider_name: str) -> str:
    return _DEFAULT_MODELS.get((provider_name or "").strip().lower(), "")


__all__ = [
    "CacheHint",
    "LLMProvider",
    "MessageStop",
    "NormalizedEvent",
    "TextDelta",
    "ToolInputDelta",
    "ToolUseStart",
    "ToolUseStop",
    "make_provider",
    "default_model",
]
