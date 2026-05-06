"""Auditor LLM caller — exercised against a stubbed LLMProvider so
no real model traffic is required."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest

from ai_assistant_client.llm import (
    CacheHint,
    LLMProvider,
    MessageStop,
    NormalizedEvent,
    TextDelta,
)
from ai_assistant_client.validation.auditor import (
    DEFAULT_AUDITOR_MODELS,
    run_auditor,
)


class _FixedResponseProvider(LLMProvider):
    def __init__(self, output: str) -> None:
        self._output = output
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {"model": model, "system": system, "messages": messages}
        )
        yield TextDelta(text=self._output)
        yield MessageStop(stop_reason="end_turn")


@pytest.mark.asyncio
async def test_parses_well_formed_auditor_output() -> None:
    payload = json.dumps(
        {
            "issues": [
                {
                    "claim": "$1,500.00",
                    "severity": "error",
                    "reason": "tool result shows $1,250.00",
                }
            ]
        }
    )
    provider = _FixedResponseProvider(payload)
    issues = await run_auditor(
        provider=provider,
        model="claude-haiku-4-5-20251001",
        response_text="The total is $1,500.00.",
        tool_results_by_id={"tu_1": {"amount": 1250.00}},
    )
    assert len(issues) == 1
    assert issues[0].kind == "auditor_finding"
    assert issues[0].severity == "error"
    assert issues[0].claim == "$1,500.00"
    assert issues[0].source == "auditor"


@pytest.mark.asyncio
async def test_returns_empty_list_when_no_issues() -> None:
    provider = _FixedResponseProvider('{"issues": []}')
    issues = await run_auditor(
        provider=provider,
        model="x",
        response_text="OK.",
        tool_results_by_id={},
    )
    assert issues == []


@pytest.mark.asyncio
async def test_tolerates_prose_around_json() -> None:
    payload = (
        "Sure, here's the audit:\n"
        '{"issues": [{"claim": "Acme", "severity": "warning", "reason": "fuzzy"}]}\n'
        "Hope that helps."
    )
    provider = _FixedResponseProvider(payload)
    issues = await run_auditor(
        provider=provider,
        model="x",
        response_text="x",
        tool_results_by_id={},
    )
    assert len(issues) == 1
    assert issues[0].severity == "warning"


@pytest.mark.asyncio
async def test_invalid_json_returns_empty() -> None:
    provider = _FixedResponseProvider("not json at all")
    issues = await run_auditor(
        provider=provider,
        model="x",
        response_text="x",
        tool_results_by_id={},
    )
    assert issues == []


@pytest.mark.asyncio
async def test_unknown_severity_falls_back_to_warning() -> None:
    payload = json.dumps(
        {"issues": [{"claim": "x", "severity": "catastrophic", "reason": "y"}]}
    )
    provider = _FixedResponseProvider(payload)
    issues = await run_auditor(
        provider=provider, model="x", response_text="x", tool_results_by_id={}
    )
    assert issues[0].severity == "warning"


def test_default_auditor_models_cover_all_providers() -> None:
    assert set(DEFAULT_AUDITOR_MODELS) == {"anthropic", "openai", "gemini", "bedrock"}
