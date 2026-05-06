"""End-to-end agent-loop tests for the validation pipeline.

Uses a stubbed LLMProvider (same shape as test_agent.py) plus a
stubbed auditor provider so no real LLM traffic is required.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest

from ai_assistant_client.agent import AgentRunConfig, run_agent
from ai_assistant_client.discovery import (
    ProgressiveToolRegistry,
    RemoteToolDescriptor,
)
from ai_assistant_client.llm import (
    CacheHint,
    LLMProvider,
    MessageStop,
    NormalizedEvent,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    ToolUseStop,
)


class _StubProvider(LLMProvider):
    """Replays scripted normalized-event lists, one per stream_turn."""

    def __init__(self, scripted: list[list[NormalizedEvent]]) -> None:
        self._scripted = list(scripted)
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
            {
                "model": model,
                "system": system,
                "messages": [m for m in messages],
                "cache_hint": cache_hint,
            }
        )
        events = (
            self._scripted.pop(0)
            if self._scripted
            else [MessageStop(stop_reason="end_turn")]
        )
        for ev in events:
            yield ev


class _FixedAuditor(LLMProvider):
    """An auditor that always returns the same JSON payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

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
        self.calls += 1
        yield TextDelta(text=json.dumps(self.payload))
        yield MessageStop(stop_reason="end_turn")


def _text_turn(text: str) -> list[NormalizedEvent]:
    return [TextDelta(text=text), MessageStop(stop_reason="end_turn")]


def _tool_use_turn(
    *, tool_id: str, tool_name: str, tool_input_json: str
) -> list[NormalizedEvent]:
    return [
        ToolUseStart(index=0, id=tool_id, name=tool_name),
        ToolInputDelta(index=0, partial_json=tool_input_json),
        ToolUseStop(index=0),
        MessageStop(stop_reason="tool_use"),
    ]


def _registry_with(name: str) -> ProgressiveToolRegistry:
    registry = ProgressiveToolRegistry(
        [
            RemoteToolDescriptor(
                name=name,
                description="x",
                input_schema={"type": "object", "properties": {}},
            )
        ]
    )
    registry.handle_meta_call("tool_load", {"names": [name]})
    return registry


# ---------------------------------------------------------------------------
# Validation off — legacy behavior unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validation_off_yields_no_validation_event() -> None:
    provider = _StubProvider([_text_turn("hi")])
    events = []
    async for ev in run_agent(
        user_message="x",
        history=[],
        registry=ProgressiveToolRegistry([]),
        dispatcher=lambda *_: None,  # type: ignore[arg-type]
        provider=provider,
        config=AgentRunConfig(),  # validation_mode default = "off"
    ):
        events.append(ev)
    assert not any(e.type == "validation" for e in events)
    assert events[-1].type == "turn_complete"


# ---------------------------------------------------------------------------
# Citation mode: deterministic only, no auditor.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_citation_mode_passes_when_value_matches() -> None:
    registry = _registry_with("get_invoice")
    provider = _StubProvider(
        [
            _tool_use_turn(
                tool_id="tu_1",
                tool_name="get_invoice",
                tool_input_json='{"id": 4711}',
            ),
            _text_turn(
                'You owe <cite tu="tu_1" path="$.amount">$1,250.00</cite>.'
            ),
        ]
    )

    async def dispatcher(name: str, args: dict[str, Any]) -> Any:
        return {"amount": 1250.00}

    events = []
    async for ev in run_agent(
        user_message="how much?",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(validation_mode="citation"),
    ):
        events.append(ev)

    [val_event] = [e for e in events if e.type == "validation"]
    assert val_event.data["passed"] is True
    assert val_event.data["citations_total"] == 1
    assert val_event.data["citations_verified"] == 1
    assert val_event.data["auditor_used"] is False


@pytest.mark.asyncio
async def test_citation_mode_fails_on_value_mismatch_without_retry() -> None:
    registry = _registry_with("get_invoice")
    provider = _StubProvider(
        [
            _tool_use_turn(
                tool_id="tu_1",
                tool_name="get_invoice",
                tool_input_json='{"id": 4711}',
            ),
            _text_turn(
                'You owe <cite tu="tu_1" path="$.amount">$9,999.00</cite>.'
            ),
        ]
    )

    async def dispatcher(name: str, args: dict[str, Any]) -> Any:
        return {"amount": 1250.00}

    events = []
    async for ev in run_agent(
        user_message="x",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(
            validation_mode="citation", max_validation_retries=0
        ),
    ):
        events.append(ev)

    val_events = [e for e in events if e.type == "validation"]
    assert len(val_events) == 1
    assert val_events[0].data["passed"] is False
    assert any(
        i["kind"] == "value_mismatch"
        for i in val_events[0].data["issues"]
    )
    # No retry allowed → only one validation event, then turn_complete.
    assert events[-1].type == "turn_complete"


# ---------------------------------------------------------------------------
# Retry loop.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_appends_feedback_and_re_runs_until_pass() -> None:
    registry = _registry_with("get_invoice")
    # First text turn cites a wrong value; second turn after the
    # validation feedback cites the right value.
    provider = _StubProvider(
        [
            _tool_use_turn(
                tool_id="tu_1",
                tool_name="get_invoice",
                tool_input_json='{"id": 4711}',
            ),
            _text_turn('Wrong: <cite tu="tu_1" path="$.amount">$9,999.00</cite>'),
            _text_turn('Right: <cite tu="tu_1" path="$.amount">$1,250.00</cite>'),
        ]
    )

    async def dispatcher(name: str, args: dict[str, Any]) -> Any:
        return {"amount": 1250.00}

    events: list[Any] = []
    history: list[dict[str, Any]] = []
    async for ev in run_agent(
        user_message="x",
        history=history,
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(
            validation_mode="citation", max_validation_retries=2
        ),
    ):
        events.append(ev)

    val_events = [e for e in events if e.type == "validation"]
    retry_events = [e for e in events if e.type == "validation_retry"]
    assert len(val_events) == 2  # one fail, one pass
    assert val_events[0].data["passed"] is False
    assert val_events[1].data["passed"] is True
    assert len(retry_events) == 1
    assert retry_events[0].data["retries_remaining"] == 1

    # The synthetic feedback user-message landed in history before the
    # retry attempt.
    assert any(
        m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and m["content"].startswith("[validation]")
        for m in history
    )


@pytest.mark.asyncio
async def test_retry_budget_exhausted_yields_failed_validation() -> None:
    registry = _registry_with("get_invoice")
    # Every assistant turn cites the wrong value.
    bad = lambda: _text_turn(  # noqa: E731
        'Wrong: <cite tu="tu_1" path="$.amount">$9,999.00</cite>'
    )
    provider = _StubProvider(
        [
            _tool_use_turn(
                tool_id="tu_1",
                tool_name="get_invoice",
                tool_input_json='{"id": 4711}',
            ),
            bad(), bad(), bad(), bad(),
        ]
    )

    async def dispatcher(name: str, args: dict[str, Any]) -> Any:
        return {"amount": 1250.00}

    events = []
    async for ev in run_agent(
        user_message="x",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(
            validation_mode="citation", max_validation_retries=2
        ),
    ):
        events.append(ev)

    val_events = [e for e in events if e.type == "validation"]
    retry_events = [e for e in events if e.type == "validation_retry"]
    # 1 initial + 2 retries = 3 validation runs.
    assert len(val_events) == 3
    assert all(e.data["passed"] is False for e in val_events)
    assert len(retry_events) == 2
    assert events[-1].type == "turn_complete"


# ---------------------------------------------------------------------------
# Hybrid mode + auditor wiring.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_invokes_auditor_only_on_citation_failure() -> None:
    registry = _registry_with("get_invoice")
    provider = _StubProvider(
        [
            _tool_use_turn(
                tool_id="tu_1",
                tool_name="get_invoice",
                tool_input_json='{"id": 4711}',
            ),
            _text_turn('Wrong: <cite tu="tu_1" path="$.amount">$9,999</cite>'),
        ]
    )
    auditor = _FixedAuditor(
        {"issues": [{"claim": "$9,999", "severity": "error", "reason": "fabricated"}]}
    )

    async def dispatcher(name: str, args: dict[str, Any]) -> Any:
        return {"amount": 1250.00}

    events = []
    async for ev in run_agent(
        user_message="x",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(
            validation_mode="hybrid", max_validation_retries=0
        ),
        provider_name="anthropic",
        auditor_provider=auditor,
    ):
        events.append(ev)

    [val] = [e for e in events if e.type == "validation"]
    assert val.data["auditor_used"] is True
    assert auditor.calls == 1
    sources = {i["source"] for i in val.data["issues"]}
    assert "citation" in sources and "auditor" in sources


@pytest.mark.asyncio
async def test_hybrid_skips_auditor_when_citations_pass() -> None:
    registry = _registry_with("get_invoice")
    provider = _StubProvider(
        [
            _tool_use_turn(
                tool_id="tu_1",
                tool_name="get_invoice",
                tool_input_json='{"id": 4711}',
            ),
            _text_turn('OK: <cite tu="tu_1" path="$.amount">$1,250.00</cite>'),
        ]
    )
    auditor = _FixedAuditor({"issues": []})

    async def dispatcher(name: str, args: dict[str, Any]) -> Any:
        return {"amount": 1250.00}

    async for _ in run_agent(
        user_message="x",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(validation_mode="hybrid"),
        provider_name="anthropic",
        auditor_provider=auditor,
    ):
        pass

    assert auditor.calls == 0


# ---------------------------------------------------------------------------
# System prompt augmentation.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_prompt_gets_citation_instructions_when_validation_on() -> None:
    provider = _StubProvider([_text_turn("ok")])
    async for _ in run_agent(
        user_message="x",
        history=[],
        registry=ProgressiveToolRegistry([]),
        dispatcher=lambda *_: None,  # type: ignore[arg-type]
        provider=provider,
        config=AgentRunConfig(validation_mode="citation"),
    ):
        pass
    sys_used = provider.calls[0]["system"]
    assert "<cite tu=" in sys_used
    assert "OK to" in sys_used or "OK not" in sys_used or "uncertainty" in sys_used


@pytest.mark.asyncio
async def test_system_prompt_unchanged_when_validation_off() -> None:
    provider = _StubProvider([_text_turn("ok")])
    config = AgentRunConfig()
    async for _ in run_agent(
        user_message="x",
        history=[],
        registry=ProgressiveToolRegistry([]),
        dispatcher=lambda *_: None,  # type: ignore[arg-type]
        provider=provider,
        config=config,
    ):
        pass
    assert provider.calls[0]["system"] == config.system_prompt
