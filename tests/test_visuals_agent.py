"""End-to-end agent-loop behavior for the render_visual meta-tool."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest

from ai_assistant_client.agent import AgentRunConfig, run_agent
from ai_assistant_client.discovery import ProgressiveToolRegistry
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
        self.calls.append({"system": system, "tools": tools, "messages": list(messages)})
        events = (
            self._scripted.pop(0)
            if self._scripted
            else [MessageStop(stop_reason="end_turn")]
        )
        for ev in events:
            yield ev


def _render_visual_turn(*, tool_use_id: str, payload: dict[str, Any]) -> list[NormalizedEvent]:
    return [
        ToolUseStart(index=0, id=tool_use_id, name="render_visual"),
        ToolInputDelta(index=0, partial_json=json.dumps(payload)),
        ToolUseStop(index=0),
        MessageStop(stop_reason="tool_use"),
    ]


def _text_turn(text: str) -> list[NormalizedEvent]:
    return [TextDelta(text=text), MessageStop(stop_reason="end_turn")]


# ---------------------------------------------------------------------------
# Happy path: chart + table + kpi
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_visual_emits_visual_event_for_chart() -> None:
    payload = {
        "schema_version": 1,
        "spec": {
            "kind": "chart",
            "chart_type": "bar",
            "title": "Revenue",
            "data": [{"q": "Q1", "rev": 100}, {"q": "Q2", "rev": 150}],
            "x_key": "q",
            "y_keys": ["rev"],
        },
    }
    provider = _StubProvider(
        [
            _render_visual_turn(tool_use_id="tu_v1", payload=payload),
            _text_turn("Here's the chart."),
        ]
    )

    async def dispatcher(name: str, args: dict[str, Any]) -> Any:
        raise AssertionError("render_visual should not reach the dispatcher")

    events = []
    async for ev in run_agent(
        user_message="show me",
        history=[],
        registry=ProgressiveToolRegistry([]),
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
    ):
        events.append(ev)

    visual_events = [e for e in events if e.type == "visual"]
    assert len(visual_events) == 1
    assert visual_events[0].data["tool_use_id"] == "tu_v1"
    assert visual_events[0].data["schema_version"] == 1
    assert visual_events[0].data["spec"]["chart_type"] == "bar"

    # The tool_result content reaching the LLM should be a short
    # confirmation, not the full spec dump.
    tr_events = [e for e in events if e.type == "tool_result"]
    assert len(tr_events) == 1
    assert "Rendered" in tr_events[0].data["content"]


@pytest.mark.asyncio
async def test_render_visual_for_kpi() -> None:
    payload = {
        "schema_version": 1,
        "spec": {
            "kind": "kpi",
            "label": "Open invoices",
            "value": "$4,340.00",
            "trend": {"direction": "up", "delta": "+12%", "period": "vs last month"},
            "status": "warn",
        },
    }
    provider = _StubProvider(
        [
            _render_visual_turn(tool_use_id="tu_kpi", payload=payload),
            _text_turn("Trending up — review past-due."),
        ]
    )

    events = []
    async for ev in run_agent(
        user_message="x",
        history=[],
        registry=ProgressiveToolRegistry([]),
        dispatcher=lambda *_: None,  # type: ignore[arg-type]
        provider=provider,
        config=AgentRunConfig(),
    ):
        events.append(ev)

    [vis] = [e for e in events if e.type == "visual"]
    assert vis.data["spec"]["kind"] == "kpi"
    assert vis.data["spec"]["status"] == "warn"
    assert vis.data["spec"]["trend"]["direction"] == "up"


# ---------------------------------------------------------------------------
# Validation failure: error fed back to LLM, no visual event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_visual_invalid_spec_produces_error_tool_result() -> None:
    bad_payload = {
        "schema_version": 1,
        "spec": {
            "kind": "chart",
            "chart_type": "bar",
            "data": [{"q": "Q1"}],  # missing y_key
            "x_key": "q",
            "y_keys": ["rev"],
        },
    }
    provider = _StubProvider(
        [
            _render_visual_turn(tool_use_id="tu_bad", payload=bad_payload),
            _text_turn("Sorry — let me describe in text."),
        ]
    )

    events = []
    async for ev in run_agent(
        user_message="x",
        history=[],
        registry=ProgressiveToolRegistry([]),
        dispatcher=lambda *_: None,  # type: ignore[arg-type]
        provider=provider,
        config=AgentRunConfig(),
    ):
        events.append(ev)

    assert not any(e.type == "visual" for e in events)
    [tr] = [e for e in events if e.type == "tool_result"]
    assert "rejected" in tr.data["content"].lower()
    assert "rev" in tr.data["content"]  # message names the missing key


@pytest.mark.asyncio
async def test_render_visual_rejects_unsafe_image_src() -> None:
    payload = {
        "schema_version": 1,
        "spec": {
            "kind": "image",
            "src": "javascript:alert(1)",
            "alt": "x",
        },
    }
    provider = _StubProvider(
        [
            _render_visual_turn(tool_use_id="tu_img", payload=payload),
            _text_turn("noted"),
        ]
    )
    events = []
    async for ev in run_agent(
        user_message="x",
        history=[],
        registry=ProgressiveToolRegistry([]),
        dispatcher=lambda *_: None,  # type: ignore[arg-type]
        provider=provider,
        config=AgentRunConfig(),
    ):
        events.append(ev)
    assert not any(e.type == "visual" for e in events)
    [tr] = [e for e in events if e.type == "tool_result"]
    assert "src" in tr.data["content"].lower()


# ---------------------------------------------------------------------------
# Schema surface: render_visual is always available
# ---------------------------------------------------------------------------


def test_render_visual_schema_in_initial_tool_window() -> None:
    registry = ProgressiveToolRegistry([])
    names = {t["name"] for t in registry.anthropic_tools()}
    assert "render_visual" in names
