"""End-to-end: when a TranscriptStore is passed to ``run_agent``,
every workflow the LLM dispatches inside the turn gets recorded.

Unrecorded path (``transcript_store=None``) stays byte-for-byte
identical to the pre-recording dispatch — the new kwarg must not
have any observable side effects when absent.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from ai_assistant_client.agent import AgentRunConfig, run_agent
from ai_assistant_client.discovery import (
    ProgressiveToolRegistry,
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
from ai_assistant_client.persistence import InMemoryTranscriptStore
from ai_assistant_client.workflows import (
    WorkflowRegistry,
    get_workflow,
    workflow,
)


class _StubProvider(LLMProvider):
    def __init__(self, scripted: list[list[NormalizedEvent]]) -> None:
        self._scripted = list(scripted)

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
        events = (
            self._scripted.pop(0)
            if self._scripted
            else [MessageStop(stop_reason="end_turn")]
        )
        for ev in events:
            yield ev


def _tool_use(
    *, tool_id: str, tool_name: str, tool_input_json: str = "{}"
) -> list[NormalizedEvent]:
    return [
        ToolUseStart(index=0, id=tool_id, name=tool_name),
        ToolInputDelta(index=0, partial_json=tool_input_json),
        ToolUseStop(index=0),
        MessageStop(stop_reason="tool_use"),
    ]


def _text(text: str) -> list[NormalizedEvent]:
    return [TextDelta(text=text), MessageStop(stop_reason="end_turn")]


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_dispatch_records_when_store_provided() -> None:
    @workflow(description="d")
    async def recorded_workflow() -> str:
        return "done"

    wf = get_workflow(recorded_workflow)
    assert wf is not None
    workflow_registry = WorkflowRegistry([wf])
    registry = ProgressiveToolRegistry(workflow_registry.as_descriptors())

    provider = _StubProvider(
        [
            _tool_use(
                tool_id="tu_1",
                tool_name=wf.name,
            ),
            _text("ok"),
        ]
    )

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError(
            "workflow dispatch should not call the remote dispatcher"
        )

    store = InMemoryTranscriptStore()

    async for _ in run_agent(
        user_message="do it",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        workflow_registry=workflow_registry,
        transcript_store=store,
    ):
        pass

    # Exactly one workflow was dispatched, so exactly one
    # transcript should be in the store.
    runs = await store.list_runs()
    assert len(runs) == 1
    transcript = await store.read(runs[0])
    assert transcript.header.workflow_name == wf.name
    assert transcript.footer is not None
    assert transcript.footer.outcome == "result"
    # Auto-generated run id includes the workflow name for legibility.
    assert wf.name in runs[0]


@pytest.mark.asyncio
async def test_workflow_dispatch_does_not_record_when_store_omitted() -> None:
    """Default behaviour (no transcript_store kwarg) stays
    pre-recording-equivalent."""

    @workflow(description="d")
    async def silent_workflow() -> str:
        return "done"

    wf = get_workflow(silent_workflow)
    assert wf is not None
    workflow_registry = WorkflowRegistry([wf])
    registry = ProgressiveToolRegistry(workflow_registry.as_descriptors())

    provider = _StubProvider(
        [
            _tool_use(
                tool_id="tu_1",
                tool_name=wf.name,
            ),
            _text("ok"),
        ]
    )

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("no remote dispatch expected")

    # Confirm the workflow runs to completion just like the
    # pre-recording path — the new kwarg is purely additive when
    # left at its default None.
    final_text = ""
    async for event in run_agent(
        user_message="do it",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        workflow_registry=workflow_registry,
    ):
        if event.type == "text_delta":
            final_text += event.data["text"]
    assert "ok" in final_text


@pytest.mark.asyncio
async def test_each_workflow_dispatch_gets_unique_run_id() -> None:
    """Two dispatches inside the same turn produce two run ids
    so the transcripts don't collide on the store side."""

    @workflow(description="d")
    async def repeatable() -> str:
        return "done"

    wf = get_workflow(repeatable)
    assert wf is not None
    workflow_registry = WorkflowRegistry([wf])
    registry = ProgressiveToolRegistry(workflow_registry.as_descriptors())

    provider = _StubProvider(
        [
            _tool_use(tool_id="tu_1", tool_name=wf.name),
            _tool_use(tool_id="tu_2", tool_name=wf.name),
            _text("ok"),
        ]
    )

    async def dispatcher(name: str, arguments: dict[str, Any]) -> Any:
        raise AssertionError("no remote dispatch expected")

    store = InMemoryTranscriptStore()
    async for _ in run_agent(
        user_message="do it twice",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(),
        workflow_registry=workflow_registry,
        transcript_store=store,
    ):
        pass

    runs = await store.list_runs()
    assert len(runs) == 2
    assert runs[0] != runs[1]
