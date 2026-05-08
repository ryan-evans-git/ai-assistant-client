"""Tests for the client-side workflow primitive.

Covers the @workflow decorator + signature → JSON Schema, the
runtime (single + multi pause, status events, errors), the
registry, and end-to-end integration through run_agent.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from ai_assistant_client.confirmation_store import ConfirmationOutcome
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
from ai_assistant_client.workflows import (
    WorkflowRegistry,
    emit_status,
    get_workflow,
    is_workflow,
    pause_for_confirmation,
    run_workflow,
    workflow,
)


# Module-level workflow so the load_workflows_from_module test
# (which walks dir(module)) finds something concrete.  Decorators
# inside individual tests register at local scope dir() can't see.
@workflow(name="module_level_ping_workflow", description="returns pong")
async def _module_level_ping_workflow() -> str:
    return "pong"


# ---------------------------------------------------------------------------
# @workflow decorator + schema derivation
# ---------------------------------------------------------------------------


def test_workflow_decorator_attaches_marker_and_schema() -> None:
    @workflow(name="echo", description="Echo input.")
    async def echo(*, msg: str) -> dict:
        return {"echo": msg}

    assert is_workflow(echo)
    wf = get_workflow(echo)
    assert wf is not None
    assert wf.name == "echo"
    assert wf.description == "Echo input."
    assert wf.input_schema["properties"]["msg"]["type"] == "string"
    assert wf.input_schema["required"] == ["msg"]


def test_workflow_decorator_uses_docstring_when_no_description() -> None:
    @workflow()
    async def hello() -> str:
        """Greet the world."""
        return "hello"

    wf = get_workflow(hello)
    assert wf is not None
    assert wf.name == "hello"
    assert wf.description == "Greet the world."


def test_workflow_decorator_requires_async_function() -> None:
    with pytest.raises(TypeError, match="async function"):
        @workflow(description="x")
        def sync_fn() -> str:
            return "x"


def test_workflow_decorator_requires_description() -> None:
    with pytest.raises(ValueError, match="description="):
        @workflow()
        async def silent(x: int) -> int:
            return x


def test_workflow_no_args_emits_empty_object_schema() -> None:
    @workflow(description="d")
    async def nothing() -> str:
        return "x"

    wf = get_workflow(nothing)
    assert wf is not None
    assert wf.input_schema == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


def test_workflow_rejects_var_args() -> None:
    async def variadic(*args: int) -> int:
        return sum(args)

    with pytest.raises(ValueError, match="args"):
        # Decorate manually so we can assert directly.
        workflow(description="x")(variadic)


# ---------------------------------------------------------------------------
# Runtime: single pause + status + result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_happy_path_no_pauses() -> None:
    @workflow(description="d")
    async def add(*, a: int, b: int) -> int:
        return a + b

    wf = get_workflow(add)
    assert wf is not None
    events = []
    async for ev in run_workflow(
        wf, {"a": 2, "b": 3}, tool_use_id="tu", confirmation_hook=None
    ):
        events.append(ev)
    assert len(events) == 1
    assert events[0].type == "result"
    assert events[0].value == 5


@pytest.mark.asyncio
async def test_runtime_emits_status_events() -> None:
    @workflow(description="d")
    async def progressive() -> str:
        await emit_status("step 1")
        await emit_status("step 2", data={"i": 2})
        return "ok"

    wf = get_workflow(progressive)
    assert wf is not None
    events = []
    async for ev in run_workflow(
        wf, {}, tool_use_id="tu", confirmation_hook=None
    ):
        events.append(ev)
    statuses = [e for e in events if e.type == "status"]
    assert len(statuses) == 2
    assert statuses[0].payload["message"] == "step 1"
    assert statuses[1].payload["data"] == {"i": 2}
    assert events[-1].type == "result"


@pytest.mark.asyncio
async def test_runtime_pause_calls_hook_and_returns_outcome() -> None:
    captured_payload: dict[str, Any] = {}

    @workflow(description="d")
    async def pause_once(*, x: str) -> dict:
        outcome = await pause_for_confirmation(
            message="Do it?", preview={"x": x}
        )
        return {"decision": outcome.decision, "note": outcome.note}

    async def hook(payload: dict[str, Any]) -> ConfirmationOutcome:
        captured_payload.update(payload)
        return ConfirmationOutcome(decision="confirm", note="LGTM")

    wf = get_workflow(pause_once)
    assert wf is not None
    events = []
    async for ev in run_workflow(
        wf, {"x": "hello"}, tool_use_id="tu_pause", confirmation_hook=hook
    ):
        events.append(ev)

    types = [e.type for e in events]
    assert types == ["confirmation_request", "confirmation_resolved", "result"]
    assert captured_payload["tool_use_id"] == "tu_pause"
    assert captured_payload["tool_name"] == "pause_once"
    assert captured_payload["tool_input"] == {"x": "hello"}
    assert captured_payload["message"] == "Do it?"
    final = events[-1].value
    assert final == {"decision": "confirm", "note": "LGTM"}


@pytest.mark.asyncio
async def test_runtime_two_pauses_each_get_unique_request_ids() -> None:
    @workflow(description="d")
    async def two_pauses() -> dict:
        a = await pause_for_confirmation(message="first?")
        b = await pause_for_confirmation(message="second?")
        return {"first": a.decision, "second": b.decision}

    decisions = iter(
        [
            ConfirmationOutcome(decision="confirm"),
            ConfirmationOutcome(decision="decline", note="changed mind"),
        ]
    )

    async def hook(payload: dict[str, Any]) -> ConfirmationOutcome:
        return next(decisions)

    wf = get_workflow(two_pauses)
    assert wf is not None
    request_ids: list[str] = []
    events = []
    async for ev in run_workflow(
        wf, {}, tool_use_id="tu", confirmation_hook=hook
    ):
        events.append(ev)
        if ev.type == "confirmation_request":
            request_ids.append(ev.payload["request_id"])
    assert len(request_ids) == 2
    assert request_ids[0] != request_ids[1]
    final = events[-1].value
    assert final == {"first": "confirm", "second": "decline"}


@pytest.mark.asyncio
async def test_runtime_pause_without_hook_auto_confirms() -> None:
    @workflow(description="d")
    async def pause_once() -> str:
        outcome = await pause_for_confirmation(message="?")
        return outcome.decision

    wf = get_workflow(pause_once)
    assert wf is not None
    events = []
    async for ev in run_workflow(
        wf, {}, tool_use_id="tu", confirmation_hook=None
    ):
        events.append(ev)
    assert events[-1].value == "confirm"


@pytest.mark.asyncio
async def test_runtime_pause_timeout_treated_as_decline() -> None:
    @workflow(description="d")
    async def pause_slowly() -> str:
        outcome = await pause_for_confirmation(
            message="?", timeout_seconds=1
        )
        return outcome.decision

    async def slow_hook(payload: dict[str, Any]) -> ConfirmationOutcome:
        await asyncio.sleep(5)
        return ConfirmationOutcome(decision="confirm")  # pragma: no cover

    wf = get_workflow(pause_slowly)
    assert wf is not None
    events = []
    # Override default timeout to keep the test fast.
    async for ev in run_workflow(
        wf,
        {},
        tool_use_id="tu",
        confirmation_hook=slow_hook,
        default_timeout_seconds=1,
    ):
        events.append(ev)
    # The pause itself uses the supplied timeout_seconds=1; we
    # short-circuit slow_hook through asyncio.wait_for inside the
    # runtime.
    assert events[-1].value == "decline"


@pytest.mark.asyncio
async def test_runtime_workflow_raising_emits_error_event() -> None:
    @workflow(description="d")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    wf = get_workflow(boom)
    assert wf is not None
    events = []
    async for ev in run_workflow(
        wf, {}, tool_use_id="tu", confirmation_hook=None
    ):
        events.append(ev)
    assert events[-1].type == "error"
    assert "kaboom" in (events[-1].error or "")


@pytest.mark.asyncio
async def test_runtime_workflow_argument_mismatch_emits_error() -> None:
    @workflow(description="d")
    async def strict(*, a: int) -> int:
        return a

    wf = get_workflow(strict)
    assert wf is not None
    events = []
    async for ev in run_workflow(
        wf, {"a": 1, "b": 2}, tool_use_id="tu", confirmation_hook=None
    ):
        events.append(ev)
    assert events[-1].type == "error"
    assert "rejected arguments" in (events[-1].error or "")


@pytest.mark.asyncio
async def test_pause_outside_workflow_raises() -> None:
    with pytest.raises(RuntimeError, match="inside a @workflow body"):
        await pause_for_confirmation(message="?")


@pytest.mark.asyncio
async def test_emit_status_outside_workflow_raises() -> None:
    with pytest.raises(RuntimeError, match="inside a @workflow body"):
        await emit_status("nope")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_lookup_and_descriptors() -> None:
    @workflow(name="alpha", description="d")
    async def alpha() -> str:
        return "a"

    @workflow(name="beta", description="d", tags=("group",))
    async def beta(*, x: int) -> int:
        return x

    wf_a = get_workflow(alpha)
    wf_b = get_workflow(beta)
    assert wf_a is not None and wf_b is not None
    reg = WorkflowRegistry([wf_a, wf_b])
    assert "alpha" in reg
    assert "ghost" not in reg
    assert len(reg) == 2
    assert sorted(reg.names()) == ["alpha", "beta"]
    descs = reg.as_descriptors()
    by_name = {d.name: d for d in descs}
    assert by_name["alpha"].input_schema["properties"] == {}
    assert by_name["beta"].tags == ("group",)
    # Workflows never carry hitl on the descriptor — gates are inside.
    assert all(d.hitl is None for d in descs)


def test_registry_rejects_duplicate_names() -> None:
    @workflow(name="dup", description="d")
    async def a() -> str:
        return "a"

    @workflow(name="dup", description="d")
    async def b() -> str:
        return "b"

    wf_a = get_workflow(a)
    wf_b = get_workflow(b)
    assert wf_a is not None and wf_b is not None
    with pytest.raises(ValueError, match="Duplicate"):
        WorkflowRegistry([wf_a, wf_b])


def test_registry_empty_default() -> None:
    reg = WorkflowRegistry()
    assert len(reg) == 0
    assert reg.as_descriptors() == []


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_load_workflows_from_directory_skips_missing_dir(tmp_path) -> None:
    from ai_assistant_client.workflows import load_workflows_from_directory

    assert load_workflows_from_directory(tmp_path / "no-such") == []


def test_load_workflows_from_directory_loads_decorated_functions(
    tmp_path,
) -> None:
    from ai_assistant_client.workflows import load_workflows_from_directory

    (tmp_path / "_helpers.py").write_text("X = 1\n")
    (tmp_path / "ok.py").write_text(
        "from ai_assistant_client.workflows import workflow\n\n"
        "@workflow(name='ping', description='returns pong')\n"
        "async def ping() -> str:\n"
        "    return 'pong'\n"
    )
    workflows = load_workflows_from_directory(tmp_path)
    assert [w.name for w in workflows] == ["ping"]


def test_load_workflows_from_directory_skips_unparseable_file(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    from ai_assistant_client.workflows import load_workflows_from_directory

    (tmp_path / "broken.py").write_text("def def def\n")
    (tmp_path / "good.py").write_text(
        "from ai_assistant_client.workflows import workflow\n\n"
        "@workflow(name='good', description='d')\n"
        "async def good() -> str:\n"
        "    return 'ok'\n"
    )
    with caplog.at_level(logging.WARNING):
        workflows = load_workflows_from_directory(tmp_path)
    assert [w.name for w in workflows] == ["good"]
    assert any("Skipping workflow file" in r.message for r in caplog.records)


def test_load_workflows_from_module() -> None:
    from ai_assistant_client.workflows import load_workflows_from_module

    # Load from this very test module — it has @workflow-decorated funcs above.
    workflows = load_workflows_from_module(__name__)
    # At least one well-known fixture-defined workflow registered above.
    assert workflows  # non-empty


# ---------------------------------------------------------------------------
# Sample workflows file smoke test
# ---------------------------------------------------------------------------


def test_sample_workflows_load_with_three_workflows() -> None:
    """Smoke: the example workflows/sample.py registers three demos."""
    from pathlib import Path

    from ai_assistant_client.workflows import load_workflows_from_directory

    workflows_dir = Path(__file__).parent.parent / "workflows"
    workflows = load_workflows_from_directory(workflows_dir)
    names = {w.name for w in workflows}
    for expected in (
        "approve_and_send_email",
        "draft_review_send_email",
        "bulk_archive_with_review",
    ):
        assert expected in names


# ---------------------------------------------------------------------------
# Agent-loop integration
# ---------------------------------------------------------------------------


def _tool_use_turn(*, tool_id: str, tool_name: str, args_json: str) -> list[NormalizedEvent]:
    return [
        ToolUseStart(index=0, id=tool_id, name=tool_name),
        ToolInputDelta(index=0, partial_json=args_json),
        ToolUseStop(index=0),
        MessageStop(stop_reason="tool_use"),
    ]


def _end_turn() -> list[NormalizedEvent]:
    return [TextDelta(text="ok"), MessageStop(stop_reason="end_turn")]


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
        events = self._scripted.pop(0) if self._scripted else _end_turn()
        for ev in events:
            yield ev


@pytest.mark.asyncio
async def test_agent_dispatches_workflow_with_pauses() -> None:
    from ai_assistant_client.agent import AgentEvent, AgentRunConfig, run_agent

    @workflow(name="approve_send", description="d")
    async def approve_send(*, to: str) -> dict:
        decision = await pause_for_confirmation(
            message=f"Send to {to}?", preview={"to": to}
        )
        if decision.decision == "decline":
            return {"sent": False}
        await emit_status("Sending…")
        return {"sent": True, "to": to}

    wf = get_workflow(approve_send)
    assert wf is not None
    workflow_registry = WorkflowRegistry([wf])
    descriptors = workflow_registry.as_descriptors()
    registry = ProgressiveToolRegistry(descriptors)

    async def dispatcher(name: str, args: dict[str, Any]) -> Any:
        # Workflows shouldn't hit the MCP dispatcher.
        raise AssertionError("dispatcher called for a workflow")

    async def hook(payload: dict[str, Any]) -> ConfirmationOutcome:
        return ConfirmationOutcome(decision="confirm")

    provider = _StubProvider([_tool_use_turn(
        tool_id="tu1", tool_name="approve_send", args_json='{"to":"a@b.c"}'
    )])

    events: list[AgentEvent] = []
    async for ev in run_agent(
        user_message="please send",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(model="x"),
        confirmation_hook=hook,
        workflow_registry=workflow_registry,
    ):
        events.append(ev)

    types = [e.type for e in events]
    # Confirmation pair fired from inside the workflow.
    assert "tool_confirmation_request" in types
    assert "tool_confirmation_resolved" in types
    # Status event from emit_status.
    assert "workflow_status" in types
    # Final tool_result with the workflow's return value.
    result = next(e for e in events if e.type == "tool_result")
    assert "sent" in result.data["content"]


@pytest.mark.asyncio
async def test_agent_workflow_decline_returns_workflow_result_not_decline_text() -> None:
    """When the workflow itself handles a decline (returning something
    rather than raising), the tool_result reflects the workflow's
    return value — the per-tool gate doesn't apply."""
    from ai_assistant_client.agent import AgentEvent, AgentRunConfig, run_agent

    @workflow(name="maybe_send", description="d")
    async def maybe_send(*, to: str) -> dict:
        decision = await pause_for_confirmation(message="ok?")
        return {"sent": False, "reason": decision.note or "user said no"}

    wf = get_workflow(maybe_send)
    assert wf is not None
    workflow_registry = WorkflowRegistry([wf])
    registry = ProgressiveToolRegistry(workflow_registry.as_descriptors())

    async def dispatcher(name: str, args: dict[str, Any]) -> Any:
        raise AssertionError("should not dispatch")

    async def hook(payload: dict[str, Any]) -> ConfirmationOutcome:
        return ConfirmationOutcome(decision="decline", note="cold feet")

    provider = _StubProvider([_tool_use_turn(
        tool_id="tu1", tool_name="maybe_send", args_json='{"to":"x"}'
    )])
    events: list[AgentEvent] = []
    async for ev in run_agent(
        user_message="x",
        history=[],
        registry=registry,
        dispatcher=dispatcher,
        provider=provider,
        config=AgentRunConfig(model="x"),
        confirmation_hook=hook,
        workflow_registry=workflow_registry,
    ):
        events.append(ev)
    # Workflow ran to completion — the tool_result is its normal return
    # value, not the per-tool-gate decline path.
    result = next(e for e in events if e.type == "tool_result")
    assert "cold feet" in result.data["content"]
    assert result.data.get("is_error") is None or result.data.get("is_error") is False
