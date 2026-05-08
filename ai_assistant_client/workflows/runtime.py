"""Workflow execution runtime.

A workflow is just an async function — what makes it different from
a plain plugin tool is that it can pause for the user mid-execution
via :func:`pause_for_confirmation` and emit progress updates via
:func:`emit_status`.  Both helpers reach a per-call
:class:`WorkflowContext` through a contextvar so the workflow author
doesn't have to pass plumbing through every step.

The runtime drives the workflow on a background task and yields
events to the agent loop through an :class:`asyncio.Queue` — that's
how a workflow's pauses and statuses reach the SSE stream.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Literal

from ai_assistant_client.confirmation_store import ConfirmationOutcome
from ai_assistant_client.workflows.decorator import Workflow


log = logging.getLogger(__name__)


# Shape of the confirmation_hook callable the agent loop hands in.
ConfirmationHook = Callable[[dict[str, Any]], Awaitable[ConfirmationOutcome]]


@dataclass
class WorkflowEvent:
    """Anything a workflow run can emit through the runtime queue."""

    type: Literal[
        "status",
        "confirmation_request",
        "confirmation_resolved",
        "result",
        "error",
    ]
    payload: dict[str, Any] | None = None
    value: Any = None
    error: str | None = None


@dataclass
class WorkflowContext:
    """Per-call state the workflow's helpers need to reach.

    Owned by :func:`run_workflow`; surfaced to the workflow body
    via a contextvar so authors don't have to thread it through.
    """

    workflow_name: str
    tool_use_id: str
    queue: "asyncio.Queue[WorkflowEvent]"
    confirmation_hook: ConfirmationHook | None
    default_timeout_seconds: int
    on_timeout_decision: Literal["confirm", "decline"] = "decline"


# Contextvar carries the active context to whatever inner await
# `pause_for_confirmation` / `emit_status` is reached through.  The
# runtime sets it before invoking the workflow and clears it after.
_current_ctx: contextvars.ContextVar[WorkflowContext | None] = contextvars.ContextVar(
    "aai_workflow_ctx", default=None
)


@contextmanager
def _bind_context(ctx: WorkflowContext) -> Iterator[None]:
    token = _current_ctx.set(ctx)
    try:
        yield
    finally:
        _current_ctx.reset(token)


# ---------------------------------------------------------------------------
# Author-facing helpers
# ---------------------------------------------------------------------------


async def pause_for_confirmation(
    *,
    message: str | None = None,
    preview: Any = None,
    timeout_seconds: int | None = None,
) -> ConfirmationOutcome:
    """Surface a confirm/decline modal in the chat UI; await the
    user's decision.

    ``message`` is shown as a one-line hint above the preview.
    ``preview`` is an arbitrary JSON-serializable object the UI
    renders alongside the prompt — typically the proposed action's
    payload (the email body, the SQL query, the charge amount).
    Falling back to a 60-second default when ``timeout_seconds``
    isn't set; a timeout counts as decline by default (see
    ``AgentRunConfig.confirmation_on_timeout``).
    """
    ctx = _current_ctx.get()
    if ctx is None:
        raise RuntimeError(
            "pause_for_confirmation() must be called inside a @workflow body "
            "running under run_workflow(...)."
        )
    request_id = f"wf_{uuid.uuid4().hex}"
    timeout = timeout_seconds or ctx.default_timeout_seconds
    payload: dict[str, Any] = {
        "request_id": request_id,
        "tool_use_id": ctx.tool_use_id,
        "tool_name": ctx.workflow_name,
        "tool_description": "",
        "tool_input": preview,
        "timeout_seconds": int(timeout),
    }
    if message:
        payload["message"] = message
    await ctx.queue.put(
        WorkflowEvent(type="confirmation_request", payload=payload)
    )

    if ctx.confirmation_hook is None:
        # No host-installed hook → mirror the agent loop's auto-confirm
        # fallback so a workflow doesn't deadlock when the host hasn't
        # wired a UI yet.  The request event still fired above.
        log.warning(
            "Workflow %r reached pause_for_confirmation but no hook is "
            "registered; auto-confirming.",
            ctx.workflow_name,
        )
        outcome = ConfirmationOutcome(decision="confirm")
    else:
        try:
            outcome = await asyncio.wait_for(
                ctx.confirmation_hook(payload), timeout=timeout
            )
        except asyncio.TimeoutError:
            outcome = ConfirmationOutcome(
                decision=ctx.on_timeout_decision,
                note="confirmation timed out",
            )
    await ctx.queue.put(
        WorkflowEvent(
            type="confirmation_resolved",
            payload={
                "request_id": request_id,
                "tool_use_id": ctx.tool_use_id,
                "decision": outcome.decision,
                "note": outcome.note,
            },
        )
    )
    return outcome


async def emit_status(message: str, *, data: Any = None) -> None:
    """Push a one-line progress update to the host.  The agent loop
    forwards each as a ``workflow_status`` SSE event so the UI can
    render an inline indicator (e.g. "Sending email…", "Page 3 of
    7…")."""
    ctx = _current_ctx.get()
    if ctx is None:
        raise RuntimeError(
            "emit_status() must be called inside a @workflow body "
            "running under run_workflow(...)."
        )
    payload = {
        "tool_use_id": ctx.tool_use_id,
        "tool_name": ctx.workflow_name,
        "message": message,
    }
    if data is not None:
        payload["data"] = data
    await ctx.queue.put(WorkflowEvent(type="status", payload=payload))


# ---------------------------------------------------------------------------
# Runtime driver
# ---------------------------------------------------------------------------


async def run_workflow(
    wf: Workflow,
    args: dict[str, Any],
    *,
    tool_use_id: str,
    confirmation_hook: ConfirmationHook | None,
    default_timeout_seconds: int = 60,
    on_timeout_decision: Literal["confirm", "decline"] = "decline",
) -> AsyncIterator[WorkflowEvent]:
    """Execute ``wf`` with ``args``; stream events back to the caller.

    The caller (the agent loop) iterates and forwards each event
    onto the SSE channel.  The final ``WorkflowEvent`` will always
    be either ``type='result'`` (with ``value``) or ``type='error'``
    (with ``error``) so the caller can stop reading.
    """
    queue: asyncio.Queue[WorkflowEvent] = asyncio.Queue()
    ctx = WorkflowContext(
        workflow_name=wf.name,
        tool_use_id=tool_use_id,
        queue=queue,
        confirmation_hook=confirmation_hook,
        default_timeout_seconds=default_timeout_seconds,
        on_timeout_decision=on_timeout_decision,
    )

    async def _run() -> None:
        try:
            with _bind_context(ctx):
                result = await wf.handler(**args)
            await queue.put(WorkflowEvent(type="result", value=result))
        except TypeError as err:
            # Argument-shape mismatch from the LLM — surface as a
            # tool error, not a workflow crash.
            await queue.put(
                WorkflowEvent(
                    type="error",
                    error=f"workflow {wf.name!r} rejected arguments: {err}",
                )
            )
        except Exception as err:  # noqa: BLE001
            log.exception("Workflow %s failed", wf.name)
            await queue.put(
                WorkflowEvent(
                    type="error",
                    error=f"workflow {wf.name!r} raised: {err}",
                )
            )

    task = asyncio.create_task(_run())
    try:
        while True:
            event = await queue.get()
            yield event
            if event.type in ("result", "error"):
                return
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
