"""Client-side multi-step human-in-the-loop workflows.

A *workflow* is a Python async function the agent can dispatch like
any other tool, except that it runs **inside the agent process**
instead of being routed to an MCP server.  Because it lives in the
same process as the agent loop and the SSE stream, it can ``await
pause_for_confirmation(...)`` to pause for user approval at any
checkpoint — not just before the first call.

Drop a file under the configured workflows directory (default
``./workflows``) with one or more ``@workflow``-decorated coroutines
and the agent picks them up at startup, exposing each as a tool
the LLM can call.

Authoring example::

    from ai_assistant_client.workflows import workflow, pause_for_confirmation

    @workflow(
        name="approve_and_send_email",
        description="Send an email after user approval.",
    )
    async def approve_and_send_email(*, to: str, subject: str, body: str) -> dict:
        outcome = await pause_for_confirmation(
            message=f"Send email to {to}?",
            preview={"subject": subject, "body": body},
        )
        if outcome.decision == "decline":
            return {"sent": False, "note": outcome.note}
        # ... real send here
        return {"sent": True}
"""

from __future__ import annotations

from ai_assistant_client.workflows.decorator import (
    Workflow,
    get_workflow,
    is_workflow,
    workflow,
)
from ai_assistant_client.workflows.loader import (
    load_workflows_from_directory,
    load_workflows_from_module,
)
from ai_assistant_client.workflows.registry import WorkflowRegistry
from ai_assistant_client.workflows.runtime import (
    WorkflowContext,
    WorkflowEvent,
    emit_status,
    pause_for_confirmation,
    run_workflow,
)


__all__ = [
    "Workflow",
    "WorkflowContext",
    "WorkflowEvent",
    "WorkflowRegistry",
    "emit_status",
    "get_workflow",
    "is_workflow",
    "load_workflows_from_directory",
    "load_workflows_from_module",
    "pause_for_confirmation",
    "run_workflow",
    "workflow",
]
