"""Sample multi-step human-in-the-loop workflows.

Drop this file into the client's workflows directory (default
``./workflows``, override with ``WORKFLOWS_DIR``) and the assistant
will surface each function as a tool the LLM can call.

Each workflow runs inside the agent process — that's what lets it
``await pause_for_confirmation(...)`` between steps.  Replace the
stub bodies with real side effects (an SMTP send, a SQL query, a
batch-archive job…) when wiring against your own backend.
"""

from __future__ import annotations

from ai_assistant_client.workflows import (
    emit_status,
    pause_for_confirmation,
    workflow,
)


# ---------------------------------------------------------------------------
# 1. Single-pause: send-email-with-approval
# ---------------------------------------------------------------------------


@workflow(
    name="approve_and_send_email",
    description=(
        "Send an email after the user approves a single confirm/decline "
        "modal showing the recipient, subject, and body."
    ),
    tags=("email", "communications"),
)
async def approve_and_send_email(*, to: str, subject: str, body: str) -> dict:
    outcome = await pause_for_confirmation(
        message=f"Send email to {to}?",
        preview={"to": to, "subject": subject, "body": body},
    )
    if outcome.decision == "decline":
        return {
            "sent": False,
            "reason": "user_declined",
            "note": outcome.note,
        }
    await emit_status("Sending email…")
    # Replace with your real send.
    return {"sent": True, "to": to, "subject": subject}


# ---------------------------------------------------------------------------
# 2. Two pauses: draft → review → confirm-send
# ---------------------------------------------------------------------------


@workflow(
    name="draft_review_send_email",
    description=(
        "Generate an email draft from instructions, ask the user to "
        "review the draft, then ask one more time before sending. "
        "Two confirmation modals total."
    ),
    tags=("email", "communications"),
)
async def draft_review_send_email(
    *, to: str, subject: str, instructions: str
) -> dict:
    # Step 1: Generate a draft body from the instructions.  Real
    # implementations would call an LLM or template engine; here we
    # stub to keep the workflow runnable without external deps.
    await emit_status("Drafting email…")
    draft_body = (
        f"Hi,\n\nFollowing up on: {instructions}\n\nBest,\nAssistant"
    )

    # Step 2: First pause — review the draft.
    review = await pause_for_confirmation(
        message=f"Review draft email to {to}?",
        preview={"to": to, "subject": subject, "draft": draft_body},
    )
    if review.decision == "decline":
        return {
            "sent": False,
            "reason": "user_rejected_draft",
            "note": review.note,
        }

    # Step 3: Final confirm before send.
    final = await pause_for_confirmation(
        message="Final confirm — send now?",
        preview={"to": to, "subject": subject, "body": draft_body},
    )
    if final.decision == "decline":
        return {
            "sent": False,
            "reason": "user_aborted_send",
            "note": final.note,
        }
    await emit_status("Sending email…")
    return {"sent": True, "to": to, "subject": subject}


# ---------------------------------------------------------------------------
# 3. Conditional pause: bulk archive only confirms when the count is large
# ---------------------------------------------------------------------------


@workflow(
    name="bulk_archive_with_review",
    description=(
        "Archive records matching a query.  Emits a status per batch; "
        "if the total exceeds ``confirmation_threshold`` rows, asks the "
        "user to confirm before proceeding."
    ),
    tags=("database", "destructive"),
)
async def bulk_archive_with_review(
    *,
    query: str,
    confirmation_threshold: int = 100,
) -> dict:
    # Stand-in for the actual record count — replace with a SELECT
    # COUNT(*) or similar.
    matched_count = 250

    await emit_status(
        f"Found {matched_count} records matching query…",
        data={"query": query, "count": matched_count},
    )

    if matched_count >= confirmation_threshold:
        outcome = await pause_for_confirmation(
            message=(
                f"About to archive {matched_count} records "
                f"(threshold {confirmation_threshold}).  Continue?"
            ),
            preview={"query": query, "matched_count": matched_count},
        )
        if outcome.decision == "decline":
            return {
                "archived": 0,
                "reason": "user_declined",
                "note": outcome.note,
            }

    # Pretend we batch through them; emit progress per chunk.
    archived = 0
    chunk_size = max(1, matched_count // 5)
    for chunk_start in range(0, matched_count, chunk_size):
        archived += min(chunk_size, matched_count - chunk_start)
        await emit_status(
            f"Archived {archived}/{matched_count}…",
            data={"archived": archived, "total": matched_count},
        )
    return {"archived": archived, "query": query}
