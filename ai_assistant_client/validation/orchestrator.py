"""High-level entry point used by the agent loop.

Three concerns live here so ``agent.py`` doesn't have to know about
citation parsing, JSONPath internals, or auditor wiring:

  - :data:`CITATION_INSTRUCTIONS` — system-prompt addendum that
    describes the citation grammar plus the "it's OK to not know"
    nudge.  The agent appends this to the user-supplied system prompt
    when ``validation_mode != "off"``.

  - :func:`run_validation` — runs the citation verifier and (when
    appropriate) the auditor LLM, returning a single
    :class:`ValidationResult`.

  - :func:`build_retry_feedback` — formats a list of issues into a
    user-facing message the agent loop appends to history when asking
    the model to revise.
"""

from __future__ import annotations

from typing import Any, Literal

from ai_assistant_client.llm import LLMProvider
from ai_assistant_client.validation.auditor import (
    DEFAULT_AUDITOR_MODELS,
    run_auditor,
)
from ai_assistant_client.validation.citations import strip_citations
from ai_assistant_client.validation.types import (
    CitationIssue,
    ValidationIssue,
    ValidationResult,
)
from ai_assistant_client.validation.verify import (
    index_tool_results,
    verify_citations,
)


ValidationMode = Literal["off", "citation", "audit", "hybrid"]
Strictness = Literal["permissive", "strict"]


CITATION_INSTRUCTIONS = """\

When you reference any specific value drawn from a tool result —
numbers, amounts, dates, names, statuses, or IDs — wrap it in a
citation tag in this exact form:

  <cite tu="<tool_use_id>" path="<jsonpath>">displayed value</cite>

Examples:
  The customer owes <cite tu="tu_1" path="$.invoices[2].amount">$1,250.00</cite>.
  Payment was due <cite tu="tu_1" path="$.invoices[2].due_date">March 15, 2026</cite>.
  There are <cite tu="tu_1" path="$.invoices|length">3</cite> open invoices.

You don't need to cite conversational text, your own inferences, or
recommendations — only specific values that appear in tool data.

If the data doesn't contain what was asked, or you don't know an
answer, say so directly.  It is always better to acknowledge
uncertainty than to guess — fabricated values are a much worse
failure than admitting "I don't know."
"""


async def run_validation(
    *,
    mode: ValidationMode,
    strictness: Strictness,
    response_text: str,
    history: list[dict[str, Any]],
    auditor_provider: LLMProvider | None,
    auditor_provider_name: str | None,
    auditor_model: str | None,
) -> ValidationResult:
    """Run the configured validation pipeline against an assistant
    response.

    ``mode`` selects which layers to run:
      - ``citation`` — deterministic only.
      - ``audit`` — auditor LLM only.
      - ``hybrid`` — citation first; auditor when citation flags issues.

    ``auditor_provider`` may be ``None`` when ``mode`` is ``citation``
    (no auditor needed).  When the auditor needs to run but no provider
    was supplied the auditor step is skipped and a warning is recorded.
    """
    tool_results_by_id = index_tool_results(history)
    issues: list[ValidationIssue] = []
    citations_total = 0
    citations_verified = 0
    auditor_used = False
    auditor_model_used: str | None = None

    needs_audit = False

    if mode in ("citation", "hybrid"):
        cites, citation_issues = verify_citations(response_text, tool_results_by_id)
        citations_total = len(cites)
        citations_verified = citations_total - sum(
            1 for ci in citation_issues if ci.severity == "error"
        )
        for ci in citation_issues:
            issues.append(_promote(ci))
        if mode == "hybrid":
            needs_audit = any(ci.severity == "error" for ci in citation_issues)
            if strictness == "strict" and citations_total == 0 and response_text.strip():
                # In strict mode, no citations on a non-trivial response
                # is itself a reason to invoke the auditor.
                needs_audit = True

    if mode == "audit":
        needs_audit = True

    if needs_audit:
        if auditor_provider is None:
            issues.append(
                ValidationIssue(
                    kind="auditor_unavailable",
                    severity="warning",
                    claim="",
                    reason="Auditor LLM requested but no auditor_provider was configured.",
                    source="auditor",
                )
            )
        else:
            auditor_model_used = auditor_model or DEFAULT_AUDITOR_MODELS.get(
                auditor_provider_name or "", ""
            )
            if not auditor_model_used:
                issues.append(
                    ValidationIssue(
                        kind="auditor_unavailable",
                        severity="warning",
                        claim="",
                        reason=(
                            "No default auditor model for provider "
                            f"{auditor_provider_name!r}; pass auditor_model explicitly."
                        ),
                        source="auditor",
                    )
                )
            else:
                auditor_used = True
                stripped = strip_citations(response_text)
                audit_issues = await run_auditor(
                    provider=auditor_provider,
                    model=auditor_model_used,
                    response_text=stripped,
                    tool_results_by_id=tool_results_by_id,
                )
                issues.extend(audit_issues)

    has_errors = any(i.severity == "error" for i in issues)
    return ValidationResult(
        method=mode if mode != "off" else "citation",  # caller never passes "off"
        passed=not has_errors,
        citations_total=citations_total,
        citations_verified=citations_verified,
        issues=issues,
        auditor_used=auditor_used,
        auditor_model=auditor_model_used,
    )


def build_retry_feedback(result: ValidationResult) -> str:
    """Format a validation result as a user-message prompt asking the
    model to revise its previous answer.

    Includes the explicit "OK not to know" reminder so the model
    doesn't double down on a fabrication when it can't find support.
    """
    bullets: list[str] = []
    for issue in result.issues:
        if issue.severity != "error":
            continue
        prefix = f"- ({issue.kind})"
        detail_parts = []
        if issue.claim:
            detail_parts.append(f"claim: {issue.claim!r}")
        if issue.tool_use_id:
            detail_parts.append(f"tu: {issue.tool_use_id}")
        if issue.path:
            detail_parts.append(f"path: {issue.path}")
        if issue.expected is not None:
            detail_parts.append(f"actual data: {issue.expected!r}")
        if issue.reason:
            detail_parts.append(issue.reason)
        bullets.append(f"{prefix} " + " — ".join(detail_parts))

    body = "\n".join(bullets) if bullets else "- general fidelity issue"
    return (
        "[validation] Your previous response contained issues that could not be "
        "verified against the tool results:\n"
        f"{body}\n\n"
        "Please revise.  If the data does not contain what was asked, say so "
        "directly — it is always better to acknowledge uncertainty than to "
        "guess.  Use <cite tu=\"...\" path=\"...\">value</cite> tags for any "
        "specific values you reference."
    )


def _promote(citation_issue: CitationIssue) -> ValidationIssue:
    return ValidationIssue(
        kind=citation_issue.kind,
        severity=citation_issue.severity,
        claim=citation_issue.claim,
        reason=citation_issue.reason,
        source="citation",
        tool_use_id=citation_issue.tool_use_id,
        path=citation_issue.path,
        expected=citation_issue.expected,
    )
