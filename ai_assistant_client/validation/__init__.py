"""Hybrid response validation: citation tracing + auditor LLM fallback.

The agent loop (when ``AgentRunConfig.validation_mode != "off"``) runs
each completed assistant turn through this package before yielding
``turn_complete``.  Two layers of defense:

1. **Citation tracing** (deterministic).  The model is asked to wrap
   any tool-derived value in an inline ``<cite tu="..." path="...">``
   tag.  :func:`verify_citations` parses each tag, evaluates the
   JSONPath against the referenced ``tool_result`` content, and
   normalizes both the displayed value and the actual value before
   comparing.

2. **Auditor LLM** (semantic).  When citation tracing flags an issue
   — broken citation, value mismatch, or (in strict mode) an uncited
   concrete value — the auditor LLM gets the response + the relevant
   tool results and is asked to identify any unsupported claim.

Public API consumed by the agent loop:

    :func:`run_validation` — top-level orchestrator (citation only,
        audit only, or hybrid).
    :class:`ValidationResult` — structured output, surfaced as an
        ``AgentEvent("validation", ...)``.
"""

from __future__ import annotations

from ai_assistant_client.validation.orchestrator import (
    CITATION_INSTRUCTIONS,
    build_retry_feedback,
    run_validation,
)
from ai_assistant_client.validation.types import (
    Citation,
    CitationIssue,
    Severity,
    ValidationIssue,
    ValidationResult,
)
from ai_assistant_client.validation.verify import index_tool_results

__all__ = [
    "CITATION_INSTRUCTIONS",
    "Citation",
    "CitationIssue",
    "Severity",
    "ValidationIssue",
    "ValidationResult",
    "build_retry_feedback",
    "index_tool_results",
    "run_validation",
]
