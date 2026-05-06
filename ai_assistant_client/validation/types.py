"""Shared dataclasses for the validation pipeline.

Kept in their own module so the citation parser, normalizer, verifier,
auditor, and agent loop can all import without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class Citation:
    """One ``<cite tu="..." path="...">value</cite>`` tag pulled from
    the assistant's response."""

    tool_use_id: str
    path: str
    displayed_text: str
    # Character offsets in the original message text.  Useful for
    # hosts that want to highlight or strip citations in the rendered
    # output.
    start: int
    end: int


@dataclass(frozen=True)
class CitationIssue:
    """One problem found by the deterministic citation verifier."""

    kind: Literal[
        "value_mismatch",
        "broken_citation",
        "unknown_tool_use_id",
        "non_json_substring_miss",
        "uncited_claim",
    ]
    severity: Severity
    claim: str
    reason: str
    tool_use_id: str | None = None
    path: str | None = None
    expected: Any = None


@dataclass(frozen=True)
class ValidationIssue:
    """Provider-neutral issue surfaced to the host.

    Wraps both citation-verifier issues and auditor-LLM issues in a
    single shape so the host doesn't need to branch.
    """

    kind: str  # mirror CitationIssue.kind, plus auditor kinds
    severity: Severity
    claim: str
    reason: str
    source: Literal["citation", "auditor"] = "citation"
    tool_use_id: str | None = None
    path: str | None = None
    expected: Any = None


@dataclass
class ValidationResult:
    """Surfaced as ``AgentEvent("validation", ...)``."""

    method: Literal["citation", "audit", "hybrid"]
    passed: bool
    citations_total: int = 0
    citations_verified: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    auditor_used: bool = False
    auditor_model: str | None = None

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)
