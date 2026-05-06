"""Auditor-LLM fallback for the hybrid validation pipeline.

When the deterministic citation verifier finds an issue (broken
citation, value mismatch, or — in strict mode — an uncited concrete
value), we ask a separate, cheap LLM to read the assistant's response
and the relevant tool results, and to flag any claim that isn't
supported by the data.

The auditor reuses the existing :class:`LLMProvider` abstraction, so
it works across all four supported providers without per-vendor code.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ai_assistant_client.llm import LLMProvider, MessageStop, TextDelta
from ai_assistant_client.validation.types import ValidationIssue


log = logging.getLogger(__name__)


# Hard-coded "cheapest in family" defaults — overridable per call via
# ``AgentRunConfig.auditor_model``.  These IDs are conservative
# (non-cross-region for Bedrock; stable model names) so the default
# works without region-specific configuration.
DEFAULT_AUDITOR_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash-lite",
    "bedrock": "anthropic.claude-3-5-haiku-20241022-v1:0",
}


_AUDITOR_SYSTEM_PROMPT = """\
You audit AI assistant responses for factual fidelity to tool data.

You will be given:
  1. The assistant's response (the user-visible text).
  2. The tool results the assistant retrieved during its turn.

Your job is to identify any specific claim in the response — numeric
values, dates, names, statuses, IDs, or direct quotes — that is NOT
supported by, or CONTRADICTS, the tool results.

Output JSON only.  No prose, no explanation, no preamble.  Schema:

  {
    "issues": [
      {"claim": "<exact text from the response>",
       "severity": "error" | "warning",
       "reason": "<short explanation>"}
    ]
  }

If nothing is wrong, return {"issues": []}.

Use "error" for direct contradictions (response says X, data shows Y)
and for fabricated values (response asserts X, no matching data).
Use "warning" for vague paraphrases you cannot verify either way.

Do NOT flag:
  - Conversational filler ("Sure, let me check that.")
  - The assistant's own inferences or recommendations.
  - Aggregate calculations that are mathematically derivable from the
    tool data.
"""


async def run_auditor(
    *,
    provider: LLMProvider,
    model: str,
    response_text: str,
    tool_results_by_id: dict[str, Any],
    max_tokens: int = 2048,
) -> list[ValidationIssue]:
    """Call the auditor LLM and parse its issues list.

    On any auditor failure (JSON-parse error, transport error,
    timeout) we log and return an empty list — the auditor's job is
    *additional* signal, never a blocker for the primary turn.
    """
    user_message = _build_user_message(response_text, tool_results_by_id)
    accumulated: list[str] = []
    try:
        async for event in provider.stream_turn(
            model=model,
            max_tokens=max_tokens,
            system=_AUDITOR_SYSTEM_PROMPT,
            tools=[],
            messages=[{"role": "user", "content": user_message}],
            cache_hint=None,
        ):
            if isinstance(event, TextDelta):
                accumulated.append(event.text)
            elif isinstance(event, MessageStop):
                break
    except Exception:  # noqa: BLE001
        log.exception("Auditor LLM call failed; skipping audit")
        return []

    return _parse_auditor_output("".join(accumulated))


def _build_user_message(
    response_text: str, tool_results_by_id: dict[str, Any]
) -> str:
    serialized_results = json.dumps(tool_results_by_id, default=str, indent=2)
    return (
        "ASSISTANT_RESPONSE:\n"
        f"{response_text}\n\n"
        "TOOL_RESULTS (keyed by tool_use_id):\n"
        f"{serialized_results}\n"
    )


def _parse_auditor_output(text: str) -> list[ValidationIssue]:
    """Pull a JSON ``{"issues": [...]}`` object out of the auditor's
    raw text response.  Tolerates leading/trailing prose despite the
    system prompt's strictness — we slice from the first ``{`` to the
    last ``}`` and try ``json.loads``.
    """
    text = text.strip()
    if not text:
        return []
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except (TypeError, ValueError):
        return []
    raw_issues = payload.get("issues") if isinstance(payload, dict) else None
    if not isinstance(raw_issues, list):
        return []
    out: list[ValidationIssue] = []
    for entry in raw_issues:
        if not isinstance(entry, dict):
            continue
        claim = str(entry.get("claim") or "")
        reason = str(entry.get("reason") or "")
        severity_raw = str(entry.get("severity") or "warning").lower()
        severity = severity_raw if severity_raw in ("error", "warning") else "warning"
        out.append(
            ValidationIssue(
                kind="auditor_finding",
                severity=severity,  # type: ignore[arg-type]
                claim=claim,
                reason=reason,
                source="auditor",
            )
        )
    return out
