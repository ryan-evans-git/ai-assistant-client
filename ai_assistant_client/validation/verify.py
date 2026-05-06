"""Verify each citation in an assistant response against the actual
tool_result data it claims to come from.

The verifier never throws on bad input — it produces structured
:class:`CitationIssue` records that the agent loop turns into
``ValidationResult.issues``.
"""

from __future__ import annotations

import json
from typing import Any

from ai_assistant_client.validation.citations import Citation, parse_citations
from ai_assistant_client.validation.jsonpath import resolve
from ai_assistant_client.validation.normalize import values_match
from ai_assistant_client.validation.types import CitationIssue


def index_tool_results(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a ``tool_use_id → parsed-content`` map from the history.

    Tool results live in user-role messages as ``{"type": "tool_result",
    "tool_use_id": "...", "content": "..."}``.  Content is typically a
    JSON-serialized string (the agent loop passes ``_to_text`` results
    through ``json.dumps``); we parse it back so JSONPath can see
    structure.  Non-JSON content is preserved as a raw string for
    substring fallback.
    """
    out: dict[str, Any] = {}
    for message in history:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tu_id = block.get("tool_use_id")
            if not tu_id:
                continue
            raw = block.get("content")
            out[tu_id] = _try_json(raw)
    return out


def _try_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def verify_citations(
    response_text: str,
    tool_results_by_id: dict[str, Any],
) -> tuple[list[Citation], list[CitationIssue]]:
    """Return ``(parsed_citations, issues)`` for ``response_text``.

    ``issues`` includes both verifiable failures (``value_mismatch``,
    ``broken_citation``, ``unknown_tool_use_id``) and substring-fallback
    misses (``non_json_substring_miss``) when a citation references a
    non-JSON tool result.
    """
    citations = parse_citations(response_text)
    issues: list[CitationIssue] = []

    for cite in citations:
        if cite.tool_use_id not in tool_results_by_id:
            issues.append(
                CitationIssue(
                    kind="unknown_tool_use_id",
                    severity="error",
                    claim=cite.displayed_text,
                    reason=f"No tool_result with id {cite.tool_use_id!r} in history.",
                    tool_use_id=cite.tool_use_id,
                    path=cite.path,
                )
            )
            continue

        body = tool_results_by_id[cite.tool_use_id]

        if isinstance(body, str):
            # Non-JSON tool result: substring contains.
            if cite.displayed_text and cite.displayed_text in body:
                continue
            issues.append(
                CitationIssue(
                    kind="non_json_substring_miss",
                    severity="warning",
                    claim=cite.displayed_text,
                    reason=(
                        "Tool result is free text; the displayed value "
                        "does not appear as a substring."
                    ),
                    tool_use_id=cite.tool_use_id,
                    path=cite.path,
                )
            )
            continue

        actual = resolve(body, cite.path)
        if actual is None and not _path_legitimately_returns_none(body, cite.path):
            issues.append(
                CitationIssue(
                    kind="broken_citation",
                    severity="error",
                    claim=cite.displayed_text,
                    reason=f"Path {cite.path!r} does not resolve in tool_result.",
                    tool_use_id=cite.tool_use_id,
                    path=cite.path,
                )
            )
            continue

        if not values_match(cite.displayed_text, actual):
            issues.append(
                CitationIssue(
                    kind="value_mismatch",
                    severity="error",
                    claim=cite.displayed_text,
                    reason=(
                        f"Path {cite.path!r} resolves to {actual!r}; "
                        f"the displayed value {cite.displayed_text!r} does not match."
                    ),
                    tool_use_id=cite.tool_use_id,
                    path=cite.path,
                    expected=actual,
                )
            )

    return citations, issues


def _path_legitimately_returns_none(body: Any, path: str) -> bool:
    """A path that actually points at a JSON ``null`` is legitimately
    ``None``-valued — distinguish it from "path didn't resolve".

    Cheap heuristic: walk one step at a time using the same segmenter
    the resolver uses; if every prefix exists, the terminal ``None`` is
    a real null rather than a missing key.
    """
    # Re-import locally to avoid widening jsonpath's public surface.
    from ai_assistant_client.validation.jsonpath import _segments  # type: ignore[attr-defined]

    cursor = body
    segments = _segments(path.split("|", 1)[0].strip())
    if not segments:
        return cursor is None
    for seg in segments:
        if cursor is None:
            return False
        if seg.startswith("[") and seg.endswith("]"):
            inner = seg[1:-1]
            if inner.lstrip("-").isdigit():
                if not isinstance(cursor, list):
                    return False
                idx = int(inner)
                if not -len(cursor) <= idx < len(cursor):
                    return False
                cursor = cursor[idx]
                continue
            # Wildcards / predicates aren't worth special-casing here —
            # treat them as "not legitimately None" (i.e., flag as broken
            # if they resolve to None, which usually means an empty match).
            return False
        if not isinstance(cursor, dict) or seg not in cursor:
            return False
        cursor = cursor[seg]
    return cursor is None
