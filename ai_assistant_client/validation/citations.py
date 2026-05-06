"""Parse inline ``<cite tu="..." path="...">value</cite>`` tags.

We use a small purpose-built tokenizer instead of pulling in a
full HTML parser — the tag shape is fixed and the surrounding text
is markdown that an HTML parser would mis-tokenize anyway.

Recognized form (whitespace tolerant inside the opening tag):

    <cite tu="<id>" path="<jsonpath>">displayed</cite>

Attributes are ``tu`` (the tool_use_id) and ``path`` (a JSONPath
expression honored by :mod:`ai_assistant_client.validation.jsonpath`).
Unknown attributes are silently ignored to leave room for future
additions (e.g. a unit hint).
"""

from __future__ import annotations

import re

from ai_assistant_client.validation.types import Citation


# Capture the full `<cite ...>...</cite>` span so we can also report
# offsets back to the host (useful when stripping or replacing tags).
# Tag attributes are matched leniently — order-insensitive, single or
# double quotes, optional whitespace.
_CITE_TAG_RE = re.compile(
    r"""
    <cite\b(?P<attrs>[^>]*)>          # opening tag with arbitrary attrs
    (?P<text>.*?)                     # displayed text (non-greedy)
    </cite>                           # closing tag
    """,
    re.VERBOSE | re.DOTALL | re.IGNORECASE,
)

_ATTR_RE = re.compile(
    r"""
    (?P<name>[A-Za-z_][\w-]*)          # attribute name
    \s*=\s*                            # =
    (?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)')  # quoted value
    """,
    re.VERBOSE,
)


def parse_citations(text: str) -> list[Citation]:
    """Return every well-formed citation in ``text`` in document order.

    Tags missing a required attribute (``tu`` or ``path``) are skipped
    silently — they're recorded as ``uncited_claim`` issues by the
    verifier rather than parser errors here.
    """
    citations: list[Citation] = []
    for match in _CITE_TAG_RE.finditer(text):
        attrs = _parse_attrs(match.group("attrs") or "")
        tu = attrs.get("tu")
        path = attrs.get("path")
        if not tu or not path:
            continue
        citations.append(
            Citation(
                tool_use_id=tu,
                path=path,
                displayed_text=match.group("text").strip(),
                start=match.start(),
                end=match.end(),
            )
        )
    return citations


def strip_citations(text: str) -> str:
    """Remove all ``<cite>`` tags, leaving only the displayed values.

    Intended for the auditor-LLM input (the auditor judges what the
    user would see, not the citation metadata).
    """
    return _CITE_TAG_RE.sub(lambda m: (m.group("text") or ""), text)


def _parse_attrs(blob: str) -> dict[str, str]:
    return {
        m.group("name").lower(): m.group("dq") if m.group("dq") is not None else m.group("sq")
        for m in _ATTR_RE.finditer(blob)
    }
