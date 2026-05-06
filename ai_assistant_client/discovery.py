"""Progressive tool discovery for large MCP catalogs.

The naive approach to MCP — load every upstream tool's schema into
the system prompt — burns context budget linearly with the catalog
size.  At ~50 tools you're already paying tens of thousands of tokens
on every turn even when only one tool is relevant.

This module exposes only **two** tools to Claude:

    tool_search(query, max_results=5)
        Returns the top-N matches by name + summary.  Cheap.

    tool_load(names)
        Returns the full input schemas for the named tools and
        marks them as *loaded* — they show up alongside
        ``tool_search`` / ``tool_load`` on the next turn.

The model's loop becomes: search → load → invoke.  The full catalog
isn't sent at any single turn.

Loaded tools persist for the lifetime of a :class:`ProgressiveToolRegistry`
instance — typically one chat session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None  # type: ignore[assignment]


log = logging.getLogger(__name__)


_META_TOOL_SEARCH_NAME = "tool_search"
_META_TOOL_LOAD_NAME = "tool_load"


@dataclass(frozen=True)
class RemoteToolDescriptor:
    """A tool the registry knows about but may not have surfaced yet."""

    name: str
    description: str
    input_schema: dict[str, Any]
    tags: tuple[str, ...] = ()


class ProgressiveToolRegistry:
    """Tracks the catalog + which tools have been loaded this session.

    Construct with the full catalog.  Pass :meth:`anthropic_tools` as
    the ``tools=`` parameter on each Claude call.  Route any
    ``tool_search`` / ``tool_load`` tool-use blocks back through
    :meth:`handle_meta_call` before dispatching anything else.
    """

    def __init__(
        self,
        catalog: Iterable[RemoteToolDescriptor],
        *,
        search_max_results: int = 5,
    ) -> None:
        descriptors = list(catalog)
        names = [d.name for d in descriptors]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate tool names in catalog")
        self._catalog: dict[str, RemoteToolDescriptor] = {
            d.name: d for d in descriptors
        }
        self._loaded: set[str] = set()
        self._search_max_results = search_max_results

    # -- Public surface ------------------------------------------------

    @property
    def loaded_tools(self) -> set[str]:
        return set(self._loaded)

    def reset_loaded(self) -> None:
        self._loaded.clear()

    def is_meta_tool(self, name: str) -> bool:
        return name in (_META_TOOL_SEARCH_NAME, _META_TOOL_LOAD_NAME)

    def is_known_tool(self, name: str) -> bool:
        return name in self._catalog or self.is_meta_tool(name)

    def anthropic_tools(self) -> list[dict[str, Any]]:
        """Tools to include in the next ``messages.create`` call.

        Always includes the two meta-tools.  Includes the full
        schema of every tool the model has explicitly loaded
        this session.
        """
        tools: list[dict[str, Any]] = [self._tool_search_schema(), self._tool_load_schema()]
        for name in sorted(self._loaded):
            descriptor = self._catalog.get(name)
            if descriptor is None:
                continue
            tools.append(
                {
                    "name": descriptor.name,
                    "description": descriptor.description,
                    "input_schema": descriptor.input_schema,
                }
            )
        return tools

    def handle_meta_call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a meta-tool call locally.  Returns the tool_result content."""
        if name == _META_TOOL_SEARCH_NAME:
            return self._do_search(arguments)
        if name == _META_TOOL_LOAD_NAME:
            return self._do_load(arguments)
        raise KeyError(f"Not a meta tool: {name}")

    # -- Search --------------------------------------------------------

    def _do_search(self, args: dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        max_results = int(args.get("max_results") or self._search_max_results)
        max_results = max(1, min(max_results, 25))
        if not query:
            # Empty query: surface a stable subset (first N
            # alphabetically) so the model can browse.
            ordered = sorted(self._catalog.values(), key=lambda d: d.name)
            results = ordered[:max_results]
        else:
            results = self._rank(query, max_results)
        lines = []
        for d in results:
            summary = _first_line(d.description)
            lines.append(
                f"- {d.name}: {summary}"
                + (f"  [tags: {', '.join(d.tags)}]" if d.tags else "")
            )
        if not lines:
            return "No tools matched."
        return "\n".join(lines)

    def _rank(self, query: str, max_results: int) -> list[RemoteToolDescriptor]:
        scored: list[tuple[float, RemoteToolDescriptor]] = []
        q = query.lower()
        for d in self._catalog.values():
            score = self._score(q, d)
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [d for _, d in scored[:max_results]]

    def _score(self, query: str, descriptor: RemoteToolDescriptor) -> float:
        name_score = _fuzzy(query, descriptor.name.lower())
        desc_score = _fuzzy(query, descriptor.description.lower())
        tag_score = max(
            (_fuzzy(query, t.lower()) for t in descriptor.tags),
            default=0.0,
        )
        # Name matches dominate; description is supporting context.
        return max(name_score, tag_score * 0.95, desc_score * 0.6)

    # -- Load ----------------------------------------------------------

    def _do_load(self, args: dict[str, Any]) -> str:
        raw_names = args.get("names") or []
        if isinstance(raw_names, str):
            raw_names = [raw_names]
        names = [str(n) for n in raw_names if n]
        if not names:
            return "tool_load requires a non-empty 'names' array."
        loaded: list[str] = []
        unknown: list[str] = []
        for n in names:
            descriptor = self._catalog.get(n)
            if descriptor is None:
                unknown.append(n)
                continue
            self._loaded.add(n)
            loaded.append(n)
        parts: list[str] = []
        if loaded:
            parts.append(
                "Loaded for this session:\n"
                + "\n".join(f"- {n}" for n in loaded)
                + "\n\nThe full schemas are now available; call them directly."
            )
        if unknown:
            parts.append("Unknown tool name(s): " + ", ".join(unknown))
        return "\n\n".join(parts)

    # -- Meta-tool schemas --------------------------------------------

    def _tool_search_schema(self) -> dict[str, Any]:
        return {
            "name": _META_TOOL_SEARCH_NAME,
            "description": (
                "Search the available tool catalog. Returns up to "
                f"{self._search_max_results} matches ranked by name + "
                "description relevance. Use this to discover tools "
                "before loading their full schemas with tool_load."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Free-text search across tool names, "
                            "descriptions, and tags. Empty query "
                            "returns a stable alphabetical subset."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 25,
                        "default": self._search_max_results,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }

    def _tool_load_schema(self) -> dict[str, Any]:
        return {
            "name": _META_TOOL_LOAD_NAME,
            "description": (
                "Load the full input schemas for one or more tools "
                "discovered via tool_search. Loaded tools are "
                "available for direct invocation for the rest of "
                "this conversation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
                "required": ["names"],
                "additionalProperties": False,
            },
        }


# ---------------------------------------------------------------------------
# Fuzzy ranking
# ---------------------------------------------------------------------------


def _fuzzy(query: str, text: str) -> float:
    if not text:
        return 0.0
    if not query:
        return 0.0
    if fuzz is not None:
        # ``token_set_ratio`` handles word reordering well; works
        # for both short names and long descriptions.
        return float(fuzz.token_set_ratio(query, text))
    # Fallback: substring + word-overlap heuristic when rapidfuzz
    # isn't available (kept simple — main path is rapidfuzz).
    if query in text:
        return 100.0
    q_tokens = set(query.split())
    t_tokens = set(text.split())
    overlap = len(q_tokens & t_tokens)
    if not q_tokens:
        return 0.0
    return 100.0 * overlap / len(q_tokens)


def _first_line(text: str) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return ""
