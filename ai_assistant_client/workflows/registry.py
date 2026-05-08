"""Holder for loaded workflows.

Kept small + free-standing so the agent loop can ask "is ``name`` a
workflow?" without poking the catalog dict directly.  Workflows are
also surfaced as :class:`RemoteToolDescriptor` entries through
:meth:`as_descriptors` so they appear in
:class:`ProgressiveToolRegistry`'s LLM-facing catalog alongside MCP
tools.
"""

from __future__ import annotations

from collections.abc import Iterable

from ai_assistant_client.discovery import RemoteToolDescriptor
from ai_assistant_client.workflows.decorator import Workflow


class WorkflowRegistry:
    """Lookup-by-name index for loaded workflows."""

    def __init__(self, workflows: Iterable[Workflow] = ()) -> None:
        descriptors = list(workflows)
        names = [w.name for w in descriptors]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate workflow names: {names}")
        self._workflows: dict[str, Workflow] = {w.name: w for w in descriptors}

    def __contains__(self, name: str) -> bool:
        return name in self._workflows

    def __len__(self) -> int:
        return len(self._workflows)

    def get(self, name: str) -> Workflow | None:
        return self._workflows.get(name)

    def names(self) -> list[str]:
        return list(self._workflows.keys())

    def as_descriptors(self) -> list[RemoteToolDescriptor]:
        """Each workflow rendered as a catalog descriptor — same
        shape as MCP-derived tools, so :class:`ProgressiveToolRegistry`
        can intermix them seamlessly.

        Workflows never carry a ``hitl`` block at the descriptor
        level: their pauses happen *inside* the body, not before
        dispatch.  The agent loop short-circuits the per-tool gate
        for workflow names and lets the runtime drive its own pauses.
        """
        return [
            RemoteToolDescriptor(
                name=w.name,
                description=w.description,
                input_schema=w.input_schema,
                tags=w.tags,
            )
            for w in self._workflows.values()
        ]
