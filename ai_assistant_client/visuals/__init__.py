"""Visuals contract: shared spec for assistant-rendered visuals.

Renderable visuals (charts, tables, KPI tiles, images) are emitted by
the LLM via the ``render_visual(kind, spec)`` meta-tool registered in
:mod:`ai_assistant_client.discovery`.  The agent loop validates the
spec through these Pydantic models, then surfaces it to the host UI
as an ``AgentEvent("visual", ...)`` and persists a ``VisualBlock`` in
the assistant message in history.

The wire shape lives in :doc:`VISUAL_SPEC.md` at the repo root and is
duplicated identically in ``ai-assistant-ui`` (Zod-validated there).
Any change here requires a coordinated bump of ``SCHEMA_VERSION`` on
both sides.
"""

from __future__ import annotations

from ai_assistant_client.visuals.types import (
    SCHEMA_VERSION,
    ChartSpec,
    ImageSpec,
    KpiSpec,
    KpiTrend,
    TableColumn,
    TableSpec,
    VisualEnvelope,
    VisualSpec,
)
from ai_assistant_client.visuals.validate import (
    DEFAULT_MAX_IMAGE_DATA_URI_KB,
    InvalidVisualSpecError,
    validate_envelope,
    validate_image_src,
)

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_MAX_IMAGE_DATA_URI_KB",
    "ChartSpec",
    "ImageSpec",
    "InvalidVisualSpecError",
    "KpiSpec",
    "KpiTrend",
    "TableColumn",
    "TableSpec",
    "VisualEnvelope",
    "VisualSpec",
    "validate_envelope",
    "validate_image_src",
]
