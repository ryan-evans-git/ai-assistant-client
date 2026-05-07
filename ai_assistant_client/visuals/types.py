"""Pydantic models for the VisualSpec wire contract.

See ``VISUAL_SPEC.md`` at the repo root for the canonical description
of each shape and the rules around versioning.  The Python models
here mirror the Zod schemas in ``ai-assistant-ui/src/visuals/schemas.ts``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


ChartType = Literal["bar", "line", "area", "pie", "donut", "scatter"]


class ChartSpec(BaseModel):
    """Cartesian-coordinate or pie/donut chart.

    For ``chart_type`` in {bar, line, area, scatter}: ``data`` is a
    list of row objects, ``x_key`` selects the categorical axis, and
    each entry in ``y_keys`` becomes a series.

    For ``chart_type`` in {pie, donut}: ``data`` rows must have
    ``label`` (str) and ``value`` (number); ``x_key``/``y_keys`` are
    ignored.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["chart"] = "chart"
    chart_type: ChartType
    title: str | None = None
    data: list[dict[str, Any]] = Field(default_factory=list)
    x_key: str | None = None
    y_keys: list[str] | None = None
    x_label: str | None = None
    y_label: str | None = None
    stacked: bool = False  # bar/area only; ignored elsewhere


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


TableColumnType = Literal["string", "number", "currency", "date", "boolean"]
TableAlign = Literal["left", "center", "right"]


class TableColumn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    type: TableColumnType = "string"
    align: TableAlign | None = None


class TableSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["table"] = "table"
    title: str | None = None
    columns: list[TableColumn]
    rows: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# KPI single-stat
# ---------------------------------------------------------------------------


KpiStatus = Literal["good", "warn", "bad", "neutral"]
TrendDirection = Literal["up", "down", "flat"]


class KpiTrend(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: TrendDirection
    delta: str | int | float
    period: str | None = None


class KpiSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["kpi"] = "kpi"
    label: str
    value: str | int | float
    unit: str | None = None
    trend: KpiTrend | None = None
    status: KpiStatus = "neutral"


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


class ImageSpec(BaseModel):
    """An image reference.  ``src`` must satisfy
    :func:`ai_assistant_client.visuals.validate.validate_image_src`
    — https URL or a non-SVG ``data:image/...;base64,...`` URI under
    the configured size cap."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["image"] = "image"
    src: str
    alt: str
    width: int | None = None
    height: int | None = None
    caption: str | None = None


# ---------------------------------------------------------------------------
# Discriminated union + envelope
# ---------------------------------------------------------------------------


VisualSpec = Annotated[
    Union[ChartSpec, TableSpec, KpiSpec, ImageSpec],
    Field(discriminator="kind"),
]


class VisualEnvelope(BaseModel):
    """The wire object surfaced to the UI.

    Carrying the version inside the envelope (rather than implicitly
    in the agent event type) lets the UI ignore unknown future
    versions gracefully without breaking the SSE contract.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    spec: VisualSpec
