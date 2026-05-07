"""VisualSpec validation: schema, chart-shape rules, image src safety."""

from __future__ import annotations

import base64

import pytest

from ai_assistant_client.visuals import (
    DEFAULT_MAX_IMAGE_DATA_URI_KB,
    InvalidVisualSpecError,
    SCHEMA_VERSION,
    validate_envelope,
    validate_image_src,
)


def _envelope(spec: dict) -> dict:
    return {"schema_version": SCHEMA_VERSION, "spec": spec}


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def test_bar_chart_passes() -> None:
    env = validate_envelope(
        _envelope(
            {
                "kind": "chart",
                "chart_type": "bar",
                "data": [{"month": "Jan", "revenue": 100}],
                "x_key": "month",
                "y_keys": ["revenue"],
            }
        )
    )
    assert env.spec.kind == "chart"


def test_pie_chart_requires_label_value_rows() -> None:
    with pytest.raises(InvalidVisualSpecError, match="label.*value"):
        validate_envelope(
            _envelope(
                {
                    "kind": "chart",
                    "chart_type": "pie",
                    "data": [{"month": "Jan", "revenue": 100}],
                }
            )
        )


def test_cartesian_chart_requires_x_key() -> None:
    with pytest.raises(InvalidVisualSpecError, match="x_key"):
        validate_envelope(
            _envelope(
                {
                    "kind": "chart",
                    "chart_type": "line",
                    "data": [{"month": "Jan", "revenue": 100}],
                    "y_keys": ["revenue"],
                }
            )
        )


def test_chart_row_missing_declared_y_key() -> None:
    with pytest.raises(InvalidVisualSpecError, match="cost"):
        validate_envelope(
            _envelope(
                {
                    "kind": "chart",
                    "chart_type": "bar",
                    "data": [{"month": "Jan", "revenue": 100}],
                    "x_key": "month",
                    "y_keys": ["revenue", "cost"],
                }
            )
        )


# ---------------------------------------------------------------------------
# Table + KPI
# ---------------------------------------------------------------------------


def test_table_spec_passes() -> None:
    env = validate_envelope(
        _envelope(
            {
                "kind": "table",
                "columns": [
                    {"key": "id", "label": "ID"},
                    {"key": "amt", "label": "Amount", "type": "currency", "align": "right"},
                ],
                "rows": [{"id": "1", "amt": 100}],
            }
        )
    )
    assert env.spec.kind == "table"


def test_kpi_spec_with_trend() -> None:
    env = validate_envelope(
        _envelope(
            {
                "kind": "kpi",
                "label": "Open invoices",
                "value": "$4,340",
                "trend": {"direction": "up", "delta": "+12%"},
                "status": "warn",
            }
        )
    )
    assert env.spec.kind == "kpi"
    assert env.spec.trend is not None
    assert env.spec.trend.direction == "up"


# ---------------------------------------------------------------------------
# Image src safety
# ---------------------------------------------------------------------------


def test_https_image_accepted() -> None:
    validate_image_src("https://example.com/x.png")


def test_http_image_rejected() -> None:
    with pytest.raises(InvalidVisualSpecError):
        validate_image_src("http://example.com/x.png")


def test_javascript_uri_rejected() -> None:
    with pytest.raises(InvalidVisualSpecError):
        validate_image_src("javascript:alert(1)")


def test_file_uri_rejected() -> None:
    with pytest.raises(InvalidVisualSpecError):
        validate_image_src("file:///etc/passwd")


def test_data_image_png_accepted() -> None:
    payload = base64.b64encode(b"hello").decode()
    validate_image_src(f"data:image/png;base64,{payload}")


def test_data_image_svg_rejected() -> None:
    payload = base64.b64encode(b"<svg/>").decode()
    with pytest.raises(InvalidVisualSpecError):
        validate_image_src(f"data:image/svg+xml;base64,{payload}")


def test_data_uri_oversize_rejected() -> None:
    # 1 KB cap; payload of 2 KB decoded should fail.
    payload = base64.b64encode(b"x" * 2048).decode()
    with pytest.raises(InvalidVisualSpecError, match="cap"):
        validate_image_src(f"data:image/png;base64,{payload}", max_image_data_uri_kb=1)


def test_data_uri_corrupt_base64_rejected() -> None:
    with pytest.raises(InvalidVisualSpecError):
        validate_image_src("data:image/png;base64,!!!not-valid!!!")


def test_image_envelope_threads_size_cap() -> None:
    payload = base64.b64encode(b"x" * 4096).decode()
    with pytest.raises(InvalidVisualSpecError, match="cap"):
        validate_envelope(
            _envelope(
                {"kind": "image", "src": f"data:image/png;base64,{payload}", "alt": "x"}
            ),
            max_image_data_uri_kb=1,
        )


# ---------------------------------------------------------------------------
# Envelope-level
# ---------------------------------------------------------------------------


def test_unknown_schema_version_rejected() -> None:
    with pytest.raises(InvalidVisualSpecError, match="schema_version"):
        validate_envelope(
            {
                "schema_version": 999,
                "spec": {
                    "kind": "kpi", "label": "x", "value": 1,
                },
            }
        )


def test_unknown_kind_rejected() -> None:
    with pytest.raises(InvalidVisualSpecError):
        validate_envelope(_envelope({"kind": "spaceship", "thrust": "max"}))


def test_default_size_cap_is_5mb() -> None:
    assert DEFAULT_MAX_IMAGE_DATA_URI_KB == 5 * 1024
