"""Spec validation + image-source safety enforcement.

Both layers convert their failures into friendly messages the agent
loop returns as ``tool_result`` content, so the LLM can correct on
the next turn rather than the host seeing a raw 500.
"""

from __future__ import annotations

import base64
import re
from typing import Any

from pydantic import ValidationError

from ai_assistant_client.visuals.types import (
    ChartSpec,
    SCHEMA_VERSION,
    VisualEnvelope,
)


DEFAULT_MAX_IMAGE_DATA_URI_KB: int = 5 * 1024  # 5 MB of decoded payload


class InvalidVisualSpecError(ValueError):
    """Raised when a render_visual call carries a malformed spec.

    The message is intentionally short + LLM-friendly — the agent loop
    surfaces it back as a tool_result so the model can revise.
    """


# ---------------------------------------------------------------------------
# Envelope validation
# ---------------------------------------------------------------------------


def validate_envelope(
    payload: dict[str, Any],
    *,
    max_image_data_uri_kb: int = DEFAULT_MAX_IMAGE_DATA_URI_KB,
) -> VisualEnvelope:
    """Validate a raw ``render_visual`` argument dict and return the
    typed envelope.  Raises :class:`InvalidVisualSpecError` on any
    failure.

    Performs three checks beyond Pydantic's structural validation:
      - Schema version is one we understand.
      - Chart specs have ``x_key`` + non-empty ``y_keys`` for cartesian
        types, or ``label``/``value`` rows for pie/donut.
      - Image specs pass :func:`validate_image_src` and are under the
        size cap.
    """
    if not isinstance(payload, dict):
        raise InvalidVisualSpecError("render_visual payload must be a JSON object.")

    try:
        envelope = VisualEnvelope.model_validate(payload)
    except ValidationError as err:
        raise InvalidVisualSpecError(
            "Spec failed schema validation: " + _condense_pydantic_error(err)
        ) from err

    if envelope.schema_version != SCHEMA_VERSION:
        raise InvalidVisualSpecError(
            f"Unsupported schema_version {envelope.schema_version}; "
            f"this server speaks v{SCHEMA_VERSION}."
        )

    spec = envelope.spec

    if isinstance(spec, ChartSpec):
        _validate_chart(spec)
    if spec.kind == "image":
        validate_image_src(spec.src, max_image_data_uri_kb=max_image_data_uri_kb)

    return envelope


def _validate_chart(spec: ChartSpec) -> None:
    if spec.chart_type in ("pie", "donut"):
        for i, row in enumerate(spec.data):
            if "label" not in row or "value" not in row:
                raise InvalidVisualSpecError(
                    f"Pie/donut row {i} must have 'label' and 'value' keys."
                )
        return
    # Cartesian charts (bar/line/area/scatter) require axis hints.
    if not spec.x_key:
        raise InvalidVisualSpecError(
            f"chart_type={spec.chart_type!r} requires 'x_key'."
        )
    if not spec.y_keys:
        raise InvalidVisualSpecError(
            f"chart_type={spec.chart_type!r} requires non-empty 'y_keys'."
        )
    for i, row in enumerate(spec.data):
        if spec.x_key not in row:
            raise InvalidVisualSpecError(
                f"Row {i} missing x_key {spec.x_key!r}."
            )
        for yk in spec.y_keys:
            if yk not in row:
                raise InvalidVisualSpecError(
                    f"Row {i} missing y_key {yk!r}."
                )


# ---------------------------------------------------------------------------
# Image source allowlist
# ---------------------------------------------------------------------------


_DATA_URI_RE = re.compile(
    r"^data:image/(?P<subtype>png|jpeg|jpg|gif|webp);base64,(?P<payload>[A-Za-z0-9+/=]+)$"
)


def validate_image_src(
    src: str,
    *,
    max_image_data_uri_kb: int = DEFAULT_MAX_IMAGE_DATA_URI_KB,
) -> None:
    """Enforce the image-src allowlist.

    Accepts:
      * ``https://`` URLs.
      * ``data:image/(png|jpeg|jpg|gif|webp);base64,<...>`` URIs under
        ``max_image_data_uri_kb`` KB of decoded payload.

    Rejects everything else, including ``http://``, ``data:image/svg+xml``,
    ``javascript:``, ``file:``, ``ftp://``, and ``data:`` URIs over the
    size cap.
    """
    if not isinstance(src, str) or not src:
        raise InvalidVisualSpecError("Image src is required.")

    if src.startswith("https://"):
        return

    match = _DATA_URI_RE.match(src)
    if not match:
        raise InvalidVisualSpecError(
            "Image src must be an https:// URL or a data:image/(png|jpeg|"
            "jpg|gif|webp);base64,... URI.  SVG data URIs are not allowed."
        )

    payload = match.group("payload")
    # base64-decoded length is roughly len*3/4; check decoded size against cap.
    try:
        decoded_bytes = len(base64.b64decode(payload, validate=True))
    except (ValueError, base64.binascii.Error) as err:  # type: ignore[attr-defined]
        raise InvalidVisualSpecError(
            "Image data URI base64 payload is malformed."
        ) from err

    if decoded_bytes > max_image_data_uri_kb * 1024:
        raise InvalidVisualSpecError(
            f"Image data URI is {decoded_bytes // 1024} KB; cap is "
            f"{max_image_data_uri_kb} KB."
        )


def _condense_pydantic_error(err: ValidationError) -> str:
    """Pydantic errors can be long.  Squash to first 3 issues."""
    issues = err.errors()[:3]
    parts = []
    for issue in issues:
        loc = ".".join(str(x) for x in issue.get("loc") or [])
        msg = issue.get("msg") or "invalid"
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)
