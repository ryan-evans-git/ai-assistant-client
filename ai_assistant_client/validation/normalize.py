"""Type-aware comparison between a citation's displayed text and the
JSONPath-resolved value from the tool result.

The displayed text is whatever the model rendered in chat — possibly
formatted ("$1,250.00" for a numeric 1250.00, "March 15, 2026" for an
ISO timestamp).  The actual value is the parsed JSON.  We normalize
both into the same comparable form, dispatching on the actual value's
type (which we trust more than the model's prose).
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from typing import Any


def values_match(displayed: str, actual: Any) -> bool:
    """Return ``True`` when the displayed text faithfully represents
    the actual value, accounting for currency / date / count / string
    formatting differences."""
    displayed = (displayed or "").strip()
    if isinstance(actual, bool):
        return _bool_match(displayed, actual)
    if isinstance(actual, (int, float)):
        return _numeric_match(displayed, actual)
    if isinstance(actual, str):
        return _string_or_date_match(displayed, actual)
    if actual is None:
        return displayed.lower() in ("none", "null", "n/a", "—", "-")
    if isinstance(actual, (list, dict)):
        return _structural_match(displayed, actual)
    return False


# ---------------------------------------------------------------------------
# Numeric (counts, currency, percentages)
# ---------------------------------------------------------------------------


_CURRENCY_SYMBOLS = "$€£¥₹"
_NUMERIC_RE = re.compile(r"-?\d{1,3}(?:[,\d]*)(?:\.\d+)?|-?\.\d+")
_PERCENT_RE = re.compile(r"-?\d+(?:\.\d+)?\s*%")


def _numeric_match(displayed: str, actual: float | int) -> bool:
    # Percentages: "47.5%" ↔ 0.475
    if _PERCENT_RE.fullmatch(displayed.strip()):
        try:
            pct = float(displayed.strip().rstrip("%").replace(",", "")) / 100.0
            return _close(pct, float(actual))
        except ValueError:
            return False

    cleaned = displayed
    for sym in _CURRENCY_SYMBOLS:
        cleaned = cleaned.replace(sym, "")
    cleaned = cleaned.replace(",", "").strip()

    # Pull the leading numeric prefix out of phrases like "3 invoices".
    match = _NUMERIC_RE.match(cleaned)
    if match:
        try:
            return _close(float(match.group(0).replace(",", "")), float(actual))
        except ValueError:
            return False
    return False


def _close(a: float, b: float) -> bool:
    # Strict by default — currency/count comparisons should be exact.
    # 1e-9 tolerance absorbs floating-point representation drift only.
    if a == b:
        return True
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------------------
# Strings / dates
# ---------------------------------------------------------------------------


def _string_or_date_match(displayed: str, actual: str) -> bool:
    # Try date interpretation first — many string-typed JSON values
    # are actually ISO timestamps.
    actual_dt = _parse_iso_datetime(actual)
    if actual_dt is not None:
        displayed_dt = _parse_freeform_datetime(displayed)
        if displayed_dt is not None:
            return _datetimes_match(displayed_dt, actual_dt)

    # Fall back to case-insensitive trimmed equality.  Numeric ID
    # strings ("4711" ↔ 4711) are handled in _numeric_match when the
    # actual side is numeric; here we're string ↔ string.
    return _norm_str(displayed) == _norm_str(actual)


def _norm_str(s: str) -> str:
    # Collapse internal whitespace so "  Acme  Corp" matches "Acme Corp".
    return " ".join((s or "").strip().lower().split())


_ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)


def _parse_iso_datetime(s: str) -> datetime | None:
    if not isinstance(s, str) or not _ISO_DATETIME_RE.match(s.strip()):
        return None
    try:
        # ``fromisoformat`` accepts the "Z" suffix on 3.11+.
        return datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_FREEFORM_DATE_RE = re.compile(
    r"""
    (?P<month>[A-Za-z]+)\s+
    (?P<day>\d{1,2})(?:st|nd|rd|th)?
    ,?\s+
    (?P<year>\d{4})
    """,
    re.VERBOSE,
)


def _parse_freeform_datetime(s: str) -> datetime | None:
    """Best-effort parse for human-rendered dates: "March 15, 2026"."""
    s = (s or "").strip()
    iso = _parse_iso_datetime(s)
    if iso is not None:
        return iso
    m = _FREEFORM_DATE_RE.search(s)
    if not m:
        return None
    month = _MONTH_NAMES.get(m.group("month").lower())
    if month is None:
        return None
    try:
        return datetime(int(m.group("year")), month, int(m.group("day")), tzinfo=timezone.utc)
    except ValueError:
        return None


def _datetimes_match(displayed: datetime, actual: datetime) -> bool:
    # If the displayed form omitted a time component (date-only parse
    # via _FREEFORM_DATE_RE), compare at date granularity.
    if displayed.hour == 0 and displayed.minute == 0 and displayed.second == 0:
        return displayed.date() == _to_aware(actual).date()
    return _to_aware(displayed) == _to_aware(actual)


def _to_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Booleans + structural
# ---------------------------------------------------------------------------


_TRUTHY = {"true", "yes", "y", "on", "1", "✓"}
_FALSY = {"false", "no", "n", "off", "0", "✗"}


def _bool_match(displayed: str, actual: bool) -> bool:
    norm = displayed.strip().lower()
    if actual is True:
        return norm in _TRUTHY
    return norm in _FALSY


def _structural_match(displayed: str, actual: Any) -> bool:
    """Last-resort: did the model just paste the JSON back?

    We don't try to "describe a list of dicts" comparisons — that's a
    job for the auditor LLM, not deterministic verification.
    """
    try:
        parsed = json.loads(displayed)
    except (TypeError, ValueError):
        return False
    return parsed == actual
