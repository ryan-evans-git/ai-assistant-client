"""Tiny JSONPath subset for citation verification.

Adding the full ``jsonpath-ng`` dependency for the few path forms we
actually need would inflate the dependency surface (and its CVE
exposure).  The grammar supported here covers the common cases the
agent will produce:

    $                       — root document
    $.foo                   — child key
    $.foo.bar               — nested key
    $.foo[0]                — array index
    $.foo[-1]               — negative array index (last)
    $.foo[*]                — all array elements (returns a list)
    $.foo[*].bar            — projection through arrays
    $.foo[?(@.id==42)]      — equality predicate (numeric or quoted string)
    $.foo|length            — pipe filter: length of a list/string
    $.foo|sum               — pipe filter: numeric sum of a list
    $.foo|count             — alias for length

Returns ``None`` when the path doesn't resolve.  Multiple results from
``[*]`` projections are returned as a Python list.
"""

from __future__ import annotations

import re
from typing import Any


_FILTERS = {
    "length": lambda v: len(v) if hasattr(v, "__len__") else None,
    "count": lambda v: len(v) if hasattr(v, "__len__") else None,
    "sum": lambda v: sum(x for x in v if isinstance(x, (int, float)))
    if isinstance(v, list)
    else None,
}

_BRACKET_RE = re.compile(
    r"""
    ^                                  # anchor
    (?:
        (?P<index>-?\d+)               # numeric index (allow negative)
      | (?P<wildcard>\*)               # wildcard
      | \?\(@\.(?P<key>[A-Za-z_]\w*)   # ?(@.key
        \s*(?P<op>==|!=|<=|>=|<|>)\s*  # comparator
        (?P<val>(?:"[^"]*"|'[^']*'|-?\d+(?:\.\d+)?))  # quoted str / number
        \)
      | "(?P<dq_key>[^"]+)"            # bracket-quoted key
      | '(?P<sq_key>[^']+)'
    )
    $
    """,
    re.VERBOSE,
)


def resolve(data: Any, path: str) -> Any:
    """Evaluate ``path`` against ``data``.

    Returns the matched value(s) or ``None`` when the path doesn't
    resolve.  ``None`` is also a legitimate return for paths that
    point at a JSON ``null`` — the caller distinguishes the two by
    catching :exc:`KeyError`/`IndexError` upstream if it cares
    (we don't, for verification purposes).
    """
    if not path or not path.startswith("$"):
        return None

    # Pipe filter (length/sum/count) — applied to the result of the
    # base path.
    base, _, filt = path.partition("|")
    base = base.strip()
    filt = filt.strip()

    cursor = data
    for segment in _segments(base):
        cursor = _step(cursor, segment)
        if cursor is None:
            return None

    if filt:
        fn = _FILTERS.get(filt)
        if fn is None:
            return None
        return fn(cursor)
    return cursor


def _segments(path: str) -> list[str]:
    """Split ``$.a.b[0][*]`` into ``["a", "b", "[0]", "[*]"]``.

    Bracket expressions stay glued to their leading ``[`` so
    :func:`_step` can dispatch on the shape.
    """
    body = path[1:]  # drop the leading $
    if body.startswith("."):
        body = body[1:]
    if not body:
        return []
    out: list[str] = []
    current = ""
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == ".":
            if current:
                out.append(current)
                current = ""
            i += 1
            continue
        if ch == "[":
            if current:
                out.append(current)
                current = ""
            # Find matching ]; bracket exprs we care about don't nest.
            close = body.find("]", i + 1)
            if close == -1:
                return []
            out.append(body[i : close + 1])
            i = close + 1
            continue
        current += ch
        i += 1
    if current:
        out.append(current)
    return out


def _step(cursor: Any, segment: str) -> Any:
    if segment.startswith("[") and segment.endswith("]"):
        return _bracket_step(cursor, segment[1:-1])
    # Plain key.
    if isinstance(cursor, list):
        # Projecting a key through a list, e.g. "$.invoices.amount"
        # after a previous wildcard step left us with a list.
        return [_step(item, segment) for item in cursor if isinstance(item, dict) and segment in item]
    if isinstance(cursor, dict):
        return cursor.get(segment)
    return None


def _bracket_step(cursor: Any, expr: str) -> Any:
    m = _BRACKET_RE.match(expr.strip())
    if not m:
        return None
    if m.group("wildcard"):
        if isinstance(cursor, list):
            return list(cursor)
        if isinstance(cursor, dict):
            return list(cursor.values())
        return None
    if m.group("index") is not None:
        if not isinstance(cursor, list):
            return None
        idx = int(m.group("index"))
        try:
            return cursor[idx]
        except IndexError:
            return None
    if m.group("dq_key") or m.group("sq_key"):
        key = m.group("dq_key") or m.group("sq_key")
        if isinstance(cursor, dict):
            return cursor.get(key)
        return None
    if m.group("key"):
        # Predicate filter on a list of objects.
        if not isinstance(cursor, list):
            return None
        op = m.group("op")
        raw = m.group("val")
        rhs: Any
        if raw.startswith(('"', "'")):
            rhs = raw[1:-1]
        elif "." in raw:
            rhs = float(raw)
        else:
            rhs = int(raw)
        return [item for item in cursor if isinstance(item, dict) and _cmp(item.get(m.group("key")), op, rhs)]
    return None


def _cmp(left: Any, op: str, right: Any) -> bool:
    try:
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right  # type: ignore[operator]
        if op == "<=":
            return left <= right  # type: ignore[operator]
        if op == ">":
            return left > right  # type: ignore[operator]
        if op == ">=":
            return left >= right  # type: ignore[operator]
    except TypeError:
        return False
    return False
