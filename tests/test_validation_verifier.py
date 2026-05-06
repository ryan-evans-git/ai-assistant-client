"""Verifier + normalizer end-to-end behavior."""

from __future__ import annotations

import json

from ai_assistant_client.validation.normalize import values_match
from ai_assistant_client.validation.verify import index_tool_results, verify_citations


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


def test_currency_match() -> None:
    assert values_match("$1,250.00", 1250.00)
    assert values_match("$1250", 1250)
    assert values_match("€890.50", 890.50)


def test_currency_mismatch() -> None:
    assert not values_match("$1,250.00", 1500.00)


def test_count_with_trailing_text() -> None:
    assert values_match("3 invoices", 3)
    assert values_match("3", 3)
    assert not values_match("4 invoices", 3)


def test_percentage() -> None:
    assert values_match("47.5%", 0.475)
    assert values_match("100%", 1.0)


def test_string_case_insensitive() -> None:
    assert values_match("Acme Corp", "acme corp")
    assert values_match("  Acme   Corp  ", "Acme Corp")


def test_iso_date_to_freeform() -> None:
    assert values_match("March 15, 2026", "2026-03-15T10:30:00Z")
    assert values_match("Mar 15, 2026", "2026-03-15")


def test_id_string_to_int_via_numeric_path() -> None:
    # When the actual is numeric, _numeric_match handles "4711".
    assert values_match("4711", 4711)


def test_boolean() -> None:
    assert values_match("yes", True)
    assert values_match("no", False)
    assert not values_match("yes", False)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def _history(tool_use_id: str, content: object) -> list[dict[str, object]]:
    text = content if isinstance(content, str) else json.dumps(content)
    return [
        {"role": "user", "content": "ask"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tool_use_id, "name": "x", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": text},
            ],
        },
    ]


def test_verify_passes_for_correct_citation() -> None:
    history = _history("tu_1", {"amount": 1250.00})
    text = 'You owe <cite tu="tu_1" path="$.amount">$1,250.00</cite>.'
    cites, issues = verify_citations(text, index_tool_results(history))
    assert len(cites) == 1
    assert issues == []


def test_verify_flags_value_mismatch() -> None:
    history = _history("tu_1", {"amount": 1500.00})
    text = 'You owe <cite tu="tu_1" path="$.amount">$1,250.00</cite>.'
    _, issues = verify_citations(text, index_tool_results(history))
    assert len(issues) == 1
    assert issues[0].kind == "value_mismatch"
    assert issues[0].severity == "error"
    assert issues[0].expected == 1500.00


def test_verify_flags_unknown_tool_use_id() -> None:
    history = _history("tu_1", {"amount": 1250.00})
    text = '<cite tu="tu_99" path="$.amount">$1,250.00</cite>'
    _, issues = verify_citations(text, index_tool_results(history))
    assert len(issues) == 1
    assert issues[0].kind == "unknown_tool_use_id"


def test_verify_flags_broken_path() -> None:
    history = _history("tu_1", {"amount": 1250.00})
    text = '<cite tu="tu_1" path="$.nope">x</cite>'
    _, issues = verify_citations(text, index_tool_results(history))
    assert len(issues) == 1
    assert issues[0].kind == "broken_citation"


def test_substring_fallback_for_non_json_tool_result() -> None:
    history = _history("tu_1", "Customer name is Acme Corp")
    text_ok = '<cite tu="tu_1" path="$.name">Acme Corp</cite>'
    _, issues = verify_citations(text_ok, index_tool_results(history))
    assert issues == []

    text_bad = '<cite tu="tu_1" path="$.name">Globex Inc</cite>'
    _, issues = verify_citations(text_bad, index_tool_results(history))
    assert len(issues) == 1
    assert issues[0].kind == "non_json_substring_miss"
    assert issues[0].severity == "warning"


def test_predicate_filter_path_against_invoices() -> None:
    history = _history(
        "tu_1",
        {"invoices": [{"id": 4711, "amount": 1250.00}, {"id": 4733, "amount": 890.00}]},
    )
    text = (
        'Outstanding: <cite tu="tu_1" path="$.invoices[*].amount|sum">$2,140.00</cite>'
    )
    _, issues = verify_citations(text, index_tool_results(history))
    assert issues == []
