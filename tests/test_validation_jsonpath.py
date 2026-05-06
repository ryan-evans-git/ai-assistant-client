"""Citation parser + JSONPath subset behavior."""

from __future__ import annotations

from ai_assistant_client.validation.citations import parse_citations, strip_citations
from ai_assistant_client.validation.jsonpath import resolve


# ---------------------------------------------------------------------------
# Citation parser
# ---------------------------------------------------------------------------


def test_parses_basic_citation() -> None:
    text = 'You owe <cite tu="tu_1" path="$.amount">$1,250.00</cite>.'
    cites = parse_citations(text)
    assert len(cites) == 1
    assert cites[0].tool_use_id == "tu_1"
    assert cites[0].path == "$.amount"
    assert cites[0].displayed_text == "$1,250.00"


def test_parses_multiple_citations_in_order() -> None:
    text = (
        'There are <cite tu="tu_1" path="$.invoices|length">3</cite> '
        'invoices totaling <cite tu="tu_1" path="$.invoices|sum">$4,340.00</cite>.'
    )
    cites = parse_citations(text)
    assert len(cites) == 2
    assert cites[0].displayed_text == "3"
    assert cites[1].displayed_text == "$4,340.00"
    assert cites[0].start < cites[1].start


def test_skips_tag_missing_tu() -> None:
    text = '<cite path="$.amount">$1</cite>'
    assert parse_citations(text) == []


def test_handles_single_quoted_attrs() -> None:
    text = "<cite tu='tu_1' path='$.foo'>x</cite>"
    cites = parse_citations(text)
    assert len(cites) == 1
    assert cites[0].path == "$.foo"


def test_strip_citations_leaves_displayed_text() -> None:
    text = 'You owe <cite tu="tu_1" path="$.amount">$1,250.00</cite> total.'
    assert strip_citations(text) == "You owe $1,250.00 total."


# ---------------------------------------------------------------------------
# JSONPath subset
# ---------------------------------------------------------------------------


SAMPLE = {
    "invoices": [
        {"id": 4711, "amount": 1250.00, "status": "open"},
        {"id": 4733, "amount": 890.00, "status": "open"},
        {"id": 4798, "amount": 2200.00, "status": "paid"},
    ],
    "customer": {"name": "Acme Corp", "tier": "gold"},
    "outstanding": 4340.00,
}


def test_resolves_root_property() -> None:
    assert resolve(SAMPLE, "$.outstanding") == 4340.00


def test_resolves_nested_key() -> None:
    assert resolve(SAMPLE, "$.customer.name") == "Acme Corp"


def test_resolves_array_index() -> None:
    assert resolve(SAMPLE, "$.invoices[0].id") == 4711


def test_resolves_negative_index() -> None:
    assert resolve(SAMPLE, "$.invoices[-1].status") == "paid"


def test_wildcard_returns_list() -> None:
    assert resolve(SAMPLE, "$.invoices[*].amount") == [1250.00, 890.00, 2200.00]


def test_pipe_length() -> None:
    assert resolve(SAMPLE, "$.invoices|length") == 3


def test_pipe_sum_on_projection() -> None:
    assert resolve(SAMPLE, "$.invoices[*].amount|sum") == 4340.00


def test_predicate_filter_numeric() -> None:
    result = resolve(SAMPLE, "$.invoices[?(@.id==4711)]")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["id"] == 4711


def test_predicate_filter_string() -> None:
    result = resolve(SAMPLE, '$.invoices[?(@.status=="paid")]')
    assert len(result) == 1
    assert result[0]["id"] == 4798


def test_missing_key_returns_none() -> None:
    assert resolve(SAMPLE, "$.nope") is None


def test_out_of_range_index_returns_none() -> None:
    assert resolve(SAMPLE, "$.invoices[99]") is None


def test_invalid_path_returns_none() -> None:
    assert resolve(SAMPLE, "not a path") is None
    assert resolve(SAMPLE, "") is None
