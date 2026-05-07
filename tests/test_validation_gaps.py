"""Targeted tests filling remaining coverage gaps in validation modules."""

from __future__ import annotations

from typing import Any

import pytest

from ai_assistant_client.validation import jsonpath, normalize, verify
from ai_assistant_client.validation.auditor import (
    _build_user_message,
    _parse_auditor_output,
    run_auditor,
)


# ---------------------------------------------------------------------------
# jsonpath: bracket variants, predicates, filters, edge cases
# ---------------------------------------------------------------------------


def test_jsonpath_unknown_filter_returns_none() -> None:
    assert jsonpath.resolve({"x": [1]}, "$.x|nope") is None


def test_jsonpath_unbalanced_bracket_returns_no_segments() -> None:
    # "[" without "]" → segmenter returns []  → resolver returns root.
    out = jsonpath.resolve({"x": 1}, "$.x[")
    # No segments past the bad bracket → None or root data; both are
    # acceptable here.  We just want the malformed path not to crash.
    assert out is None or out == {"x": 1}


def test_jsonpath_root_path_returns_data() -> None:
    assert jsonpath.resolve({"a": 1}, "$") == {"a": 1}


def test_jsonpath_empty_path_returns_none() -> None:
    assert jsonpath.resolve({"a": 1}, "") is None


def test_jsonpath_index_out_of_range() -> None:
    assert jsonpath.resolve({"a": [1, 2]}, "$.a[5]") is None


def test_jsonpath_negative_index_works() -> None:
    assert jsonpath.resolve({"a": [10, 20, 30]}, "$.a[-1]") == 30


def test_jsonpath_wildcard_on_dict_returns_values() -> None:
    assert sorted(jsonpath.resolve({"a": 1, "b": 2}, "$[*]")) == [1, 2]


def test_jsonpath_wildcard_on_non_iterable_returns_none() -> None:
    assert jsonpath.resolve({"x": 7}, "$.x[*]") is None


def test_jsonpath_bracket_quoted_key_supports_special_chars() -> None:
    assert jsonpath.resolve({"x": {"a-b": 1}}, '$.x["a-b"]') == 1
    assert jsonpath.resolve({"x": {"a-b": 1}}, "$.x['a-b']") == 1


def test_jsonpath_bracket_quoted_key_on_non_dict_returns_none() -> None:
    assert jsonpath.resolve([1, 2], '$["a"]') is None


def test_jsonpath_predicate_filters_list_of_dicts() -> None:
    data = {"items": [{"id": 1}, {"id": 2}, {"id": 3}]}
    assert jsonpath.resolve(data, "$.items[?(@.id==2)]") == [{"id": 2}]


def test_jsonpath_predicate_with_string_quoted_value() -> None:
    data = {"items": [{"name": "a"}, {"name": "b"}]}
    assert jsonpath.resolve(data, '$.items[?(@.name=="a")]') == [{"name": "a"}]


def test_jsonpath_predicate_with_float_value() -> None:
    data = {"items": [{"x": 1.5}, {"x": 2.5}]}
    assert jsonpath.resolve(data, "$.items[?(@.x==1.5)]") == [{"x": 1.5}]


def test_jsonpath_predicate_with_inequality_and_lt_le_gt_ge() -> None:
    data = {"items": [{"x": 1}, {"x": 2}, {"x": 3}]}
    assert jsonpath.resolve(data, "$.items[?(@.x>1)]") == [{"x": 2}, {"x": 3}]
    assert jsonpath.resolve(data, "$.items[?(@.x>=2)]") == [{"x": 2}, {"x": 3}]
    assert jsonpath.resolve(data, "$.items[?(@.x<3)]") == [{"x": 1}, {"x": 2}]
    assert jsonpath.resolve(data, "$.items[?(@.x<=2)]") == [{"x": 1}, {"x": 2}]
    assert jsonpath.resolve(data, "$.items[?(@.x!=2)]") == [{"x": 1}, {"x": 3}]


def test_jsonpath_predicate_on_non_list_returns_none() -> None:
    assert jsonpath.resolve({"x": 1}, "$.x[?(@.id==1)]") is None


def test_jsonpath_pipe_filter_length_and_sum_and_count() -> None:
    data = {"xs": [1, 2, 3, 4]}
    assert jsonpath.resolve(data, "$.xs|length") == 4
    assert jsonpath.resolve(data, "$.xs|count") == 4
    assert jsonpath.resolve(data, "$.xs|sum") == 10


def test_jsonpath_pipe_sum_on_non_list_returns_none() -> None:
    assert jsonpath.resolve({"x": "abc"}, "$.x|sum") is None


def test_jsonpath_pipe_length_on_non_lengthable_returns_none() -> None:
    assert jsonpath.resolve({"x": 7}, "$.x|length") is None


def test_jsonpath_step_into_list_with_key_projects() -> None:
    data = {"items": [{"a": 1}, {"a": 2}, {"b": 3}]}
    # "$.items[*].a" first wildcards then projects "a" through the list.
    assert jsonpath.resolve(data, "$.items[*].a") == [1, 2]


def test_jsonpath_invalid_bracket_expression_returns_none() -> None:
    # "[bad expr]" doesn't match the bracket regex.
    assert jsonpath.resolve({"x": [1]}, "$.x[bogus]") is None


def test_jsonpath_step_into_non_dict_non_list_returns_none() -> None:
    assert jsonpath.resolve({"x": 1}, "$.x.y") is None


def test_jsonpath_cmp_handles_typeerror() -> None:
    """The ``_cmp`` helper swallows TypeError from incompatible types."""
    # We can call it directly to exercise the except branch.
    from ai_assistant_client.validation.jsonpath import _cmp

    assert _cmp("a", "<", 1) is False


# ---------------------------------------------------------------------------
# normalize: percent + currency + booleans + null + structural + dates
# ---------------------------------------------------------------------------


def test_normalize_none_displayed_as_em_dash() -> None:
    assert normalize.values_match("—", None) is True
    assert normalize.values_match("None", None) is True
    assert normalize.values_match("hello", None) is False


def test_normalize_bool_match() -> None:
    assert normalize.values_match("Yes", True) is True
    assert normalize.values_match("✓", True) is True
    assert normalize.values_match("✗", False) is True
    assert normalize.values_match("maybe", True) is False


def test_normalize_percent_match() -> None:
    assert normalize.values_match("47.5%", 0.475) is True
    assert normalize.values_match("100%", 1.0) is True


def test_normalize_percent_unparseable_returns_false() -> None:
    # The pattern matches the whole input; if float() somehow fails, the
    # ValueError branch is taken.
    assert normalize.values_match("--50%", 0.5) is False


def test_normalize_currency_with_commas_matches_int() -> None:
    assert normalize.values_match("$1,234", 1234) is True


def test_normalize_numeric_phrase_extracts_leading_count() -> None:
    assert normalize.values_match("3 invoices", 3) is True


def test_normalize_numeric_unparseable_returns_false() -> None:
    assert normalize.values_match("nope", 5) is False


def test_normalize_string_case_insensitive_collapses_whitespace() -> None:
    assert normalize.values_match("  Acme   Corp  ", "acme corp") is True


def test_normalize_iso_datetime_match() -> None:
    assert normalize.values_match("March 15, 2026", "2026-03-15") is True


def test_normalize_iso_datetime_match_with_z_suffix() -> None:
    assert normalize.values_match(
        "2026-03-15T12:00:00Z", "2026-03-15T12:00:00Z"
    ) is True


def test_normalize_iso_datetime_invalid_returns_none() -> None:
    """``_parse_iso_datetime`` returns None for malformed iso strings."""
    assert normalize._parse_iso_datetime("2026-13-99") is None
    assert normalize._parse_iso_datetime("not a date") is None


def test_normalize_freeform_unknown_month_returns_none() -> None:
    assert normalize._parse_freeform_datetime("Foo 1, 2026") is None


def test_normalize_freeform_invalid_day_returns_none() -> None:
    assert normalize._parse_freeform_datetime("February 30, 2026") is None


def test_normalize_structural_match_pasted_json() -> None:
    assert normalize.values_match('[1,2,3]', [1, 2, 3]) is True
    assert normalize.values_match("not json", [1, 2, 3]) is False


def test_normalize_unsupported_type_returns_false() -> None:
    """Custom objects fall through to the trailing False branch."""
    class _Weird:
        pass

    assert normalize.values_match("anything", _Weird()) is False


# ---------------------------------------------------------------------------
# verify: index_tool_results edge cases + path-legitimately-None heuristic
# ---------------------------------------------------------------------------


def test_index_tool_results_skips_non_user_messages() -> None:
    history = [{"role": "assistant", "content": "ignored"}]
    assert verify.index_tool_results(history) == {}


def test_index_tool_results_skips_string_user_content() -> None:
    history = [{"role": "user", "content": "plain text"}]
    assert verify.index_tool_results(history) == {}


def test_index_tool_results_skips_block_without_id_or_wrong_type() -> None:
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "tool_result", "content": "no id here"},
                {"type": "tool_result", "tool_use_id": "tu_1", "content": '{"x":1}'},
            ],
        }
    ]
    assert verify.index_tool_results(history) == {"tu_1": {"x": 1}}


def test_index_tool_results_preserves_non_json_content() -> None:
    history = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "free text"},
            ],
        }
    ]
    assert verify.index_tool_results(history) == {"tu_1": "free text"}


def test_path_legitimately_returns_none_for_explicit_null() -> None:
    body = {"a": {"b": None}}
    issues = verify.verify_citations(
        '<cite tu="tu" path="$.a.b">none</cite>',
        {"tu": body},
    )[1]
    # Path resolves to None legitimately, displayed "none" matches → no issue.
    assert issues == []


def test_path_legitimately_returns_none_false_for_missing_key() -> None:
    body = {"a": {}}
    issues = verify.verify_citations(
        '<cite tu="tu" path="$.a.missing">x</cite>',
        {"tu": body},
    )[1]
    assert any(i.kind == "broken_citation" for i in issues)


def test_verify_handles_predicate_index_paths() -> None:
    """A predicate-resolved-to-none path should be flagged broken."""
    body = {"items": [{"id": 1}]}
    issues = verify.verify_citations(
        '<cite tu="tu" path="$.items[?(@.id==99)]">x</cite>',
        {"tu": body},
    )[1]
    # Predicate returns [] (not None), so the value-mismatch branch fires.
    assert issues  # at least one issue


def test_verify_legitimately_none_with_wildcard_segment_treated_as_broken() -> None:
    """``_path_legitimately_returns_none`` returns False for wildcard
    segments that hit None — the citation is reported as broken."""
    body = {"a": []}
    issues = verify.verify_citations(
        '<cite tu="tu" path="$.a[*]">x</cite>',
        {"tu": body},
    )[1]
    assert issues


def test_verify_legitimately_none_with_index_oob_is_broken() -> None:
    body = {"a": [1, 2]}
    issues = verify.verify_citations(
        '<cite tu="tu" path="$.a[5]">x</cite>',
        {"tu": body},
    )[1]
    assert any(i.kind == "broken_citation" for i in issues)


def test_verify_legitimately_none_for_root_null_data() -> None:
    """``$``-only path against ``None`` body should be considered legit."""
    issues = verify.verify_citations(
        '<cite tu="tu" path="$">none</cite>',
        {"tu": None},
    )[1]
    # The displayed "none" matches actual None, no issue.
    assert issues == []


# ---------------------------------------------------------------------------
# auditor: parse paths + run_auditor with stub provider
# ---------------------------------------------------------------------------


def test_auditor_parse_empty_text_returns_empty() -> None:
    assert _parse_auditor_output("") == []
    assert _parse_auditor_output("    ") == []


def test_auditor_parse_text_without_braces_returns_empty() -> None:
    assert _parse_auditor_output("just prose") == []


def test_auditor_parse_invalid_json_returns_empty() -> None:
    assert _parse_auditor_output("{ not valid json") == []


def test_auditor_parse_payload_without_issues_returns_empty() -> None:
    assert _parse_auditor_output('{"other": "thing"}') == []


def test_auditor_parse_skips_non_dict_entries() -> None:
    out = _parse_auditor_output(
        '{"issues": ["bare string", {"claim": "x", "reason": "y", "severity": "error"}]}'
    )
    assert len(out) == 1
    assert out[0].claim == "x"
    assert out[0].severity == "error"


def test_auditor_parse_unknown_severity_defaults_to_warning() -> None:
    out = _parse_auditor_output(
        '{"issues": [{"claim": "x", "reason": "y", "severity": "panic"}]}'
    )
    assert out[0].severity == "warning"


def test_auditor_build_user_message_includes_both_sections() -> None:
    msg = _build_user_message("the response", {"tu_1": {"k": 1}})
    assert "ASSISTANT_RESPONSE" in msg
    assert "TOOL_RESULTS" in msg
    assert "the response" in msg
    assert '"tu_1"' in msg


@pytest.mark.asyncio
async def test_run_auditor_returns_empty_when_provider_raises() -> None:
    from ai_assistant_client.llm.base import LLMProvider

    class _BoomProvider(LLMProvider):
        async def stream_turn(self, **_kw: Any):  # type: ignore[override]
            raise RuntimeError("network fail")
            if False:
                yield  # pragma: no cover

    out = await run_auditor(
        provider=_BoomProvider(),
        model="m",
        response_text="r",
        tool_results_by_id={},
    )
    assert out == []


@pytest.mark.asyncio
async def test_run_auditor_streams_and_parses() -> None:
    from ai_assistant_client.llm.base import LLMProvider, MessageStop, TextDelta

    class _ScriptedProvider(LLMProvider):
        async def stream_turn(self, **_kw: Any):  # type: ignore[override]
            yield TextDelta(text='{"issues": [{"claim":"X", "reason":"Y", "severity":"error"}]}')
            yield MessageStop(stop_reason="end_turn")

    out = await run_auditor(
        provider=_ScriptedProvider(),
        model="m",
        response_text="r",
        tool_results_by_id={"tu_1": {"x": 1}},
    )
    assert len(out) == 1
    assert out[0].claim == "X"
    assert out[0].severity == "error"
