"""Progressive tool discovery tests."""

from __future__ import annotations

import pytest

from ai_assistant_client.discovery import (
    ProgressiveToolRegistry,
    RemoteToolDescriptor,
)


def _descriptor(
    name: str, description: str = "", tags: tuple[str, ...] = ()
) -> RemoteToolDescriptor:
    return RemoteToolDescriptor(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {}},
        tags=tags,
    )


def test_meta_tools_always_present() -> None:
    registry = ProgressiveToolRegistry([])
    tool_names = [t["name"] for t in registry.anthropic_tools()]
    assert "tool_search" in tool_names
    assert "tool_load" in tool_names


def test_unloaded_catalog_tools_not_in_window() -> None:
    catalog = [_descriptor("get_pet", "Find pet by ID")]
    registry = ProgressiveToolRegistry(catalog)
    names = {t["name"] for t in registry.anthropic_tools()}
    # Catalog tools are NOT exposed until loaded.
    assert "get_pet" not in names
    assert names == {"tool_search", "tool_load"}


def test_search_returns_top_matches() -> None:
    catalog = [
        _descriptor("get_pet", "Find pet by ID"),
        _descriptor("list_pets", "List pets by status"),
        _descriptor("get_invoice", "Look up an invoice"),
        _descriptor("list_users", "List users"),
    ]
    registry = ProgressiveToolRegistry(catalog, search_max_results=3)
    out = registry.handle_meta_call("tool_search", {"query": "pet"})
    assert "get_pet" in out
    assert "list_pets" in out
    # Unrelated tools shouldn't dominate.
    assert "get_invoice" not in out or out.index("get_pet") < out.index("get_invoice")


def test_search_empty_query_returns_alphabetical_subset() -> None:
    catalog = [_descriptor(n) for n in ("zebra", "alpha", "mike", "delta")]
    registry = ProgressiveToolRegistry(catalog, search_max_results=2)
    out = registry.handle_meta_call("tool_search", {"query": ""})
    # First two alphabetically: alpha, delta.
    assert "alpha" in out
    assert "delta" in out


def test_load_marks_tools_as_loaded() -> None:
    catalog = [_descriptor("get_pet", "Find pet by ID")]
    registry = ProgressiveToolRegistry(catalog)
    registry.handle_meta_call("tool_load", {"names": ["get_pet"]})
    assert "get_pet" in registry.loaded_tools
    names = {t["name"] for t in registry.anthropic_tools()}
    assert "get_pet" in names


def test_load_unknown_name_returns_message() -> None:
    registry = ProgressiveToolRegistry([])
    out = registry.handle_meta_call("tool_load", {"names": ["nonexistent"]})
    assert "Unknown" in out
    assert "nonexistent" in out


def test_load_accepts_string_or_array() -> None:
    catalog = [_descriptor("get_pet")]
    registry = ProgressiveToolRegistry(catalog)
    out = registry.handle_meta_call("tool_load", {"names": "get_pet"})
    assert "Loaded" in out
    assert "get_pet" in registry.loaded_tools


def test_load_empty_returns_helpful_message() -> None:
    registry = ProgressiveToolRegistry([])
    out = registry.handle_meta_call("tool_load", {"names": []})
    assert "non-empty" in out


def test_is_meta_tool() -> None:
    registry = ProgressiveToolRegistry([_descriptor("get_pet")])
    assert registry.is_meta_tool("tool_search")
    assert registry.is_meta_tool("tool_load")
    assert not registry.is_meta_tool("get_pet")


def test_is_known_tool() -> None:
    registry = ProgressiveToolRegistry([_descriptor("get_pet")])
    assert registry.is_known_tool("tool_search")
    assert registry.is_known_tool("get_pet")
    assert not registry.is_known_tool("not_a_thing")


def test_reset_loaded_clears_session_state() -> None:
    catalog = [_descriptor("get_pet")]
    registry = ProgressiveToolRegistry(catalog)
    registry.handle_meta_call("tool_load", {"names": ["get_pet"]})
    assert registry.loaded_tools == {"get_pet"}
    registry.reset_loaded()
    assert registry.loaded_tools == set()


def test_duplicate_catalog_names_rejected() -> None:
    with pytest.raises(ValueError):
        ProgressiveToolRegistry(
            [_descriptor("dup"), _descriptor("dup")]
        )


def test_search_uses_tags() -> None:
    catalog = [
        _descriptor("op_a", "operation a", tags=("invoicing",)),
        _descriptor("op_b", "operation b", tags=("shipping",)),
    ]
    registry = ProgressiveToolRegistry(catalog, search_max_results=1)
    out = registry.handle_meta_call("tool_search", {"query": "invoicing"})
    assert "op_a" in out


def test_anthropic_tools_includes_loaded_schemas() -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    catalog = [
        RemoteToolDescriptor(
            name="search_widgets",
            description="Search widgets.",
            input_schema=schema,
        )
    ]
    registry = ProgressiveToolRegistry(catalog)
    registry.handle_meta_call("tool_load", {"names": ["search_widgets"]})
    tools = registry.anthropic_tools()
    by_name = {t["name"]: t for t in tools}
    assert by_name["search_widgets"]["input_schema"] == schema
