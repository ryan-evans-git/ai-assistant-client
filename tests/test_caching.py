"""Cache-marker translation in the Anthropic + Bedrock adapters.

The agent loop's cache strategy is provider-neutral (it just builds a
``CacheHint``) — these tests verify each adapter inserts the right
native markers in the right places, without mutating the caller's
history list.
"""

from __future__ import annotations

import copy
from typing import Any

from ai_assistant_client.llm.anthropic_provider import _apply_cache_markers
from ai_assistant_client.llm.base import CacheHint
from ai_assistant_client.llm.bedrock_provider import (
    _apply_cache_points,
    _to_bedrock_messages,
)


SAMPLE_TOOLS = [
    {"name": "alpha", "description": "first", "input_schema": {"type": "object"}},
    {"name": "beta", "description": "second", "input_schema": {"type": "object"}},
]


def _sample_history() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "tu_1", "name": "alpha", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_marks_system_tools_and_history() -> None:
    history = _sample_history()
    history_snapshot = copy.deepcopy(history)
    tools_snapshot = copy.deepcopy(SAMPLE_TOOLS)

    system, tools, messages = _apply_cache_markers(
        "you are helpful",
        SAMPLE_TOOLS,
        history,
        CacheHint(cache_history_through_index=-2),  # mark assistant turn
    )

    # System promoted to typed-block array with cache_control marker.
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[0]["text"] == "you are helpful"

    # Last tool entry marked.
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in tools[0]

    # Last block of the targeted message marked.
    target_blocks = messages[-2]["content"]
    assert target_blocks[-1]["cache_control"] == {"type": "ephemeral"}

    # Original inputs untouched (caller can keep using them across turns).
    assert history == history_snapshot
    assert SAMPLE_TOOLS == tools_snapshot


def test_anthropic_disabled_hint_is_passthrough() -> None:
    system, tools, messages = _apply_cache_markers(
        "sys", SAMPLE_TOOLS, _sample_history(), None
    )
    assert system == "sys"
    assert tools is SAMPLE_TOOLS
    assert all("cache_control" not in m for m in messages)


def test_anthropic_string_content_message_promoted_to_block() -> None:
    history = [{"role": "user", "content": "hello"}]
    _, _, messages = _apply_cache_markers(
        "", [], history, CacheHint(cache_history_through_index=0)
    )
    blocks = messages[0]["content"]
    assert isinstance(blocks, list)
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "hello"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------


def test_bedrock_appends_cache_points() -> None:
    history = _sample_history()
    bedrock_messages = _to_bedrock_messages(history)
    bedrock_system = [{"text": "you are helpful"}]
    bedrock_tools = [
        {"toolSpec": {"name": "alpha", "description": "", "inputSchema": {"json": {}}}},
        {"toolSpec": {"name": "beta", "description": "", "inputSchema": {"json": {}}}},
    ]

    system, tools, messages = _apply_cache_points(
        bedrock_system,
        bedrock_tools,
        bedrock_messages,
        CacheHint(cache_history_through_index=-2),
    )

    assert system[-1] == {"cachePoint": {"type": "default"}}
    assert tools[-1] == {"cachePoint": {"type": "default"}}
    # cache point appended inside the targeted message's content array.
    assert messages[-2]["content"][-1] == {"cachePoint": {"type": "default"}}
    # Other messages untouched.
    assert all(
        b != {"cachePoint": {"type": "default"}}
        for m in (messages[0], messages[-1])
        for b in m["content"]
    )


def test_bedrock_skips_when_arrays_empty() -> None:
    system, tools, messages = _apply_cache_points(
        [], None, [], CacheHint()
    )
    assert system == []
    assert tools is None
    assert messages == []
