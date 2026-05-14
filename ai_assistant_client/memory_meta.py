"""LLM-callable meta-tools for per-user memory.

Three tools the model can invoke through the standard tool-use
contract:

* ``memory_recall(tags?)`` — return the user's memories (optionally
  filtered to those carrying any of the given tags).
* ``memory_remember(key, value, tags?)`` — write a new memory and
  return its assigned id.
* ``memory_forget(memory_id)`` — delete a memory by id.

These live alongside the existing meta-tools (``tool_search`` /
``tool_load`` / ``render_visual``) but are **enabled only when**
the host wires a :class:`MemoryStore` + ``user_id`` through
``run_agent``.  When either is missing, the memory tools are not
surfaced and the model can't accidentally call them.

Security model:

* Every dispatch path takes ``user_id`` from the agent's context
  (not from a tool argument).  The model can't escalate by
  passing a different user id — the value is closed over by the
  host's call to ``run_agent``.
* Memory ids are server-assigned opaque tokens (see
  :func:`user_memory_local._new_memory_id`).
* Cross-user access raises :class:`KeyError`, surfaced to the
  model as a "Memory not found" tool_result so a probing prompt
  can't tell "wrong owner" from "doesn't exist".
"""

from __future__ import annotations

import json
from typing import Any

from ai_assistant_client.persistence.user_memory import Memory, MemoryStore


# Tool-name constants the agent loop uses to route dispatch.
MEMORY_RECALL = "memory_recall"
MEMORY_REMEMBER = "memory_remember"
MEMORY_UPDATE = "memory_update"
MEMORY_FORGET = "memory_forget"


MEMORY_META_TOOL_NAMES = frozenset(
    {MEMORY_RECALL, MEMORY_REMEMBER, MEMORY_UPDATE, MEMORY_FORGET}
)


def is_memory_meta_tool(name: str) -> bool:
    return name in MEMORY_META_TOOL_NAMES


def memory_meta_tool_schemas() -> list[dict[str, Any]]:
    """JSON Schemas for the memory meta-tools.

    Same shape the LLM sees for any other tool — name +
    description + input_schema.  Returned in a stable order so
    the prefix cache stays valid across turns.
    """
    return [
        {
            "name": MEMORY_RECALL,
            "description": (
                "Recall persisted notes about the current user. "
                "Returns a JSON array of {memory_id, key, value, "
                "tags, created_at, updated_at} objects.  Pass "
                "``tags`` to filter to memories carrying any of "
                "the given tags."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional: filter to memories with any of "
                            "these tags (union, not intersection)."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        },
        {
            "name": MEMORY_REMEMBER,
            "description": (
                "Persist a new note about the current user.  "
                "``key`` is a short label the UI can render; "
                "``value`` is any JSON-serializable payload.  "
                "Returns the assigned memory_id."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": (
                            "Short label, e.g. 'role', 'timezone', "
                            "'favorite_language'."
                        ),
                    },
                    "value": {
                        "description": (
                            "JSON-serializable note content.  Prefer "
                            "structured payloads (e.g. {\"kind\": "
                            "\"preference\", ...}) over free-text "
                            "blobs so future calls can key on the "
                            "structure."
                        ),
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
        },
        {
            "name": MEMORY_UPDATE,
            "description": (
                "Replace the value of an existing memory by id.  "
                "Use this when the user's stored note is wrong or "
                "out of date — preferred over forget+remember "
                "because it preserves the memory_id, key, tags, "
                "and ``created_at``.  Returns the updated record."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": (
                            "The memory_id returned from "
                            "memory_remember or memory_recall."
                        ),
                    },
                    "value": {
                        "description": (
                            "New JSON-serializable value.  Fully "
                            "replaces the existing value — there's "
                            "no partial / merge semantics."
                        ),
                    },
                },
                "required": ["memory_id", "value"],
                "additionalProperties": False,
            },
        },
        {
            "name": MEMORY_FORGET,
            "description": (
                "Delete one memory by id.  Returns a confirmation "
                "string.  Use sparingly — prefer ``memory_remember`` "
                "with the same key to overwrite an outdated value."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": (
                            "The memory_id returned from "
                            "memory_remember or memory_recall."
                        ),
                    },
                },
                "required": ["memory_id"],
                "additionalProperties": False,
            },
        },
    ]


async def handle_memory_meta_call(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    store: MemoryStore,
    user_id: str,
) -> str:
    """Dispatch a memory meta-tool call.

    Returns the ``tool_result`` content string.  Errors are
    surfaced as strings (not exceptions) so the LLM can react
    and recover — same convention as the other meta-tools.

    The ``user_id`` argument comes from the agent's context, not
    from the LLM-supplied ``arguments`` dict — that's load-
    bearing for the security model.  A user_id field in
    ``arguments`` is ignored.
    """
    if tool_name == MEMORY_RECALL:
        return await _recall(store, user_id, arguments)
    if tool_name == MEMORY_REMEMBER:
        return await _remember(store, user_id, arguments)
    if tool_name == MEMORY_UPDATE:
        return await _update(store, user_id, arguments)
    if tool_name == MEMORY_FORGET:
        return await _forget(store, user_id, arguments)
    return f"Unknown memory tool: {tool_name}"


async def _recall(
    store: MemoryStore, user_id: str, args: dict[str, Any]
) -> str:
    tags = args.get("tags") if isinstance(args.get("tags"), list) else None
    records = await store.list(user_id=user_id, tags=tags)
    return json.dumps(
        [_memory_to_dict(r) for r in records], default=str, ensure_ascii=False
    )


async def _remember(
    store: MemoryStore, user_id: str, args: dict[str, Any]
) -> str:
    key = args.get("key")
    if not isinstance(key, str) or not key.strip():
        return "memory_remember requires a non-empty 'key' string."
    if "value" not in args:
        return "memory_remember requires a 'value' field."
    tags_raw = args.get("tags") or []
    tags = tuple(t for t in tags_raw if isinstance(t, str))
    record = await store.add(
        user_id=user_id, key=key, value=args["value"], tags=tags
    )
    return json.dumps(
        {"memory_id": record.memory_id, "created_at": record.created_at},
        default=str,
        ensure_ascii=False,
    )


async def _update(
    store: MemoryStore, user_id: str, args: dict[str, Any]
) -> str:
    memory_id = args.get("memory_id")
    if not isinstance(memory_id, str) or not memory_id.strip():
        return "memory_update requires a 'memory_id' string."
    if "value" not in args:
        return "memory_update requires a 'value' field."
    try:
        record = await store.update(
            user_id=user_id, memory_id=memory_id, value=args["value"]
        )
    except KeyError:
        # Same anti-enumeration property as forget — wrong owner
        # and missing-id both surface as "not found" so a
        # probing prompt can't distinguish them.
        return f"Memory not found: {memory_id}"
    return json.dumps(
        _memory_to_dict(record), default=str, ensure_ascii=False
    )


async def _forget(
    store: MemoryStore, user_id: str, args: dict[str, Any]
) -> str:
    memory_id = args.get("memory_id")
    if not isinstance(memory_id, str) or not memory_id.strip():
        return "memory_forget requires a 'memory_id' string."
    try:
        await store.remove(user_id=user_id, memory_id=memory_id)
    except KeyError:
        # Don't leak whether the id is wrong or just not the
        # caller's — same anti-enumeration property the store
        # itself enforces.
        return f"Memory not found: {memory_id}"
    return f"Forgot {memory_id}."


def _memory_to_dict(record: Memory) -> dict[str, Any]:
    return {
        "memory_id": record.memory_id,
        "key": record.key,
        "value": record.value,
        "tags": list(record.tags),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }
