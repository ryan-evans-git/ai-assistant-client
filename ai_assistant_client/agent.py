"""Streaming chat loop with Claude + progressive MCP tool dispatch.

Drives a single user turn from prompt → Claude streaming response
→ tool-use detection → tool_result → Claude streaming response →
…until Claude returns a stop_reason that isn't ``tool_use``.

Yields :class:`AgentEvent` instances suitable for SSE serialization.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

from ai_assistant_client.discovery import ProgressiveToolRegistry


log = logging.getLogger(__name__)


ToolDispatcher = Callable[[str, dict[str, Any]], Awaitable[Any]]


@dataclass
class AgentEvent:
    """A single SSE event in the agent stream."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> dict[str, str]:
        import json

        return {"event": self.type, "data": json.dumps(asdict(self)["data"])}


@dataclass
class AgentRunConfig:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    system_prompt: str = (
        "You are a helpful assistant.  When you need to take an action, "
        "first call `tool_search` to discover relevant tools, then call "
        "`tool_load` with the names you want to use.  Loaded tools become "
        "directly callable for the rest of the conversation."
    )
    max_tool_iterations: int = 16


async def run_agent(
    *,
    user_message: str,
    history: list[dict[str, Any]],
    registry: ProgressiveToolRegistry,
    dispatcher: ToolDispatcher,
    anthropic_client: Any,  # AsyncAnthropic
    config: AgentRunConfig,
) -> AsyncIterator[AgentEvent]:
    """Run one user turn end-to-end.

    ``history`` should be the prior conversation turns (without the
    new user message).  This function appends the new user message
    + the assistant + tool blocks it produces, mutating the list
    in place so callers can persist after each turn.

    ``dispatcher`` is the function that runs *non-meta* tool calls.
    Typically :meth:`McpPool.call_tool`.  Meta-tool calls
    (``tool_search`` / ``tool_load``) are handled inline by the
    registry and never reach the dispatcher.
    """
    history.append({"role": "user", "content": user_message})
    yield AgentEvent("user_message", {"content": user_message})

    for iteration in range(config.max_tool_iterations):
        tools = registry.anthropic_tools()
        log.debug("Iteration %d: %d tools in window", iteration, len(tools))

        assistant_blocks: list[dict[str, Any]] = []
        accumulated_text_by_block_index: dict[int, str] = {}
        tool_use_blocks: list[dict[str, Any]] = []
        partial_tool_inputs: dict[int, str] = {}
        stop_reason: str | None = None

        async with anthropic_client.messages.stream(
            model=config.model,
            max_tokens=config.max_tokens,
            system=config.system_prompt,
            tools=tools,
            messages=history,
        ) as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    block_type = getattr(block, "type", None)
                    index = getattr(event, "index", None)
                    if block_type == "tool_use":
                        # Capture the (eventual) tool-use id + name
                        # now; the input streams in across deltas.
                        tool_use_blocks.append(
                            {
                                "type": "tool_use",
                                "index": index,
                                "id": getattr(block, "id", None),
                                "name": getattr(block, "name", None),
                                "input": {},
                            }
                        )
                        partial_tool_inputs[index] = ""

                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None)
                    index = getattr(event, "index", None)
                    if delta_type == "text_delta":
                        text = getattr(delta, "text", "")
                        accumulated_text_by_block_index[index] = (
                            accumulated_text_by_block_index.get(index, "") + text
                        )
                        yield AgentEvent("text_delta", {"text": text})
                    elif delta_type == "input_json_delta":
                        partial_tool_inputs[index] = (
                            partial_tool_inputs.get(index, "")
                            + getattr(delta, "partial_json", "")
                        )

                elif event_type == "content_block_stop":
                    index = getattr(event, "index", None)
                    if index in partial_tool_inputs:
                        # Finalize the tool-use block's input.
                        for tu in tool_use_blocks:
                            if tu["index"] == index:
                                tu["input"] = _safe_json(partial_tool_inputs[index])

                elif event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    sr = getattr(delta, "stop_reason", None)
                    if sr is not None:
                        stop_reason = sr

        # Reconstruct the assistant message in MCP-friendly form.
        for index, text in accumulated_text_by_block_index.items():
            if text:
                assistant_blocks.append({"type": "text", "text": text})
        for tu in tool_use_blocks:
            assistant_blocks.append(
                {
                    "type": "tool_use",
                    "id": tu["id"],
                    "name": tu["name"],
                    "input": tu["input"],
                }
            )
        history.append({"role": "assistant", "content": assistant_blocks})

        if stop_reason != "tool_use" or not tool_use_blocks:
            yield AgentEvent("turn_complete", {"stop_reason": stop_reason or "end_turn"})
            return

        # Dispatch each tool call and append the tool_result.
        tool_result_content: list[dict[str, Any]] = []
        for tu in tool_use_blocks:
            tool_name = tu["name"]
            tool_input = tu["input"]
            tu_id = tu["id"]
            yield AgentEvent(
                "tool_use", {"id": tu_id, "name": tool_name, "input": tool_input}
            )
            try:
                if registry.is_meta_tool(tool_name):
                    result_text = registry.handle_meta_call(tool_name, tool_input)
                elif registry.is_known_tool(tool_name):
                    result = await dispatcher(tool_name, tool_input)
                    result_text = result if isinstance(result, str) else _to_text(result)
                else:
                    result_text = (
                        f"Tool '{tool_name}' isn't available. "
                        "Use tool_search to discover available tools."
                    )
                tool_result_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu_id,
                        "content": result_text,
                    }
                )
                yield AgentEvent("tool_result", {"id": tu_id, "content": result_text})
            except Exception as err:  # noqa: BLE001
                log.exception("Tool %s failed", tool_name)
                tool_result_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu_id,
                        "is_error": True,
                        "content": f"Tool error: {err}",
                    }
                )
                yield AgentEvent(
                    "tool_error", {"id": tu_id, "name": tool_name, "error": str(err)}
                )

        history.append({"role": "user", "content": tool_result_content})

    yield AgentEvent(
        "turn_complete",
        {"stop_reason": "max_iterations", "iterations": config.max_tool_iterations},
    )


def _safe_json(text: str) -> Any:
    import json

    if not text:
        return {}
    try:
        return json.loads(text)
    except ValueError:
        return {"_raw": text}


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    import json

    try:
        return json.dumps(value, default=str, indent=2)
    except (TypeError, ValueError):
        return str(value)
