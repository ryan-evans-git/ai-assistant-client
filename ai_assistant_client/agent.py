"""Streaming chat loop with a pluggable LLM + progressive MCP tool dispatch.

Drives a single user turn from prompt → LLM streaming response
→ tool-use detection → tool_result → LLM streaming response →
…until the LLM returns a stop_reason that isn't ``tool_use``.

The LLM is accessed through :class:`~ai_assistant_client.llm.LLMProvider`,
which yields a normalized event stream regardless of vendor.  The
agent loop itself no longer knows or cares whether it's talking to
Anthropic, OpenAI, Gemini, or Bedrock.

Yields :class:`AgentEvent` instances suitable for SSE serialization.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

from ai_assistant_client.discovery import ProgressiveToolRegistry
from ai_assistant_client.llm import (
    CacheHint,
    LLMProvider,
    MessageStop,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    ToolUseStop,
)
from ai_assistant_client.validation import (
    CITATION_INSTRUCTIONS,
    ValidationResult,
    build_retry_feedback,
    run_validation,
)


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
    # Opt in to provider-level prompt caching.  Anthropic + Bedrock
    # honor this by marking the system prompt, tool catalog, and last
    # completed message as a cached prefix; subsequent turns re-read
    # that prefix at ~10% of the input cost (5-minute TTL).  OpenAI
    # and Gemini 2.5+ cache automatically and ignore the flag.
    enable_prompt_caching: bool = True

    # ---------------------------------------------------------------
    # Hybrid response validation
    # ---------------------------------------------------------------
    # ``off``      — no validation; legacy behavior.
    # ``citation`` — deterministic citation tracing only.
    # ``audit``    — auditor LLM only.
    # ``hybrid``   — citation first; auditor on failure.
    validation_mode: Literal["off", "citation", "audit", "hybrid"] = "off"
    # Permissive: missing citations are OK; cited values are checked.
    # Strict: an uncited concrete value is itself a failure.
    citation_strictness: Literal["permissive", "strict"] = "permissive"
    # Override the auditor model.  Default is the cheapest model in the
    # primary provider's family (see DEFAULT_AUDITOR_MODELS in
    # validation.auditor).
    auditor_model: str | None = None
    # Number of automatic retries when validation finds errors.  Each
    # retry appends a synthetic feedback user-message asking the model
    # to revise (with the "OK to not know" reminder).  ``0`` = emit the
    # validation event but never retry.
    max_validation_retries: int = 2


async def run_agent(
    *,
    user_message: str,
    history: list[dict[str, Any]],
    registry: ProgressiveToolRegistry,
    dispatcher: ToolDispatcher,
    provider: LLMProvider,
    config: AgentRunConfig,
    provider_name: str = "",
    auditor_provider: LLMProvider | None = None,
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

    ``provider_name`` (one of ``"anthropic"``, ``"openai"``, ``"gemini"``,
    ``"bedrock"``) is used to look up the default auditor model when
    validation is enabled.  Empty string is fine when validation is off.

    ``auditor_provider`` is an optional secondary :class:`LLMProvider`
    used for the auditor LLM when ``validation_mode`` is ``"audit"`` or
    ``"hybrid"``.  Defaults to ``provider``.

    History is stored in Anthropic-style block form regardless of
    which provider is in use; each provider adapter translates on
    the way out.
    """
    history.append({"role": "user", "content": user_message})
    yield AgentEvent("user_message", {"content": user_message})

    effective_system_prompt = (
        config.system_prompt + CITATION_INSTRUCTIONS
        if config.validation_mode != "off"
        else config.system_prompt
    )
    retries_remaining = config.max_validation_retries

    while True:
        # ----- Inner agent loop: run until stop_reason != tool_use --
        last_assistant_text = ""
        terminated_normally = False

        for iteration in range(config.max_tool_iterations):
            tools = registry.anthropic_tools()
            log.debug("Iteration %d: %d tools in window", iteration, len(tools))

            accumulated_text: list[str] = []
            # tool_use blocks indexed by the provider's stable index.
            tool_uses: dict[int, dict[str, Any]] = {}
            partial_inputs: dict[int, str] = {}
            stop_reason: str | None = None

            # Build the cache hint for this turn.  We cache the system
            # prompt, the tool catalog, and the most recently *completed*
            # message in history.  The in-flight last message (the user
            # turn we just appended, or the most recent tool_result
            # block) is intentionally left uncached so each new turn
            # forms a fresh suffix beyond the cached prefix.  Anthropic
            # supports up to 4 cache breakpoints per request — three is
            # well within budget.
            cache_hint = (
                CacheHint(cache_history_through_index=len(history) - 2)
                if config.enable_prompt_caching and len(history) >= 2
                else None
            )

            async for event in provider.stream_turn(
                model=config.model,
                max_tokens=config.max_tokens,
                system=effective_system_prompt,
                tools=tools,
                messages=history,
                cache_hint=cache_hint,
            ):
                if isinstance(event, TextDelta):
                    if event.text:
                        accumulated_text.append(event.text)
                        yield AgentEvent("text_delta", {"text": event.text})

                elif isinstance(event, ToolUseStart):
                    tool_uses[event.index] = {
                        "id": event.id,
                        "name": event.name,
                        "input": {},
                    }
                    partial_inputs.setdefault(event.index, "")

                elif isinstance(event, ToolInputDelta):
                    partial_inputs[event.index] = (
                        partial_inputs.get(event.index, "") + event.partial_json
                    )

                elif isinstance(event, ToolUseStop):
                    if event.index in tool_uses:
                        tool_uses[event.index]["input"] = _safe_json(
                            partial_inputs.get(event.index, "")
                        )

                elif isinstance(event, MessageStop):
                    stop_reason = event.stop_reason

            # Reconstruct the assistant message in MCP-friendly form.
            assistant_blocks: list[dict[str, Any]] = []
            joined_text = "".join(accumulated_text)
            if joined_text:
                assistant_blocks.append({"type": "text", "text": joined_text})
            ordered_tool_uses = [tool_uses[i] for i in sorted(tool_uses)]
            for tu in ordered_tool_uses:
                # Backfill input for adapters that emitted ToolInputDelta
                # but no ToolUseStop (defensive — current adapters all
                # emit Stop, but it's cheap insurance).
                if not tu["input"]:
                    index = next(
                        (idx for idx, blk in tool_uses.items() if blk is tu), None
                    )
                    if index is not None:
                        tu["input"] = _safe_json(partial_inputs.get(index, ""))
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tu["id"],
                        "name": tu["name"],
                        "input": tu["input"],
                    }
                )
            history.append({"role": "assistant", "content": assistant_blocks})

            if stop_reason != "tool_use" or not ordered_tool_uses:
                last_assistant_text = joined_text
                terminated_normally = True
                inner_stop_reason = stop_reason or "end_turn"
                break

            # Dispatch each tool call and append the tool_result.
            tool_result_content: list[dict[str, Any]] = []
            for tu in ordered_tool_uses:
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

        if not terminated_normally:
            yield AgentEvent(
                "turn_complete",
                {
                    "stop_reason": "max_iterations",
                    "iterations": config.max_tool_iterations,
                },
            )
            return

        # ----- Validation (when enabled) -------------------------------
        if config.validation_mode == "off":
            yield AgentEvent("turn_complete", {"stop_reason": inner_stop_reason})
            return

        result: ValidationResult = await run_validation(
            mode=config.validation_mode,
            strictness=config.citation_strictness,
            response_text=last_assistant_text,
            history=history,
            auditor_provider=auditor_provider or provider,
            auditor_provider_name=provider_name,
            auditor_model=config.auditor_model,
        )
        yield AgentEvent(
            "validation",
            {
                "method": result.method,
                "passed": result.passed,
                "citations_total": result.citations_total,
                "citations_verified": result.citations_verified,
                "auditor_used": result.auditor_used,
                "auditor_model": result.auditor_model,
                "issues": [asdict(i) for i in result.issues],
            },
        )

        if result.passed or retries_remaining <= 0:
            yield AgentEvent("turn_complete", {"stop_reason": inner_stop_reason})
            return

        # Append a synthetic user message asking the model to revise,
        # then loop.  The feedback message is a real history entry —
        # hosts that want to hide it can detect the "[validation] "
        # prefix.
        feedback = build_retry_feedback(result)
        history.append({"role": "user", "content": feedback})
        retries_remaining -= 1
        yield AgentEvent(
            "validation_retry",
            {"retries_remaining": retries_remaining, "feedback_preview": feedback[:200]},
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
