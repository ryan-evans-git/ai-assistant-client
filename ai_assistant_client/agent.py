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

import asyncio
import logging
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

from ai_assistant_client.confirmation_store import ConfirmationOutcome
from ai_assistant_client.discovery import ProgressiveToolRegistry
from ai_assistant_client.memory_meta import (
    handle_memory_meta_call,
    is_memory_meta_tool,
    memory_meta_tool_schemas,
)
from ai_assistant_client.persistence import (
    ConversationStore,
    MemoryStore,
    TranscriptStore,
)
from ai_assistant_client.workflows import WorkflowRegistry, run_workflow
from ai_assistant_client.workflows.replay import run_workflow_recording
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
from ai_assistant_client.visuals import (
    DEFAULT_MAX_IMAGE_DATA_URI_KB,
    InvalidVisualSpecError,
    VisualEnvelope,
    validate_envelope,
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

    # ---------------------------------------------------------------
    # Visuals (render_visual meta-tool)
    # ---------------------------------------------------------------
    # Per-image cap on data: URIs the LLM is allowed to embed.  Bound
    # exists to keep history payload + LLM context spend in check
    # (5 MB default — typical Claude Sonnet limit is ~5 MB per image
    # block too).
    max_image_data_uri_kb: int = DEFAULT_MAX_IMAGE_DATA_URI_KB

    # ---------------------------------------------------------------
    # Human-in-the-loop confirmation
    # ---------------------------------------------------------------
    # Tools whose upstream metadata says ``requires_confirmation=True``
    # pause the agent before dispatch.  The agent emits a
    # ``tool_confirmation_request`` event the host routes to the UI;
    # the host answers via the registered confirmation hook.
    # ``confirmation_default_timeout_seconds`` is the wait used when
    # the tool's own ``timeout_seconds`` is unset.  ``confirmation_on_timeout``
    # decides whether an unanswered prompt counts as a decline (safe
    # default — destructive ops shouldn't auto-run) or a confirm.
    confirmation_default_timeout_seconds: int = 60
    confirmation_on_timeout: Literal["decline", "confirm"] = "decline"


ConfirmationHook = Callable[[dict[str, Any]], Awaitable["ConfirmationOutcome"]]


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
    confirmation_hook: ConfirmationHook | None = None,
    workflow_registry: WorkflowRegistry | None = None,
    conversation_store: ConversationStore | None = None,
    conversation_id: str | None = None,
    transcript_store: TranscriptStore | None = None,
    memory_store: MemoryStore | None = None,
    user_id: str | None = None,
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

    ``conversation_store`` + ``conversation_id``: when both are set,
    every message appended to ``history`` is mirrored into the
    store under that id.  This is *write-only* from the agent's
    perspective — the caller is responsible for loading prior
    turns out of the store and seeding ``history`` (so the
    store-read path stays a host-side decision, not a hidden
    side-effect of ``run_agent``).  Either ``None`` disables
    persistence entirely.

    ``transcript_store``: when set, every workflow dispatched
    inside this turn is recorded via
    :func:`run_workflow_recording`.  Run ids are auto-generated
    as ``wf_{workflow_name}_{uuid}`` so they're human-recognisable
    in store listings.  ``None`` (the default) keeps the
    pre-recording dispatch path unchanged.

    ``memory_store`` + ``user_id``: when **both** are set, the
    LLM-callable memory meta-tools (``memory_recall``,
    ``memory_remember``, ``memory_forget``) are surfaced to the
    model.  All operations close over ``user_id`` from the
    agent's context — the model can't pass a different one — so
    per-user isolation is preserved at the dispatch layer too.
    Either ``None`` and the memory tools stay invisible.

    History is stored in Anthropic-style block form regardless of
    which provider is in use; each provider adapter translates on
    the way out.
    """
    # Only persist when BOTH a store and an id are provided — the
    # combination is the well-formed shape, and silently ignoring
    # half-configured persistence is exactly the kind of "did the
    # data make it to disk?" ambiguity that bites later.
    persist = conversation_store is not None and conversation_id is not None

    # Same well-formed-shape rule for memory: tools only surface
    # when both a store and a user id are wired, so a misconfig
    # can't accidentally expose the recall surface without
    # isolation.
    memory_enabled = memory_store is not None and user_id is not None

    async def _record(message: dict[str, Any]) -> None:
        if persist:
            # ``persist`` only becomes True when both checks pass,
            # but the type-checker can't see that across the
            # closure; the ``assert`` is for it, not for runtime.
            assert conversation_store is not None and conversation_id is not None
            await conversation_store.append(conversation_id, message)

    user_turn = {"role": "user", "content": user_message}
    history.append(user_turn)
    await _record(user_turn)
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
            # Surface the memory meta-tools only when the host has
            # wired both a store and a user_id.  Prepending keeps
            # them at a stable position in the catalog so the
            # prompt prefix stays cache-friendly turn over turn.
            if memory_enabled:
                tools = memory_meta_tool_schemas() + tools
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
            assistant_turn = {"role": "assistant", "content": assistant_blocks}
            history.append(assistant_turn)
            await _record(assistant_turn)

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
                # ── Human-in-the-loop confirmation gate ─────────────
                # Meta- and visual-tools never gate (they're inline).
                # Workflows skip the per-tool gate too — their pauses
                # are driven from inside the workflow body via
                # pause_for_confirmation(), not before dispatch.
                # Remote tools may carry HITL metadata that pauses
                # dispatch until the user approves via the UI.
                gate_outcome: ConfirmationOutcome | None = None
                is_workflow_call = bool(
                    workflow_registry and workflow_registry.get(tool_name)
                )
                if (
                    not registry.is_meta_tool(tool_name)
                    and not registry.is_visual_tool(tool_name)
                    and not is_workflow_call
                    and not (memory_enabled and is_memory_meta_tool(tool_name))
                ):
                    descriptor = registry.get_descriptor(tool_name)
                    hitl = (descriptor.hitl if descriptor else None) or None
                    if hitl and hitl.get("requires_confirmation"):
                        async for ev, outcome in _gate_confirmation(
                            tu_id=tu_id,
                            tool_name=tool_name,
                            tool_description=descriptor.description if descriptor else "",
                            tool_input=tool_input,
                            hitl=hitl,
                            confirmation_hook=confirmation_hook,
                            config=config,
                        ):
                            if ev is not None:
                                yield ev
                            if outcome is not None:
                                gate_outcome = outcome
                if gate_outcome is not None and gate_outcome.decision == "decline":
                    note = gate_outcome.note or ""
                    decline_text = (
                        f"User declined: {note}" if note else "User declined."
                    )
                    tool_result_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu_id,
                            "is_error": True,
                            "content": decline_text,
                        }
                    )
                    yield AgentEvent(
                        "tool_result",
                        {"id": tu_id, "content": decline_text, "is_error": True},
                    )
                    continue
                try:
                    if memory_enabled and is_memory_meta_tool(tool_name):
                        # Memory meta-tools run inline like the other
                        # meta-tools.  ``user_id`` comes from the
                        # agent's context (closed over), not from
                        # the LLM-supplied input — load-bearing for
                        # cross-user isolation.
                        assert memory_store is not None and user_id is not None
                        memory_args = (
                            tool_input if isinstance(tool_input, dict) else {}
                        )
                        result_text = await handle_memory_meta_call(
                            tool_name,
                            memory_args,
                            store=memory_store,
                            user_id=user_id,
                        )
                    elif registry.is_meta_tool(tool_name):
                        result_text = registry.handle_meta_call(tool_name, tool_input)
                    elif registry.is_visual_tool(tool_name):
                        # Validate the spec; emit the visual event when
                        # it passes; surface the InvalidVisualSpecError
                        # message as the tool_result on failure so the
                        # model can correct.
                        try:
                            envelope = validate_envelope(
                                tool_input,
                                max_image_data_uri_kb=config.max_image_data_uri_kb,
                            )
                        except InvalidVisualSpecError as ve:
                            result_text = (
                                f"render_visual rejected: {ve}. "
                                "Fix the spec and try again, or fall back to text."
                            )
                        else:
                            yield AgentEvent(
                                "visual",
                                {
                                    "tool_use_id": tu_id,
                                    "schema_version": envelope.schema_version,
                                    "spec": envelope.spec.model_dump(),
                                },
                            )
                            result_text = _summarize_visual(envelope)
                    elif is_workflow_call:
                        wf = workflow_registry.get(tool_name)  # type: ignore[union-attr]
                        assert wf is not None  # guarded by is_workflow_call
                        wf_value: Any = None
                        wf_error: str | None = None
                        # Pick recording vs. plain dispatch based on
                        # whether the host wired in a transcript
                        # store.  Both return the same event
                        # AsyncIterator shape, so the rest of the
                        # loop is identical.
                        wf_args = (
                            tool_input if isinstance(tool_input, dict) else {}
                        )
                        if transcript_store is not None:
                            run_id = f"wf_{wf.name}_{uuid.uuid4().hex}"
                            wf_iter = run_workflow_recording(
                                wf,
                                wf_args,
                                tool_use_id=tu_id,
                                confirmation_hook=confirmation_hook,
                                store=transcript_store,
                                run_id=run_id,
                                default_timeout_seconds=config.confirmation_default_timeout_seconds,
                                on_timeout_decision=config.confirmation_on_timeout,
                            )
                        else:
                            wf_iter = run_workflow(
                                wf,
                                wf_args,
                                tool_use_id=tu_id,
                                confirmation_hook=confirmation_hook,
                                default_timeout_seconds=config.confirmation_default_timeout_seconds,
                                on_timeout_decision=config.confirmation_on_timeout,
                            )
                        async for wf_event in wf_iter:
                            if wf_event.type == "status":
                                yield AgentEvent(
                                    "workflow_status", wf_event.payload or {}
                                )
                            elif wf_event.type == "confirmation_request":
                                yield AgentEvent(
                                    "tool_confirmation_request",
                                    wf_event.payload or {},
                                )
                            elif wf_event.type == "confirmation_resolved":
                                yield AgentEvent(
                                    "tool_confirmation_resolved",
                                    wf_event.payload or {},
                                )
                            elif wf_event.type == "result":
                                wf_value = wf_event.value
                            elif wf_event.type == "error":
                                wf_error = wf_event.error
                        if wf_error is not None:
                            raise RuntimeError(wf_error)
                        result_text = (
                            wf_value if isinstance(wf_value, str) else _to_text(wf_value)
                        )
                    elif registry.is_known_tool(tool_name):
                        result = await dispatcher(tool_name, tool_input)
                        result_text = result if isinstance(result, str) else _to_text(result)
                        if gate_outcome is not None and gate_outcome.note:
                            # User attached a note when confirming —
                            # prepend it so the model sees the operator's
                            # rationale alongside the tool's output.
                            result_text = (
                                f"User confirmed (note: {gate_outcome.note})\n"
                                f"{result_text}"
                            )
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

            tool_result_turn = {"role": "user", "content": tool_result_content}
            history.append(tool_result_turn)
            await _record(tool_result_turn)

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
        feedback_turn = {"role": "user", "content": feedback}
        history.append(feedback_turn)
        await _record(feedback_turn)
        retries_remaining -= 1
        yield AgentEvent(
            "validation_retry",
            {"retries_remaining": retries_remaining, "feedback_preview": feedback[:200]},
        )


async def _gate_confirmation(
    *,
    tu_id: str,
    tool_name: str,
    tool_description: str,
    tool_input: Any,
    hitl: dict[str, Any],
    confirmation_hook: ConfirmationHook | None,
    config: AgentRunConfig,
) -> AsyncIterator[tuple["AgentEvent | None", "ConfirmationOutcome | None"]]:
    """Pause the agent before dispatching a HITL-flagged tool.

    Yields ``(event, outcome)`` tuples — at least one event (the
    confirmation_request) followed by a ``(None, outcome)`` carrying
    the user's decision.  When no hook is registered, treats the
    request as auto-confirmed (so a host that hasn't wired HITL
    yet doesn't deadlock — it just behaves like the old immediate-
    dispatch path).

    The request_id is a fresh UUID per pause; the UI's `tool_use_id`
    field maps to the LLM's tool_use id so multiple pauses inside
    one workflow can each have a unique id.
    """
    request_id = f"conf_{uuid.uuid4().hex}"
    timeout = (
        hitl.get("timeout_seconds")
        or config.confirmation_default_timeout_seconds
    )
    payload: dict[str, Any] = {
        "request_id": request_id,
        "tool_use_id": tu_id,
        "tool_name": tool_name,
        "tool_description": tool_description,
        "tool_input": tool_input,
        "timeout_seconds": int(timeout),
    }
    if hitl.get("message"):
        payload["message"] = hitl["message"]
    yield AgentEvent("tool_confirmation_request", payload), None

    if confirmation_hook is None:
        # No host-installed hook → fall back to confirm so the chat
        # doesn't wedge.  The agent still emitted the request event,
        # so an attached UI gets a chance to show the modal even if
        # the host hasn't wired the resolution path yet.
        log.warning(
            "Tool %r marked HITL but no confirmation_hook registered; "
            "auto-confirming.  Wire confirmation_hook to enforce.",
            tool_name,
        )
        outcome = ConfirmationOutcome(decision="confirm")
    else:
        try:
            outcome = await asyncio.wait_for(
                confirmation_hook(payload), timeout=timeout
            )
        except asyncio.TimeoutError:
            outcome = ConfirmationOutcome(
                decision=config.confirmation_on_timeout,
                note="confirmation timed out",
            )
    yield AgentEvent(
        "tool_confirmation_resolved",
        {
            "request_id": request_id,
            "tool_use_id": tu_id,
            "decision": outcome.decision,
            "note": outcome.note,
        },
    ), None
    yield None, outcome


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


def _summarize_visual(envelope: VisualEnvelope) -> str:
    """Short, model-facing confirmation that a visual was rendered.

    Intentionally compact — the LLM doesn't need to see the full spec
    again (it just sent it).  The summary helps it craft a follow-up
    text turn that complements the visual.
    """
    spec = envelope.spec
    if spec.kind == "chart":
        rows = len(spec.data)
        series = ", ".join(spec.y_keys) if spec.y_keys else "label/value"
        return (
            f"Rendered: {spec.chart_type} chart with {rows} row(s); "
            f"series: {series}.  Add a one-sentence text takeaway "
            "below the visual."
        )
    if spec.kind == "table":
        return (
            f"Rendered: table with {len(spec.columns)} column(s) and "
            f"{len(spec.rows)} row(s).  Add a one-sentence text "
            "takeaway below the visual."
        )
    if spec.kind == "kpi":
        return (
            f"Rendered: KPI tile {spec.label!r} = {spec.value}"
            + (f" {spec.unit}" if spec.unit else "")
            + ".  Add brief context below."
        )
    if spec.kind == "image":
        return f"Rendered: image ({spec.alt!r})."
    return "Rendered."
