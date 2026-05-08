"""``@workflow`` decorator + signature → JSON Schema derivation.

Mirrors ai-assistant-server's ``@tool`` shape so the authoring
experience is identical for callers — only the import path
differs (workflows live client-side; tools live server-side).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from pydantic import Field, create_model


# Sentinel attribute the loader scans module namespaces for.
_WORKFLOW_ATTR = "_aai_workflow"


@dataclass(frozen=True)
class Workflow:
    """A registered workflow.

    ``handler`` is the original async function — the runtime calls
    it with the parsed input arguments.  ``input_schema`` is derived
    from the signature at decoration time (same pipeline as plugin
    tools server-side)."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    tags: tuple[str, ...] = field(default_factory=tuple)
    source_module: str = ""


def workflow(
    *,
    name: str | None = None,
    description: str | None = None,
    tags: tuple[str, ...] | list[str] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a coroutine as a multi-step HITL workflow.

    The decorated function MUST be ``async def`` — the runtime
    awaits it directly.  Inside the function, call
    :func:`pause_for_confirmation` to gate steps and
    :func:`emit_status` to push progress updates over the SSE
    stream.

    Parameters
    ----------
    name:
        Tool name surfaced to the LLM.  Defaults to ``fn.__name__``.
    description:
        Human-readable description.  Defaults to the function's
        docstring (first paragraph).
    tags:
        Optional tags for progressive-discovery ranking on the
        client side.
    """

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"@workflow expects an async function; "
                f"{fn.__qualname__!r} is sync.  Use ``async def`` so the "
                "runtime can await pause_for_confirmation()."
            )
        wf_name = name or fn.__name__
        doc = description if description is not None else (fn.__doc__ or "")
        wf_desc = doc.strip()
        if not wf_desc:
            raise ValueError(
                f"@workflow({wf_name!r}) requires either an explicit "
                "description= or a non-empty docstring."
            )
        input_schema = _derive_input_schema(fn)
        wf = Workflow(
            name=wf_name,
            description=wf_desc,
            input_schema=input_schema,
            handler=fn,
            tags=tuple(tags),
            source_module=fn.__module__,
        )
        setattr(fn, _WORKFLOW_ATTR, wf)
        return fn

    return wrap


def is_workflow(obj: Any) -> bool:
    """Return ``True`` when ``obj`` was decorated with ``@workflow``."""
    return callable(obj) and hasattr(obj, _WORKFLOW_ATTR)


def get_workflow(obj: Any) -> Workflow | None:
    """Return the :class:`Workflow` attached by ``@workflow``."""
    wf = getattr(obj, _WORKFLOW_ATTR, None)
    return wf if isinstance(wf, Workflow) else None


# ---------------------------------------------------------------------------
# Schema derivation (parallel to ai-assistant-server's derive_input_schema)
# ---------------------------------------------------------------------------


def _derive_input_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON Schema from a coroutine's typed signature.

    Same pipeline as the server-side plugin decorator — Pydantic
    handles every annotation it normally accepts (``Literal``,
    ``Optional``, ``list[T]``, ``Enum``, BaseModel, …).  Defaults
    become the schema's ``default``; parameters without a default
    land in ``required``."""
    sig = inspect.signature(fn)
    try:
        resolved_hints = get_type_hints(fn, include_extras=True)
    except Exception:  # noqa: BLE001
        resolved_hints = {}

    field_definitions: dict[str, tuple[Any, Any]] = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError(
                f"@workflow target {fn.__qualname__!r} cannot use *args / "
                "**kwargs; declare each parameter explicitly."
            )
        annotation = resolved_hints.get(
            param_name,
            param.annotation
            if param.annotation is not inspect.Parameter.empty
            else str,
        )
        if param.default is inspect.Parameter.empty:
            field_definitions[param_name] = (annotation, Field(...))
        else:
            field_definitions[param_name] = (annotation, Field(default=param.default))

    if not field_definitions:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    Model = create_model(f"{fn.__name__}__InputModel", **field_definitions)
    schema = Model.model_json_schema()
    schema.pop("title", None)
    schema.setdefault("additionalProperties", False)
    return schema
