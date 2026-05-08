"""Discovery for ``@workflow``-decorated coroutines.

Two entry points, mirroring the server-side plugin loader:

* :func:`load_workflows_from_module` — import a dotted module path
  and collect every workflow registered in its namespace.  Use when
  workflows are an installed Python package.
* :func:`load_workflows_from_directory` — import every ``*.py`` under
  the directory as a freestanding module and collect their
  workflows.  Quick iteration, no packaging required; ``_*.py`` is
  skipped so authors can drop helpers alongside.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

from ai_assistant_client.workflows.decorator import Workflow, get_workflow


log = logging.getLogger(__name__)


def load_workflows_from_module(module_path: str) -> list[Workflow]:
    """Import ``module_path`` (dotted form) and return every workflow
    registered in it via the ``@workflow`` decorator.

    Propagates :class:`ImportError` so the caller can surface a
    clear startup error instead of silently skipping a misconfigured
    module path.
    """
    module = importlib.import_module(module_path)
    workflows: list[Workflow] = []
    for attr_name in dir(module):
        wf = get_workflow(getattr(module, attr_name))
        if wf is not None:
            workflows.append(wf)
    log.info(
        "Loaded %d workflow(s) from module %s", len(workflows), module_path
    )
    return workflows


def load_workflows_from_directory(directory: str | Path) -> list[Workflow]:
    """Import every ``*.py`` under ``directory`` and return all
    workflows found.  Missing directory is silently OK; broken
    files are warned + skipped (one bad file shouldn't take down
    the rest of the catalog).
    """
    root = Path(directory)
    if not root.is_dir():
        return []

    workflows: list[Workflow] = []
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        mod_name = f"_aai_workflow_{root.name}_{path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            log.warning("Could not load workflow file %s", path)
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as err:  # noqa: BLE001
            log.warning("Skipping workflow file %s: %s", path.name, err)
            del sys.modules[mod_name]
            continue
        for attr_name in dir(module):
            wf = get_workflow(getattr(module, attr_name))
            if wf is not None:
                workflows.append(wf)
    log.info("Loaded %d workflow(s) from %s", len(workflows), root)
    return workflows
