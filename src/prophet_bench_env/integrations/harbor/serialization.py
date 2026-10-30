"""Frozen public/private task records. No Python objects are pickled."""

from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ...core.contracts import (
    PublicTask,
    RubricBundle,
    RubricCriterion,
    SubmissionPolicy,
    ToolDefinition,
)


def read_public(value: dict[str, Any]) -> PublicTask:
    value = dict(value)
    value["submission_policy"] = SubmissionPolicy(**value.get("submission_policy", {}))
    value["cutoff"] = datetime.fromisoformat(value["cutoff"])
    if value["cutoff"].tzinfo is None:
        raise ValueError("task cutoff must include a timezone")
    function = value["submit_tool"]["function"]
    value["submit_tool"] = ToolDefinition(
        function["name"],
        function["description"],
        function["parameters"],
        side_effect="submission",
        parallel_safe=False,
    )
    return PublicTask(**value)


def read_rubric(value: dict[str, Any]) -> RubricBundle:
    value = dict(value)
    value["criteria"] = tuple(RubricCriterion(**row) for row in value["criteria"])
    value["answer_evidence_refs"] = tuple(value.get("answer_evidence_refs", ()))
    return RubricBundle(**value)


def load_entrypoint(reference: str):
    """Load a trusted operator-selected entrypoint, never a tool argument."""
    module, separator, name = reference.partition(":")
    if not separator or not module or not name or name.startswith("_"):
        raise ValueError("entrypoint must be module:public_attribute")
    return getattr(importlib.import_module(module), name)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)
