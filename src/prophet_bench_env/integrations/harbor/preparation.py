"""Bounded authoring/export only; rollout scheduling belongs to Harbor."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .tasks import export_task


def prepare_tasks(
    assignments,
    *,
    authoring,
    verifier_factory,
    output: Path,
    workers: int = 8,
    agent_timeout_sec: int = 1800,
    tool_factory: str | None = None,
    tool_kwargs: dict | None = None,
):
    if workers < 1:
        raise ValueError("authoring workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        bundles = list(
            pool.map(lambda a: authoring.generate(a, a.seed or 0), assignments)
        )
    identifiers = [b.public_task.task_id.casefold() for b in bundles]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("authoring produced duplicate task IDs")
    for bundle in bundles:
        report = authoring.validate(bundle)
        if not report.ok:
            raise ValueError("authoring validation failed: " + "; ".join(report.issues))
        export_task(
            bundle,
            output / bundle.public_task.task_id,
            verifier_factory(bundle),
            agent_timeout_sec=agent_timeout_sec,
            tool_factory=tool_factory,
            tool_kwargs=tool_kwargs,
        )
    return bundles
