"""Artifact storage port used by all pipeline phases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ArtifactStore(Protocol):
    """Resolve and persist artifacts without prescribing a physical backend."""

    def task_path(self, run_id: str, task_id: str) -> Path: ...

    def rubric_path(self, run_id: str, task_id: str) -> Path: ...

    def trajectory_path(
        self,
        run_id: str,
        task_id: str,
        harness_id: str,
        model_id: str,
        episode_id: str,
    ) -> Path: ...

    def submission_path(
        self,
        run_id: str,
        task_id: str,
        harness_id: str,
        model_id: str,
        episode_id: str,
    ) -> Path: ...

    def evaluation_path(
        self,
        run_id: str,
        task_id: str,
        harness_id: str,
        model_id: str,
        episode_id: str,
    ) -> Path: ...

    def plugin_artifact_path(
        self,
        phase: str,
        plugin_id: str,
        run_id: str,
        task_id: str,
        harness_id: str,
        model_id: str,
        episode_id: str,
    ) -> Path: ...

    def workspace_path(self, episode_id: str) -> Path: ...

    def write_task(self, bundle: Any, *, run_id: str) -> Path: ...

    def write_rubric(self, bundle: Any, *, run_id: str) -> Path: ...

    def write_submission(self, path: Path, payload: dict[str, Any]) -> None: ...

    def write_evaluation(self, path: Path, payload: dict[str, Any]) -> None: ...
