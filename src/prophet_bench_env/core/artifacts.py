"""Deterministic artifact layout for authoring, rollout and evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import TaskBundle


def _component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "unknown"


@dataclass(frozen=True, slots=True)
class FilesystemArtifactStore:
    """Write public task and private run outputs without mixing their paths."""

    root: Path = Path("var")

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())

    def task_path(self, run_id: str, task_id: str) -> Path:
        return (
            self.root
            / "authoring"
            / "public"
            / _component(run_id)
            / _component(task_id)
            / "task.json"
        )

    def rubric_path(self, run_id: str, task_id: str) -> Path:
        return (
            self.root
            / "authoring"
            / "evaluator"
            / _component(run_id)
            / _component(task_id)
            / "rubric.json"
        )

    def trajectory_path(
        self,
        run_id: str,
        task_id: str,
        harness_id: str,
        model_id: str,
        episode_id: str,
    ) -> Path:
        return (
            self.root
            / "rollouts"
            / _component(run_id)
            / _component(task_id)
            / _component(harness_id)
            / _component(model_id)
            / f"{_component(episode_id)}.trajectory.jsonl"
        )

    def submission_path(
        self,
        run_id: str,
        task_id: str,
        harness_id: str,
        model_id: str,
        episode_id: str,
    ) -> Path:
        return self.root / self._episode_path(
            "submissions",
            run_id,
            task_id,
            harness_id,
            model_id,
            episode_id,
            "submission.json",
        )

    def evaluation_path(
        self,
        run_id: str,
        task_id: str,
        harness_id: str,
        model_id: str,
        episode_id: str,
    ) -> Path:
        return self.root / self._episode_path(
            "evaluations",
            run_id,
            task_id,
            harness_id,
            model_id,
            episode_id,
            "evaluation.json",
        )

    def workspace_path(self, episode_id: str) -> Path:
        return self.root / "workspaces" / _component(episode_id)

    def plugin_artifact_path(
        self,
        phase: str,
        plugin_id: str,
        run_id: str,
        task_id: str,
        harness_id: str,
        model_id: str,
        episode_id: str,
    ) -> Path:
        """Return one isolated directory for plugin-owned audit artifacts."""

        return (
            self.root
            / "plugin-artifacts"
            / _component(phase)
            / _component(plugin_id)
            / _component(run_id)
            / _component(task_id)
            / _component(harness_id)
            / _component(model_id)
            / _component(episode_id)
        )

    def write_task(self, bundle: TaskBundle, *, run_id: str) -> Path:
        path = self.task_path(run_id, bundle.public_task.task_id)
        self._write_json(
            path,
            {
                "public_task": bundle.public_task.as_dict(),
                "capability_set": bundle.capability_set,
            },
        )
        return path

    def write_rubric(self, bundle: TaskBundle, *, run_id: str) -> Path:
        path = self.rubric_path(run_id, bundle.public_task.task_id)
        self._write_json(
            path,
            {
                "rubric": bundle.rubric.as_dict(),
                "rubric_sha256": bundle.rubric.sha256,
            },
        )
        path.chmod(0o600)
        return path

    def write_submission(self, path: Path, payload: dict[str, Any]) -> None:
        self._write_json(path, payload)

    def write_evaluation(self, path: Path, payload: dict[str, Any]) -> None:
        self._write_json(path, payload)

    @staticmethod
    def _episode_path(
        kind: str,
        run_id: str,
        task_id: str,
        harness_id: str,
        model_id: str,
        episode_id: str,
        filename: str,
    ) -> Path:
        return (
            Path(kind)
            / _component(run_id)
            / _component(task_id)
            / _component(harness_id)
            / _component(model_id)
            / _component(episode_id)
            / filename
        )

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        # Custom stores may still return relative paths; the filesystem store does not.
        if not path.is_absolute():
            path = self.root / path
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
