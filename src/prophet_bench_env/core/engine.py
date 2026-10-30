"""Episode engine composition for authoring, harness and evaluator plugins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..ports.artifacts import ArtifactStore
from ..ports.plugins import AuthoringPipeline, Evaluator, ValidationReport
from ..ports.tooling import ToolGateway, ToolRegistration
from .artifacts import FilesystemArtifactStore
from .contracts import (
    Assignment,
    AssistantAction,
    EpisodeContext,
    Observation,
    RegisteredTool,
    StepResult,
    TaskBundle,
)
from .harness import CanonicalHarness
from .trajectory import JsonlTrajectorySink

ToolProvider = Callable[
    [TaskBundle, EpisodeContext], tuple[RegisteredTool, ...] | ToolRegistration
]


@dataclass(slots=True)
class EpisodeEngine:
    authoring: AuthoringPipeline
    evaluator: Evaluator | None = None
    tool_provider: ToolProvider | None = None
    artifacts: ArtifactStore | None = None
    trajectory_root: Path = Path("var/trajectories")
    workspace_root: Path = Path("var/workspaces")
    _harness: CanonicalHarness | None = None
    _bundle: TaskBundle | None = None
    _tool_closer: Callable[[], None] | None = None

    def reset(
        self,
        assignment: Assignment,
        *,
        seed: int = 0,
        episode_id: str | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        if self._harness is not None:
            self.close()
        bundle = self.authoring.generate(assignment, seed)
        report = self.authoring.validate(bundle)
        if not report.ok:
            raise ValueError(f"authoring validation failed: {list(report.issues)}")
        if bundle.public_task.cutoff is None:
            raise ValueError(
                "task cutoff must be a real event or information availability time"
            )
        if assignment.task_type is not None and (
            bundle.public_task.task_type != assignment.task_type
        ):
            raise ValueError(
                "authoring returned a task type different from the assignment"
            )
        if bundle.rubric.task_id != bundle.public_task.task_id:
            raise ValueError("rubric task_id must match public task")
        if bundle.rubric.task_type != bundle.public_task.task_type:
            raise ValueError("rubric task_type must match public task")
        if self.evaluator is not None:
            rubric_report = self.evaluator.validate_inputs(
                bundle.public_task,
                bundle.rubric,
            )
            if not rubric_report.ok:
                raise ValueError(
                    "evaluator rejected authoring rubric before episode: "
                    + "; ".join(rubric_report.issues)
                )
        episode_id = episode_id or f"ep_{uuid4().hex}"
        run_id = assignment.metadata.get("run_id") or f"run_{uuid4().hex[:12]}"
        artifacts = self.artifacts or FilesystemArtifactStore(
            self.trajectory_root.parent
        )
        artifacts.write_task(bundle, run_id=str(run_id))
        artifacts.write_rubric(bundle, run_id=str(run_id))
        workspace = self.workspace_root / episode_id
        workspace.mkdir(mode=0o700, parents=True, exist_ok=True)
        context = EpisodeContext(
            episode_id=episode_id,
            run_id=str(run_id),
            task=bundle.public_task,
            workspace_root=str(workspace),
            metadata={},
        )
        tools = list(bundle.registered_tools)
        if self.tool_provider is not None:
            provided = self.tool_provider(bundle, context)
            if isinstance(provided, ToolRegistration):
                self._tool_closer = provided.close
                tools.extend(provided.tools)
            else:
                tools.extend(provided)
        if not tools:
            self.close()
            raise ValueError(
                "task bundle must register its task-specific submit tool and other tools"
            )
        if not any(tool.definition.side_effect == "submission" for tool in tools):
            self.close()
            raise ValueError("task bundle must provide an executable submission tool")
        submissions = [
            tool.definition
            for tool in tools
            if tool.definition.side_effect == "submission"
        ]
        if len(submissions) != 1 or submissions[0] != bundle.public_task.submit_tool:
            self.close()
            raise ValueError(
                "episode must expose exactly this task's public submission tool"
            )
        try:
            gateway = ToolGateway.from_tools(
                tools, submission_policy=bundle.public_task.submission_policy
            )
        except BaseException:
            self.close()
            raise
        trajectory = JsonlTrajectorySink(
            self.trajectory_root / f"{episode_id}.jsonl",
            episode_id,
        )
        trajectory.append(
            "episode_started",
            {
                "assignment": {
                    "benchmark_id": assignment.benchmark_id,
                    "task_id": assignment.task_id,
                    "task_type": assignment.task_type,
                    "pipeline_id": assignment.pipeline_id,
                    "seed": seed,
                },
                "authoring_plugin": self.authoring.manifest().plugin_id,
            },
            step_id=0,
        )
        self._harness = CanonicalHarness(
            task=bundle.public_task,
            rubric=bundle.rubric,
            private_context_ref=bundle.private_context_ref,
            context=context,
            gateway=gateway,
            evaluator=self.evaluator,
            trajectory=trajectory,
        )
        self._bundle = bundle
        return self._harness.reset(), {
            "episode_id": episode_id,
            "task": bundle.public_task.as_dict(),
        }

    def step(self, action: AssistantAction | dict[str, Any]) -> StepResult:
        if self._harness is None:
            raise RuntimeError("reset must be called before step")
        return self._harness.step(action)

    def close(self) -> None:
        try:
            if self._harness is not None:
                self._harness.close()
        finally:
            self._harness = None
            self._bundle = None
            closer, self._tool_closer = self._tool_closer, None
            if closer:
                closer()

    @property
    def task_bundle(self) -> TaskBundle | None:
        return self._bundle


def accepted_report() -> ValidationReport:
    """Convenience for tiny plugins and examples."""

    return ValidationReport(ok=True)
