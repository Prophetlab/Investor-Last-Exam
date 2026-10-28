"""A small canonical harness that can be driven by an external model policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ports.plugins import Evaluator
from ..ports.tooling import ToolGateway
from .contracts import (
    AssistantAction,
    EpisodeContext,
    EvaluationContext,
    EvaluationResult,
    Observation,
    PublicTask,
    RubricBundle,
    StepResult,
    ToolCall,
)
from .submission import artifact_from_record, received
from .trajectory import JsonlTrajectorySink


@dataclass(slots=True)
class CanonicalHarness:
    task: PublicTask
    rubric: RubricBundle
    private_context_ref: str
    context: EpisodeContext
    gateway: ToolGateway
    evaluator: Evaluator | None = None
    trajectory: JsonlTrajectorySink | None = None
    max_steps: int | None = None
    _step_id: int = 0
    _done: bool = False

    def reset(self) -> Observation:
        self._step_id = 0
        self._done = False
        self._event("task_presented", {"task": self.task.as_dict()}, step_id=0)
        return Observation(
            episode_id=self.context.episode_id,
            step_id=0,
            kind="task",
            messages=({"type": "task", "task": self.task.as_dict()},),
            available_tools=tuple(tool.name for tool in self.gateway.definitions()),
            budgets={
                "max_submission_attempts": self.gateway.submission.policy.max_attempts
            },
        )

    def step(self, action: AssistantAction | Mapping[str, Any]) -> StepResult:
        if self._done:
            return StepResult(
                self._observation("done", ()),
                terminated=True,
                info={"already_done": True},
            )
        if self.max_steps is not None and self._step_id >= self.max_steps:
            self._done = True
            self._event(
                "episode_truncated", {"reason": "max_steps"}, step_id=self._step_id
            )
            return StepResult(
                self._observation("truncated", ()),
                truncated=True,
                info={"reason": "max_steps"},
            )
        parsed, correction = _parse_action(
            action, submission_name=self.task.submit_tool.name
        )
        self._step_id += 1
        if correction is not None:
            self._event(
                "malformed_tool_batch",
                {"reason": correction},
                step_id=self._step_id,
                actor="model",
            )
            observation = self._observation(
                "correction", ({"type": "correction", "message": correction},)
            )
            return StepResult(observation, info={"correction": correction})
        assert parsed is not None
        self._event(
            "model_response",
            {
                "text": parsed.text,
                "tool_calls": [_call_dict(call) for call in parsed.tool_calls],
            },
            step_id=self._step_id,
            actor="model",
        )
        results = self.gateway.execute_batch(parsed.tool_calls)
        for result in results:
            self._event(
                "tool_result", result.as_dict(), step_id=self._step_id, actor="tool"
            )
        messages = tuple(
            {"type": "tool_result", **result.as_dict()} for result in results
        )
        submission_result = next(
            (
                result
                for result in results
                if result.name == self.task.submit_tool.name and received(result)
            ),
            None,
        )
        if submission_result is not None:
            record = self.gateway.submission.record(
                self.task,
                episode_id=self.context.episode_id,
                model_id="external-policy",
                harness_id="external-policy",
            )
            evaluation = self._evaluate_submission(record)
            self._done = True
            reward = (
                evaluation.reward
                if evaluation is not None
                and type(evaluation.reward) in {int, float}
                and math.isfinite(evaluation.reward)
                and not evaluation.reward_pending
                and not evaluation.failure_classification
                else None
            )
            info: dict[str, Any] = {
                "submission": submission_result.output,
                "submission_record": record,
                "submission_state": {
                    **record["submission_state"],
                    "grading_status": "completed"
                    if reward is not None
                    else "error"
                    if evaluation and evaluation.failure_classification
                    else "pending",
                },
            }
            if evaluation is not None:
                info["evaluation"] = _evaluation_dict(evaluation)
                if evaluation.reward_pending:
                    info["reward_pending"] = True
            else:
                info["evaluation_pending"] = True
            self._event(
                "episode_terminated",
                {"reason": "submission", **info},
                step_id=self._step_id,
            )
            return StepResult(
                self._observation("submission", messages, done=True),
                reward=reward,
                terminated=True,
                info=info,
            )
        if self.gateway.submission.final is not None:
            self._done = True
            record = self.gateway.submission.record(
                self.task,
                episode_id=self.context.episode_id,
                model_id="external-policy",
                harness_id="external-policy",
            )
            from .artifacts import FilesystemArtifactStore

            path = (
                Path(self.context.workspace_root)
                / "plugin-artifacts"
                / "evaluator"
                / "submission.json"
            )
            FilesystemArtifactStore(path.parent).write_submission(path, record)
            return StepResult(
                self._observation("submission_rejected", messages, done=True),
                reward=None,
                terminated=True,
                info={
                    "submission_record": record,
                    "submission_state": record["submission_state"],
                    "failure_classification": record["submission_state"][
                        "failure_classification"
                    ]
                    or "submission_rejected",
                },
            )
        return StepResult(self._observation("tool_results", messages))

    def close(self) -> None:
        if self.trajectory is not None:
            self._event("episode_closed", {"reason": "close"}, step_id=self._step_id)
            self.trajectory.close()

    def _evaluate_submission(self, record) -> EvaluationResult | None:
        submission = artifact_from_record(record, self.task, require_evidence=True)
        artifact_root = (
            Path(self.context.workspace_root) / "plugin-artifacts" / "evaluator"
        )
        artifact_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        from .artifacts import FilesystemArtifactStore

        FilesystemArtifactStore(artifact_root).write_submission(
            artifact_root / "submission.json", record
        )
        if self.evaluator is None:
            return None
        self._event(
            "submission_received",
            {"payload_sha256": submission.payload_sha256},
            step_id=self._step_id,
        )
        self._event(
            "evaluation_started",
            {"evaluator": self.evaluator.manifest().plugin_id},
            step_id=self._step_id,
        )
        artifact_root = (
            Path(self.context.workspace_root)
            / "plugin-artifacts"
            / "evaluator"
            / self.evaluator.manifest().plugin_id
        )
        artifact_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            result = self.evaluator.evaluate(
                self.task,
                self.rubric,
                submission,
                EvaluationContext(
                    run_id=self.context.run_id,
                    episode_id=self.context.episode_id,
                    harness_id="external-policy",
                    model_id="external-policy",
                    private_context_ref=self.private_context_ref,
                    artifact_root=str(artifact_root),
                ),
            )
            if result.reward is not None and (
                type(result.reward) not in {int, float}
                or not math.isfinite(result.reward)
            ):
                raise ValueError("evaluator reward must be a finite number or None")
        except Exception as error:  # noqa: BLE001 - evaluator failures are not zero scores
            manifest = self.evaluator.manifest()
            result = EvaluationResult(
                manifest.plugin_id,
                manifest.version,
                (),
                None,
                None,
                failure_classification="evaluator_error",
                metadata={"error_type": type(error).__name__},
            )
        self._event(
            "evaluation_finished", _evaluation_dict(result), step_id=self._step_id
        )
        return result

    def _observation(
        self, kind: str, messages: tuple[dict[str, Any], ...], *, done: bool = False
    ) -> Observation:
        return Observation(
            episode_id=self.context.episode_id,
            step_id=self._step_id,
            kind=kind,
            messages=messages,
            available_tools=tuple(tool.name for tool in self.gateway.definitions()),
            budgets={
                "submission_attempts": self.gateway.submission.attempts,
                "submission_attempts_remaining": max(
                    0,
                    self.gateway.submission.policy.max_attempts
                    - self.gateway.submission.attempts,
                ),
            },
            done=done,
        )

    def _event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        step_id: int | None = None,
        actor: str = "environment",
    ) -> None:
        if self.trajectory is not None:
            self.trajectory.append(event_type, payload, step_id=step_id, actor=actor)


def _parse_action(
    action: AssistantAction | Mapping[str, Any],
    *,
    submission_name=None,
) -> tuple[AssistantAction | None, str | None]:
    if isinstance(action, AssistantAction):
        return action, None
    if not isinstance(action, Mapping):
        return None, "Action must be an assistant_response object."
    if action.get("type", "assistant_response") != "assistant_response":
        return None, "Action type must be assistant_response."
    raw_calls = action.get("tool_calls", ())
    if raw_calls is None:
        raw_calls = ()
    if not isinstance(raw_calls, (list, tuple)):
        return (
            None,
            "tool_calls must be a list; resend the complete assistant response.",
        )
    calls: list[ToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, Mapping):
            return (
                None,
                "Every tool call must be an object; resend the complete assistant response.",
            )
        call_id = raw.get("id", raw.get("call_id"))
        name = raw.get("name")
        arguments = raw.get("arguments", {})
        if (
            not isinstance(call_id, str)
            or not call_id
            or not isinstance(name, str)
            or not name
            or (name != submission_name and not isinstance(arguments, Mapping))
        ):
            return (
                None,
                "Each tool call needs an id, name, and JSON-object arguments; resend the complete assistant response.",
            )
        calls.append(
            ToolCall(
                call_id=call_id,
                name=name,
                arguments=dict(arguments)
                if isinstance(arguments, Mapping)
                else arguments,
            )
        )
    text = action.get("text", "")
    return AssistantAction(text=str(text), tool_calls=tuple(calls)), None


def _call_dict(call: ToolCall) -> dict[str, Any]:
    return {"id": call.call_id, "name": call.name, "arguments": call.arguments}


def _evaluation_dict(result: EvaluationResult) -> dict[str, Any]:
    return {
        "evaluator_id": result.evaluator_id,
        "evaluator_version": result.evaluator_version,
        "total_score": result.total_score,
        "reward": result.reward,
        "reward_pending": result.reward_pending,
        "failure_classification": result.failure_classification,
        "rubric_results": [
            {
                "rubric_id": item.rubric_id,
                "score": item.score,
                "max_score": item.max_score,
                "status": item.status,
                "diagnostic": item.diagnostic,
                "evidence": list(item.evidence),
            }
            for item in result.rubric_results
        ],
    }
