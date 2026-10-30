"""Provider-neutral contracts shared by the environment and plugins.

The public surface deliberately uses plain dataclasses and JSON-compatible values. Provider SDK
objects stay inside adapters and never become part of an episode observation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol

JsonObject = dict[str, Any]
ToolSideEffect = Literal["read_only", "workspace_write", "submission"]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: JsonObject
    side_effect: ToolSideEffect = "read_only"
    parallel_safe: bool = True

    def as_provider_schema(self) -> JsonObject:
        """Return the OpenAI-style function schema used by most model adapters."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Any


@dataclass(frozen=True, slots=True)
class ToolError:
    code: str
    message: str
    retryable: bool = False
    trace_id: str | None = None
    details: JsonObject = field(default_factory=dict)

    def as_dict(self) -> JsonObject:
        result: JsonObject = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.details:
            result["details"] = self.details
        return result


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    name: str
    ok: bool
    output: Any = None
    error: ToolError | None = None
    metadata: JsonObject = field(default_factory=dict)

    def as_dict(self) -> JsonObject:
        result: JsonObject = {
            "call_id": self.call_id,
            "name": self.name,
            "ok": self.ok,
            "output": self.output,
        }
        if self.error is not None:
            result["error"] = self.error.as_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass(frozen=True, slots=True)
class SubmissionPolicy:
    max_attempts: int = 3
    on_exhausted: Literal["reject", "seal_last"] = "reject"

    def __post_init__(self):
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if self.on_exhausted not in {"reject", "seal_last"}:
            raise ValueError("unknown submission exhaustion policy")

    def as_dict(self):
        return {"max_attempts": self.max_attempts, "on_exhausted": self.on_exhausted}


@dataclass(frozen=True, slots=True)
class PublicTask:
    benchmark_id: str
    task_id: str
    task_type: str
    task_version: str
    prompt: str
    cutoff: datetime
    timezone: str
    target_definition: JsonObject
    capability_set: JsonObject
    submit_tool: ToolDefinition
    metadata: JsonObject = field(default_factory=dict)
    submission_policy: SubmissionPolicy = field(default_factory=SubmissionPolicy)

    def as_dict(self) -> JsonObject:
        return {
            "benchmark_id": self.benchmark_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_version": self.task_version,
            "prompt": self.prompt,
            "cutoff": self.cutoff.isoformat(),
            "timezone": self.timezone,
            "target_definition": self.target_definition,
            "capability_set": self.capability_set,
            "submit_tool": self.submit_tool.as_provider_schema(),
            "metadata": self.metadata,
            "submission_policy": self.submission_policy.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class Assignment:
    benchmark_id: str
    task_id: str | None = None
    pipeline_id: str | None = None
    task_type: str | None = None
    seed: int | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RubricCriterion:
    """One author-defined scoring criterion executed by an evaluator plugin."""

    criterion_id: str
    description: str
    max_score: float
    scorer_ref: str
    weight: float | None = None
    required: bool = True
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.criterion_id.strip() or not self.description.strip():
            raise ValueError("rubric criterion id and description are required")
        if not self.scorer_ref.strip():
            raise ValueError("rubric criterion scorer_ref is required")
        if self.max_score <= 0:
            raise ValueError("rubric criterion max_score must be positive")
        if self.weight is not None and self.weight < 0:
            raise ValueError("rubric criterion weight cannot be negative")

    def as_dict(self) -> JsonObject:
        return {
            "criterion_id": self.criterion_id,
            "description": self.description,
            "max_score": self.max_score,
            "scorer_ref": self.scorer_ref,
            "weight": self.weight,
            "required": self.required,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class RubricBundle:
    """Complete per-task scoring package emitted by authoring and hidden from rollout."""

    rubric_id: str
    rubric_version: str
    task_id: str
    task_type: str
    criteria: tuple[RubricCriterion, ...]
    max_score: float
    aggregation: JsonObject
    reference_answer: JsonObject
    answer_version: str
    answer_evidence_refs: tuple[str, ...] = ()
    payload: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rubric_id.strip() or not self.rubric_version.strip():
            raise ValueError("rubric id and version are required")
        if not self.task_id.strip() or not self.task_type.strip():
            raise ValueError("rubric task identity is required")
        if not self.criteria:
            raise ValueError("rubric must contain at least one criterion")
        criterion_ids = [item.criterion_id for item in self.criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("rubric criterion ids must be unique")
        if self.max_score <= 0:
            raise ValueError("rubric max_score must be positive")
        if not self.aggregation:
            raise ValueError("rubric aggregation policy is required")
        if not self.reference_answer:
            raise ValueError("rubric reference_answer is required")
        if not self.answer_version.strip():
            raise ValueError("rubric answer_version is required")

    def as_dict(self) -> JsonObject:
        return {
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "criteria": [item.as_dict() for item in self.criteria],
            "max_score": self.max_score,
            "aggregation": self.aggregation,
            "reference_answer": self.reference_answer,
            "answer_version": self.answer_version,
            "answer_evidence_refs": list(self.answer_evidence_refs),
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TaskBundle:
    public_task: PublicTask
    rubric: RubricBundle
    private_context_ref: str
    capability_set: JsonObject
    tool_definitions: tuple[ToolDefinition, ...]
    registered_tools: tuple[RegisteredTool, ...] = ()
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EpisodeContext:
    episode_id: str
    run_id: str
    task: PublicTask
    workspace_root: str
    metadata: JsonObject = field(default_factory=dict)
    resources: Any | None = field(default=None, repr=False, compare=False)
    cancel_requested: Callable[[], bool] | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class RolloutToolContext:
    """Public tool-construction input that deliberately excludes private rubric data."""

    public_task: PublicTask
    capability_set: JsonObject
    tool_definitions: tuple[ToolDefinition, ...]
    registered_tools: tuple[RegisteredTool, ...]
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AssistantAction:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


Action = AssistantAction


@dataclass(frozen=True, slots=True)
class SubmissionArtifact:
    task_id: str
    episode_id: str
    payload: Any
    received_at: str
    schema_version: str
    payload_sha256: str
    raw_payload: Any = None
    validation_report: JsonObject = field(default_factory=dict)
    reference_receipts: tuple[JsonObject, ...] = ()
    attachment_receipts: tuple[JsonObject, ...] = ()
    evidence: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """Stable run identity and private inputs supplied to an evaluator plugin.

    The evaluator owns its internal verifier/judge implementation. Core only
    provides identity, an isolated artifact directory and caller-supplied
    private values; no benchmark-specific scoring fields live here.
    """

    run_id: str
    episode_id: str
    harness_id: str
    model_id: str
    private_context_ref: str
    artifact_root: str
    metadata: JsonObject = field(default_factory=dict)
    resources: Any | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class RubricResult:
    rubric_id: str
    score: float | None
    max_score: float
    status: str
    diagnostic: str | None = None
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    evaluator_id: str
    evaluator_version: str
    rubric_results: tuple[RubricResult, ...]
    total_score: float | None
    reward: float | None
    reward_pending: bool = False
    failure_classification: str | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Observation:
    episode_id: str
    step_id: int
    kind: str
    messages: tuple[JsonObject, ...] = ()
    available_tools: tuple[str, ...] = ()
    budgets: JsonObject = field(default_factory=dict)
    done: bool = False
    metadata: JsonObject = field(default_factory=dict)

    def as_dict(self) -> JsonObject:
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "kind": self.kind,
            "messages": list(self.messages),
            "available_tools": list(self.available_tools),
            "budgets": self.budgets,
            "done": self.done,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: Observation
    reward: float | None = 0.0
    terminated: bool = False
    truncated: bool = False
    info: JsonObject = field(default_factory=dict)

    def as_dict(self) -> JsonObject:
        return {
            "observation": self.observation.as_dict(),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }


class Reader(Protocol):
    def read(
        self,
        *,
        query: str,
        content: str,
        metadata: Mapping[str, Any],
        image: bytes | None = None,
    ) -> str: ...


ToolHandler = Callable[[ToolCall], ToolResult]


@dataclass(frozen=True, slots=True)
class SubmissionReview:
    """Trusted plugin validation/evidence hook, invoked even when schema rejects input."""

    validator_id: str
    validator_version: str
    issues: tuple[Any, ...] = ()
    reference_receipts: tuple[JsonObject, ...] = ()
    attachment_receipts: tuple[JsonObject, ...] = ()


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    definition: ToolDefinition
    handler: ToolHandler
    submission_review: Callable[[ToolCall], SubmissionReview] | None = None


def as_json_object(value: Mapping[str, Any]) -> JsonObject:
    return dict(value)


def as_json_list(values: Sequence[Mapping[str, Any]]) -> list[JsonObject]:
    return [dict(value) for value in values]
