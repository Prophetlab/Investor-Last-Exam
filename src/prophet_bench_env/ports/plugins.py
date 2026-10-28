"""Plugin ports for authoring, harness and evaluation implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.contracts import (
    Assignment,
    EpisodeContext,
    EvaluationContext,
    EvaluationResult,
    JsonObject,
    Observation,
    PublicTask,
    RegisteredTool,
    RubricBundle,
    StepResult,
    SubmissionArtifact,
    TaskBundle,
    ToolDefinition,
)


@dataclass(frozen=True, slots=True)
class PluginManifest:
    plugin_id: str
    version: str
    contract_version: str
    capabilities: JsonObject = field(default_factory=dict)
    source_commit: str | None = None
    fixture_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    ok: bool
    issues: tuple[str, ...] = ()
    metadata: JsonObject = field(default_factory=dict)


class AuthoringPipeline(Protocol):
    def manifest(self) -> PluginManifest: ...

    def generate(self, assignment: Assignment, seed: int) -> TaskBundle: ...

    def validate(self, bundle: TaskBundle) -> ValidationReport: ...


class Evaluator(Protocol):
    def manifest(self) -> PluginManifest: ...

    def validate_inputs(
        self,
        task: PublicTask,
        rubric: RubricBundle,
    ) -> ValidationReport: ...

    def evaluate(
        self,
        task: PublicTask,
        rubric: RubricBundle,
        submission: SubmissionArtifact,
        context: EvaluationContext,
    ) -> EvaluationResult: ...


class Harness(Protocol):
    def manifest(self) -> PluginManifest: ...

    def definitions(self, context: EpisodeContext) -> tuple[ToolDefinition, ...]: ...

    def reset(self, task: PublicTask, context: EpisodeContext) -> Observation: ...

    def step(self, action: Any) -> StepResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentResult:
    """The only result a bound harness/model agent exposes to the rollout runner."""

    submission: Any
    raw_output: str | None = None
    metadata: JsonObject = field(default_factory=dict)


class Agent(Protocol):
    """A managed model+interaction policy for one episode."""

    def manifest(self) -> PluginManifest: ...

    def run(
        self,
        *,
        task: PublicTask,
        context: EpisodeContext,
        tools: Any,
        trajectory: Any,
    ) -> AgentResult: ...


class HarnessPlugin(Protocol):
    """Factory for a harness implementation, parameterized by a model adapter."""

    def manifest(self) -> PluginManifest: ...

    def build(self, *, model: Any) -> Agent: ...


@dataclass(frozen=True, slots=True)
class PipelineManifest:
    pipeline_id: str
    authoring: PluginManifest
    harness: PluginManifest
    evaluator: PluginManifest
    tool_spec_sha256: str
    public_bundle_sha256: str
    capabilities: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HarnessContext:
    episode: EpisodeContext
    tools: tuple[RegisteredTool, ...]
