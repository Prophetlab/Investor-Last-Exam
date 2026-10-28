"""BenchResearch hv3.8 Compact ReAct as a composable harness plugin."""

from __future__ import annotations

from dataclasses import dataclass, replace

from prophet_bench_env.core.contracts import EpisodeContext, PublicTask
from prophet_bench_env.core.trajectory import JsonlTrajectorySink
from prophet_bench_env.ports.data_specs import DATA_TOOL_NAMES
from prophet_bench_env.ports.model import ModelAdapter
from prophet_bench_env.ports.plugins import AgentResult, PluginManifest
from prophet_bench_env.ports.tooling import ToolGateway

from .loop import CompactReactConfig, CompactReactLoop


def _manifest() -> PluginManifest:
    return PluginManifest(
        plugin_id="compact-react-hv3.8",
        version="0.1.0",
        contract_version="harness.v1",
        capabilities={
            "interaction": "compact_react",
            "model_binding": "injected",
            "data_tools": list(DATA_TOOL_NAMES),
        },
    )


@dataclass(frozen=True, slots=True)
class CompactReactAgent:
    """One bound `(Compact ReAct harness, model adapter)` combination."""

    model: ModelAdapter
    system_prompt: str | None = None
    config: CompactReactConfig | None = None

    def manifest(self) -> PluginManifest:
        return _manifest()

    def run(
        self,
        *,
        task: PublicTask,
        context: EpisodeContext,
        tools: ToolGateway,
        trajectory: JsonlTrajectorySink,
    ) -> AgentResult:
        result = CompactReactLoop(
            model=self.model,
            tools=tools,
            trajectory=trajectory,
            config=self.config,
            cancel_requested=context.cancel_requested,
        ).run(
            task_prompt=task.prompt,
            system_prompt=self.system_prompt,
        )
        return AgentResult(
            submission=result.submission,
            raw_output=result.raw_output,
            metadata={
                "episode_id": context.episode_id,
                "malformed_corrections": result.malformed_corrections,
                "submit_attempts": result.submit_attempts,
            },
        )


@dataclass(frozen=True, slots=True)
class CompactReactHarnessPlugin:
    """Factory allowing one harness implementation to bind many models."""

    system_prompt: str | None = None
    config: CompactReactConfig | None = None

    def manifest(self) -> PluginManifest:
        return _manifest()

    def build(self, *, model: ModelAdapter) -> CompactReactAgent:
        config = self.config or CompactReactConfig()
        settings = getattr(model, "settings", None)
        context_window_tokens = getattr(settings, "context_window_tokens", None)
        if config.context_window_characters is None and isinstance(
            context_window_tokens, int
        ):
            config = replace(
                config,
                context_window_characters=context_window_tokens * 4,
            )
        return CompactReactAgent(
            model=model,
            system_prompt=self.system_prompt,
            config=config,
        )
