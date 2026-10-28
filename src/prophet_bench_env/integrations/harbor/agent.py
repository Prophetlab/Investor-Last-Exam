"""Native Harbor external agent; no nested runner or second sandbox."""

from __future__ import annotations

import asyncio
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import Field

from ...core.contracts import EpisodeContext, RegisteredTool
from ...core.trajectory import JsonlTrajectorySink
from ...ports.tooling import ToolGateway
from .backend import HarborBackend
from .runtime import TrialRuntime
from .serialization import load_entrypoint, write_json
from .trajectory import export_atif
from .workspace import prepare_workspace


class FinancialAgentOptions(AgentOptions):
    runtime_factory: str
    runtime_kwargs: dict = Field(default_factory=dict)


def resolve_runtime_kwargs(value, task_root: Path):
    """Resolve only declared runtime/ assets, never private scoring files."""
    if isinstance(value, dict):
        if set(value) == {"$runtime_file"}:
            relative = Path(value["$runtime_file"])
            root = (task_root / "runtime").resolve()
            path = (root / relative).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not path.is_relative_to(root)
                or not path.is_file()
            ):
                raise ValueError(
                    "runtime file must be a regular file inside this task's runtime directory"
                )
            return str(path)
        return {
            key: resolve_runtime_kwargs(item, task_root) for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_runtime_kwargs(item, task_root) for item in value]
    return value


class FinancialAgent(BaseAgent):
    options_model = FinancialAgentOptions
    capabilities = AgentCapabilities(atif=True)

    @staticmethod
    def name() -> str:
        return "prophet-financial-agent"

    def version(self) -> str:
        return "0.2.0a0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if environment.network_policy.network_mode.value != "no-network":
            raise ValueError("financial agent requires an offline workspace")
        self.task_root, self.public_task = await prepare_workspace(
            environment, self.logs_dir
        )
        result = await environment.exec(
            "python -c 'import numpy, scipy, pandas, openpyxl, pyarrow, duckdb'",
            timeout_sec=90,
        )
        if result.return_code:
            raise RuntimeError("sandbox is missing required Python packages")

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if instruction.strip() != self.public_task.prompt.strip():
            raise ValueError("instruction does not match frozen public task")
        backend = HarborBackend(environment, asyncio.get_running_loop())
        episode_id = str(
            self.context_id or self.session_id or self.logs_dir.parent.name
        )
        trajectory = JsonlTrajectorySink(self.logs_dir / "events.jsonl", episode_id)
        trajectory.append("task_presented", {"task": self.public_task.as_dict()})
        runtime: TrialRuntime | None = None
        definitions: list[dict] = []
        gateway = None

        def work():
            nonlocal runtime, definitions, gateway
            options = resolve_runtime_kwargs(
                self.options.runtime_kwargs, self.task_root
            )
            if set(options) & {"task", "backend", "trajectory", "model_id", "logs_dir"}:
                raise ValueError(
                    "runtime kwargs cannot override environment-owned arguments"
                )
            runtime = load_entrypoint(self.options.runtime_factory)(
                task=self.public_task,
                backend=backend,
                trajectory=trajectory,
                model_id=self.model_name,
                logs_dir=self.logs_dir,
                **options,
            )
            try:
                tools = list(runtime.tools.tools)
                submits = [t for t in tools if t.definition.side_effect == "submission"]
                if (
                    len(submits) != 1
                    or submits[0].definition != self.public_task.submit_tool
                ):
                    raise ValueError(
                        "trial must register exactly this task's submission tool"
                    )

                def wrap(tool):
                    def receive(call):
                        if backend.cancelled.is_set():
                            raise asyncio.CancelledError("trial cancelled")
                        return tool.handler(call)

                    return RegisteredTool(
                        tool.definition, receive, tool.submission_review
                    )

                tools = [wrap(tool) for tool in tools]
                gateway = ToolGateway.from_tools(
                    tools, submission_policy=self.public_task.submission_policy
                )
                definitions = [
                    item.as_provider_schema() for item in gateway.definitions()
                ]
                runtime.agent.run(
                    task=self.public_task,
                    context=EpisodeContext(
                        episode_id,
                        episode_id,
                        self.public_task,
                        "/app/workspace",
                        metadata={"runtime": "harbor"},
                        cancel_requested=backend.cancelled.is_set,
                    ),
                    tools=gateway,
                    trajectory=trajectory,
                )
                if gateway.submission.final is None:
                    raise RuntimeError(
                        "agent returned without an accepted submission tool call"
                    )
            finally:
                runtime.close()

        try:
            await asyncio.to_thread(work)
            record = gateway.submission.record(
                self.public_task,
                episode_id=episode_id,
                harness_id=runtime.agent.manifest().plugin_id,
                model_id=self.model_name,
            )
            write_json(self.logs_dir / "submission.json", record)
            await environment.upload_file(
                self.logs_dir / "submission.json", "/logs/artifacts/submission.json"
            )
            context.metadata = {
                "task_id": self.public_task.task_id,
                "submission_received": record["submission_state"]["received"],
                "submission_state": record["submission_state"],
            }
        except BaseException as error:
            if gateway is not None and gateway.submission.final is not None:
                record = gateway.submission.record(
                    self.public_task,
                    episode_id=episode_id,
                    harness_id=runtime.agent.manifest().plugin_id,
                    model_id=self.model_name,
                )
                write_json(self.logs_dir / "submission.json", record)
            trajectory.append("rollout_failed", {"error_type": type(error).__name__})
            context.metadata = {
                "task_id": self.public_task.task_id,
                "failure_classification": type(error).__name__,
                "submission_state": record["submission_state"]
                if gateway is not None and gateway.submission.final is not None
                else None,
            }
            raise
        finally:
            backend.close()
            export_atif(
                trajectory.path,
                self.logs_dir / "trajectory.json",
                episode_id=episode_id,
                model_name=self.model_name,
                tool_definitions=definitions,
            )
            trajectory.close()
