"""Host-owned tools and models; the solver's Docker workspace has no network or keys."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from threading import BoundedSemaphore

from plugins.harness.compact_react import CompactReactConfig
from prophet_bench_env import RegisteredTool
from prophet_bench_env.adapters.data_tools import DataLakeTools
from prophet_bench_env.adapters.sdk_data_lake import SDKDataLakeClient
from prophet_bench_env.core.model_audit import bind_model_audit
from prophet_bench_env.core.resources import LimitedModelAdapter, ResourceGuards
from prophet_bench_env.integrations.harbor.runtime import TrialRuntime
from prophet_bench_env.ports.plugins import AgentResult, PluginManifest
from prophet_bench_env.ports.tooling import ToolRegistration

from .config import boolean, data_environment, route, values
from .corrected_checks import table_receipt
from .harness import ILELoop
from .models import model
from .public_submission import bind_submission

RESOURCES = ResourceGuards(BoundedSemaphore(8), BoundedSemaphore(8), None)


@lru_cache
def reader_class():
    from prophet_bench_env.adapters import model_reader

    name = "prophet_bench_env.adapters._ile_document_first"
    spec = importlib.util.spec_from_file_location(name, model_reader.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    def document_first(query, content, metadata):
        return (
            "Source material and coverage:\n"
            + content
            + "\n\nSource identifiers and requested scopes:\n"
            + json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            + "\n\nReading request:\n"
            + query
        )

    module._user_text = document_first
    return module.ModelBackedReader


@dataclass
class Agent:
    adapter: object
    system: str
    env: dict
    seconds: int

    def manifest(self):
        return PluginManifest("ile-public-react", "1.0.0", "harness.v1")

    def run(self, *, task, context, tools, trajectory):
        window = self.adapter.settings.context_window_tokens
        config = CompactReactConfig(
            context_window_characters=window * 4 if window else None,
            compaction_ratio=float(self.env.get("ILE_COMPACTION_RATIO", ".85")),
            recent_messages_after_compaction=int(self.env.get("ILE_RECENT_MESSAGES", "12")),
            max_submit_retries=task.submission_policy.max_attempts,
            max_malformed_corrections=int(self.env.get("ILE_MAX_MALFORMED_CORRECTIONS", "5")),
            max_tool_calls_per_response=int(self.env.get("ILE_MAX_TOOL_CALLS_PER_RESPONSE", "16")),
            wall_clock_seconds=self.seconds,
            enforce_tool_allowlist=True,
        )
        result = ILELoop(
            model=self.adapter,
            tools=tools,
            trajectory=trajectory,
            config=config,
            cancel_requested=context.cancel_requested,
        ).run(task_prompt=task.prompt, system_prompt=self.system)
        return AgentResult(
            result.submission, result.raw_output, {"submit_attempts": result.submit_attempts}
        )


def create_runtime(*, task, backend, trajectory, model_id, logs_dir, public_config):
    config = json.loads(Path(public_config).read_text())
    env = values()
    if config["task_id"] != task.task_id:
        raise ValueError("Task/runtime mismatch")
    data_environment(env)
    solver_route = route("SOLVER", env)
    reader_route = (
        solver_route if boolean(env.get("READER_REUSE_SOLVER", "true")) else route("READER", env)
    )
    solver = reader = client = data = None
    try:
        solver = LimitedModelAdapter(model(solver_route), RESOURCES)
        reader = LimitedModelAdapter(model(reader_route), RESOURCES)
        bind_model_audit(solver, trajectory=trajectory, root=Path(logs_dir) / "model-audit")
        bind_model_audit(reader, trajectory=trajectory, root=Path(logs_dir) / "reader-audit")
        client = SDKDataLakeClient.from_scope(
            cutoff=task.cutoff,
            datasets=tuple(task.capability_set["datasets"]),
            run_id=trajectory.episode_id,
            timezone=task.timezone,
            base_url=env.get("DATALAKE_BASE_URL", "https://datalake.prophet-lab.cn"),
            env_file="/dev/null",
        )
        with RESOURCES.datalake():
            grant = client.preflight_scope()
        trajectory.append("datalake_preflight", grant, actor="resource_manager")
        data = DataLakeTools(
            client=client,
            backend=backend,
            command_backend=backend,
            reader=reader_class().from_default(reader, trajectory=trajectory),
            resource_guard=RESOURCES.datalake,
            allowed_datasets=tuple(task.capability_set["datasets"]),
            trajectory=trajectory,
        )
        definitions = {t["function"]["name"]: t["function"] for t in config["tool_definitions"]}
        receipts = {}
        registered = []
        for tool in data.registered_tools():
            original = tool.handler
            name = tool.definition.name

            def handler(call, fn=original):
                result = fn(call)
                if result.name in ("table_read", "table_download"):
                    receipt = table_receipt(result, data)
                    if receipt:
                        receipts[receipt["ref"]] = receipt
                return result

            fn = definitions[name]
            definition = replace(
                tool.definition, description=fn["description"], input_schema=fn["parameters"]
            )
            registered.append(RegisteredTool(definition, handler, tool.submission_review))
        registered.append(bind_submission(task, {}, data, backend, receipts))
    except BaseException:
        if data:
            data.close()
        elif client:
            client.close()
        if reader:
            reader.close()
        if solver:
            solver.close()
        raise

    def close_tools():
        try:
            data.close()
        finally:
            reader.close()

    return TrialRuntime(
        Agent(solver, config["system_prompt"], env, config["wall_clock_seconds"]),
        ToolRegistration(tuple(registered), close_tools),
        solver.close,
    )
