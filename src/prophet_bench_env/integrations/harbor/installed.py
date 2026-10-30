"""Thin native Harbor agent adapters: shared tools, no replacement solver."""

import asyncio
import json

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.claude_code import ClaudeCode, ClaudeCodeOptions
from harbor.agents.installed.codex import Codex, CodexOptions
from harbor.agents.options import AgentOptions
from pydantic import Field

from ...core.trajectory import JsonlTrajectorySink
from ...ports.tooling import ToolRegistration
from .agent import resolve_runtime_kwargs
from .backend import HarborBackend
from .mcp_bridge import MailboxBridge
from .serialization import load_entrypoint, write_json
from .tool_session import ToolSession
from .workspace import prepare_workspace


class ToolOptions(AgentOptions):
    tool_factory: str | None = None
    tool_kwargs: dict = Field(default_factory=dict)
    omit_tools: list[str] = Field(default_factory=lambda: ["bash"])
    tool_timeout_seconds: int = Field(default=900, ge=1, le=3600)


class NativeClaudeOptions(ClaudeCodeOptions, ToolOptions):
    pass


class NativeCodexOptions(CodexOptions, ToolOptions):
    pass


class FinancialToolsMixin:
    # Resume needs stateful document references, so don't advertise support until
    # that state can be restored. Native configuration and ATIF remain available.
    capabilities = AgentCapabilities(atif=True, native_config=True)

    async def setup(self, environment):
        self.task_root, self.public_task = await prepare_workspace(
            environment, self.logs_dir
        )
        await super().setup(environment)

    async def run(self, instruction, environment, context):
        binding_path = self.task_root / "runtime/tools.json"
        binding = json.loads(binding_path.read_text()) if binding_path.is_file() else {}
        factory = self.options.tool_factory or binding.get("factory")
        if not factory:
            raise ValueError(
                "task needs a tools-only factory (runtime/tools.json or tool_factory)"
            )
        kwargs = resolve_runtime_kwargs(
            {**binding.get("kwargs", {}), **self.options.tool_kwargs}, self.task_root
        )
        if set(kwargs) & {"task", "backend", "trajectory", "model_id", "logs_dir"}:
            raise ValueError("tool kwargs cannot override environment-owned arguments")
        if any(server.name == "prophet" for server in self.mcp_servers):
            raise ValueError("MCP server name 'prophet' is reserved for task tools")
        backend = HarborBackend(environment, asyncio.get_running_loop())
        sink = JsonlTrajectorySink(
            self.logs_dir / "prophet-events.jsonl",
            str(self.context_id or self.session_id),
        )
        registration = None
        bridge = None
        pump = native = None
        try:
            factory_task = asyncio.create_task(
                asyncio.to_thread(
                    load_entrypoint(factory),
                    task=self.public_task,
                    backend=backend,
                    trajectory=sink,
                    model_id=self.model_name,
                    logs_dir=self.logs_dir,
                    **kwargs,
                )
            )
            try:
                registration = await asyncio.shield(factory_task)
            except asyncio.CancelledError:
                registration = await factory_task
                raise
            if not isinstance(registration, ToolRegistration):
                raise TypeError(
                    "tool factory must return ToolRegistration, not a solver runtime"
                )
            session = ToolSession(
                self.public_task,
                ToolRegistration(
                    tuple(
                        t
                        for t in registration.tools
                        if t.definition.name not in self.options.omit_tools
                    )
                ),
                sink,
            )
            self.tool_session = session
            bridge = MailboxBridge(
                environment,
                session,
                self.logs_dir,
                timeout=self.options.tool_timeout_seconds,
            )
            self.mcp_servers.append(await bridge.setup())
            pump = asyncio.create_task(bridge.pump())
            native = asyncio.create_task(super().run(instruction, environment, context))
            done, _ = await asyncio.wait(
                {pump, native}, return_when=asyncio.FIRST_COMPLETED
            )
            if pump in done:
                pump.result()
                raise RuntimeError("tool transport stopped before agent completion")
            await native
            record = session.submission_record(
                model_id=self.model_name, harness_id=self.name()
            )
            write_json(self.logs_dir / "submission.json", record)
            await environment.upload_file(
                self.logs_dir / "submission.json", "/logs/artifacts/submission.json"
            )
            context.metadata = {
                **(context.metadata or {}),
                "submission_state": record["submission_state"],
                "submission_received": record["submission_state"]["received"],
            }
        finally:
            try:
                if (
                    hasattr(self, "tool_session")
                    and self.tool_session.gateway.submission.final is not None
                ):
                    record = self.tool_session.submission_record(
                        model_id=self.model_name, harness_id=self.name()
                    )
                    write_json(self.logs_dir / "submission.json", record)
                    context.metadata = {
                        **(context.metadata or {}),
                        "submission_state": record["submission_state"],
                    }
            finally:
                backend.close()
                for task in (native, pump):
                    if task and not task.done():
                        task.cancel()
                await asyncio.gather(
                    *(t for t in (native, pump) if t), return_exceptions=True
                )
                if bridge:
                    await bridge.finish()
                if registration and registration.close:
                    await asyncio.to_thread(registration.close)
                sink.close()

    def populate_context_post_run(self, context):
        super().populate_context_post_run(context)
        from .trajectory import attach_external_tool_audit

        if (
            hasattr(self, "tool_session")
            and (self.logs_dir / "trajectory.json").is_file()
        ):
            attach_external_tool_audit(
                self.logs_dir / "trajectory.json",
                self.logs_dir / "prophet-events.jsonl",
            )


class ClaudeCodeWithTools(FinancialToolsMixin, ClaudeCode):
    options_model = NativeClaudeOptions


class CodexWithTools(FinancialToolsMixin, Codex):
    options_model = NativeCodexOptions

    def _build_effective_config(self, openai_base_url=None):
        config = super()._build_effective_config(openai_base_url)
        if "prophet" in config.get("mcp_servers", {}):
            config["mcp_servers"]["prophet"].update(
                startup_timeout_sec=30,
                tool_timeout_sec=self.options.tool_timeout_seconds,
                required=True,
            )
        return config
