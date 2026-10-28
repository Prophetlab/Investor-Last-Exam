"""Customer-neutral one-command launcher for native agents and task tools."""

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..integrations.harbor.jobs import run_config
from ..integrations.harbor.serialization import load_entrypoint
from .prepare import prepare_run
from .profile import DeploymentProfile

AGENTS = {
    "claude-code": "prophet_bench_env.integrations.harbor.installed:ClaudeCodeWithTools",
    "codex": "prophet_bench_env.integrations.harbor.installed:CodexWithTools",
}
SETUP_HOSTS = [
    "deb.debian.org",
    "security.debian.org",
    "registry.npmjs.org",
    "nodejs.org",
    "raw.githubusercontent.com",
    "github.com",
    "codeload.github.com",
    "downloads.claude.ai",
    "storage.googleapis.com",
]


class LaunchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    agent: str = "claude-code"
    agent_mode: Literal["prophet", "harbor"] = "prophet"
    agent_import_path: str | None = None
    model: str | None = None
    backend: Literal["docker", "e2b"] = "docker"
    tool_factory: str | None = None
    tool_kwargs: dict = Field(default_factory=dict)
    agent_kwargs: dict = Field(default_factory=dict)
    model_hosts: list[str] = Field(default_factory=list)
    setup_hosts: list[str] = Field(default_factory=lambda: list(SETUP_HOSTS))
    external_web: bool = False
    agent_image: str | None = None
    verifier_images: dict[str, str] = Field(default_factory=dict)
    concurrency: int = Field(default=1, ge=1)
    attempts: int = Field(default=1, ge=1)
    lease_seconds: int = Field(default=3600, ge=60, le=14400)

    @model_validator(mode="after")
    def no_embedded_credentials(self):
        if self.agent_mode == "prophet":
            if self.agent not in AGENTS or self.agent_import_path:
                raise ValueError(
                    "use agent_mode=harbor for native/custom Harbor agents"
                )
        elif self.tool_factory or self.tool_kwargs:
            raise ValueError(
                "Harbor mode uses task-owned tools, not Prophet tool_factory"
            )

        def check(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if re.search(
                        r"(?i)(api[_-]?key|authorization|password|secret|access[_-]?token|auth[_-]?token)$",
                        key,
                    ):
                        raise ValueError(
                            "credentials belong in receiver environment, not launch configuration"
                        )
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)

        check(self.model_dump())
        return self


def prepare_native_run(package_root: Path, config: LaunchConfig, output: Path):
    if config.agent_mode == "harbor":
        from .native import prepare_harbor_run

        return prepare_harbor_run(package_root, config, output)
    root = package_root.resolve()
    if not config.model:
        raise ValueError("select an actual model with --model or launch configuration")
    task_dirs = sorted(p for p in (root / "tasks").iterdir() if p.is_dir())
    if not task_dirs:
        raise ValueError("package has no tasks")
    for task in task_dirs:
        binding_path = task / "runtime/tools.json"
        binding = json.loads(binding_path.read_text()) if binding_path.is_file() else {}
        factory = config.tool_factory or binding.get("factory")
        if not factory:
            raise ValueError(
                f"{task.name}: supply tool_factory or freeze runtime/tools.json"
            )
        if not callable(load_entrypoint(factory)):
            raise TypeError(f"{task.name}: tool_factory is not callable")
    prefix, default_host = (
        ("ANTHROPIC", "api.anthropic.com")
        if config.agent == "claude-code"
        else ("OPENAI", "api.openai.com")
    )
    url = os.getenv(prefix + "_BASE_URL")
    host = urlsplit(url).hostname if url else default_host
    if not host or host in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(
            "model endpoint must be reachable from the selected sandbox, not host loopback"
        )
    kwargs = dict(config.agent_kwargs)
    if set(kwargs) & {"tool_factory", "tool_kwargs"}:
        raise ValueError(
            "tool binding belongs in tool_factory/tool_kwargs, not agent_kwargs"
        )
    kwargs.update(tool_factory=config.tool_factory, tool_kwargs=config.tool_kwargs)
    if not config.external_web:
        if config.agent == "codex":
            kwargs["web_search"] = "disabled"
        else:
            existing = str(kwargs.get("disallowed_tools") or "")
            kwargs["disallowed_tools"] = ",".join(
                filter(None, [existing, "WebSearch", "WebFetch"])
            )
    # Validate options before any sandbox creation or model billing.
    load_entrypoint(AGENTS[config.agent]).parse_options(kwargs)
    template = {
        "agents": [
            {
                "import_path": AGENTS[config.agent],
                "model_name": config.model,
                "extra_allowed_hosts": list(dict.fromkeys([host, *config.model_hosts])),
                "kwargs": kwargs,
            }
        ]
    }
    return prepare_run(
        tasks_dir=root / "tasks",
        run_template=template,
        profile=DeploymentProfile(
            backend=config.backend,
            agent_image=config.agent_image,
            verifier_images=config.verifier_images,
            concurrency=config.concurrency,
            attempts=config.attempts,
            lease_seconds=config.lease_seconds,
        ),
        output=output,
        job_name="evaluation",
        native_setup_hosts=tuple(config.setup_hosts),
    )


def launch(
    package_root, *, config_file=None, overrides=None, output=None, prepare_only=False
):
    root = Path(package_root).resolve()
    defaults = root / "environment/launch.json"
    values = json.loads(defaults.read_text()) if defaults.is_file() else {}
    if config_file:
        values.update(json.loads(Path(config_file).read_text()))
    values.update(
        {key: value for key, value in (overrides or {}).items() if value is not None}
    )
    config = LaunchConfig.model_validate(values)
    if not prepare_only:
        credential_names = (
            ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
            if config.agent == "claude-code"
            else ("OPENAI_API_KEY", "CODEX_AUTH_JSON_PATH")
        )
        if config.agent_mode == "prophet" and not any(
            os.getenv(name) for name in credential_names
        ):
            raise ValueError(
                "configure model credentials in the receiving machine's environment first"
            )
        if config.backend == "e2b" and not os.getenv("E2B_API_KEY"):
            raise ValueError("E2B_API_KEY is required for the selected backend")
    target = (
        Path(output)
        if output
        else root
        / "runs"
        / (datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8])
    )
    job = prepare_native_run(root, config, target)
    if prepare_only:
        return job
    result = run_config(job)
    if any(row.exception_info is not None for row in result.trial_results):
        raise RuntimeError(
            f"some trials failed; original results are saved under {target}"
        )
    return target
