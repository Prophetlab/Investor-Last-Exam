"""Prepare task-authored Harbor jobs without injecting Prophet tools or policy."""

import hashlib
import shutil
import tempfile
import tomllib
from pathlib import Path

import toml
from dirhash import dirhash
from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.models.job.config import JobConfig
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig

from ..integrations.harbor.docker import managed_docker_config
from ..integrations.harbor.serialization import write_json
from ..integrations.harbor.tasks import component


def prepare_harbor_run(package_root, config, output):
    """Copy trusted tasks; preserve Compose, MCP, artifacts and network policies.

    No imports from customer plugins are required in a task. Only an explicitly
    selected custom Harbor agent is imported on the trusted controller. This is
    preparation, not an execution/availability or customer acceptance check.
    """
    tasks_root = Path(package_root) / "tasks"
    if tasks_root.is_symlink():
        raise ValueError("task source cannot contain symlinks")
    tasks_root = tasks_root.resolve()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(tasks_root):
        raise ValueError("output must not be inside the source task directory")
    if getattr(config, "docker_network", "harbor") != "harbor":
        raise ValueError(
            "Harbor mode preserves task network policy; use docker_network=harbor"
        )
    if config.external_web:
        raise ValueError("Harbor mode uses task.toml network policy, not external_web")
    agent = AgentConfig(
        name=None if config.agent_import_path else config.agent,
        import_path=config.agent_import_path,
        model_name=config.model,
        kwargs=config.agent_kwargs,
        extra_allowed_hosts=config.model_hosts,
    )
    agent_class = AgentFactory.get_agent_class_from_config(agent)
    if not issubclass(agent_class, BaseAgent):
        raise TypeError("custom agent must implement Harbor BaseAgent")
    agent_class.parse_options(config.agent_kwargs)
    if (
        not config.model
        and not config.agent_import_path
        and config.agent not in {"nop", "oracle"}
    ):
        raise ValueError("select an actual model with --model or launch configuration")
    tasks = []
    for path in sorted(tasks_root.iterdir()):
        if not path.is_dir():
            continue
        for item in (path, *path.rglob("*")):
            if item.is_symlink():
                raise ValueError("task source cannot contain symlinks")
        # Do not silently omit malformed author tasks from the requested dataset.
        component(path.name)
        task = Task(path)
        if config.backend == "e2b" and (
            any(
                (path / role / "docker-compose.yaml").exists()
                for role in ("environment", "tests")
            )
            or any(
                artifact.service not in (None, "main")
                for artifact in task.config.artifacts
            )
            or any(
                artifact.service not in (None, "main")
                for step in (task.config.steps or [])
                for artifact in step.artifacts
            )
        ):
            raise ValueError(
                "E2B does not support Compose/sidecars; select the Docker backend"
            )
        tasks.append((path, task.name))
    if not tasks:
        raise ValueError("package has no Harbor tasks")
    names = [name for _, name in tasks]
    if len(names) != len(set(names)):
        raise ValueError("duplicate Harbor task identities")
    if set(config.verifier_images) - set(names):
        raise ValueError("verifier_images contains a task outside this dataset")
    environment = managed_docker_config(config.lease_seconds)
    if config.backend == "e2b":
        environment = {
            "import_path": "prophet_bench_env.integrations.harbor.e2b:LeasedE2BEnvironment",
            "delete": True,
            "kwargs": {"lease_seconds": config.lease_seconds},
        }
    job = JobConfig.model_validate(
        {
            "job_name": "evaluation",
            "jobs_dir": str(output / "jobs"),
            "n_concurrent_trials": config.concurrency,
            "n_attempts": config.attempts,
            "retry": {"max_retries": 0},
            "datasets": [{"path": str(output / "tasks")}],
            "agents": [agent],
            "environment": environment,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".prophet-native-", dir=output.parent
    ) as temp:
        stage = Path(temp) / "run"
        (stage / "tasks").mkdir(parents=True)
        manifest = {
            "schema_version": "prepared-run.v1",
            "agent_mode": "harbor",
            "tasks": {},
        }
        for path, name in tasks:
            original = dirhash(path, "sha256")
            target = stage / "tasks" / path.name
            shutil.copytree(path, target)
            # Explicit operator overrides only. Keep authored image references
            # and shared/separate verifier mode intact otherwise.
            selected_verifier = config.verifier_images.get(name)
            if config.agent_image or selected_verifier:
                values = tomllib.loads((target / "task.toml").read_text())
                if config.agent_image:
                    values.setdefault("environment", {})["docker_image"] = (
                        config.agent_image
                    )
                if selected_verifier:
                    verifier = values.get("verifier", {})
                    if verifier.get("environment_mode") != "separate":
                        raise ValueError(
                            "verifier image override requires a separate verifier"
                        )
                    verifier.setdefault("environment", {})["docker_image"] = (
                        selected_verifier
                    )
                (target / "task.toml").write_text(toml.dumps(values))
                Task(target)
            if dirhash(path, "sha256") != original:
                raise ValueError("source task changed while preparing the deployment")
            copied = dirhash(target, "sha256")
            if not config.agent_image and not selected_verifier and copied != original:
                raise ValueError("task copy differs from its frozen source")
            manifest["tasks"][name] = {
                "source_sha256": original,
                "prepared_sha256": copied,
            }
        write_json(stage / "job.json", job.model_dump(mode="json", exclude_none=True))
        manifest["job_sha256"] = hashlib.sha256(
            (stage / "job.json").read_bytes()
        ).hexdigest()
        write_json(stage / "deployment.json", manifest)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        stage.rename(output)
    return output / "job.json"
