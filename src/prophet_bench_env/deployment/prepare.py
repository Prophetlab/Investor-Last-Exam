"""Materialize a new Harbor run from frozen tasks and an explicit deployment profile."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import tomllib
from pathlib import Path

import toml
from dirhash import dirhash
from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig
from harbor.models.task.task import Task

from ..integrations.harbor.docker import managed_docker_config
from ..integrations.harbor.serialization import write_json
from ..integrations.harbor.tasks import component
from .profile import DeploymentProfile

# These are ordinary Harbor task-authored Compose overrides. Harbor still owns
# setup, exec, resource limits, artifact collection, verification and teardown.
# Both services are detached from bridge/host networks. No host firewall edits.
DOCKER_NONE_COMPOSE = (
    "services:\n"
    "  main:\n"
    "    network_mode: none\n"
    "  harbor-docker-egress-control-sidecar:\n"
    "    network_mode: none\n"
)


def _enforce_offline(config: dict) -> None:
    """The none profile cannot satisfy online verifiers or phase network changes."""
    if config.get("steps"):
        raise ValueError(
            "docker_network=none currently supports single-step tasks only"
        )
    environment = config["environment"]
    verifier = config.get("verifier", {})
    baselines = [environment, verifier.get("environment") or environment]
    for entry in baselines:
        if entry.get("network_mode", "public") != "no-network" or entry.get(
            "allowed_hosts"
        ):
            raise ValueError(
                "docker_network=none cannot satisfy a task requiring network access"
            )
    for entry in (config.get("agent", {}), verifier):
        if entry.get("network_mode") not in (None, "no-network") or entry.get(
            "allowed_hosts"
        ):
            raise ValueError(
                "docker_network=none cannot satisfy a task requiring network access"
            )


def _image_for_role(config: dict, role_dir: Path, selected: str | None) -> None:
    if selected is not None:
        config["docker_image"] = selected
    elif (role_dir / "Dockerfile").is_file():
        # Captured machine-local image choices are evidence of the source run,
        # not a deployment default for an unknown receiving customer.
        config.pop("docker_image", None)
    elif config.get("docker_image"):
        raise ValueError(
            "prebuilt-only tasks require an explicit target image selection"
        )


def prepare_run(
    *,
    tasks_dir: Path,
    run_template: dict,
    profile: DeploymentProfile,
    output: Path,
    job_name: str = "review",
    native_setup_hosts: tuple[str, ...] | None = None,
) -> Path:
    """Prepare only: no model, Docker, API, billing, host changes or execution.

    Public tasks/rubrics remain byte-identical. Only deployment fields in a new
    copy change. Source and resulting task hashes are recorded. Existing output
    paths are never overwritten, including after a partially failed preparation.
    """
    component(job_name)
    tasks_dir = tasks_dir.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(tasks_dir):
        raise ValueError("output must not be inside the source task directory")
    tasks = sorted(path for path in tasks_dir.iterdir() if Task.is_valid_dir(path))
    if not tasks:
        raise ValueError("no valid Harbor tasks")
    identities = [
        json.loads((path / "public-task.json").read_text())["task_id"] for path in tasks
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate public task identities")
    if set(profile.verifier_images) - set(identities):
        raise ValueError("verifier_images contains a task outside this dataset")
    allowed_template_keys = {
        "job_name",
        "jobs_dir",
        "n_concurrent_trials",
        "n_attempts",
        "datasets",
        "agents",
        "retry",
    }
    if set(run_template) - allowed_template_keys or not run_template.get("agents"):
        raise ValueError(
            "run template must contain agent bindings, not a host environment configuration"
        )
    agents = run_template["agents"]
    for agent in agents:
        if native_setup_hosts is None and (
            agent.get("extra_allowed_hosts")
            or agent.get("env")
            or agent.get("mcp_servers")
        ):
            raise ValueError(
                "agent runtime extensions must be configured and reviewed separately"
            )
    env = (
        managed_docker_config(profile.lease_seconds)
        if profile.backend == "docker"
        else {
            "import_path": "prophet_bench_env.integrations.harbor.e2b:LeasedE2BEnvironment",
            "delete": True,
            "kwargs": {"lease_seconds": profile.lease_seconds},
        }
    )
    if profile.backend == "docker" and profile.docker_network == "none":
        env = {
            "import_path": "prophet_bench_env.integrations.harbor.docker:OfflineProbeDockerEnvironment",
            "delete": True,
            "kwargs": {"lease_seconds": profile.lease_seconds},
        }
    job = JobConfig.model_validate(
        {
            "job_name": job_name,
            "jobs_dir": str(output / "jobs"),
            "n_concurrent_trials": profile.concurrency,
            "n_attempts": profile.attempts,
            "environment": env,
            "retry": {"max_retries": 0},
            "datasets": [{"path": str(output / "tasks")}],
            "agents": agents,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publication also avoids leaving half-exported task sets to resume.
    with tempfile.TemporaryDirectory(
        prefix=".prophet-prepare-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / "run"
        (stage / "tasks").mkdir(parents=True)
        manifest = {
            "schema_version": "prepared-run.v1",
            "profile": profile.model_dump(),
            "tasks": {},
        }
        for task_dir, task_id in zip(tasks, identities, strict=True):
            for path in (task_dir, *task_dir.rglob("*")):
                if path.is_symlink():
                    raise ValueError("task source cannot contain symlinks")
            original_hash = dirhash(task_dir, "sha256")
            target = stage / "tasks" / component(task_dir.name)
            shutil.copytree(
                task_dir,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", ".git"),
            )
            config = tomllib.loads((target / "task.toml").read_text())
            if profile.backend == "docker" and profile.docker_network == "none":
                _enforce_offline(config)
            environment = config.setdefault("environment", {})
            verifier = config.setdefault("verifier", {})
            if (
                verifier.get("environment_mode") == "shared"
                or "environment" not in verifier
            ):
                raise ValueError(
                    "deployment preparation requires an explicit separate verifier environment"
                )
            _image_for_role(environment, target / "environment", profile.agent_image)
            if native_setup_hosts is not None:
                if profile.docker_network == "none":
                    raise ValueError("installed native agents require model API egress")
                # Install before the model sees the workspace. During agent.run,
                # only explicitly configured model hosts are added by Harbor.
                environment["network_mode"] = (
                    "allowlist" if native_setup_hosts else "no-network"
                )
                environment["allowed_hosts"] = list(native_setup_hosts)
                config.setdefault("agent", {})["network_mode"] = "no-network"
                config["agent"]["allowed_hosts"] = []
            _image_for_role(
                verifier["environment"],
                target / "tests",
                profile.verifier_images.get(task_id),
            )
            for role in ("environment", "tests"):
                compose = target / role / "docker-compose.yaml"
                if compose.exists():
                    if compose.read_text() != DOCKER_NONE_COMPOSE:
                        raise ValueError(
                            "custom Compose task requires a separately reviewed deployment adapter"
                        )
                    compose.unlink()  # only in the new staging copy; original evidence is untouched
                if profile.backend == "docker" and profile.docker_network == "none":
                    compose.write_text(DOCKER_NONE_COMPOSE)
            TaskConfig.model_validate(config)
            (target / "task.toml").write_text(toml.dumps(config))
            for name in ("public-task.json", "tests/verification.json"):
                if (target / name).read_bytes() != (task_dir / name).read_bytes():
                    raise ValueError(
                        "deployment preparation changed task/scoring semantics"
                    )
            if dirhash(task_dir, "sha256") != original_hash:
                raise ValueError("source task changed while preparing the deployment")
            manifest["tasks"][task_id] = {
                "source_sha256": original_hash,
                "prepared_sha256": dirhash(target, "sha256"),
            }
        write_json(stage / "job.json", job.model_dump(mode="json", exclude_none=True))
        manifest["job_sha256"] = hashlib.sha256(
            (stage / "job.json").read_bytes()
        ).hexdigest()
        write_json(stage / "deployment.json", manifest)
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return output / "job.json"
