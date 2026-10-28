"""Freeze authoring output into native Harbor tasks, not a nested runner."""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import toml
from harbor.models.task.config import TaskConfig
from harbor.models.task.task import Task

from ...core.contracts import TaskBundle
from .serialization import write_json

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_IMAGE = "python:3.12-slim-bookworm"
SCIENTIFIC_PACKAGES = (
    "numpy==2.2.6",
    "scipy==1.15.3",
    "pandas==2.2.3",
    "openpyxl==3.1.5",
    "pyarrow==20.0.0",
    "duckdb==1.3.2",
)


@dataclass(frozen=True)
class VerifierSpec:
    """Operator-owned evaluator factory and build requirements (no credentials)."""

    factory: str
    private_files: dict[str, Path] = field(default_factory=dict)
    private_context: str | None = None
    env_names: tuple[str, ...] = ()
    allowed_hosts: tuple[str, ...] = ()
    # None installs the project's locked dependencies. A plugin may specify a
    # smaller evaluator runtime. The evidence validator is always installed.
    dependencies: tuple[str, ...] | None = None
    timeout_sec: int = 600
    factory_kwargs: dict = field(default_factory=dict)


def component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", value):
        raise ValueError("invalid portable file name component")
    return value


def snapshot_runtime(destination: Path, source: Path = PACKAGE_ROOT) -> None:
    """Copy only explicit runtime roots, never a working tree or .env."""
    destination.mkdir(parents=True)
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", ".env", ".env.*", "tasks")
    for name in ("prophet_bench_env", "plugins", "vendor"):
        shutil.copytree(source / name, destination / name, ignore=ignored)
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copy2(source / name, destination / name)


def export_task(
    bundle: TaskBundle,
    destination: Path,
    verifier: VerifierSpec,
    *,
    base_image: str = DEFAULT_BASE_IMAGE,
    agent_timeout_sec: int | None = 1800,
    public_files: dict[str, Path] | None = None,
    runtime_files: dict[str, Path] | None = None,
    tool_factory: str | None = None,
    tool_kwargs: dict | None = None,
) -> Path:
    """Export once; existing tasks are immutable. Runtime edits require a new export."""
    task = bundle.public_task
    component(task.task_id)
    if task.cutoff.tzinfo is None:
        raise ValueError("task cutoff must be timezone-aware")
    if (
        bundle.rubric.task_id != task.task_id
        or bundle.rubric.task_type != task.task_type
    ):
        raise ValueError("rubric does not belong to this task")
    if destination.exists():
        raise FileExistsError(destination)
    if not re.fullmatch(r"[A-Za-z0-9_./:@-]+", base_image):
        raise ValueError("invalid base image reference")
    destination.mkdir(parents=True)
    env_dir, tests_dir = destination / "environment", destination / "tests"
    env_dir.mkdir()
    tests_dir.mkdir()
    public = task.as_dict()
    write_json(destination / "public-task.json", public)
    # Public task is loaded by the trusted external agent before any agent code
    # executes. It does not contain evaluator imports, rubric or private paths.
    (destination / "instruction.md").write_text(task.prompt + "\n")
    packages = " ".join(SCIENTIFIC_PACKAGES)
    (env_dir / "Dockerfile").write_text(
        f"FROM {base_image}\n"
        f"RUN pip install --no-cache-dir {packages}\n"
        "RUN (id -u agent >/dev/null 2>&1 || useradd -m agent) && mkdir -p /app/workspace /logs/artifacts "
        "&& chown -R agent:agent /app /logs/artifacts\n"
        "WORKDIR /app/workspace\n"
    )
    for name, source in (public_files or {}).items():
        target = destination / "public-data" / component(name)
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
    # Trusted host-side factory configuration, not an agent workspace upload.
    for name, source in (runtime_files or {}).items():
        target = destination / "runtime" / component(name)
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
    if tool_factory:
        if (destination / "runtime/tools.json").exists():
            raise ValueError("runtime/tools.json is reserved for the tool binding")
        write_json(
            destination / "runtime/tools.json",
            {"factory": tool_factory, "kwargs": tool_kwargs or {}},
        )
    for name, source in verifier.private_files.items():
        target = tests_dir / "private" / component(name)
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
    if (
        verifier.private_context
        and verifier.private_context not in verifier.private_files
    ):
        raise ValueError("private_context must name an exported private file")
    write_json(
        tests_dir / "verification.json",
        {
            "public_task": public,
            "rubric": bundle.rubric.as_dict(),
            "rubric_sha256": bundle.rubric.sha256,
            "submission_evidence_required": True,
            "evaluator_factory": verifier.factory,
            "evaluator_kwargs": verifier.factory_kwargs,
            "private_context_ref": (
                "/tests/private/" + verifier.private_context
                if verifier.private_context
                else ""
            ),
        },
    )
    snapshot_runtime(tests_dir / "runtime")
    from importlib.metadata import version

    dependencies = verifier.dependencies
    if dependencies is not None and not any(
        re.match(r"jsonschema(?:[<>=!\[]|$)", d) for d in dependencies
    ):
        dependencies = (*dependencies, "jsonschema==" + version("jsonschema"))
    install = (
        "RUN pip install --no-cache-dir uv==0.11.26 "
        "&& cd /opt/prophetbench && uv sync --frozen --no-dev\n"
        "ENV PATH=/opt/prophetbench/.venv/bin:$PATH\n"
        if verifier.dependencies is None
        else (
            "RUN pip install --no-cache-dir " + " ".join(dependencies) + "\n"
            if dependencies
            else ""
        )
    )
    if verifier.dependencies and any(
        not re.fullmatch(r"[A-Za-z0-9_.\[\]=<>!-]+", dep)
        for dep in verifier.dependencies
    ):
        raise ValueError(
            "verifier dependencies must be package requirements, not commands"
        )
    (tests_dir / "Dockerfile").write_text(
        f"FROM {base_image}\nCOPY runtime/ /opt/prophetbench/\n"
        + install
        + "ENV PYTHONPATH=/opt/prophetbench\nCOPY verification.json /tests/verification.json\n"
        + ("COPY private/ /tests/private/\n" if verifier.private_files else "")
        + "COPY test.sh /tests/test.sh\nWORKDIR /app\n"
    )
    (tests_dir / "test.sh").write_text(
        "#!/bin/sh\nset -eu\n"
        "cd /opt/prophetbench\nexport PYTHONPATH=/opt/prophetbench\n"
        "PYTHON=python\nif [ -x /opt/prophetbench/.venv/bin/python ]; then PYTHON=/opt/prophetbench/.venv/bin/python; fi\n"
        'exec "$PYTHON" -m prophet_bench_env.integrations.harbor.verifier '
        "--spec /tests/verification.json\n"
    )
    (tests_dir / "test.sh").chmod(0o755)
    verifier_env = {name: "${" + name + "}" for name in verifier.env_names}
    config = {
        "schema_version": "1.4",
        "metadata": {
            "prophet_task_id": task.task_id,
            "task_version": task.task_version,
            "rubric_sha256": bundle.rubric.sha256,
            "public_task_sha256": hashlib.sha256(
                (destination / "public-task.json").read_bytes()
            ).hexdigest(),
            "cutoff": task.cutoff.isoformat(),
            "harbor_version": "0.23.0",
        },
        "artifacts": ["/app/workspace"],
        "agent": {
            "timeout_sec": agent_timeout_sec,
            "user": "agent",
            "network_mode": "no-network",
        },
        "environment": {
            "network_mode": "no-network",
            "cpus": 2,
            "memory_mb": 2048,
            "build_timeout_sec": 900,
        },
        "verifier": {
            "timeout_sec": verifier.timeout_sec,
            "environment_mode": "separate",
            "env": verifier_env,
            "environment": {
                "network_mode": "allowlist" if verifier.allowed_hosts else "no-network",
                "allowed_hosts": list(verifier.allowed_hosts),
                "cpus": 2,
                "memory_mb": 2048,
                "build_timeout_sec": 900,
            },
        },
    }
    TaskConfig.model_validate(config)
    (destination / "task.toml").write_text(toml.dumps(config))
    if not Task.is_valid_dir(destination):
        raise ValueError("export did not produce a valid Harbor task")
    return destination
