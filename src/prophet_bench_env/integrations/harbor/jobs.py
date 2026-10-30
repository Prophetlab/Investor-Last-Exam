"""Native Harbor job composition; no Prophet rollout worker pool."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from harbor.job import Job
from harbor.models.job.config import JobConfig

from .docker import managed_docker_config
from .outcomes import finalize_job_result, write_incomplete_summary
from .serialization import write_json


def job_config(
    *,
    tasks: Path,
    jobs: Path,
    job_name: str,
    runtime_factory: str,
    model_ids: tuple[str, ...],
    backend: str = "docker",
    concurrency: int = 4,
    per_model_concurrency: int | None = None,
    attempts: int = 1,
    lease_seconds: int = 1800,
    renew_lease: bool = False,
    runtime_kwargs: dict | None = None,
) -> JobConfig:
    if backend not in {"e2b", "docker"}:
        raise ValueError("backend must be docker or e2b")
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids must be nonempty and unique")
    environment = (
        managed_docker_config(lease_seconds)
        if backend == "docker"
        else {
            "import_path": "prophet_bench_env.integrations.harbor.e2b:LeasedE2BEnvironment",
            "delete": True,
            "kwargs": {"lease_seconds": lease_seconds, "renew_lease": renew_lease},
        }
    )
    return JobConfig.model_validate(
        {
            "job_name": job_name,
            # Compose resolves relative copy paths against its project directory,
            # not the controller's cwd. Freeze absolute host paths for new jobs.
            "jobs_dir": str(jobs.resolve()),
            "n_attempts": attempts,
            "n_concurrent_trials": concurrency,
            "environment": environment,
            "retry": {"max_retries": 0},
            "datasets": [{"path": str(tasks.resolve())}],
            "agents": [
                {
                    "import_path": "prophet_bench_env.integrations.harbor.agent:FinancialAgent",
                    "model_name": model,
                    "n_concurrent": per_model_concurrency,
                    "kwargs": {
                        "runtime_factory": runtime_factory,
                        **(
                            {"runtime_kwargs": runtime_kwargs} if runtime_kwargs else {}
                        ),
                    },
                }
                for model in model_ids
            ],
        }
    )


async def run_job(config: JobConfig):
    """Harbor persists config/lock/results and resumes the same native job."""
    os.environ.setdefault("HARBOR_TELEMETRY", "0")
    job = await Job.create(config)
    try:
        result = await job.run()
    except BaseException as error:
        try:
            write_incomplete_summary(job.job_dir)
        except Exception as summary_error:  # noqa: BLE001 - preserve the original job failure
            error.add_note("Incomplete summary failed: " + type(summary_error).__name__)
        raise
    return finalize_job_result(result, job.job_dir)


def run_config(path: Path):
    config = JobConfig.model_validate(json.loads(path.read_text()))
    return asyncio.run(run_job(config))


def save_config(config: JobConfig, path: Path) -> None:
    write_json(path, config.model_dump(mode="json", exclude_none=True))
