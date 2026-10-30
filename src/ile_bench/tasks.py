from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import toml
from jsonschema import Draft202012Validator

from prophet_bench_env.core.contracts import PublicTask, SubmissionPolicy, ToolDefinition
from prophet_bench_env.integrations.harbor.serialization import write_json
from prophet_bench_env.ports.data_specs import DATASETS

from .config import ROOT


def load_tasks(codes=None):
    rows = [json.loads(p.read_text()) for p in sorted((ROOT / "tasks").glob("*.json"))]
    if len(rows) != 86 or len({t["id"] for t in rows}) != 86:
        raise ValueError("Expected 86 distinct released tasks")
    if sum(len(t["rubrics"]) for t in rows) != 1472:
        raise ValueError("Rubric count mismatch")
    if codes and codes != ["all"]:
        unknown = set(codes) - {t["id"] for t in rows}
        if unknown:
            raise ValueError("Unknown tasks: " + ",".join(sorted(unknown)))
        rows = [t for t in rows if t["id"] in codes]
    for t in rows:
        Draft202012Validator.check_schema(t["submission_schema"])
    return rows


def public_task(t):
    fn = next(x["function"] for x in t["tools"] if x["function"]["name"] == "submit_answer")
    return PublicTask(
        benchmark_id="investor-last-exam-v1",
        task_id=t["id"],
        task_type=t["submission_mode"],
        task_version="1.0.0",
        prompt=t["question"],
        cutoff=datetime.fromisoformat(t["cutoff"].replace("Z", "+00:00")),
        timezone=json.loads((ROOT / "environment/task-timezones.json").read_text())[t["id"]],
        target_definition={"submission_mode": t["submission_mode"]},
        capability_set={"datasets": list(DATASETS)},
        submit_tool=ToolDefinition(
            "submit_answer",
            fn["description"],
            t["submission_schema"],
            side_effect="submission",
            parallel_safe=False,
        ),
        submission_policy=SubmissionPolicy(
            max_attempts=t["evaluation_policy"]["submission_attempts"]
        ),
    )


def export_task(t, dest: Path, wall_clock: int, image=None):
    from harbor.models.task.config import TaskConfig

    dest.mkdir(parents=True, exist_ok=False)
    (dest / "environment").mkdir()
    task = public_task(t)
    write_json(dest / "public-task.json", task.as_dict())
    (dest / "instruction.md").write_text(task.prompt + "\n")
    write_json(
        dest / "runtime/config.json",
        {
            "task_id": t["id"],
            "system_prompt": t["solver_system_prompt"],
            "tool_definitions": t["tools"],
            "wall_clock_seconds": wall_clock,
        },
    )
    shutil_source = ROOT / "environment/Dockerfile"
    (dest / "environment/Dockerfile").write_bytes(shutil_source.read_bytes())
    config = {
        "schema_version": "1.4",
        "metadata": {"task_code": t["id"], "source_sha256": t["source_sha256"]},
        "artifacts": ["/app/workspace"],
        "agent": {"timeout_sec": wall_clock, "user": "agent", "network_mode": "no-network"},
        "environment": {
            "network_mode": "no-network",
            "cpus": 2,
            "memory_mb": 2048,
            "build_timeout_sec": 1200,
        },
        "verifier": {"environment_mode": "separate"},
    }
    if image:
        config["environment"]["docker_image"] = image
    TaskConfig.model_validate(config)
    (dest / "task.toml").write_text(toml.dumps(config))
    # No verifier truth, reference documents, API keys or host bind mounts in this directory.
    return hashlib.sha256(json.dumps(t, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
