"""Tencent/Hunyuan COS sample ZIP contract, independent of financial runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dirhash import dirhash
from harbor.models.task.task import Task

from ..deployment.handoff import runtime_handoff_files
from ..integrations.harbor.tasks import component
from ..integrations.harbor.trajectory import validate_atif_bundle
from ..ports.delivery import DeliveryReceipt, DeliverySource
from .audit import audit_reference_index


def from_config(*, options: dict) -> TencentExporter:
    values = dict(options)
    values["delivery_date"] = date.fromisoformat(values["delivery_date"])
    return TencentExporter(**values)


@dataclass(frozen=True)
class TencentExporter:
    vendor: str
    category: str
    batch: int
    version: int
    delivery_date: date
    max_total_bytes: int = 512 * 1024 * 1024

    def export(self, source: DeliverySource, destination: Path) -> DeliveryReceipt:
        """Fail closed on missing/mismatched artifacts or likely secrets; never upload."""
        component(self.vendor)
        component(self.category)
        if self.batch < 1 or self.version < 1:
            raise ValueError("batch and version must be positive")
        tasks = sorted(
            path for path in source.tasks_dir.iterdir() if Task.is_valid_dir(path)
        )
        if not tasks:
            raise ValueError("no valid Harbor tasks")
        name = (
            f"{self.vendor}_Tencent_{self.category}_sample_batch{self.batch}"
            f"_taskNum{len(tasks)}_fix-v{self.version}_{self.delivery_date.isoformat()}"
        )
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / (name + ".zip")
        if output.exists():
            raise FileExistsError(output)
        entries: dict[str, bytes] = {}
        total = 0
        task_ids = set()
        task_paths = {}

        def add(name: str, data: bytes):
            nonlocal total
            if name in entries:
                raise ValueError("duplicate delivery path")
            _check_content(name, data)
            total += len(data)
            if total > self.max_total_bytes:
                raise ValueError("delivery exceeds configured size budget")
            entries[name] = data

        def tree(root: Path, prefix: str):
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise ValueError("delivery cannot include symlinks")
                if not path.is_file():
                    continue
                relative = path.relative_to(root)
                if any(
                    part in {"__pycache__", ".git", ".venv"} for part in relative.parts
                ):
                    continue
                if path.stat().st_size > self.max_total_bytes - total:
                    raise ValueError("artifact exceeds delivery size budget")
                add(prefix + "/" + relative.as_posix(), path.read_bytes())

        for task in tasks:
            public = json.loads((task / "public-task.json").read_text())
            if public["task_id"] in task_ids:
                raise ValueError("duplicate task identity")
            task_ids.add(public["task_id"])
            task_paths[public["task_id"]] = task
            tree(task, "tasks/" + component(task.name))
        represented = set()
        agent_configs = {}
        for job in source.job_dirs:
            for trial in sorted(job.iterdir()):
                result_path = trial / "result.json"
                if not trial.is_dir() or not result_path.is_file():
                    continue
                result = json.loads(result_path.read_text())
                config = result["config"]["agent"]
                public_config = {
                    key: config[key]
                    for key in ("import_path", "name", "model_name", "kwargs")
                    if config.get(key) is not None
                }
                agent_configs[json.dumps(public_config, sort_keys=True)] = public_config
                public_path = trial / "agent/public-task.json"
                if not public_path.is_file():
                    raise ValueError(
                        "trial has no frozen task identity; cannot silently omit it"
                    )
                public = json.loads(public_path.read_text())
                task_id = public["task_id"]
                if task_id not in task_ids:
                    raise ValueError("trial belongs to a task outside this delivery")
                frozen = json.loads(
                    (task_paths[task_id] / "public-task.json").read_text()
                )
                if frozen != public:
                    raise ValueError("trial public task differs from exported task")
                if result.get("task_checksum") != dirhash(
                    task_paths[task_id], "sha256"
                ):
                    raise ValueError("task contents changed after this trial")
                trajectory_path = trial / "agent/trajectory.json"
                validate_atif_bundle(trajectory_path)
                if not (trial / "artifacts").is_dir():
                    raise ValueError("trial artifacts were not collected")
                artifact_manifest = json.loads(
                    (trial / "artifacts/manifest.json").read_text()
                )
                if not artifact_manifest or any(
                    item.get("status") != "ok" for item in artifact_manifest
                ):
                    raise ValueError("artifact collection is incomplete")
                rewards = (result.get("verifier_result") or {}).get("rewards")
                if not result.get("exception_info") and rewards is None:
                    raise ValueError("completed trial has no verifier reward")
                if rewards is not None:
                    _check_scored_trial(trial, task_id, rewards)
                prefix = (
                    "trajectory/" + component(job.name) + "/" + component(trial.name)
                )
                for subdir in ("agent", "verifier", "artifacts"):
                    if (trial / subdir).is_dir():
                        tree(trial / subdir, prefix + "/" + subdir)
                add(
                    prefix + "/audit-references.json",
                    json.dumps(
                        audit_reference_index(entries, prefix),
                        ensure_ascii=False,
                        indent=2,
                    ).encode(),
                )
                # Native configs can contain machine paths and environment
                # values. Export explicit result fields instead, keeping errors.
                projected = {
                    key: result.get(key)
                    for key in (
                        "id",
                        "task_name",
                        "trial_name",
                        "task_checksum",
                        "agent_info",
                        "agent_result",
                        "verifier_result",
                        "verifier_environment_mode",
                        "started_at",
                        "finished_at",
                        "environment_setup",
                        "agent_execution",
                        "verifier",
                    )
                }
                if result.get("exception_info"):
                    projected["exception_info"] = {
                        "exception_type": result["exception_info"]["exception_type"]
                    }
                add(
                    prefix + "/result.json",
                    json.dumps(projected, ensure_ascii=False, indent=2).encode(),
                )
                represented.add(task_id)
        if represented != task_ids:
            raise ValueError(
                "each exported task must have at least one collected trajectory"
            )
        runtime = tasks[0] / "tests/runtime"
        if not (runtime / "prophet_bench_env/deployment/prepare.py").is_file():
            raise ValueError(
                "frozen runtime predates deployment.v1; export and validate a new task bundle"
            )
        # One host runtime drives the whole dataset. Do not silently choose the
        # first one when different tasks were frozen from different code.
        runtime_hash = dirhash(runtime, "sha256")
        if any(
            dirhash(task / "tests/runtime", "sha256") != runtime_hash
            for task in tasks[1:]
        ):
            raise ValueError(
                "tasks contain different runtime snapshots; split the delivery or re-freeze consistently"
            )
        tree(runtime, "environment/runtime")
        for handoff_path, data in runtime_handoff_files(
            list(agent_configs.values())
        ).items():
            add(handoff_path, data)
        for prefix, files in (
            ("environment", source.environment_files),
            ("docs", source.documentation_files),
        ):
            for filename, path in files.items():
                if path.is_symlink():
                    raise ValueError("delivery source cannot be a symlink")
                add(prefix + "/" + component(filename), path.read_bytes())
        manifest = {
            "exporter": "tencent-hunyuan.v2",
            "deployment_contract": "deployment.v1",
            "task_ids": sorted(task_ids),
            "harbor_version": "0.23.0",
            "files": {
                path: {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
                for path, data in entries.items()
            },
        }
        add(
            "docs/manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode(),
        )
        with tempfile.NamedTemporaryFile(
            dir=destination, suffix=".zip", delete=False
        ) as f:
            temporary = Path(f.name)
        try:
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                for folder in ("tasks", "trajectory", "environment", "docs"):
                    archive.writestr(f"{name}/{folder}/", b"")
                for path, data in sorted(entries.items()):
                    info = zipfile.ZipInfo(name + "/" + path)
                    info.external_attr = (
                        0o100755 if path.endswith(".sh") else 0o100644
                    ) << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, data)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return DeliveryReceipt(
            output,
            hashlib.sha256(output.read_bytes()).hexdigest(),
            len(tasks),
            f"/supplier_{self.vendor}/{self.vendor}_sample_data/{self.category}/{output.name}",
        )


def _check_scored_trial(trial: Path, task_id: str, rewards: dict) -> None:
    required = (
        "agent/submission.json",
        "artifacts/logs/artifacts/submission.json",
        "verifier/evaluation.json",
        "verifier/reward.json",
    )
    if any(not (trial / name).is_file() for name in required):
        raise ValueError("scored trial is missing submission or verifier artifacts")
    submission, collected, evaluation, reward = (
        json.loads((trial / name).read_text()) for name in required
    )
    if submission != collected or submission.get("task_id") != task_id:
        raise ValueError("collected submission differs from the accepted submission")
    if (
        reward != rewards
        or evaluation.get("reward") != rewards.get("reward")
        or evaluation.get("reward_pending")
        or evaluation.get("failure_classification")
    ):
        raise ValueError("verifier artifacts differ from the recorded reward")


def _check_content(name: str, data: bytes) -> None:
    if any(
        part == ".env"
        or part.startswith(".env.")
        or part in {"auth.json", "id_rsa", "id_ed25519"}
        for part in Path(name).parts
    ):
        raise ValueError("secret/configuration file cannot be delivered")
    if re.search(
        rb"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{24,}|e2b_[A-Za-z0-9]{20,}|dl_live_[A-Za-z0-9_-]{16,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)",
        data,
    ):
        raise ValueError(
            "possible credential in delivery; review source before retrying"
        )
    for key, value in os.environ.items():
        if (
            re.search(r"(API_KEY|TOKEN|SECRET|PASSWORD)$", key)
            and len(value) >= 12
            and value.encode() in data
        ):
            raise ValueError("configured credential found in delivery")
