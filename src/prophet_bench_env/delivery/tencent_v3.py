"""Tencent September 2026 V3 packaging over frozen Harbor tasks and trials.

This adapter never runs models, changes task semantics, creates QA evidence, or
uploads files. Its review manifest supplies human provenance and run labels;
scores, counts, task checksums and harness identities come from actual artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import tempfile
import tomllib
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import toml
from dirhash import dirhash
from harbor.models.trajectories.trajectory import Trajectory

from ..integrations.harbor.tasks import component
from ..integrations.harbor.trajectory import validate_atif_bundle
from ..ports.delivery import DeliveryReceipt, DeliverySource
from . import tencent_v3_checks as checks
from .tencent import _check_content
from .tencent_v3_report import DeliveryReport

METADATA_FIELDS = {
    "name",
    "instance_id",
    "task",
    "source_repo",
    "language",
    "programming_language",
    "prompt_creation_time",
}
DATASET_DOCS = (
    "README.md",
    "analysis.md",
    "failure_analysis_metadata.md",
    "reward_design_and_hacking_analysis.md",
)
TASK_DOCS = ("README.md", "report.md", "context_info.md", "failure_analysis/README.md")


class DeliveryNotReady(ValueError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__("Cannot safely package delivery:\n- " + "\n- ".join(issues))


class DeliveryRequirementsNotMet(DeliveryNotReady):
    """A formal-delivery gate, distinct from non-bypassable packaging safety."""

    def __init__(self, report):
        self.issues = tuple(report.blockers)
        self.report = report.as_dict()
        ValueError.__init__(
            self,
            "Formal delivery requirements not met; no ZIP generated. "
            "Use --draft for a review package (safety checks still apply):\n- "
            + "\n- ".join(self.issues),
        )


def from_config(*, options: dict):
    values = dict(options)
    values["delivery_date"] = date.fromisoformat(values["delivery_date"])
    values["materials_dir"] = Path(values["materials_dir"])
    if values.get("previous_manifest"):
        values["previous_manifest"] = Path(values["previous_manifest"])
    if values.get("runtime_dir"):
        values["runtime_dir"] = Path(values["runtime_dir"])
    return TencentV3Exporter(**values)


@dataclass(frozen=True)
class TencentV3Exporter:
    vendor: str
    category: str
    delivery_date: date
    materials_dir: Path
    package_kind: str = "sample"
    batch: int = 1
    version: int = 0
    previous_manifest: Path | None = None
    max_total_bytes: int = 2 * 1024**3
    launch_options: dict = field(default_factory=dict)
    runtime_dir: Path | None = None
    draft: bool = False

    def __post_init__(self):
        if type(self.draft) is not bool:
            raise TypeError("draft must be a boolean")

    def export(self, source: DeliverySource, destination: Path) -> DeliveryReceipt:
        entries, manifest, task_count, report = self._prepare(source)
        if not self.draft and report.blockers:
            raise DeliveryRequirementsNotMet(report)
        batch_count = len(manifest["batch_task_ids"])
        name = (
            f"{self.vendor}_Tencent_{self.category}_{self.package_kind}"
            f"_batch{self.batch}_taskNum{batch_count}_fix-v{self.version}"
            f"_{self.delivery_date.isoformat()}"
        )
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / (name + ".zip")
        if output.exists():
            raise FileExistsError(output)
        manifest["files"] = {p: _receipt(v) for p, v in sorted(entries.items())}
        entries["docs/delivery-manifest.json"] = _json(manifest)
        if sum(_size(value) for value in entries.values()) > self.max_total_bytes:
            raise DeliveryNotReady(["Package exceeds max_total_bytes"])
        with tempfile.NamedTemporaryFile(
            dir=destination, suffix=".zip", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                for folder in ("tasks", "trajectory", "environment", "docs"):
                    archive.writestr(f"{name}/{folder}/", b"")
                for task_id in manifest["included_task_ids"]:
                    archive.writestr(f"{name}/tasks/{task_id}/environment/", b"")
                for path, value in sorted(entries.items()):
                    info = zipfile.ZipInfo(f"{name}/{path}")
                    executable = (
                        path.endswith(".sh")
                        or isinstance(value, Path)
                        and value.stat().st_mode & 0o111
                    )
                    info.external_attr = (0o100755 if executable else 0o100644) << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    digest = hashlib.sha256()
                    with archive.open(info, "w", force_zip64=True) as out:
                        for chunk in _chunks(value):
                            digest.update(chunk)
                            out.write(chunk)
                    expected = manifest["files"].get(path)
                    if expected and digest.hexdigest() != expected["sha256"]:
                        raise DeliveryNotReady(
                            [f"Input changed while packaging: {path}"]
                        )
            # Do not replace an existing customer delivery, even after a race.
            import os

            os.link(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return DeliveryReceipt(
            output,
            _receipt(output)["sha256"],
            task_count,
            f"/supplier_{self.vendor}/{self.vendor}_{self.package_kind}_data/{self.category}/{output.name}",
        )

    def preflight(self, source: DeliverySource) -> list[str]:
        """No ZIP or remote calls; report missing or inconsistent evidence."""
        try:
            _, _, _, report = self._prepare(source)
        except DeliveryNotReady as error:
            return list(error.issues)
        return list(report)

    def preflight_report(self, source: DeliverySource) -> dict:
        """Read-only structured result; warnings are not formal-delivery blockers."""
        try:
            _, _, _, report = self._prepare(source)
        except DeliveryNotReady as error:
            return {
                "safe_to_package": False,
                "formal_ready": False,
                "blockers": list(error.issues),
                "warnings": [],
            }
        return {"safe_to_package": True, **report.as_dict()}

    def _prepare(self, source):
        issues = DeliveryReport()
        for label, value in (("vendor", self.vendor), ("category", self.category)):
            _safe_component(value)
            try:
                component(value)
            except ValueError:
                issues.append(f"Non-portable {label} component: {value}")
        # These values form the archive path, not a task-quality admission rule.
        if (
            self.package_kind not in {"sample", "delivery"}
            or self.batch < 1
            or self.version < 0
        ):
            raise DeliveryNotReady(["Use sample/delivery, batch >= 1 and version >= 0"])
        if "_" in self.vendor:
            issues.append("Vendor must not contain underscores")
        if self.max_total_bytes < 1:
            raise DeliveryNotReady(["max_total_bytes must be positive"])
        entries = {}
        _tree(self.materials_dir / "docs", "docs", entries, issues)
        _tree(
            self.materials_dir / "environment",
            "environment",
            entries,
            issues,
            required=False,
        )
        for prefix, mapping in (
            ("docs", source.documentation_files),
            ("environment", source.environment_files),
        ):
            for filename, path in mapping.items():
                _add(entries, f"{prefix}/{_relative(filename)}", path, issues)
        from ..deployment.bundle import launcher_files

        runtime_options = {"runtime_root": self.runtime_dir} if self.runtime_dir else {}
        for name, value in launcher_files(
            self.launch_options, **runtime_options
        ).items():
            _add(entries, name, value, issues)
        with issues.section("一键运行的主机依赖材料"):
            for name in (
                "run.sh",
                "environment/runtime/pyproject.toml",
                "environment/runtime/uv.lock",
                "environment/runtime/prophet_bench_env/deployment/launch.py",
                "environment/runtime/prophet_bench_env/integrations/harbor/installed.py",
            ):
                _required(entries, name, issues)
        with issues.section("数据集文档 / Dataset documentation"):
            for name in DATASET_DOCS:
                _required(entries, "docs/" + name, issues)
        review = _load_json(self.materials_dir / "review.json", issues)
        if not review.get("reviewer") or not review.get("reviewed_at"):
            issues.append(
                "review.json requires reviewer and reviewed_at; QA is supplied, not generated"
            )
        if (self.materials_dir / "review.json").is_file():
            _add(
                entries,
                "docs/delivery-review.json",
                self.materials_dir / "review.json",
                issues,
            )
        try:
            datetime.fromisoformat(review.get("reviewed_at", ""))
        except (TypeError, ValueError):
            issues.append("reviewed_at must be an ISO timestamp")
        configs, tasks = {}, {}
        if source.tasks_dir.is_symlink():
            raise DeliveryNotReady(["tasks_dir cannot be a symlink"])
        if not source.tasks_dir.is_dir():
            issues.append("Missing tasks_dir")
        else:
            for task in sorted(source.tasks_dir.iterdir()):
                if task.is_symlink():
                    raise DeliveryNotReady([f"Symlink in tasks: {task}"])
                if not task.is_dir():
                    issues.append(f"Unexpected task entry: {task.name}")
                    _add(entries, f"tasks/{task.name}", task, issues)
                    continue
                task_id = _safe_component(task.name)
                tasks[task_id] = task
                _tree(task, f"tasks/{task_id}", entries, issues)
                try:
                    config = tomllib.loads((task / "task.toml").read_text())
                except (OSError, ValueError):
                    issues.append(f"{task_id}: missing or invalid task.toml")
                else:
                    configs[task_id] = config
                    issues.check("任务配置字段", checks.validate_task, task_id, config)
                    # A pinned image, not this old build recipe, is the V3
                    # runtime authority. Preserve the original for audit only.
                    recipe = f"tasks/{task_id}/environment/Dockerfile"
                    image = _mapping(config.get("environment")).get("docker_image")
                    if image and recipe in entries:
                        _add(
                            entries,
                            f"docs/source-tasks/{task_id}/environment/Dockerfile",
                            entries.pop(recipe),
                            issues,
                        )
                with issues.section("每题必需材料"):
                    for filename in (
                        "task.toml",
                        "instruction.md",
                        "tests/test.sh",
                        "solution/solve.sh",
                    ):
                        _required(entries, f"tasks/{task_id}/{filename}", issues)
                    for filename in TASK_DOCS:
                        _required(entries, f"docs/{task_id}/{filename}", issues)
                solve = task / "solution/solve.sh"
                if (
                    solve.is_file()
                    and not solve.is_symlink()
                    and b"reward.json" in solve.read_bytes()
                ):
                    issues.warn(
                        f"{task_id}: solve.sh must solve the task, not access reward.json"
                    )
                for p in tuple(entries):
                    if p.startswith(f"tasks/{task_id}/environment/") and any(
                        part
                        in {
                            "tests",
                            "solution",
                            "reward.json",
                            "trajectory.json",
                            "Dockerfile",
                        }
                        for part in Path(p).parts[3:]
                    ):
                        issues.warn(
                            f"Workspace contains build instructions or private material: {p}"
                        )
                with issues.advisory():
                    issues.check(
                        "编辑历史（建议项）", checks.check_history, task_id, entries
                    )
        if not tasks:
            issues.append("No tasks supplied")
        issues.check(
            "镜像清单与材料（未实际加载）",
            checks.validate_images,
            configs,
            review.get("images", []),
            entries,
        )
        records = self._trials(source, tasks, review, entries, issues)
        for task_id, task_path in tasks.items():
            task_records = [r for r in records if r["task_id"] == task_id]
            baseline = {}
            with issues.section("Oracle / NOP"):
                for role, expected in (("oracle", 1.0), ("nop", 0.0)):
                    runs = [r for r in task_records if r["role"] == role]
                    scores = [r["reward"] for r in runs if not r["excluded"]]
                    normalized = [
                        checks.normalized_reward(r) for r in runs if not r["excluded"]
                    ]
                    if (
                        not scores
                        or any(v != expected for v in normalized)
                        or any(r["excluded"] for r in runs)
                    ):
                        issues.append(
                            f"{task_id}: observed {role} rewards={scores}, normalized={normalized}; requested {expected}"
                        )
                    if scores:
                        baseline[role] = {"rewards": scores, "rollout_n": len(scores)}
                        if len(set(scores)) == 1:
                            baseline[role]["reward"] = scores[0]
                        if any(r.get("reward_scale") not in (None, 1) for r in runs):
                            baseline[role]["normalized_rewards"] = normalized
            performance = issues.check(
                "运行次数与难度统计",
                checks.performance,
                task_id,
                task_records,
                review,
                entries,
            )
            if task_id in configs:
                config = dict(configs[task_id])
                config["rollout_baseline"] = baseline
                _add(
                    entries,
                    f"docs/source-tasks/{task_id}/task.toml",
                    task_path / "task.toml",
                    issues,
                )
                entries[f"tasks/{task_id}/task.toml"] = toml.dumps(config).encode()
            original = entries.get(f"tasks/{task_id}/performance.toml")
            if original is not None:
                _add(
                    entries,
                    f"docs/source-tasks/{task_id}/performance.toml",
                    original,
                    issues,
                )
            entries[f"tasks/{task_id}/performance.toml"] = toml.dumps(
                performance or {"instance_id": task_id, "results": []}
            ).encode()
        issues.check(
            "失败模式与 hint 证据关联（非人工因果复核）",
            checks.failure_evidence,
            records,
            entries,
        )
        manifest = {
            "exporter": "tencent-hy4.v3",
            "specification_date": "2026-09-26",
            "vendor": self.vendor,
            "category": self.category,
            "package_kind": self.package_kind,
            "draft": self.draft,
            "batch": self.batch,
            "version": self.version,
            "delivery_date": self.delivery_date.isoformat(),
            "included_task_ids": sorted(tasks),
            "batch_task_ids": sorted(tasks),
            "source_task_sha256": {
                key: _directory_hash(path) for key, path in tasks.items()
            },
            "task_sha256": {key: _task_digest(entries, key) for key in tasks},
            "instance_sha256": {key: _instance_digest(entries, key) for key in tasks},
            "runs": records,
            "summary_path": "docs/delivery-summary.md",
        }
        if self.version:
            previous = (
                _load_json(self.previous_manifest, issues)
                if self.previous_manifest
                else {}
            )
            if not previous:
                issues.append(
                    "A revision requires previous_manifest from this exporter"
                )
            else:
                for key in ("exporter", "vendor", "category", "package_kind", "batch"):
                    if previous.get(key) != manifest[key]:
                        issues.append(f"Revision changed batch identity: {key}")
                old_version = previous.get("version")
                if type(old_version) is not int or self.version != old_version + 1:
                    issues.append("Revision version must follow previous_manifest")
                original_list = previous.get("batch_task_ids", [])
                if not isinstance(original_list, list) or any(
                    not isinstance(value, str) for value in original_list
                ):
                    issues.append("Previous manifest has invalid batch_task_ids")
                    original_list = []
                original_ids = set(original_list)
                if not set(tasks) <= original_ids:
                    issues.append("Revision introduced new instance_ids")
                for key in tasks:
                    if (
                        _mapping(previous.get("instance_sha256")).get(key)
                        == manifest["instance_sha256"][key]
                    ):
                        issues.warn(f"Revision contains unchanged task: {key}")
                manifest["batch_task_ids"] = sorted(original_ids | set(tasks))
                manifest["task_sha256"] = {
                    **_mapping(previous.get("task_sha256")),
                    **manifest["task_sha256"],
                }
                manifest["instance_sha256"] = {
                    **_mapping(previous.get("instance_sha256")),
                    **manifest["instance_sha256"],
                }
        elif self.previous_manifest:
            issues.append("version=0 cannot have previous_manifest")
        for name, value in entries.items():
            _scan(name, value, issues)
        if sum(_size(value) for value in entries.values()) > self.max_total_bytes:
            raise DeliveryNotReady(["Package exceeds configured max_total_bytes"])
        _add(
            entries,
            "docs/delivery-summary.md",
            issues.markdown(draft=self.draft),
            issues,
        )
        _scan("docs/delivery-summary.md", entries["docs/delivery-summary.md"], issues)
        return entries, manifest, len(tasks), issues

    def _trials(self, source, tasks, review, entries, issues):
        records = []
        counters = defaultdict(int)
        seen = set()
        annotations = _mapping(review.get("runs"))
        profiles = _mapping(review.get("models"))
        folders = {}
        for job in source.job_dirs:
            if job.is_symlink():
                raise DeliveryNotReady([f"Symlink job directory: {job}"])
            if not job.is_dir():
                issues.append(f"Missing job directory: {job.name}")
                continue
            for trial in sorted(job.iterdir()):
                if not trial.is_dir():
                    continue
                key = f"{_safe_component(job.name)}/{_safe_component(trial.name)}"
                if key in seen:
                    raise DeliveryNotReady(
                        [f"Ambiguous duplicate trial identity: {key}"]
                    )
                seen.add(key)
                with issues.section("运行证据"):
                    record = self._trial(
                        trial,
                        key,
                        tasks,
                        annotations,
                        profiles,
                        folders,
                        counters,
                        entries,
                        issues,
                    )
                records.append(record)
                issues.check("指定模型与运行配置", checks.model_profile, record)
        unknown = set(annotations) - seen
        if unknown:
            issues.append(
                "Review references missing runs: " + ", ".join(sorted(unknown))
            )
        issues.check("评分量纲", checks.annotate_reward_units, records, review)
        return records

    def _trial(
        self,
        trial,
        key,
        tasks,
        annotations,
        profiles,
        folders,
        counters,
        entries,
        issues,
    ):
        # Every supplied trial is retained, even when association or QA is missing.
        result = _load_json(trial / "result.json", issues)
        annotation = annotations.get(key)
        if not isinstance(annotation, dict):
            issues.append(f"{key}: missing QA annotation in review.json runs")
            annotation = {}
        hint = annotation.get("hint")
        if hint is not None and not isinstance(hint, dict):
            issues.append(f"{key}: invalid hint annotation")
            hint = {"unparsed": hint}
        task_id = (
            annotation.get("task_id", result.get("task_name"))
            if hint
            else result.get("task_name")
        )
        if not isinstance(task_id, str):
            task_id = None
        if task_id not in tasks:
            issues.append(f"{key}: task outside delivery")
        elif not hint and annotation.get("task_id", task_id) != task_id:
            issues.append(
                f"{key}: review task_id differs from recorded task; using recorded task"
            )
        snapshot = tasks.get(task_id)
        if hint and isinstance(hint.get("snapshot"), str) and hint["snapshot"]:
            snapshot = self.materials_dir / "hint_tasks" / _relative(hint["snapshot"])
        elif hint:
            snapshot = None
        snapshot_matches = not (
            snapshot is None
            or not snapshot.is_dir()
            or result.get("task_checksum") != _directory_hash(snapshot)
        )
        if not snapshot_matches:
            issues.append(
                f"{key}: frozen task checksum mismatch or missing hint snapshot"
            )
        if hint and snapshot is not None and task_id in tasks:
            try:
                original_prompt = (tasks[task_id] / "instruction.md").read_text()
                hinted_prompt = (snapshot / "instruction.md").read_text()
                if (
                    original_prompt not in hinted_prompt
                    or hinted_prompt == original_prompt
                ):
                    issues.append(
                        f"{key}: hint snapshot must preserve the original question and add a hint"
                    )
            except (OSError, ValueError):
                issues.append(f"{key}: hint/original prompt unavailable")
            hint_prefix = (
                f"docs/{task_id}/hint-task-snapshots/{_relative(hint['snapshot'])}"
            )
            if not any(p.startswith(hint_prefix + "/") for p in entries):
                _tree(snapshot, hint_prefix, entries, issues)
        agent = _mapping(_mapping(result.get("config")).get("agent"))
        harness = str(
            agent.get("name")
            or _mapping(result.get("agent_info")).get("name")
            or "unknown"
        ).replace("-", "_")
        model = str(agent.get("model_name") or "unknown")
        profile = _mapping(profiles.get(model))
        if harness in {"oracle", "nop"}:
            role, folder = harness, harness
        else:
            role, folder = profile.get("role"), profile.get("directory")
            if (
                role not in ("Hy4-preview", "sota")
                or not isinstance(folder, str)
                or not folder
            ):
                issues.warn(f"{key}: missing Hy4/SOTA profile for actual model")
            if role not in ("Hy4-preview", "sota"):
                role = "other"
            folder = (
                _safe_component(folder)
                if isinstance(folder, str) and folder
                else _label(model)
            )
            if role == "Hy4-preview" and folder != "Hy4-preview":
                issues.append(f"{key}: Hy4 folder must be Hy4-preview")
        owner = folders.setdefault(folder, model)
        if owner != model:
            issues.append(
                f"{key}: multiple actual models share directory {folder}; kept in separate directories"
            )
            folder = _label(model)
        hacked = annotation.get("reward_hacked")
        if not isinstance(hacked, bool):
            issues.append(f"{key}: explicitly review reward_hacked (true/false)")
        system_error = bool(result.get("exception_info"))
        result_reward = _mapping(
            _mapping(result.get("verifier_result")).get("rewards")
        ).get("reward")
        reward_source = "verifier/reward.json"
        text_reward = trial / "verifier/reward.txt"
        if not (trial / reward_source).exists() and text_reward.is_file():
            # Same precedence as Harbor: JSON first, text only when JSON absent.
            reward_source = "verifier/reward.txt"
            try:
                reward = float(text_reward.read_text().strip())
                if not math.isfinite(reward):
                    reward = None
            except (ValueError, OSError):
                reward = None
        else:
            observed = _load_json(trial / reward_source, issues)
            reward = observed.get("reward")
        if reward != result_reward:
            issues.append(
                f"{key}: {reward_source} differs from Harbor result; both retained, statistics use {reward_source}"
            )
        if not _reward(reward):
            issues.append(
                f"{key}: missing finite verifier reward; run retained without a statistical reward"
            )
        evidence = {}
        evaluation_path = trial / "verifier/evaluation.json"
        if evaluation_path.is_file():
            evaluation = _load_json(evaluation_path, issues)
            evidence = {
                name: evaluation.get(name)
                for name in ("reward", "total_score", "max_score")
                if _reward(evaluation.get(name))
            }
            if "max_score" in evidence:
                evidence["max_score_source"] = "verifier/evaluation.json:max_score"
        if snapshot_matches and "max_score" not in evidence:
            verification_path = snapshot / "tests/verification.json"
            if verification_path.is_file():
                verification = _load_json(verification_path, issues)
                maximum = _mapping(verification.get("rubric")).get("max_score")
                if _reward(maximum):
                    evidence.update(
                        max_score=maximum,
                        max_score_source="tests/verification.json:rubric.max_score",
                    )
        reasons = []
        if not result.get("finished_at"):
            reasons.append("unfinished_run")
            issues.append(
                f"{key}: unfinished run retained without contributing to statistics"
            )
        if system_error:
            reasons.append("system_error")
        if hacked is True:
            reasons.append("reward_hacked")
        if not _reward(reward):
            reasons.append("unavailable_reward")
        if not snapshot_matches:
            reasons.append("unmatched_task_snapshot")
        counters[(task_id, folder, harness, bool(hint))] += 1
        n = counters[(task_id, folder, harness, bool(hint))]
        rollout = f"rollout_{_label(harness) if not re.fullmatch(r'[A-Za-z0-9_-]+', harness) else harness}_{'hint_' if hint else ''}{n}"
        prefix = f"trajectory/{task_id}/{folder}/{rollout}"
        if task_id not in tasks:
            prefix = f"docs/unassigned-runs/{key}"
        elif role in {"oracle", "nop"}:
            prefix = f"docs/{task_id}/baselines/{role}/{key}"
        elif system_error:
            prefix = f"docs/{task_id}/excluded-runs/{key}"
        record = {
            "run": key,
            "task_id": task_id,
            "role": role,
            "model": model,
            "harness": harness,
            "reward": reward if _reward(reward) else None,
            "verifier_reward": reward,
            "result_reward": result_reward,
            "reward_source": reward_source,
            "reward_json_derived": reward_source.endswith(".txt") and _reward(reward),
            "reward_evidence": evidence,
            "task_checksum": result.get("task_checksum"),
            "task_snapshot_matches": snapshot_matches,
            "hint": hint,
            "excluded": bool(reasons),
            "exclusion_reason": ",".join(reasons) or None,
            "reward_hacked": hacked if isinstance(hacked, bool) else None,
            "failure_modes": annotation.get("failure_modes")
            if isinstance(annotation.get("failure_modes"), list)
            else [],
            "path": prefix,
            "finished_at": result.get("finished_at"),
            "reasoning_effort": _field(
                result,
                profile.get("effort_field", "config.agent.kwargs.reasoning_effort"),
            ),
            "context_window_tokens": _field(
                result,
                profile.get("context_field", "config.agent.kwargs.context_window"),
            ),
        }
        # Preserve the original trial files, including malformed/mismatched
        # results. The credential scanner runs before anything is written.
        _tree(trial, prefix, entries, issues)
        if role not in {"oracle", "nop"} and not system_error:
            trajectory_path = trial / "agent/trajectory.json"
            try:
                validate_atif_bundle(trajectory_path)
                trajectory = Trajectory.model_validate_json(trajectory_path.read_text())
                if (
                    trajectory.agent.model_name != model
                    or trajectory.agent.name.replace("-", "_") != harness
                ):
                    issues.append(f"{key}: ATIF model/harness differs from actual run")
            except (OSError, ValueError):
                issues.append(f"{key}: missing or invalid ATIF trajectory")
            else:
                record["steps"] = sum(
                    step.source == "agent" for step in trajectory.steps
                )
                if trajectory.final_metrics is not None:
                    record["metrics"] = trajectory.final_metrics.model_dump(
                        exclude_none=True
                    )
            manifest = _load_json(
                trial / "artifacts/manifest.json", issues, object_only=False
            )
            if (
                not isinstance(manifest, list)
                or not manifest
                or any(
                    not isinstance(x, dict) or x.get("status") not in {"ok", "empty"}
                    for x in manifest
                )
            ):
                issues.append(f"{key}: environment artifact collection incomplete")
        if (
            trial / "verifier/reward.json"
        ).is_file() and prefix + "/reward.json" not in entries:
            _add(
                entries,
                prefix + "/reward.json",
                trial / "verifier/reward.json",
                issues,
            )
        elif (
            reward_source.endswith(".txt")
            and _reward(reward)
            and prefix + "/reward.json" not in entries
        ):
            # Customer-format view only; retain the native text and its hash.
            _add(entries, prefix + "/reward.json", _json({"reward": reward}), issues)
        issues.check("ATIF 审计关联", checks.capture_references, entries, prefix)
        return record


def _field(value, path):
    for part in str(path).split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _safe_component(value):
    """Reject traversal, not harmless customer filename/style deviations."""
    if (
        not isinstance(value, str)
        or value in {"", ".", ".."}
        or any(c in value for c in "/\\\x00")
    ):
        raise DeliveryNotReady(["Unsafe archive path component"])
    return value


def _label(value):
    text = str(value)
    return (
        re.sub(r"[^A-Za-z0-9_-]+", "-", text)[:64].strip("-")
        + "-"
        + hashlib.sha256(text.encode()).hexdigest()[:10]
    )


def _reward(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()


def _relative(value):
    path = Path(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(p in {"..", "."} for p in path.parts)
    ):
        raise DeliveryNotReady(["Expected a relative path within supplied materials"])
    return path.as_posix()


def _load_json(path, issues, *, object_only=True):
    if path.is_symlink():
        raise DeliveryNotReady([f"Symlink is not deliverable: {path}"])
    try:

        def reject_constant(value):
            raise ValueError(f"Non-finite JSON number: {value}")

        value = json.loads(path.read_text(), parse_constant=reject_constant)
        if object_only and not isinstance(value, dict):
            raise ValueError("not object")
        return value
    except (OSError, ValueError):
        issues.append(f"Missing or invalid JSON: {path}")
        return {}


def _add(entries, name, value, issues):
    _relative(name)
    if name in entries:
        raise DeliveryNotReady(
            [f"Duplicate delivery path would overwrite evidence: {name}"]
        )
    elif isinstance(value, Path) and value.is_symlink():
        raise DeliveryNotReady([f"Symlink is not deliverable: {name}"])
    elif isinstance(value, Path) and not value.is_file():
        issues.append(f"Missing file: {name}")
    else:
        entries[name] = value


def _tree(root, prefix, entries, issues, *, required=True):
    if root.is_symlink():
        raise DeliveryNotReady([f"Symlink is not deliverable: {root}"])
    if not root.is_dir():
        if required:
            issues.append(f"Missing directory: {root}")
        return
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DeliveryNotReady([f"Symlink is not deliverable: {path}"])
        elif path.is_file() and not {"__pycache__", ".git", ".venv"}.intersection(
            path.relative_to(root).parts
        ):
            _add(
                entries, prefix + "/" + path.relative_to(root).as_posix(), path, issues
            )


def _required(entries, path, issues):
    if path not in entries or _size(entries[path]) == 0:
        issues.append(f"Missing required material: {path}")


def _chunks(value):
    if isinstance(value, bytes):
        yield value
    else:
        with value.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                yield chunk


def _size(value):
    return len(value) if isinstance(value, bytes) else value.stat().st_size


def _receipt(value):
    digest = hashlib.sha256()
    size = 0
    for chunk in _chunks(value):
        digest.update(chunk)
        size += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def _scan(name, value, issues):
    tail = b""
    rows = 0
    for chunk in _chunks(value):
        try:
            _check_content(name, tail + chunk)
        except ValueError as error:
            raise DeliveryNotReady([f"{name}: {error}"]) from error
        tail = (tail + chunk)[-16384:]
        rows += chunk.count(b"\n")
    try:
        if name.endswith(".csv"):
            stream = (
                io.StringIO(value.decode("utf-8-sig"), newline="")
                if isinstance(value, bytes)
                else value.open(encoding="utf-8-sig", newline="")
            )
            with stream:
                reader = csv.reader(stream)
                next(reader, None)
                rows = sum(1 for _ in reader)
            if rows > 10000:
                issues.append(f"{name}: exceeds 10,000 CSV records")
        elif (
            name.endswith(".jsonl")
            and rows + bool(tail and not tail.endswith(b"\n")) > 10000
        ):
            issues.append(
                f"{name}: exceeds 10,000 lines; split at source with references preserved"
            )
    except (ValueError, csv.Error) as error:
        issues.append(f"{name}: {error}")


def _task_digest(entries, task_id):
    prefix = f"tasks/{task_id}/"
    return hashlib.sha256(
        _json(
            {
                p[len(prefix) :]: _receipt(v)
                for p, v in sorted(entries.items())
                if p.startswith(prefix)
            }
        )
    ).hexdigest()


def _directory_hash(root):
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise DeliveryNotReady([f"Symlink in frozen task: {root}"])
    return dirhash(root, "sha256")


def _instance_digest(entries, task_id):
    prefixes = (f"tasks/{task_id}/", f"docs/{task_id}/", f"trajectory/{task_id}/")
    return hashlib.sha256(
        _json(
            {
                p: _receipt(v)
                for p, v in sorted(entries.items())
                if p.startswith(prefixes)
            }
        )
    ).hexdigest()
