"""Evidence checks for Tencent's 2026-09-26 formal/draft delivery modes."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

from .tencent_v3_report import warn


def validate_task(task_id, config, issues):
    from .tencent_v3 import METADATA_FIELDS

    metadata = config.get("metadata", {})
    if set(metadata) != METADATA_FIELDS or any(
        not isinstance(v, str) for v in metadata.values()
    ):
        issues.append(
            f"{task_id}: metadata must contain exactly the seven V3 identity fields"
        )
    if metadata.get("instance_id") != task_id:
        issues.append(f"{task_id}: metadata.instance_id must equal the directory name")
    if (
        not metadata.get("name")
        or not metadata.get("task")
        or metadata.get("language") not in {"en", "zh", "other"}
    ):
        issues.append(f"{task_id}: missing author/task or invalid language")
    try:
        value = metadata["prompt_creation_time"]
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError("date")
    except (KeyError, ValueError, TypeError):
        issues.append(f"{task_id}: prompt_creation_time must be YYYY-MM-DD")
    if "topic" in config or "version" in config:
        issues.append(f"{task_id}: V3 has no topic block or version field")
    for role in ("agent", "verifier"):
        timeout = config.get(role, {}).get("timeout_sec")
        if (
            not isinstance(timeout, (float, int))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            issues.append(f"{task_id}: {role}.timeout_sec must be finite and positive")
    for role, environment in (
        ("agent", config.get("environment", {})),
        ("verifier", config.get("verifier", {}).get("environment", {})),
    ):
        if (
            role == "verifier"
            and config.get("verifier", {}).get("environment_mode") != "separate"
        ):
            continue
        image = environment.get("docker_image", "")
        last = image.split("@")[0].rsplit("/", 1)[-1]
        pinned_digest = re.search(r"@sha256:[0-9a-f]{64}$", image)
        if not pinned_digest and (
            ":" not in last or last.rsplit(":", 1)[-1] in {"", "latest"}
        ):
            issues.append(
                f"{task_id}: {role} image needs an explicit non-latest tag or digest"
            )
        if (
            type(environment.get("memory_mb")) is not int
            or environment["memory_mb"] <= 0
        ):
            issues.append(f"{task_id}: {role} memory_mb must be a positive integer")
        if not environment.get("workdir") or not environment.get("network_mode"):
            issues.append(f"{task_id}: {role} workdir/network_mode must be declared")
    judge = config.get("judge", {})
    if judge.get("type") not in {"programmatic", "llm", "agent", "mixed"} or judge.get(
        "reward_kind"
    ) not in {"binary", "continuous", "multi_component"}:
        issues.append(f"{task_id}: declare judge.type and judge.reward_kind")


def validate_images(configs, inventory, entries, issues):
    from .tencent_v3 import _relative, _required

    images = {}
    for image in inventory:
        reference = image.get("reference")
        if not reference or reference in images:
            issues.append("Image inventory needs unique references")
            continue
        images[reference] = image
    for task_id, config in configs.items():
        environments = [config.get("environment", {})]
        if config.get("verifier", {}).get("environment_mode") == "separate":
            environments.append(config["verifier"].get("environment", {}))
        for env in environments:
            reference = env.get("docker_image")
            image = images.get(reference)
            if not image or image.get("visibility") not in {"public", "private"}:
                issues.append(
                    f"{task_id}: missing explicit public/private image inventory for {reference}"
                )
                continue
            if env.get("image_digest") and image.get("digest") != env["image_digest"]:
                issues.append(f"{task_id}: image inventory digest mismatch")
            if image.get("digest") and not re.fullmatch(
                r"sha256:[0-9a-f]{64}", image["digest"]
            ):
                issues.append(f"{task_id}: invalid image digest")
            if image["visibility"] == "private":
                _required(entries, "environment/README.md", issues)
                directory = image.get("directory")
                if not directory:
                    issues.append(f"{task_id}: private image needs a bundled directory")
                    continue
                prefix = "environment/" + _relative(directory) + "/"
                manifest = prefix + "manifest.md"
                _required(entries, manifest, issues)
                if manifest in entries:
                    value = entries[manifest]
                    text = (
                        value.decode()
                        if isinstance(value, bytes)
                        else value.read_text()
                    )
                    if (
                        reference not in text
                        or image.get("digest")
                        and image["digest"] not in text
                    ):
                        issues.append(
                            f"{task_id}: image manifest does not match tag/digest"
                        )
                if not any(
                    p.startswith(prefix)
                    and (
                        p.endswith((".tar", ".tar.gz", ".oci"))
                        or p in {prefix + "Dockerfile", prefix + "index.json"}
                    )
                    for p in entries
                ):
                    issues.append(
                        f"{task_id}: private image artifact or complete build context missing"
                    )


def check_history(task_id, entries, issues):
    from .tencent_v3 import _required

    prefix = f"docs/{task_id}/edit_history/"
    versions = sorted(
        {
            int(p[len(prefix) :].split("/", 1)[0][1:])
            for p in entries
            if p.startswith(prefix) and re.match(r"v\d+/", p[len(prefix) :])
        }
    )
    # History is strongly encouraged, not mandatory. If supplied, validate its
    # structure; never manufacture a v0 or a human/LLM review.
    if not versions:
        return
    if versions != list(range(versions[-1] + 1)):
        issues.append(f"{task_id}: edit history versions must be contiguous from v0")
    for version in versions:
        root = prefix + f"v{version}/"
        for path in (
            "task.toml",
            "instruction.md",
            "tests/test.sh",
            "solution/solve.sh",
        ):
            _required(entries, root + "task/" + path, issues)
        if version:
            if not any(p.startswith(root + "review/") for p in entries):
                issues.append(f"{task_id}: v{version} is missing review material")
            _required(entries, root + f"v{version - 1}_v{version}_patch.diff", issues)


def annotate_reward_units(records, review, issues):
    """Derive display units, never change rewards or statistical eligibility."""
    from .tencent_v3 import _mapping, _reward

    declared = _mapping(review.get("reward_scales"))
    inferred = defaultdict(set)
    for run in records:
        raw, task_id = run["reward"], run["task_id"]
        evidence = run["reward_evidence"]
        maximum, total = evidence.get("max_score"), evidence.get("total_score")
        scale, source = None, None
        if task_id in declared:
            if _reward(declared[task_id]) and declared[task_id] > 0:
                scale, source = declared[task_id], "review.reward_scales"
            else:
                issues.append(
                    f"{run['run']}: invalid declared reward scale; raw values retained"
                )
        elif (
            _reward(raw)
            and raw != 0
            and _reward(maximum)
            and maximum > 0
            and _reward(total)
            and evidence.get("reward") == raw
        ):
            if math.isclose(raw, total, rel_tol=1e-9, abs_tol=1e-12):
                scale = maximum
            elif math.isclose(raw, total / maximum, rel_tol=1e-9, abs_tol=1e-12):
                scale = 1
            if scale is not None:
                source = evidence.get("max_score_source")
            else:
                issues.append(
                    f"{run['run']}: reward does not match total_score or total_score/max_score; normalization unresolved"
                )
        run.update(reward_scale=scale, reward_scale_source=source)
        if scale is not None and run["task_snapshot_matches"]:
            inferred[(task_id, run["task_checksum"])].add(scale)

    for run in records:
        raw, scale = run["reward"], run["reward_scale"]
        peers = inferred[(run["task_id"], run["task_checksum"])]
        if scale is None and run["task_id"] not in declared:
            # Zero cannot reveal its own units. Only use consistent evidence
            # from the exact same frozen task; never infer a divisor from size.
            if raw == 0 and run["task_snapshot_matches"] and len(peers) == 1:
                scale = next(iter(peers))
                run["reward_scale_source"] = "same_frozen_task"
            elif _reward(raw) and 0 <= raw <= 1:
                scale = 1
                run["reward_scale_source"] = "unit_interval"
        value = raw / scale if _reward(raw) and scale is not None else None
        if value is not None and not math.isfinite(value):
            value = None
        run.update(reward_scale=scale, normalized_reward=value)
        if _reward(raw) and value is None:
            issues.append(
                f"{run['run']}: normalization unavailable; raw reward={raw} retained in statistics"
            )
        elif value is not None and not 0 <= value <= 1:
            issues.append(
                f"{run['run']}: normalized reward={value} lies outside [0, 1]; not clipped"
            )


def normalized_reward(run):
    if "normalized_reward" in run:
        return run["normalized_reward"]
    value = run.get("reward")
    return (
        value
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
        else None
    )


def performance(task_id, records, review, entries, issues):
    """Record all model profiles; validity exclusions affect statistics, not export."""
    groups = defaultdict(list)
    for run in records:
        if run["role"] not in {"oracle", "nop"}:
            settings = json.dumps(
                {
                    key: run.get(key)
                    for key in (
                        "reasoning_effort",
                        "context_window_tokens",
                        "reward_scale",
                    )
                },
                sort_keys=True,
            )
            groups[(run["role"], run["model"], run["harness"], settings)].append(run)
    rows = []
    for (role, model, harness, settings), runs in sorted(groups.items()):
        valid = [r for r in runs if not r["excluded"] and not r["hint"]]
        scores = [r["reward"] for r in valid]
        normalized = [normalized_reward(r) for r in valid]
        rows.append(
            {
                "role": role,
                "model": model,
                "harness": harness,
                **{
                    key: value
                    for key, value in json.loads(settings).items()
                    if value is not None
                },
                "recorded_run_n": len(runs),
                "hint_run_n": sum(bool(r["hint"]) for r in runs),
                "excluded_run_n": sum(r["excluded"] for r in runs),
                "rollout_n": len(scores),
                "success_n": sum(v == 1 for v in normalized),
                "success_definition": "normalized_reward == 1.0",
                "pass_at_4_method": "1 - C(n - success_n, 4) / C(n, 4); n >= 4",
            }
        )
        if not scores:
            issues.append(
                f"{task_id}: {model}/{harness}: no usable ordinary rewards; statistics unavailable"
            )
            continue
        rows[-1].update(
            reward_mean=statistics.mean(scores),
            reward_std=statistics.pstdev(scores),
            reward_min=min(scores),
            reward_max=max(scores),
            evaluated_at=max(str(r["finished_at"] or "") for r in valid),
        )
        if all(v is not None for v in normalized):
            rows[-1]["normalized_reward_mean"] = statistics.mean(normalized)
            rows[-1]["pass_rate"] = sum(v == 1 for v in normalized) / len(normalized)
        else:
            rows[-1].pop("success_n", None)
            issues.append(
                f"{task_id}: {model}: normalized statistics unavailable; raw rewards retained"
            )
        if len(scores) >= 4 and "pass_rate" in rows[-1]:
            rows[-1]["pass_at_4"] = 1 - math.comb(
                len(scores) - rows[-1]["success_n"], 4
            ) / math.comb(len(scores), 4)
        if not rows[-1]["evaluated_at"]:
            issues.append(f"{task_id}: missing recorded evaluation timestamp")
        if all("steps" in r for r in valid):
            rows[-1]["mean_steps"] = statistics.mean(r["steps"] for r in valid)
        for target, original in (
            ("mean_cost_usd", "total_cost_usd"),
            ("mean_prompt_tokens", "total_prompt_tokens"),
            ("mean_completion_tokens", "total_completion_tokens"),
        ):
            if all(original in r.get("metrics", {}) for r in valid):
                rows[-1][target] = statistics.mean(
                    r["metrics"][original] for r in valid
                )
    hy = [r for r in rows if r["role"] == "Hy4-preview"]
    sota = [r for r in rows if r["role"] == "sota"]
    if not hy or not sota:
        issues.append(
            f"{task_id}: requested Hy4-preview and Claude SOTA profiles not both present"
        )
    for row in hy + sota:
        minimum = 4 if row["role"] == "Hy4-preview" else 5
        if row["rollout_n"] < minimum:
            issues.append(
                f"{task_id}: {row['model']}: {row['rollout_n']} valid runs; formal export requires >= {minimum} "
                + (
                    "to compute pass@4"
                    if row["role"] == "Hy4-preview"
                    else "(conservative project default following the document example)"
                )
            )
        elif row["role"] == "Hy4-preview" and row["rollout_n"] < 8:
            warn(
                issues,
                f"{task_id}: {row['model']}: {row['rollout_n']} valid runs; document tree illustrates 8, but pass@4 is computable",
            )
        metric, limit = (
            ("pass_at_4", 0.5)
            if row["role"] == "Hy4-preview"
            else ("normalized_reward_mean", 0.6)
        )
        if metric not in row:
            issues.append(f"{task_id}: {row['model']}: {metric} unavailable")
        elif row[metric] >= limit:
            issues.append(
                f"{task_id}: {row['model']}: {metric}={row[metric]:.6g}, latest specification requires < {limit}"
            )
        if row["role"] == "sota" and row.get("reward_max") == 0:
            evidence_map = review.get("all_zero_evidence", {})
            evidence = (
                evidence_map.get(task_id) if isinstance(evidence_map, dict) else None
            )
            validated = (
                isinstance(evidence, str)
                and evidence in entries
                and any(
                    r["role"] == "sota"
                    and r["model"] == row["model"]
                    and r["hint"]
                    and not r["excluded"]
                    and normalized_reward(r) == 1
                    for r in records
                )
            )
            if not validated:
                issues.append(
                    f"{task_id}: {row['model']}: all-zero rewards need documented solvability and successful hint validation"
                )
    if review.get("difficulty_agreement"):
        agreement = review["difficulty_agreement"]
        if not isinstance(agreement, str) or agreement not in entries:
            issues.append(
                f"{task_id}: referenced Tencent difficulty agreement is missing: {agreement}"
            )
    return {"instance_id": task_id, "results": rows}


def model_profile(record, issues):
    """Compare observed settings only; review role labels cannot rename a model."""
    role, model = record["role"], record["model"].lower()
    if role in {"oracle", "nop"}:
        return
    if role not in {"Hy4-preview", "sota"}:
        warn(
            issues,
            f"{record['run']}: model is not assigned a requested Hy4/Claude profile",
        )
        return
    pattern = (
        r"hy4[-_ ]?preview" if role == "Hy4-preview" else r"claude[-_ ]opus[-_ ]5[.-]5"
    )
    effort = "high" if role == "Hy4-preview" else "xhigh"
    if not re.search(pattern, model):
        issues.append(
            f"{record['run']}: actual model {record['model']} does not identify the requested {role} version"
        )
    if record["harness"] != "claude_code":
        issues.append(
            f"{record['run']}: actual harness {record['harness']}; latest specification requests Claude Code"
        )
    if record.get("reasoning_effort") != effort:
        issues.append(
            f"{record['run']}: observed reasoning effort {record.get('reasoning_effort')!r}; requested {effort}"
        )
    if record.get("context_window_tokens") not in {1_000_000, 1_048_576}:
        issues.append(
            f"{record['run']}: requested 1M context configuration not verified in runtime records"
        )


def failure_evidence(records, entries, issues):
    from .tencent_v3 import _relative, _required

    by_key = {r["run"]: r for r in records}
    for run in records:
        if (
            run["role"] in {"oracle", "nop"}
            or run["excluded"]
            or run["hint"]
            or normalized_reward(run) == 1
        ):
            continue
        if not run["failure_modes"]:
            issues.append(f"{run['run']}: failure needs reviewed failure mode labels")
        for mode in run["failure_modes"]:
            prefix = f"docs/{run['task_id']}/failure_analysis/{Path(run['path']).parts[2]}/{_relative(mode)}/"
            _required(entries, prefix + "README.md", issues)
            _required(entries, prefix + "hint_v1.md", issues)
            validations = [
                r
                for r in records
                if r["task_id"] == run["task_id"]
                and r["model"] == run["model"]
                and r["hint"]
                and r["hint"].get("failure_mode") == mode
                and run["run"] in r["hint"].get("source_runs", [])
                and r["hint"].get("surpassed_failure_point") is True
                and not r["excluded"]
            ]
            if not validations:
                issues.append(
                    f"{run['run']}: no documented hint rerun validates mode {mode}"
                )
    for run in records:
        hint = run["hint"]
        if not hint:
            continue
        version = hint.get("version")
        if type(version) is not int or version < 1:
            issues.append(f"{run['run']}: invalid hint version")
        for key in hint.get("source_runs", []):
            source = by_key.get(key)
            if (
                not source
                or source["task_id"] != run["task_id"]
                or source["model"] != run["model"]
                or source["hint"]
            ):
                issues.append(
                    f"{run['run']}: hint refers to a different task/model or missing source"
                )
        if not hint.get("source_runs"):
            issues.append(f"{run['run']}: hint needs original run references")
        prefix = f"docs/{run['task_id']}/failure_analysis/{Path(run['path']).parts[2]}/{_relative(hint.get('failure_mode', 'missing'))}/"
        _required(entries, prefix + f"hint_v{version}.md", issues)


def capture_references(entries, prefix, issues):
    """Keep portable, hash-verified provider captures without changing originals."""
    from ..integrations.harbor.trajectory import validate_subagent_capture
    from .audit import audit_reference_index
    from .tencent_v3 import _json

    files = {
        p: v
        for p, v in entries.items()
        if p.startswith((prefix + "/agent/", prefix + "/verifier/"))
    }
    # Audit captures/logs are small relative to image archives; only this trial's
    # files are loaded by the existing customer-neutral integrity checker.
    data = {p: v if isinstance(v, bytes) else v.read_bytes() for p, v in files.items()}
    try:
        index = audit_reference_index(data, prefix)
    except (ValueError, KeyError) as error:
        issues.append(f"{prefix}: {error}")
        return
    if index["references"]:
        entries[prefix + "/audit-references.json"] = _json(index)
    root = data.get(prefix + "/agent/trajectory.json")
    if root:
        parsed = json.loads(root)
        events = []
        for name, content in data.items():
            if name.startswith(prefix + "/agent/") and name.endswith(".jsonl"):
                for line in content.splitlines():
                    if line.strip():
                        event = json.loads(line)
                        if (
                            isinstance(event, dict)
                            and event.get("schema_version") == "trajectory.v1"
                        ):
                            events.append(event)
        try:
            validate_subagent_capture(parsed, events)
        except ValueError as error:
            issues.append(f"{prefix}: {error}")
