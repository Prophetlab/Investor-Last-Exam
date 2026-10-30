"""Explicit trial accounting; absent/faulted evaluations never become zero rewards."""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from harbor.models.job.result import JobStats

from .serialization import write_json


def classify_trial(row, evaluation=None):
    state = ((row.get("agent_result") or {}).get("metadata") or {}).get(
        "submission_state"
    ) or {}
    exception = row.get("exception_info")
    rewards = (row.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    finite = bool(rewards) and all(
        type(v) in {int, float} and math.isfinite(v) for v in rewards.values()
    )
    outcome = row.get("_bench_outcome") or {}
    evaluation = evaluation or row.get("_bench_evaluation")
    validation_error = state.get("validation_status") == "error"
    exception_type = (exception or {}).get("exception_type")
    # Harbor reports missing reward files for intentionally unscored outcomes.
    # A genuine container/judge exception must still win over a pending receipt.
    missing_reward = exception_type == "RewardFileNotFoundError"
    delivery_invalid = (
        state.get("received")
        and state.get("accepted") is False
        and not validation_error
    )
    if validation_error:
        status = "infrastructure_failure"
        failure = state.get("failure_classification") or "submission_validation_error"
    elif delivery_invalid and (
        not exception
        or exception_type == "SubmissionRetriesExceededError"
        or (
            missing_reward
            and outcome.get("failure_classification") == "submission_rejected"
        )
    ):
        status, failure = "submission_rejected", None
    elif (
        outcome.get("grading_status") == "pending"
        and (not exception or missing_reward)
        and not outcome.get("failure_classification")
        and not (evaluation or {}).get("failure_classification")
    ):
        status, failure = "evaluation_pending", None
    elif (
        exception
        or (evaluation or {}).get("failure_classification")
        or outcome.get("failure_classification")
    ):
        status, failure = (
            "infrastructure_failure",
            (evaluation or {}).get("failure_classification")
            or outcome.get("failure_classification")
            or (exception or {}).get("exception_type"),
        )
    elif evaluation is not None and (
        evaluation.get("reward_pending") or evaluation.get("reward") is None
    ):
        status, failure = "evaluation_pending", None
    elif finite:
        status, failure = "scored", None
    else:
        status, failure = "unscored", None
    return {
        "trial_name": row.get("trial_name"),
        "status": status,
        "reward": reward if status == "scored" else None,
        "rewards": rewards if status == "scored" else None,
        "failure_classification": failure,
        "error_type": state.get("error_type")
        or outcome.get("error_type")
        or exception_type,
        "submission_accepted": state.get("accepted"),
        "submission_received": state.get("received"),
        "format_valid": state.get("format_valid"),
        "grading_completed": status == "scored",
    }


def summarize_trials(rows):
    summaries = [classify_trial(row) for row in rows]
    scored = [row for row in summaries if row["grading_completed"]]
    by_key = defaultdict(list)
    for row in scored:
        for key, value in row["rewards"].items():
            by_key[key].append(value)
    means = {k: sum(values) / len(values) for k, values in by_key.items()}
    return {
        "schema_version": "trial-summary.v1",
        "n_trials": len(summaries),
        "n_scored": len(scored),
        "n_unscored": len(summaries) - len(scored),
        "status_counts": dict(Counter(row["status"] for row in summaries)),
        "failure_counts": dict(
            Counter(
                row["failure_classification"]
                for row in summaries
                if row["failure_classification"]
            )
        ),
        "error_type_counts": dict(
            Counter(row["error_type"] for row in summaries if row["error_type"])
        ),
        "mean_scored_only": means.get("reward"),
        "means_scored_only": means,
        "trials": summaries,
    }


def finalize_job_result(result, job_dir: Path):
    """Keep native trials intact; replace final aggregates, not the Harbor scheduler."""
    rows = [r.model_dump(mode="json") for r in result.trial_results]
    for row in rows:
        _read_outcomes(row, job_dir)
    groups = defaultdict(list)
    for row in rows:
        agent = row["agent_info"]
        key = JobStats.format_agent_evals_key(
            agent["name"],
            (agent.get("model_info") or {}).get("name"),
            row.get("source") or "adhoc",
        )
        groups[key].append(row)
    summary = summarize_trials(rows)
    summary["groups"] = {k: summarize_trials(v) for k, v in groups.items()}
    for key, group in summary["groups"].items():
        stats = result.stats.evals[key]
        original = stats.metrics if not group["n_unscored"] else []
        stats.metrics = [
            {
                k: group[k]
                for k in (
                    "mean_scored_only",
                    "means_scored_only",
                    "n_scored",
                    "n_unscored",
                )
            },
            *original,
        ]
        stats.reward_stats = {}
        for row in group["trials"]:
            for key, value in (row["rewards"] or {}).items():
                stats.reward_stats.setdefault(key, {}).setdefault(value, []).append(
                    row["trial_name"]
                )
        # Native pass@k treats missing answers as failures. Do not present that
        # figure as an evaluation metric for a group containing system faults.
        if group["n_unscored"]:
            stats.pass_at_k = {}
    summary["n_expected"] = result.n_total_trials
    summary["n_unrecorded"] = max(0, result.n_total_trials - len(rows))
    write_json(job_dir / "bench-summary.json", summary)
    write_json(
        job_dir / "result.json",
        result.model_dump(mode="json", exclude={"trial_results"}),
    )
    return result


def _read_outcomes(row, job_dir):
    name = row.get("trial_name", "")
    if not name or Path(name).name != name:
        return
    for filename, key in (
        ("outcome.json", "_bench_outcome"),
        ("evaluation.json", "_bench_evaluation"),
    ):
        path = job_dir / name / "verifier" / filename
        if path.is_file():
            try:
                row[key] = json.loads(path.read_text())
                if not isinstance(row[key], dict):
                    raise TypeError("evaluation receipt must be an object")
            except (OSError, ValueError, TypeError):
                row.pop(key, None)
                row["_bench_outcome"] = {
                    "failure_classification": "invalid_evaluation_receipt"
                }


def write_incomplete_summary(job_dir):
    rows = []
    for path in job_dir.glob("*/result.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(row, dict) and "trial_name" in row:
            _read_outcomes(row, job_dir)
            rows.append(row)
    summary = summarize_trials(rows)
    summary["job_status"] = "interrupted"
    result_path = job_dir / "result.json"
    if result_path.is_file():
        native = json.loads(result_path.read_text())
        summary["n_expected"] = native.get("n_total_trials")
        # Incomplete native aggregates may contain implicit zero rewards. They
        # remain unavailable until resumed; the explicit receipt counts are usable.
        for group in (native.get("stats") or {}).get("evals", {}).values():
            group["metrics"] = []
            group["pass_at_k"] = {}
            group["reward_stats"] = {}
        write_json(result_path, native)
    summary["n_unrecorded"] = (
        max(0, summary["n_expected"] - len(rows))
        if summary.get("n_expected") is not None
        else None
    )
    write_json(job_dir / "bench-summary.json", summary)
