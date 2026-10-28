"""Harbor verifier entrypoint; runs only in the separate scoring environment."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from ...core.contracts import EvaluationContext
from ...core.submission import artifact_from_record
from .serialization import load_entrypoint, read_public, read_rubric, write_json


def verify(spec_path: Path, submission_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    # Rechecks must not mix current outcomes with earlier rewards or receipts.
    previous = "previous-" + uuid4().hex + "-"
    for name in (
        "reward.json",
        "reward.txt",
        "evaluation.json",
        "outcome.json",
        "failure.json",
        "submission-evidence.json",
    ):
        path = output / name
        if path.exists():
            path.replace(output / (previous + name))
    state = {
        "schema_version": "evaluation.status.v1",
        "submission_received": submission_path.is_file(),
        "format_valid": None,
        "grading_status": "not_started",
        "failure_classification": None,
    }
    try:
        rewards = _verify(spec_path, submission_path, output, state)
    except Exception as error:
        pending = state["grading_status"] == "pending"
        state.update(
            grading_status="pending" if pending else "error",
            failure_classification=None
            if pending
            else state["failure_classification"]
            or (
                "evaluator_error"
                if state["grading_status"] == "running"
                else "verification_error"
            ),
            error_type=type(error).__name__,
        )
        write_json(output / "outcome.json", state)
        write_json(output / "failure.json", state)
        raise
    state.update(grading_status="completed", failure_classification=None)
    write_json(output / "outcome.json", state)
    return rewards


def _verify(spec_path, submission_path, output, state):
    spec = json.loads(spec_path.read_text())
    task = read_public(spec["public_task"])
    rubric = read_rubric(spec["rubric"])
    if rubric.sha256 != spec["rubric_sha256"]:
        raise ValueError("rubric integrity mismatch")
    evaluator = load_entrypoint(spec["evaluator_factory"])(
        **spec.get("evaluator_kwargs", {})
    )
    report = evaluator.validate_inputs(task, rubric)
    if not report.ok:
        raise ValueError("evaluator rejected rubric: " + "; ".join(report.issues))
    output.mkdir(parents=True, exist_ok=True)
    if not submission_path.is_file():
        raise RuntimeError("missing submission; no score was fabricated")
    if submission_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("submission exceeds 16 MiB")
    record = json.loads(submission_path.read_text())
    state.update(
        format_valid=record.get("submission_state", {}).get("format_valid"),
        submission_accepted=record.get("submission_state", {}).get("accepted"),
    )
    if record.get("submission_state", {}).get("accepted") is False:
        state["failure_classification"] = (
            record["submission_state"].get("failure_classification")
            or "submission_rejected"
        )
    submission = artifact_from_record(
        record, task, require_evidence=spec.get("submission_evidence_required", False)
    )
    write_json(output / "submission-evidence.json", record)
    state.update(
        grading_status="running", evidence_sha256=record.get("evidence_sha256")
    )
    result = evaluator.evaluate(
        task,
        rubric,
        submission,
        EvaluationContext(
            run_id=record["episode_id"],
            episode_id=record["episode_id"],
            harness_id=record.get("harness_id", "harbor"),
            model_id=record.get("model_id", "unknown"),
            private_context_ref=spec["private_context_ref"],
            artifact_root=str(output / "audit"),
        ),
    )
    if result.reward is not None and (
        type(result.reward) not in {int, float} or not math.isfinite(result.reward)
    ):
        raise ValueError("evaluator reward must be a finite number or None")
    write_json(
        output / "evaluation.json",
        {
            **asdict(result),
            "rubric_sha256": rubric.sha256,
            "submission_evidence_sha256": record.get("evidence_sha256"),
            "submission_state": record.get("submission_state"),
        },
    )
    if result.reward_pending or result.reward is None or result.failure_classification:
        state.update(
            grading_status="error" if result.failure_classification else "pending",
            failure_classification=result.failure_classification,
        )
        raise RuntimeError("evaluation pending or failed; see evaluation.json")
    rewards = {"reward": result.reward}
    write_json(output / "reward.json", rewards)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--submission", type=Path, default=Path("/logs/artifacts/submission.json")
    )
    parser.add_argument("--output", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    try:
        verify(args.spec, args.submission, args.output)
    except Exception:  # noqa: BLE001 - sanitized CLI failure boundary
        # Exception text from third-party providers may contain secrets.
        raise SystemExit("Verifier failed; no numeric reward produced.") from None


if __name__ == "__main__":
    main()
