"""One submission lifecycle for all transports and harnesses; never assigns scores."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version
from threading import RLock

from .contracts import (
    SubmissionArtifact,
    SubmissionPolicy,
    ToolCall,
    ToolError,
    ToolResult,
)
from .validation import ValidationIssue, schema_issues

RECORD_VERSION = "submission.record.v2"
EVIDENCE_VERSION = "submission.evidence.v1"
VALIDATOR_VERSION = version("jsonschema")


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def received(result):
    return bool(result.metadata.get("submission", {}).get("accepted"))


class SubmissionController:
    def __init__(self, policy=None):
        self.policy = policy or SubmissionPolicy()
        self.explicit_policy = policy is not None
        self.attempts = 0
        self.history = []
        self.final = None
        self.lock = RLock()
        self.call_ids = set()

    def configure(self, policy, *, explicit=True):
        with self.lock:
            if self.attempts:
                if self.policy != policy:
                    raise ValueError("cannot change an active submission policy")
                return
            self.policy, self.explicit_policy = policy, explicit

    def execute(self, call, tool, invoke, precheck=None):
        with self.lock:
            if call.call_id in self.call_ids:
                return ToolResult(
                    call.call_id,
                    call.name,
                    False,
                    error=ToolError(
                        "duplicate_submission_call",
                        "Use a new call ID for a new attempt.",
                    ),
                )
            self.call_ids.add(call.call_id)
            self.attempts += 1  # before object/schema checks, hooks or handler dispatch
            if self.final is not None:
                return ToolResult(
                    call.call_id,
                    call.name,
                    False,
                    error=ToolError(
                        "submission_closed", "The final submission is already closed."
                    ),
                    metadata={
                        "submission": {
                            "attempt": self.attempts,
                            "accepted": False,
                            "terminal": True,
                        }
                    },
                )
            raw = copy.deepcopy(call.arguments)
            validation_error = None
            try:
                issues, truncated = schema_issues(tool.definition.input_schema, raw)
            except Exception as error:  # noqa: BLE001 - schema resolver failures are infrastructure
                issues, truncated, validation_error = [], False, type(error).__name__
            if not isinstance(raw, dict) and not issues:
                issues = [
                    ValidationIssue(
                        "schema.type",
                        (),
                        (),
                        "Tool arguments must be an object",
                        "schema",
                        "type",
                    ).as_dict()
                ]
            schema_valid = None if validation_error else not issues
            if precheck:
                issues.append(
                    ValidationIssue(
                        precheck.code, message=precheck.message, source="transport"
                    ).as_dict()
                )
            review = None
            execution_error = None
            result = None
            try:
                canonical(raw)  # no lossy stringification of invalid/non-JSON values
                if validation_error:
                    raise RuntimeError("schema validation could not run")
                if tool.submission_review is not None:
                    review = tool.submission_review(
                        ToolCall(call.call_id, call.name, copy.deepcopy(raw))
                    )
                    if not review.validator_id or not review.validator_version:
                        raise ValueError(
                            "review requires validator identity and version"
                        )
                    issues.extend(i.as_dict() for i in review.issues)
                    canonical(
                        [issues, review.reference_receipts, review.attachment_receipts]
                    )
                if not issues:
                    result = invoke(
                        ToolCall(call.call_id, call.name, copy.deepcopy(raw))
                    )
                    canonical(result.as_dict())
                    if not result.ok:
                        if result.error and result.error.code in {
                            "invalid_submission",
                            "invalid_arguments",
                        }:
                            issues.append(
                                ValidationIssue(
                                    result.error.code,
                                    message=result.error.message,
                                    source="handler",
                                ).as_dict()
                            )
                        else:
                            execution_error = (
                                result.metadata.get("exception_type", result.error.code)
                                if result.error
                                else "invalid_tool_result"
                            )
                elif precheck:
                    result = ToolResult(call.call_id, call.name, False, error=precheck)
                else:
                    result = ToolResult(
                        call.call_id,
                        call.name,
                        False,
                        error=ToolError(
                            "invalid_arguments"
                            if not schema_valid
                            else "invalid_submission",
                            "Submission did not pass validation.",
                            details={
                                "issues": issues[:8],
                                "issues_truncated": truncated or len(issues) > 8,
                            },
                        ),
                    )
            except Exception as error:  # noqa: BLE001 - validation services are not scoring
                execution_error = type(error).__name__
                result = ToolResult(
                    call.call_id,
                    call.name,
                    False,
                    error=ToolError(
                        "submission_validation_error",
                        "Submission checking failed; no validation success or score was inferred.",
                    ),
                )
            status = "error" if execution_error else "invalid" if issues else "valid"
            valid = status == "valid"
            exhausted = self.attempts >= self.policy.max_attempts
            accepted = valid or (
                exhausted
                and self.policy.on_exhausted == "seal_last"
                and status == "invalid"
            )
            terminal = accepted or exhausted or execution_error is not None
            state = {
                "attempt": self.attempts,
                "attempts_remaining": max(0, self.policy.max_attempts - self.attempts),
                "received": True,
                "accepted": accepted,
                "format_valid": valid if status != "error" else None,
                "validation_status": status,
                "grading_status": "pending" if accepted else "not_started",
                "sealed_with_issues": accepted and not valid,
                "terminal": terminal,
                "failure_classification": "submission_validation_error"
                if execution_error
                else None,
                "error_type": execution_error,
            }
            report = {
                "schema_version": "submission.validation.v1",
                "schema_validator": {
                    "id": "jsonschema.Draft202012Validator",
                    "version": VALIDATOR_VERSION,
                },
                "schema_sha256": digest(tool.definition.input_schema),
                "schema_valid": schema_valid,
                "format_valid": state["format_valid"],
                "status": status,
                "issues": issues,
                "issues_truncated": truncated,
                "validator": {
                    "id": review.validator_id,
                    "version": review.validator_version,
                }
                if review
                else None,
                "execution_error": execution_error,
            }
            snapshot = {
                "schema_version": EVIDENCE_VERSION,
                "call_id": call.call_id,
                "tool_name": call.name,
                "attempt": self.attempts,
                "received_at": datetime.now(UTC).isoformat(),
                "policy": self.policy.as_dict(),
                "state": state,
                "raw_payload": raw,
                "payload": result.output if valid else raw,
                "validation_report": report,
                "reference_receipts": list(review.reference_receipts) if review else [],
                "attachment_receipts": list(review.attachment_receipts)
                if review
                else [],
                "review_status": "completed"
                if review and not execution_error
                else "error"
                if execution_error
                else "not_configured",
                "handler_result": result.as_dict(),
            }
            # Detached copies stop a handler or transport from changing sealed evidence.
            self.history.append(copy.deepcopy(snapshot))
            if terminal:
                self.final = copy.deepcopy(snapshot)
            returned_error = result.error
            if execution_error:
                returned_error = ToolError(
                    "submission_validation_error",
                    "Submission checking failed; no validation success or score was inferred.",
                )
            elif accepted and not valid:
                returned_error = ToolError(
                    result.error.code if result.error else "invalid_submission",
                    "Final answer sealed with validation issues; it has not passed validation. Do not resubmit.",
                    details={
                        "issues": issues[:8],
                        "issues_truncated": truncated or len(issues) > 8,
                    },
                )
            return ToolResult(
                call.call_id,
                call.name,
                valid,
                result.output
                if valid
                else raw
                if accepted
                else None
                if execution_error
                else result.output,
                returned_error,
                {**result.metadata, "submission": state},
            )

    def record(self, task, *, episode_id, model_id, harness_id):
        with self.lock:
            if self.final is None:
                raise RuntimeError("agent finished without a final submission")
            evidence = copy.deepcopy(self.final)
        binding = {
            "task_id": task.task_id,
            "task_version": task.task_version,
            "task_sha256": digest(task.as_dict()),
            "submission_schema_sha256": digest(task.submit_tool.input_schema),
            "episode_id": episode_id,
            "model_id": model_id,
            "harness_id": harness_id,
        }
        evidence["binding"] = binding
        evidence["component_sha256"] = {
            key: digest(evidence[key])
            for key in (
                "raw_payload",
                "payload",
                "validation_report",
                "reference_receipts",
                "attachment_receipts",
            )
        }
        return {
            "schema_version": RECORD_VERSION,
            **binding,
            "received_at": evidence["received_at"],
            "payload": evidence["payload"],
            "submission_state": evidence["state"],
            "submission_evidence": evidence,
            "evidence_sha256": digest(evidence),
        }


def artifact_from_record(record, task, *, require_evidence=False):
    if record.get("task_id") != task.task_id:
        raise ValueError("submission belongs to another task")
    if record.get("schema_version") != RECORD_VERSION:
        if require_evidence:
            raise ValueError("submission evidence is required for this task")
        return SubmissionArtifact(
            task.task_id,
            record["episode_id"],
            record["payload"],
            record.get("received_at", ""),
            "submission.v1",
            digest(record["payload"]),
            validation_report={"status": "legacy_unbound"},
        )
    evidence = record["submission_evidence"]
    if (
        evidence.get("schema_version") != EVIDENCE_VERSION
        or digest(evidence) != record["evidence_sha256"]
    ):
        raise ValueError("submission evidence integrity mismatch")
    expected = {
        "task_id": task.task_id,
        "task_version": task.task_version,
        "task_sha256": digest(task.as_dict()),
        "submission_schema_sha256": digest(task.submit_tool.input_schema),
        "episode_id": record["episode_id"],
        "model_id": record["model_id"],
        "harness_id": record["harness_id"],
    }
    if evidence["binding"] != expected or any(
        record[k] != v for k, v in expected.items()
    ):
        raise ValueError("submission version binding mismatch")
    if (
        evidence["policy"] != task.submission_policy.as_dict()
        or record["payload"] != evidence["payload"]
        or record["submission_state"] != evidence["state"]
    ):
        raise ValueError("submission policy/payload binding mismatch")
    keys = {
        "raw_payload",
        "payload",
        "validation_report",
        "reference_receipts",
        "attachment_receipts",
    }
    if set(evidence["component_sha256"]) != keys:
        raise ValueError("submission component manifest is incomplete")
    state, report = evidence["state"], evidence["validation_report"]
    if not state["accepted"] or state["validation_status"] == "error":
        raise RuntimeError(
            "submission was rejected or checking failed; no score inferred"
        )
    if (
        state["accepted"] is not True
        or state["received"] is not True
        or state["terminal"] is not True
        or type(evidence["attempt"]) is not int
        or not 1 <= evidence["attempt"] <= task.submission_policy.max_attempts
        or state["attempt"] != evidence["attempt"]
        or evidence["tool_name"] != task.submit_tool.name
        or record["received_at"] != evidence["received_at"]
        or report["schema_version"] != "submission.validation.v1"
    ):
        raise ValueError("invalid submission lifecycle evidence")
    if type(state["format_valid"]) is not bool or state["sealed_with_issues"] != (
        not state["format_valid"]
    ):
        raise ValueError("forced acceptance must retain invalid format state")
    issues, _ = schema_issues(task.submit_tool.input_schema, evidence["raw_payload"])
    schema_valid = isinstance(evidence["raw_payload"], dict) and not issues
    if (
        report["schema_sha256"] != digest(task.submit_tool.input_schema)
        or report["schema_valid"] != schema_valid
        or state["format_valid"] != report["format_valid"]
        or state["validation_status"] != report["status"]
    ):
        raise ValueError("submission validation state mismatch")
    if state["format_valid"] is True and (
        not schema_valid or report["issues"] or report["status"] != "valid"
    ):
        raise ValueError("invalid submission cannot be labeled valid")
    if state["sealed_with_issues"] and (
        task.submission_policy.on_exhausted != "seal_last"
        or evidence["attempt"] != task.submission_policy.max_attempts
        or state["format_valid"] is not False
        or evidence["payload"] != evidence["raw_payload"]
    ):
        raise ValueError("invalid forced-submission state")
    for key, checksum in evidence["component_sha256"].items():
        if digest(evidence[key]) != checksum:
            raise ValueError("submission component integrity mismatch")
    return SubmissionArtifact(
        task.task_id,
        record["episode_id"],
        evidence["payload"],
        evidence["received_at"],
        "submission.v2",
        digest(evidence["payload"]),
        raw_payload=evidence["raw_payload"],
        validation_report=evidence["validation_report"],
        reference_receipts=tuple(evidence["reference_receipts"]),
        attachment_receipts=tuple(evidence["attachment_receipts"]),
        evidence=evidence,
    )
