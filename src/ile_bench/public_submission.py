"""Public-envelope-only admission; financial completeness belongs to rubrics.

The v1 implementation remains frozen for historical plans. New runs use this
explicit adapter, never the private task schema or minimum-document counter.
"""

from copy import deepcopy

from .submission import bind_submission as bind_legacy_checks

VERSION = "public-envelope-20261011-v2"


def public_contract(task):
    if task.target_definition["submission_mode"] != "structured":
        return {}
    return {
        "structured_output_schema": deepcopy(
            task.submit_tool.input_schema["properties"]["structured_output"]
        ),
        "minimum_evidence_documents": 0,
    }


def bind_submission(task, contract, data, backend, table_receipts=None):
    # Ignore even a mistakenly supplied private contract: admission is a
    # projection of the exact tool schema visible in the initial request.
    return bind_legacy_checks(task, public_contract(task), data, backend, table_receipts)


def grading_only_issue(issue):
    return issue.get("source") == "task_schema" or issue.get("code") == (
        "reference.insufficient_documents"
    )
