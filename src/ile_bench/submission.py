"""ILE task checks using main's lifecycle; no force-success or judge-specific rules."""

from __future__ import annotations

import json
import shlex

from prophet_bench_env import (
    RegisteredTool,
    SubmissionReview,
    ToolResult,
    ValidationIssue,
)
from prophet_bench_env.adapters.data_lake import DataLakeError
from prophet_bench_env.core.validation import schema_issues

# Executes only in the already-created, offline Harbor workspace. No host paths/mounts.
# -I prevents user-created Python modules from replacing the stdlib checker.
ATTACHMENT_CHECK = r"""
import os, sys, stat, errno, hashlib, json
p=json.loads(sys.argv[1]); fds=[]
out={'path':p,'status':'invalid'}
try:
 if not isinstance(p,str) or not p or p.startswith('/') or '\\' in p or '\x00' in p or any(x in ('','.','..') for x in p.split('/')):
  out['code']='attachment.unsafe_path'
 else:
  fds.append(os.open('/app/workspace',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW))
  for part in p.split('/')[:-1]: fds.append(os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fds[-1]))
  fd=os.open(p.split('/')[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fds[-1])
  if not stat.S_ISREG(os.fstat(fd).st_mode):
   os.close(fd); out['code']='attachment.unsafe_path'
  else:
   with os.fdopen(fd,'rb') as f:
    before=os.fstat(f.fileno())
    h=hashlib.sha256(); size=0
    while b:=f.read(1048576): h.update(b); size+=len(b)
    after=os.fstat(f.fileno())
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise RuntimeError('file changed while checking')
    out.update(status='valid',size_bytes=size,sha256=h.hexdigest())
except OSError as e:
 if e.errno in (errno.ENOENT,errno.ENOTDIR,errno.ELOOP): out['code']='attachment.missing' if e.errno==errno.ENOENT else 'attachment.unsafe_path'
 else: raise
finally:
 for fd in reversed(fds): os.close(fd)
print(json.dumps(out))
"""


def check_attachment(backend, path, field):
    result = backend.execute(
        "python -I -c " + shlex.quote(ATTACHMENT_CHECK) + " " + shlex.quote(json.dumps(path)),
        timeout=60,
    )
    if result.exit_code != 0 or result.timed_out:
        raise RuntimeError("Workspace attachment inspection failed")
    receipt = json.loads(result.stdout)
    if receipt.get("path") != path or receipt.get("status") not in ("valid", "invalid"):
        raise RuntimeError("Invalid workspace attachment receipt")
    receipt["checker"] = "ile-workspace-file-v1"
    issues = (
        ()
        if receipt["status"] == "valid"
        else (
            ValidationIssue(receipt["code"], field, message=receipt["code"], source="attachment"),
        )
    )
    return receipt, issues


def bind_submission(task, contract, data, backend, table_receipts=None):
    structured = task.target_definition["submission_mode"] == "structured"
    table_receipts = table_receipts if table_receipts is not None else {}

    def review(call):
        raw = call.arguments
        issues, references, attachments = [], [], []
        if isinstance(raw, dict) and structured:
            if "structured_output" in raw:
                values, truncated = schema_issues(
                    contract["structured_output_schema"], raw["structured_output"]
                )
                if truncated:
                    # Known invalid input stays a model submission error; a long list
                    # must not turn it into an excluded infrastructure failure.
                    issues.append(
                        ValidationIssue(
                            "task_schema.issues_truncated",
                            ("structured_output",),
                            message="Further validation issues were omitted; correct the reported fields and resubmit.",
                            source="task_schema",
                        )
                    )
                issues.extend(
                    ValidationIssue(
                        v["code"],
                        ("structured_output", *v["path"]),
                        tuple(v["schema_path"]),
                        v["message"],
                        "task_schema",
                        v["validator"],
                    )
                    for v in values
                )
            docs = set()
            for n, item in enumerate(
                raw.get("evidence") if isinstance(raw.get("evidence"), list) else []
            ):
                if not isinstance(item, dict) or not isinstance(item.get("document_ref"), str):
                    continue
                ref = item["document_ref"]
                field = ("evidence", n, "document_ref")
                if ref.startswith("table-"):
                    if ref in table_receipts:
                        references.append(dict(table_receipts[ref]))
                    else:
                        issues.append(
                            ValidationIssue(
                                "reference.unresolved",
                                field,
                                message="Use a table_ref successfully read in this run.",
                            )
                        )
                        references.append({"ref": ref, "kind": "table", "resolved": False})
                    continue
                try:
                    identity = data.resolve_document_ref(ref)
                except DataLakeError as error:
                    if error.code != "unknown_document_ref":
                        raise
                    issues.append(
                        ValidationIssue(
                            "reference.unresolved",
                            field,
                            message="Use a document_ref returned in this run.",
                        )
                    )
                    references.append({"ref": ref, "kind": "document", "resolved": False})
                else:
                    docs.add(identity)
                    references.append(
                        {
                            "ref": ref,
                            "kind": "document",
                            "resolved": True,
                            "document_id": identity,
                        }
                    )
            minimum = contract["minimum_evidence_documents"]
            if len(docs) < minimum:
                issues.append(
                    ValidationIssue(
                        "reference.insufficient_documents",
                        ("evidence",),
                        message=f"At least {minimum} distinct cutoff-database documents are required.",
                    )
                )
            for n, artifact in enumerate(
                raw.get("artifacts") if isinstance(raw.get("artifacts"), list) else []
            ):
                if not isinstance(artifact, dict) or "workspace_path" not in artifact:
                    continue
                receipt, errors = check_attachment(
                    backend,
                    artifact["workspace_path"],
                    ("artifacts", n, "workspace_path"),
                )
                attachments.append(receipt)
                issues.extend(errors)
        elif (
            isinstance(raw, dict)
            and not structured
            and isinstance(raw.get("answer"), str)
            and not raw["answer"].strip()
        ):
            issues.append(
                ValidationIssue(
                    "answer.empty",
                    ("answer",),
                    message="Submit a nonempty research report.",
                )
            )
        return SubmissionReview(
            "ile-v1-submission",
            "1",
            tuple(issues),
            tuple(references),
            tuple(attachments),
        )

    def accept(call):
        # The upstream controller only calls this after schema + review pass.
        return ToolResult(call.call_id, call.name, True, output=call.arguments)

    return RegisteredTool(task.submit_tool, accept, review)
