"""Portable references for captured model calls without rewriting signed events."""

from __future__ import annotations

import hashlib
import json


def audit_reference_index(entries: dict[str, bytes], prefix: str) -> dict:
    """Resolve audited bytes inside one trial only, never read an original path.

    Provider-attempt hashes bind host/reader and container/verifier captures to
    the collected files. The original JSONL and its hash chain stay untouched.
    Consumers resolve paths relative to the extracted delivery root.
    """
    files = {
        path: data
        for path, data in entries.items()
        if path.startswith((prefix + "/agent/", prefix + "/verifier/"))
    }
    by_hash: dict[str, list[str]] = {}
    for path, data in files.items():
        by_hash.setdefault(hashlib.sha256(data).hexdigest(), []).append(path)
    events = []
    for path, data in files.items():
        if not path.endswith(".jsonl"):
            continue
        for line_number, line in enumerate(data.splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if (
                isinstance(event, dict)
                and event.get("schema_version") == "trajectory.v1"
            ):
                events.append((path, line_number, event))

    bindings: dict[str, dict[str, str]] = {}
    for _source, _line, event in events:
        if event.get("event_type") != "model_provider_attempt":
            continue
        payload = event["payload"]
        for kind in ("request", "response"):
            reference = payload.get(kind + "_ref")
            if reference is None and kind == "response":
                continue  # Connection failure may have no response bytes.
            digest = payload.get(kind + "_sha256")
            candidates = by_hash.get(digest, [])
            if not isinstance(reference, str) or not reference or not candidates:
                raise ValueError("model audit capture is missing or its hash differs")
            # Prefer its original filename; identical byte copies are equivalent.
            filename = reference.replace("\\", "/").rsplit("/", 1)[-1]
            path = min(candidates, key=lambda p: (p.rsplit("/", 1)[-1] != filename, p))
            binding = {"path": path, "sha256": digest}
            if reference in bindings and bindings[reference]["sha256"] != digest:
                raise ValueError("model audit reference has conflicting captures")
            bindings[reference] = binding

    references = []
    for source, line_number, event in events:
        payload = event.get("payload", {})
        for field in ("request_ref", "response_ref", "raw_capture_ref"):
            reference = payload.get(field)
            if reference is None:
                continue
            if not isinstance(reference, str) or reference not in bindings:
                raise ValueError("model audit reference has no verified capture")
            references.append(
                {
                    "source": source,
                    "line": line_number,
                    "event_id": event.get("event_id"),
                    "field": field,
                    "original_ref": reference,
                    **bindings[reference],
                }
            )
    return {
        "schema_version": "audit-references.v1",
        "path_base": "delivery-root",
        "references": references,
    }
