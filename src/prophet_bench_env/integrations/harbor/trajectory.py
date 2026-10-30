"""Compatible main ATIF plus explicit subagents and continuation segments."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from harbor.models.trajectories.trajectory import Trajectory

from .serialization import write_json
from .trajectory_content import ContentWriter, context_steps, response_step


def _identifier(episode_id, value):
    return hashlib.sha256((episode_id + ":" + value).encode()).hexdigest()[:32]


def _document(
    steps,
    *,
    episode_id,
    model_name,
    trajectory_id,
    name="prophet-financial-agent",
    tools=(),
):
    if not steps:
        steps = [
            {
                "source": "system",
                "message": "",
                "extra": {"no_interaction_recorded": True},
            }
        ]
    return {
        "schema_version": "ATIF-v1.8",
        "session_id": episode_id,
        "trajectory_id": trajectory_id,
        "agent": {
            "name": name,
            "version": "0.2.0a0",
            "model_name": model_name,
            "tool_definitions": list(tools),
        },
        "steps": [{**step, "step_id": i} for i, step in enumerate(steps, 1)],
    }


def _subagent_identity(event):
    kind, payload = event["event_type"], event["payload"]
    for prefix, role, id_field in (
        ("reader_model_", "reader", "reader_invocation_id"),
        ("compaction_model_", "compactor", "compaction_id"),
        (
            "subagent_model_",
            payload.get("subagent_role", "subagent"),
            "subagent_invocation_id",
        ),
    ):
        if kind in {prefix + suffix for suffix in ("request", "response", "error")}:
            return role, payload.get(id_field) or payload.get("subagent_invocation_id")
    return None


def _subagents(events, *, episode_id, content):
    groups = defaultdict(list)
    for event in events:
        kind, p = event["event_type"], event["payload"]
        identity = _subagent_identity(event)
        if identity:
            role, invocation = identity
            groups[(role, invocation or "unpaired-" + event["event_id"])].append(event)
        elif kind == "context_compaction" and not p.get("compaction_id"):
            # The legacy summary is recorded, but cannot be paired with a
            # particular compactor request without an explicit invocation ID.
            groups[("compactor", "summary-" + event["event_id"])].append(
                {
                    **event,
                    "event_type": "compaction_model_response",
                    "payload": {**p, "content": p.get("summary", "")},
                }
            )
    children = []
    for (role, invocation), rows in groups.items():
        steps, parents, model_names = [], set(), set()
        for event in rows:
            p = event["payload"]
            parent = p.get("parent_tool_call_id") or event.get("parent_id")
            if parent:
                parents.add(parent)
            if p.get("model_name"):
                model_names.add(p["model_name"])
            if event["event_type"].endswith("_request"):
                steps.extend(
                    context_steps(
                        p.get("messages", []), content, event_id=event["event_id"]
                    )
                )
            elif event["event_type"].endswith("_response"):
                steps.append(response_step(event, content))
        if len(parents) > 1 or len(model_names) > 1:
            raise ValueError("subagent invocation has conflicting recorded identities")
        child = _document(
            steps,
            episode_id=episode_id,
            model_name=next(iter(model_names), None),
            trajectory_id=_identifier(episode_id, role + ":" + invocation),
            name={
                "reader": "financial-document-reader",
                "compactor": "context-compactor",
            }.get(role, role),
        )
        child["extra"] = {
            "invocation_id": invocation,
            "role": role,
            "parent_tool_call_id": next(iter(parents), None),
            "audit_event_ids": [r["event_id"] for r in rows],
            "request_recorded": any(r["event_type"].endswith("_request") for r in rows),
            "response_recorded": any(
                r["event_type"].endswith("_response") for r in rows
            ),
            "errors": [
                r["payload"] for r in rows if r["event_type"].endswith("_error")
            ],
            "link_status": "pending"
            if not invocation.startswith("unpaired-")
            else "missing_invocation_id",
        }
        children.append(child)
    return children


def _references(steps):
    return {
        ref["trajectory_id"]
        for s in steps
        for r in s.get("observation", {}).get("results", [])
        for ref in r.get("subagent_trajectory_ref", [])
        if ref.get("trajectory_id")
    }


def _attach_children(steps, children, warnings):
    calls, compactions = defaultdict(list), {}
    for step in steps:
        for call in step.get("tool_calls", []):
            calls[call["tool_call_id"]].append(step)
        if step.get("extra", {}).get("compaction_id"):
            compactions[step["extra"]["compaction_id"]] = step
    for child in children:
        metadata = child["extra"]
        parent_id = metadata.get(
            "native_parent_tool_call_id", metadata["parent_tool_call_id"]
        )
        candidates = calls.get(parent_id, [])
        step = candidates[0] if len(candidates) == 1 else None
        if metadata["role"] == "compactor":
            step = compactions.get(metadata["invocation_id"])
            parent_id = None
        if step is None:
            metadata["link_status"] = "unlinked"
            warnings.append(
                {
                    "code": "subagent_parent_unresolved",
                    "trajectory_id": child["trajectory_id"],
                    "parent_tool_call_id": parent_id,
                }
            )
            continue
        results = step.setdefault("observation", {"results": []})["results"]
        result = next(
            (r for r in results if r.get("source_call_id") == parent_id), None
        )
        if result is None:
            result = {
                "source_call_id": parent_id,
                "extra": {"tool_result_not_recorded": parent_id is not None},
            }
            results.append(result)
        result.setdefault("subagent_trajectory_ref", []).append(
            {"trajectory_id": child["trajectory_id"]}
        )
        metadata["link_status"] = "linked"


def _snapshot(events, boundary):
    p = events[boundary]["payload"]
    messages = p.get("messages_after_compaction")
    if isinstance(messages, list):
        return messages, p.get("copied_message_indices", []), "compaction_event"
    # An actual subsequent request is exact evidence, not a guessed reconstruction.
    for event in events[boundary + 1 :]:
        if event["event_type"] == "model_request":
            messages = event["payload"].get("messages", [])
            summary = "Earlier research summary:\n" + p.get("summary", "")
            copied = [
                i
                for i, m in enumerate(messages)
                if not (m.get("role") == "system" and m.get("content") == summary)
            ]
            return messages, copied, "next_model_request"
        if event["event_type"] in {"model_response", "context_compaction"}:
            break
    return None, [], "missing"


def _rebase_media(value, old_parent, new_parent):
    if isinstance(value, dict):
        if value.get("type") in {"image", "audio"} and isinstance(
            value.get("source"), dict
        ):
            value["source"]["path"] = os.path.relpath(
                old_parent / value["source"]["path"], new_parent
            )
        else:
            for item in value.values():
                _rebase_media(item, old_parent, new_parent)
    elif isinstance(value, list):
        for item in value:
            _rebase_media(item, old_parent, new_parent)


def export_atif(
    source: Path,
    destination: Path,
    *,
    episode_id: str,
    model_name: str | None,
    tool_definitions: list[dict],
) -> dict:
    """Preserve the legacy filename/full main view, adding loss-aware graph views."""
    source, destination = Path(source), Path(destination)
    events = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    warnings = []
    content = ContentWriter(source, destination, warnings)
    steps, calls, boundaries = [], {}, []
    request_seen = False
    for index, event in enumerate(events):
        kind, p = event["event_type"], event["payload"]
        if kind == "task_presented" and "task" in p:
            steps.append({"source": "user", "message": p["task"]["prompt"]})
        elif kind == "model_request" and not request_seen:
            request_seen = True
            messages = p.get("messages", [])
            if steps:
                messages = [m for m in messages if m.get("role") == "system"]
            steps[0:0] = context_steps(messages, content, event_id=event["event_id"])
        elif kind == "context_compaction":
            step = {
                "source": "system",
                "message": p.get("summary", ""),
                "extra": {
                    "event_type": kind,
                    "audit_event_id": event["event_id"],
                    "compaction_id": p.get("compaction_id")
                    or "summary-" + event["event_id"],
                },
            }
            steps.append(step)
            boundaries.append((step, index))
        elif kind == "model_response" and event.get("actor") in {"model", "assistant"}:
            step = response_step(event, content)
            steps.append(step)
            calls.update({c["tool_call_id"]: step for c in step.get("tool_calls", [])})
        elif kind == "tool_result":
            call_id = p.get("call_id", p.get("id")) or event.get("parent_id")
            step = calls.get(call_id)
            if step is not None:
                step.setdefault("observation", {"results": []})["results"].append(
                    {
                        "source_call_id": call_id,
                        "content": json.dumps(p, ensure_ascii=False),
                    }
                )
        elif kind == "correction_message":
            steps.append(
                {
                    "source": "user",
                    "message": content.convert(p.get("content", ""), event["event_id"]),
                }
            )
    children = _subagents(events, episode_id=episode_id, content=content)
    _attach_children(steps, children, warnings)
    root = _document(
        steps,
        episode_id=episode_id,
        model_name=model_name,
        trajectory_id=_identifier(episode_id, "main"),
        tools=tool_definitions,
    )
    root["extra"] = {
        "native_events": source.name,
        "export_version": "atif-graph.v1",
        "warnings": warnings,
    }
    if children:
        root["subagent_trajectories"] = children
    if boundaries:
        directory = destination.parent / (destination.stem + ".segments")
        segments, start, prefix = [], 0, []
        for boundary, event_index in boundaries:
            stop = next(i for i, step in enumerate(steps) if step is boundary)
            segments.append(prefix + steps[start:stop])
            messages, copied, evidence = _snapshot(events, event_index)
            if messages is None:
                warnings.append(
                    {
                        "code": "compacted_context_missing",
                        "event_id": events[event_index]["event_id"],
                    }
                )
                prefix = [copy.deepcopy(boundary)]
                prefix[0]["extra"]["context_incomplete"] = True
            else:
                prefix = context_steps(
                    messages,
                    content,
                    copied_indices=copied,
                    event_id=events[event_index]["event_id"],
                )
                for step in prefix:
                    step.setdefault("extra", {})["context_evidence"] = evidence
                summary = next(
                    (s for s in prefix if not s.get("is_copied_context")), None
                )
                if summary is not None and boundary.get("observation"):
                    summary["observation"] = copy.deepcopy(boundary["observation"])
            start = stop + 1
        segments.append(prefix + steps[start:])
        for i, block in enumerate(segments):
            segment = _document(
                copy.deepcopy(block),
                episode_id=episode_id,
                model_name=model_name,
                trajectory_id=_identifier(episode_id, f"segment:{i}"),
                tools=tool_definitions,
            )
            refs = _references(block)
            embedded = [
                copy.deepcopy(c)
                for c in children
                if c["trajectory_id"] in refs
                or (i == 0 and c["extra"]["link_status"] == "unlinked")
            ]
            if embedded:
                segment["subagent_trajectories"] = embedded
            segment["extra"] = {
                "export_version": "atif-graph.v1",
                "view": "canonical_segment",
                "segment_index": i,
                "native_events": "../" + source.name,
            }
            if i + 1 < len(segments):
                segment["continued_trajectory_ref"] = f"segment-{i + 2:04d}.json"
            _rebase_media(segment, destination.parent, directory)
            write_json(
                directory / f"segment-{i + 1:04d}.json",
                Trajectory.model_validate(segment).to_json_dict(),
            )
        root["extra"].update(
            view="legacy_flat",
            canonical_trajectory_ref=f"{directory.name}/segment-0001.json",
            segment_count=len(segments),
            training_note="Use canonical segments instead of combining this compatibility view with them.",
        )
    value = Trajectory.model_validate(root).to_json_dict()
    write_json(destination, value)
    return value


def validate_atif_bundle(path: Path):
    """Validate local graph/media closure; old standalone ATIF stays valid."""
    path = Path(path).resolve()
    root, seen, active = path.parent, set(), set()

    def local(parent, reference):
        value = (parent / reference).resolve()
        if not value.is_relative_to(root) or not value.is_file():
            raise ValueError(
                "ATIF reference is missing or escapes the collected bundle"
            )
        return value

    def document(value, parent):
        parsed = Trajectory.model_validate(value)
        children = {c.trajectory_id: c for c in parsed.subagent_trajectories or []}
        if len(children) != len(parsed.subagent_trajectories or []) or None in children:
            raise ValueError("ATIF embedded subagents require unique trajectory IDs")
        for step in parsed.steps:
            contents = [step.message]
            for result in step.observation.results if step.observation else []:
                contents.append(result.content)
                for ref in result.subagent_trajectory_ref or []:
                    if ref.trajectory_id not in children:
                        if not ref.trajectory_path:
                            raise ValueError(
                                "unresolved embedded ATIF subagent reference"
                            )
                        target = local(parent, ref.trajectory_path)
                        if (
                            json.loads(target.read_text()).get("trajectory_id")
                            != ref.trajectory_id
                        ):
                            raise ValueError("ATIF external subagent identity mismatch")
                        visit(target)
            for content in contents:
                if isinstance(content, list):
                    for part in content:
                        if part.type in {"image", "audio"}:
                            local(parent, part.source.path)
        for child in parsed.subagent_trajectories or []:
            document(child.to_json_dict(), parent)
        for reference in (
            parsed.continued_trajectory_ref,
            (parsed.extra or {}).get("canonical_trajectory_ref"),
        ):
            if reference:
                visit(local(parent, reference))

    def visit(file):
        if file in active:
            raise ValueError("cyclic ATIF continuation")
        if file in seen:
            return
        active.add(file)
        document(json.loads(file.read_text()), file.parent)
        active.remove(file)
        seen.add(file)

    visit(path)
    return seen


def validate_subagent_capture(value: dict, events: list[dict]) -> None:
    """Strict audit-to-ATIF completeness for tool-triggered helper calls.

    Export itself preserves incomplete old logs with warnings. A delivery policy
    may call this to require explicit identities and links for every recorded
    request/response/error. No framework, dataset or tool name is special-cased.
    """
    children = value.get("subagent_trajectories") or []
    references = {
        (result.get("source_call_id"), ref.get("trajectory_id"))
        for step in value.get("steps", [])
        for result in step.get("observation", {}).get("results", [])
        for ref in result.get("subagent_trajectory_ref", [])
    }
    for event in events:
        identity = _subagent_identity(event)
        if identity is None or identity[0] == "compactor":
            continue
        role, invocation = identity
        candidates = [
            child
            for child in children
            if event["event_id"] in child.get("extra", {}).get("audit_event_ids", [])
        ]
        if len(candidates) != 1:
            raise ValueError(
                "ATIF omits subagent audit events or represents them more than once"
            )
        child = candidates[0]
        metadata = child.get("extra", {})
        parent = event["payload"].get("parent_tool_call_id") or event.get("parent_id")
        native_parent = metadata.get("native_parent_tool_call_id", parent)
        aliases = value.get("extra", {}).get("tool_call_aliases", {})
        if (
            not invocation
            or not parent
            or metadata.get("invocation_id") != invocation
            or metadata.get("role") != role
            or metadata.get("parent_tool_call_id") != parent
            or metadata.get("link_status") != "linked"
            or (native_parent != parent and aliases.get(parent) != native_parent)
            or (native_parent, child.get("trajectory_id")) not in references
        ):
            raise ValueError(
                "ATIF subagent audit lacks a complete recorded parent/invocation link"
            )


def attach_external_tool_audit(destination: Path, source: Path) -> None:
    """Link Reader traces to native calls via explicit IDs in recorded responses.

    No matching by time, call order or similar query text. If a native log omits
    the response ID, preserve the child as unlinked rather than invent topology.
    """
    value = Trajectory.model_validate_json(destination.read_text()).model_dump(
        exclude_none=True
    )
    if value.get("extra", {}).get("external_audit_source") == source.name:
        return
    events = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    issued = {
        e["payload"]["call_id"]: e["payload"]["name"]
        for e in events
        if e["event_type"] == "external_tool_call"
    }
    calls = {
        c["tool_call_id"]: c["function_name"]
        for s in value["steps"]
        for c in s.get("tool_calls", [])
    }

    def ids(content):
        if isinstance(content, str):
            try:
                yield from ids(json.loads(content))
            except ValueError:
                # Native logs may prepend timing/status lines to a JSON tool
                # result. Accept a complete JSON suffix, not a guessed call ID
                # or a substring regex over arbitrary document prose.
                for match in re.finditer(r"(?m)^[ \t]*(?=[{\[])", content):
                    try:
                        payload = json.loads(content[match.end() :])
                    except ValueError:
                        continue
                    yield from ids(payload)
                    break
        elif isinstance(content, list):
            for part in content:
                yield from ids(part)
        elif isinstance(content, dict):
            token = content.get("call_id")
            if (
                isinstance(token, str)
                and token in issued
                and content.get("name") == issued[token]
                and "ok" in content
            ):
                yield token
            for part in content.values():
                if isinstance(part, (str, list, dict)):
                    yield from ids(part)

    candidates = defaultdict(set)
    for step in value["steps"]:
        for result in step.get("observation", {}).get("results", []):
            native_id = result.get("source_call_id")
            for token in ids(result.get("content")):
                if (
                    native_id in calls
                    and calls[native_id].split("__")[-1] == issued[token]
                ):
                    candidates[token].add(native_id)
    aliases = {
        token: next(iter(matches))
        for token, matches in candidates.items()
        if len(matches) == 1
    }
    warnings = []
    episode_id = events[0]["episode_id"] if events else value["session_id"]
    children = _subagents(
        events,
        episode_id=episode_id,
        content=ContentWriter(source, destination, warnings),
    )
    for child in children:
        parent = child["extra"]["parent_tool_call_id"]
        if parent in aliases:
            child["extra"]["native_parent_tool_call_id"] = aliases[parent]
    _attach_children(value["steps"], children, warnings)
    if children:
        value.setdefault("subagent_trajectories", []).extend(children)
    value.setdefault("extra", {}).update(
        external_audit_source=source.name,
        tool_call_aliases=aliases,
        external_audit_warnings=warnings,
    )
    Trajectory.model_validate(value)
    original = destination.with_name("native-trajectory.json")
    if not original.exists():
        original.write_bytes(destination.read_bytes())
    write_json(destination, value)
