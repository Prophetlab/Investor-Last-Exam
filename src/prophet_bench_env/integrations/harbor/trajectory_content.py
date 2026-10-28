"""Loss-aware conversion of recorded model content into ATIF fields."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ContentWriter:
    def __init__(self, source: Path, destination: Path, warnings: list):
        self.source, self.destination, self.warnings = source, destination, warnings

    def convert(self, value, event_id=None):
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        if not isinstance(value, list):
            return json.dumps(value, ensure_ascii=False)
        result = []
        for part in value:
            if not isinstance(part, dict):
                result.append({"type": "text", "text": str(part)})
            elif part.get("type") == "text":
                result.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") in {"image", "audio"}:
                source = part.get("source") or {}
                path = part.get("path") or source.get("path")
                media_type = part.get("media_type") or source.get("media_type")
                if path:
                    local = (self.source.parent / path).resolve()
                    if (
                        not local.is_relative_to(self.source.parent.resolve())
                        or not local.is_file()
                    ):
                        raise ValueError(
                            "trajectory media must exist inside its audit directory"
                        )
                    data = local.read_bytes()
                    folder = self.destination.parent / (
                        self.destination.stem + ".assets"
                    )
                    folder.mkdir(parents=True, exist_ok=True)
                    target = folder / (hashlib.sha256(data).hexdigest() + local.suffix)
                    if not target.exists():
                        target.write_bytes(data)
                    result.append(
                        {
                            "type": part["type"],
                            "source": {
                                "media_type": media_type,
                                "path": str(
                                    target.relative_to(self.destination.parent)
                                ),
                            },
                        }
                    )
                else:
                    self.warnings.append(
                        {"code": "media_not_captured", "event_id": event_id}
                    )
            else:
                self.warnings.append(
                    {
                        "code": "unsupported_content_part",
                        "event_id": event_id,
                        "type": part.get("type"),
                    }
                )
        return result or ""


def tool_calls(payload):
    result = []
    for call in payload.get("tool_calls", []):
        function = call.get("function", call)
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        result.append(
            {
                "tool_call_id": call.get("id", call.get("call_id")),
                "function_name": function["name"],
                "arguments": arguments,
            }
        )
    return result


def metrics(payload):
    usage = payload.get("usage") or {}
    result = {}
    for aliases, target in (
        (("prompt_tokens", "input_tokens"), "prompt_tokens"),
        (("completion_tokens", "output_tokens"), "completion_tokens"),
    ):
        value = next((usage[n] for n in aliases if type(usage.get(n)) is int), None)
        if value is not None:
            result[target] = value
    details = (
        usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    )
    cached = usage.get(
        "cache_read_input_tokens",
        usage.get("prompt_cache_hit_tokens", details.get("cached_tokens")),
    )
    if type(cached) is int:
        result["cached_tokens"] = cached
    output = (
        usage.get("completion_tokens_details")
        or usage.get("output_tokens_details")
        or {}
    )
    reasoning = usage.get("reasoning_tokens", output.get("reasoning_tokens"))
    if usage:
        result["extra"] = {"usage": usage}
        if reasoning is not None:
            result["extra"]["reasoning_tokens"] = reasoning
    return result


def response_step(event, content):
    p = event["payload"]
    step = {
        "source": "agent",
        "message": content.convert(
            p.get("content", p.get("text", "")), event.get("event_id")
        ),
        "extra": {"audit_event_id": event.get("event_id")},
        "llm_call_count": 1,
    }
    if event.get("timestamp"):
        step["timestamp"] = event["timestamp"]
    calls = tool_calls(p)
    if calls:
        step["tool_calls"] = calls
    usage = metrics(p)
    if usage:
        step["metrics"] = usage
    metadata = p.get("provider_metadata") or {}
    reasoning = p.get("reasoning") or metadata.get("reasoning_content")
    if isinstance(reasoning, str):
        step["reasoning_content"] = reasoning
    for key in (
        "raw_capture_ref",
        "response_sha256",
        "provider_metadata",
        "summary",
        "finish_reason",
    ):
        if p.get(key) is not None:
            step["extra"][key] = p[key]
    return step


def context_steps(messages, content, *, copied_indices=(), event_id=None):
    """Keep exact context, including truncated histories starting with a tool."""
    steps, calls = [], {}
    for index, message in enumerate(messages):
        role = message.get("role")
        converted = content.convert(message.get("content"), event_id)
        if role == "tool":
            call_id = message.get("tool_call_id", message.get("call_id"))
            parent = calls.get(call_id)
            if parent is not None:
                parent.setdefault("observation", {"results": []})["results"].append(
                    {"source_call_id": call_id, "content": converted}
                )
                continue
            # Do not invent the assistant message absent from the actual context.
            step = {
                "source": "system",
                "message": "",
                "observation": {"results": [{"content": converted}]},
                "extra": {"original_role": "tool", "original_call_id": call_id},
            }
        elif role in {"system", "user", "assistant"}:
            step = {
                "source": "agent" if role == "assistant" else role,
                "message": converted,
            }
            if role == "assistant":
                raw = tool_calls(message)
                if raw:
                    step["tool_calls"] = raw
                    calls.update({c["tool_call_id"]: step for c in raw})
        else:
            raise ValueError("unsupported recorded message role")
        if index in copied_indices:
            step["is_copied_context"] = True
        step.setdefault("extra", {})["context_message_index"] = index
        steps.append(step)
    return steps
