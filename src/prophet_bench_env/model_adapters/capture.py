"""Provider reasoning capture and response normalization helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def provider_value(value: object, *keys: str) -> object | None:
    if isinstance(value, Mapping):
        for key in keys:
            if key in value and value[key] is not None:
                return value[key]
        for item in value.values():
            found = provider_value(item, *keys)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = provider_value(item, *keys)
            if found is not None:
                return found
    return None


def _response_messages(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return []
    messages: list[dict[str, Any]] = []
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        for key in ("message", "delta"):
            value = choice.get(key)
            if isinstance(value, Mapping):
                messages.append(dict(value))
    return messages


def _reasoning_details_values(details: object) -> tuple[object | None, object | None]:
    if not isinstance(details, list):
        return None, None
    summaries: list[str] = []
    encrypted: list[object] = []
    for item in details:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type")
        data = item.get("data")
        if kind == "reasoning.summary" and isinstance(data, str):
            summaries.append(data)
        elif kind == "reasoning.encrypted_content" and data is not None:
            encrypted.append(data)
    summary = "\n".join(summaries) if summaries else None
    encrypted_value = encrypted[0] if len(encrypted) == 1 else encrypted or None
    return summary, encrypted_value


def _response_reasoning_values(
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str | None, str | None, object | None]:
    output = payload.get("output")
    if not isinstance(output, list):
        return [], None, None, None
    items = [
        dict(item)
        for item in output
        if isinstance(item, Mapping) and item.get("type") == "reasoning"
    ]
    reasoning_text: list[str] = []
    summaries: list[str] = []
    encrypted: list[object] = []
    for item in items:
        content = item.get("content")
        if isinstance(content, str) and content:
            reasoning_text.append(content)
        elif isinstance(content, list):
            reasoning_text.extend(
                str(block["text"])
                for block in content
                if isinstance(block, Mapping)
                and block.get("type") in {"reasoning_text", "text"}
                and block.get("text") is not None
            )
        summary = item.get("summary")
        if isinstance(summary, str) and summary:
            summaries.append(summary)
        elif isinstance(summary, list):
            summaries.extend(
                str(block["text"])
                for block in summary
                if isinstance(block, Mapping) and block.get("text") is not None
            )
        if item.get("encrypted_content") is not None:
            encrypted.append(item["encrypted_content"])
    encrypted_value = encrypted[0] if len(encrypted) == 1 else encrypted or None
    return (
        items,
        "\n".join(reasoning_text) or None,
        "\n".join(summaries) or None,
        encrypted_value,
    )


def capture_provider_response(
    payload: Mapping[str, Any], *, raw_body: bytes | None = None
) -> dict[str, Any]:
    """Capture provider-native reasoning without flattening or inventing it."""

    encoded = raw_body or json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    provider_reasoning: dict[str, Any] = {}
    for key in (
        "reasoning_content",
        "reasoning",
        "reasoning_details",
        "thinking",
        "raw_text",
        "encrypted_content",
        "signature",
    ):
        values = [
            message[key]
            for message in _response_messages(payload)
            if message.get(key) is not None
        ]
        if values:
            provider_reasoning[key] = values[0] if len(values) == 1 else values

    content_values = [payload.get("content")]
    content_values.extend(
        message.get("content") for message in _response_messages(payload)
    )
    content_blocks: list[dict[str, Any]] = []
    for content in content_values:
        if not isinstance(content, list):
            continue
        content_blocks.extend(
            dict(item)
            for item in content
            if isinstance(item, Mapping)
            and item.get("type") in {"thinking", "redacted_thinking", "reasoning"}
        )
    if content_blocks:
        provider_reasoning["content_blocks"] = content_blocks

    response_items, response_text, response_summary, response_encrypted = (
        _response_reasoning_values(payload)
    )
    if response_items:
        provider_reasoning["output"] = response_items
    details_summary, details_encrypted = _reasoning_details_values(
        provider_reasoning.get("reasoning_details")
    )
    reasoning_content = (
        provider_reasoning.get("raw_text")
        or provider_reasoning.get("reasoning_content")
        or provider_reasoning.get("reasoning")
        or provider_reasoning.get("thinking")
        or response_text
        or provider_value(content_blocks, "thinking", "text")
    )
    encrypted_content = (
        details_encrypted
        or response_encrypted
        or provider_value(provider_reasoning, "encrypted_content")
        or provider_value(content_blocks, "encrypted_content", "data")
    )
    reasoning_tokens = None
    for usage in (payload.get("usage"), payload.get("usage_metadata")):
        reasoning_tokens = provider_value(usage, "reasoning_tokens")
        if reasoning_tokens is None:
            reasoning_tokens = provider_value(usage, "thinking_tokens")
        if reasoning_tokens is not None:
            break
    return {
        "provider_response_sha256": hashlib.sha256(encoded).hexdigest(),
        "reasoning_content": reasoning_content,
        "summary": details_summary or response_summary,
        "encrypted_content": encrypted_content,
        "signature": provider_value(provider_reasoning, "signature")
        or provider_value(content_blocks, "signature"),
        "reasoning_tokens": reasoning_tokens,
        "provider_reasoning": provider_reasoning,
    }


def visible_text(value: object) -> str:
    """Extract visible text from string or provider content blocks."""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                text = (
                    item.get("text") or item.get("content") or item.get("output_text")
                )
                if text is not None:
                    parts.append(visible_text(text))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(value, Mapping):
        for key in ("text", "content", "output_text"):
            if key in value:
                return visible_text(value[key])
    return "" if value is None else str(value)


def parse_json_arguments(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        # Keep the malformed raw string.  The harness will discard/correct the
        # response instead of pretending the call was valid.
        return value
