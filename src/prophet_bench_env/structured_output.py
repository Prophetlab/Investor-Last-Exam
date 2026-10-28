"""Provider-neutral structured final-output helpers for plugin-owned agents."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel


def schema_instruction(schema: type[BaseModel]) -> str:
    """Return a strict final-JSON instruction without adding benchmark semantics."""

    encoded = json.dumps(
        schema.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "\n\nReturn the final result as one JSON object matching this JSON Schema. "
        "Do not wrap it in prose or Markdown fences.\n" + encoded
    )


def parse_final_json[StructuredModel: BaseModel](
    text: str, schema: type[StructuredModel]
) -> StructuredModel:
    """Parse one final object, accepting a single fenced JSON block as fallback."""

    candidates = [text.strip()]
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text.strip(), re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    for candidate in candidates:
        try:
            return schema.model_validate(json.loads(candidate))
        except (TypeError, ValueError):
            continue
    raise ValueError("model did not return a valid final JSON object")
