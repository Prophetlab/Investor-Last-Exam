"""Provider wire encoders for the environment's unified image content part."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def openai_chat_content(value: Any, *, supports_images: bool) -> Any:
    if not isinstance(value, list):
        return value
    encoded: list[dict[str, Any]] = []
    for part in value:
        kind, item = _part(part)
        if kind == "text":
            encoded.append({"type": "text", "text": str(item.get("text") or "")})
        elif kind == "image":
            _require_image_support(supports_images)
            encoded.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url(item),
                        "detail": str(item.get("detail") or "auto"),
                    },
                }
            )
        else:
            raise ValueError(f"unsupported unified content part: {kind}")
    return encoded


def openai_responses_content(value: Any, *, supports_images: bool) -> Any:
    if not isinstance(value, list):
        return value
    encoded: list[dict[str, Any]] = []
    for part in value:
        kind, item = _part(part)
        if kind == "text":
            encoded.append({"type": "input_text", "text": str(item.get("text") or "")})
        elif kind == "image":
            _require_image_support(supports_images)
            encoded.append(
                {
                    "type": "input_image",
                    "image_url": _data_url(item),
                    "detail": str(item.get("detail") or "auto"),
                }
            )
        else:
            raise ValueError(f"unsupported unified content part: {kind}")
    return encoded


def anthropic_content(value: Any, *, supports_images: bool) -> Any:
    if not isinstance(value, list):
        return value
    encoded: list[dict[str, Any]] = []
    for part in value:
        kind, item = _part(part)
        if kind == "text":
            encoded.append({"type": "text", "text": str(item.get("text") or "")})
        elif kind == "image":
            _require_image_support(supports_images)
            encoded.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": str(item.get("media_type") or "image/png"),
                        "data": str(item.get("data_base64") or ""),
                    },
                }
            )
        else:
            raise ValueError(f"unsupported unified content part: {kind}")
    return encoded


def _part(value: Any) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError("multimodal content parts must be objects")
    return str(value.get("type") or ""), value


def _data_url(part: Mapping[str, Any]) -> str:
    media_type = str(part.get("media_type") or "image/png")
    data = str(part.get("data_base64") or "")
    if not data:
        raise ValueError("image content part has no base64 data")
    return f"data:{media_type};base64,{data}"


def _require_image_support(supports_images: bool) -> None:
    if not supports_images:
        raise ValueError("configured model does not support image input")
