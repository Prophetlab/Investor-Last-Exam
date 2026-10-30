"""Reusable single-call reader for bounded financial-document content."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any
from uuid import uuid4

from ..core.audit_context import correlation, subagent_scope
from ..core.trajectory import JsonlTrajectorySink
from ..ports.model import ModelAdapter, ModelRequest


@dataclass(frozen=True, slots=True)
class ModelBackedReader:
    model: ModelAdapter
    system_prompt: str
    trajectory: JsonlTrajectorySink | None = None
    supports_images: bool = False

    @staticmethod
    def measure_content(text: str) -> int:
        """Conservative UTF-8 byte estimate, not a provider-specific tokenizer.

        Avoid a characters/4 assumption that underestimates Chinese, numbers and
        identifiers. Role framing and provider-specific overhead are reserved
        separately. Custom readers may supply their own counting implementation.
        """
        return len(text.encode("utf-8"))

    def content_budget(self, *, query: str, metadata: dict[str, Any]) -> int | None:
        settings = getattr(self.model, "settings", None)
        context = getattr(settings, "context_window_tokens", None)
        if context is None:
            return None
        output = getattr(settings, "max_output_tokens", 24_000)
        overhead = self.measure_content(self.system_prompt) + self.measure_content(
            _user_text(query, "", metadata)
        )
        budget = context - output - overhead - 2048
        if budget < 1024:
            raise ValueError(
                "Reader context budget is too small after reserving output and prompt; "
                "check context_window_tokens and max_output_tokens"
            )
        return budget

    @classmethod
    def from_default(
        cls,
        model: ModelAdapter,
        *,
        trajectory: JsonlTrajectorySink | None = None,
    ) -> ModelBackedReader:
        prompt = (
            files("prophet_bench_env.adapters.prompts")
            .joinpath("financial_document_reader.md")
            .read_text(encoding="utf-8")
            .strip()
        )
        settings = getattr(model, "settings", None)
        return cls(
            model=model,
            system_prompt=prompt,
            trajectory=trajectory,
            supports_images=bool(getattr(settings, "supports_images", False)),
        )

    def read(
        self,
        *,
        query: str,
        content: str,
        metadata: dict[str, Any],
        image: bytes | None = None,
        parent_call_id: str | None = None,
    ) -> str:
        budget = self.content_budget(query=query, metadata=metadata)
        if budget is not None and self.measure_content(content) > budget:
            raise ValueError(
                "Reader content exceeds the configured model context budget"
            )
        user_text = _user_text(query, content, metadata)
        user_content: Any = user_text
        if image is not None:
            if not self.supports_images:
                raise ValueError("reader model does not support image input")
            user_content = [
                {"type": "text", "text": user_text},
                {
                    "type": "image",
                    "media_type": "image/png",
                    "data_base64": base64.b64encode(image).decode("ascii"),
                    "detail": "auto",
                },
            ]
        request = ModelRequest(
            messages=(
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": user_content,
                },
            ),
            tools=(),
        )
        invocation_id = "reader-" + uuid4().hex
        parent_call_id = parent_call_id or correlation()["parent_tool_call_id"]
        image_ref = None
        if image is not None and self.trajectory is not None:
            asset = (
                self.trajectory.path.parent
                / "reader-assets"
                / (hashlib.sha256(image).hexdigest() + ".png")
            )
            asset.parent.mkdir(parents=True, exist_ok=True)
            if not asset.exists():
                asset.write_bytes(image)
                asset.chmod(0o600)
            image_ref = str(asset.relative_to(self.trajectory.path.parent))
        link = {
            "reader_invocation_id": invocation_id,
            "parent_tool_call_id": parent_call_id,
            "model_name": getattr(getattr(self.model, "settings", None), "model", None),
        }
        if self.trajectory is not None:
            self.trajectory.append(
                "reader_model_request",
                {
                    **link,
                    "messages": _audit_messages(request.messages, image_ref=image_ref),
                    "image_ref": image_ref,
                    "content_budget": budget,
                    "budget_measure": "utf8_bytes_conservative"
                    if budget is not None
                    else "unknown",
                    "has_image": image is not None,
                    "image_sha256": hashlib.sha256(image).hexdigest()
                    if image is not None
                    else None,
                    "image_size_bytes": len(image) if image is not None else None,
                },
                actor="reader",
                parent_id=parent_call_id,
            )
        try:
            with subagent_scope(invocation_id, parent_call_id):
                response = self.model.generate(request)
        except Exception as error:
            if self.trajectory is not None:
                self.trajectory.append(
                    "reader_model_error",
                    {**link, "error_type": type(error).__name__},
                    actor="reader",
                    parent_id=parent_call_id,
                )
            raise
        if self.trajectory is not None:
            self.trajectory.append(
                "reader_model_response",
                {
                    **link,
                    "call_id": response.provider_metadata.get("model_call_id"),
                    "content": response.text,
                    "reasoning": response.provider_metadata.get("reasoning_content"),
                    "summary": response.provider_metadata.get("summary"),
                    "usage": response.usage,
                    "provider": response.provider,
                    "provider_metadata": response.provider_metadata,
                    "raw_capture_ref": response.raw_capture_ref,
                    "response_sha256": response.response_sha256,
                },
                actor="reader",
                parent_id=parent_call_id,
            )
        text = response.text.strip()
        if not text:
            if self.trajectory is not None:
                self.trajectory.append(
                    "reader_model_error",
                    {**link, "error_type": "EmptyReaderResponse"},
                    actor="reader",
                    parent_id=parent_call_id,
                )
            raise ValueError("reader returned no visible text")
        return text


def _user_text(query, content, metadata):
    if "sources" in metadata:
        return (
            "Documents and requested scopes: "
            + json.dumps(metadata["sources"], ensure_ascii=False)
            + f"\n\nReading request:\n{query}\n\nSource material and coverage:\n{content}"
        )
    return (
        f"Document title: {metadata.get('title') or ''}\n\n"
        f"Content format: {metadata.get('format') or ''}\n"
        f"Content location: {metadata.get('navigation')}\n\n"
        f"Reading request:\n{query}\n\nDocument content window:\n{content}"
    )


def _audit_messages(
    messages: tuple[dict[str, Any], ...], *, image_ref: str | None = None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        copy = dict(message)
        content = copy.get("content")
        if isinstance(content, list):
            copy["content"] = [
                {
                    **{
                        key: value
                        for key, value in part.items()
                        if key != "data_base64"
                    },
                    **(
                        {"path": image_ref}
                        if part.get("type") == "image" and image_ref
                        else {}
                    ),
                }
                if isinstance(part, dict)
                else part
                for part in content
            ]
        result.append(copy)
    return result
