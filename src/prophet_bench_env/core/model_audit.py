"""Raw model-call artifacts and telemetry kept outside model conversation state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from .audit_context import correlation
from .trajectory import JsonlTrajectorySink


@dataclass(frozen=True, slots=True)
class ModelAttemptAudit:
    request_ref: str
    response_ref: str | None
    attempt: int
    status_code: int | None
    latency_ms: float


@dataclass(slots=True)
class FileModelAuditSink:
    """Persist exact provider payload/body without headers or credentials."""

    root: Path
    trajectory: JsonlTrajectorySink
    _attempts: dict[str, int] = field(default_factory=dict, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = self.root.expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def record_attempt(
        self,
        *,
        call_id: str,
        provider: str,
        model: str,
        endpoint: str,
        request_payload: dict[str, Any],
        response_body: bytes | None,
        status_code: int | None,
        latency_ms: float,
        error: str | None = None,
    ) -> ModelAttemptAudit:
        with self._lock:
            attempt = self._attempts.get(call_id, 0) + 1
            self._attempts[call_id] = attempt
        call_root = self.root / _safe_component(call_id)
        call_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        request_path = call_root / f"attempt-{attempt:02d}.request.json"
        request_path.write_text(
            json.dumps(
                {
                    "provider": provider,
                    "model": model,
                    "endpoint": endpoint,
                    "payload": request_payload,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        request_path.chmod(0o600)
        response_path: Path | None = None
        if response_body is not None:
            response_path = call_root / f"attempt-{attempt:02d}.response.bin"
            response_path.write_bytes(response_body)
            response_path.chmod(0o600)
        request_ref = str(request_path)
        response_ref = str(response_path) if response_path is not None else None
        self.trajectory.append(
            "model_provider_attempt",
            {
                "call_id": call_id,
                **correlation(),
                "provider": provider,
                "model": model,
                "attempt": attempt,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 3),
                "request_ref": request_ref,
                "request_sha256": _sha256(request_path.read_bytes()),
                "response_ref": response_ref,
                "response_sha256": _sha256(response_body)
                if response_body is not None
                else None,
                "error": error,
            },
            actor="model_provider",
            parent_id=call_id,
        )
        return ModelAttemptAudit(
            request_ref=request_ref,
            response_ref=response_ref,
            attempt=attempt,
            status_code=status_code,
            latency_ms=latency_ms,
        )

    def complete_call(
        self,
        *,
        call_id: str,
        provider: str,
        model: str,
        latency_ms: float,
        response: Any | None,
        error: BaseException | None,
    ) -> None:
        usage = dict(getattr(response, "usage", {}) or {}) if response else {}
        metadata = (
            dict(getattr(response, "provider_metadata", {}) or {}) if response else {}
        )
        self.trajectory.append(
            "model_call_completed" if error is None else "model_call_failed",
            {
                "call_id": call_id,
                **correlation(),
                "provider": provider,
                "model": model,
                "attempt_count": self._attempts.get(call_id, 0),
                "latency_ms": round(latency_ms, 3),
                "status_code": metadata.get("http_status"),
                "usage": usage,
                "input_tokens": _token_value(usage, "input_tokens", "prompt_tokens"),
                "output_tokens": _token_value(
                    usage, "output_tokens", "completion_tokens"
                ),
                "reasoning_tokens": _token_value(
                    usage, "reasoning_tokens", "thinking_tokens"
                ),
                "empty_response_retries": metadata.get("empty_response_retries", 0),
                "raw_capture_ref": getattr(response, "raw_capture_ref", None)
                if response
                else None,
                "error_type": type(error).__name__ if error else None,
                "error": str(error) if error else None,
            },
            actor="model_provider",
            parent_id=call_id,
        )


def bind_model_audit(
    model: Any,
    *,
    root: Path,
    trajectory: JsonlTrajectorySink,
) -> bool:
    """Bind environment-owned raw capture to any compatible model adapter."""

    setter = getattr(model, "set_audit_sink", None)
    if not callable(setter):
        return False
    setter(FileModelAuditSink(root, trajectory))
    return True


def _safe_component(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _token_value(usage: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    for nested_name in ("input_tokens_details", "output_tokens_details"):
        nested = usage.get(nested_name)
        if isinstance(nested, dict):
            value = _token_value(nested, *names)
            if value is not None:
                return value
    return None
