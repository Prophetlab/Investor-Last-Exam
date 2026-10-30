"""Provider-neutral model adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.contracts import JsonObject, ToolDefinition


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: tuple[JsonObject, ...]
    tools: tuple[ToolDefinition, ...]
    reasoning_effort: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    tool_calls: tuple[JsonObject, ...] = ()
    finish_reason: str | None = None
    usage: JsonObject = field(default_factory=dict)
    provider: str | None = None
    raw_capture_ref: str | None = None
    response_sha256: str | None = None
    # Provider-native state that must be replayed on a subsequent tool turn.  For
    # example, Responses reasoning items and Anthropic thinking blocks cannot be
    # reconstructed from the normalized tool calls alone.
    provider_replay: JsonObject = field(default_factory=dict)
    # Redacted provider metadata used by trajectory/reporting layers.  Credentials
    # and request headers never belong here.
    provider_metadata: JsonObject = field(default_factory=dict)


class ModelAdapter(Protocol):
    """Structural interface implemented by a provider-specific model adapter."""

    def generate(self, request: ModelRequest) -> ModelResponse: ...


class ModelAuditSink(Protocol):
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
    ) -> Any: ...

    def complete_call(
        self,
        *,
        call_id: str,
        provider: str,
        model: str,
        latency_ms: float,
        response: Any | None,
        error: BaseException | None,
    ) -> None: ...


class AsyncModelAdapter(Protocol):
    """Optional async companion for async harnesses."""

    async def agenerate(self, request: ModelRequest) -> ModelResponse: ...
