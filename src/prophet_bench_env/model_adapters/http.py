"""Small HTTP base shared by provider adapters."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self
from uuid import uuid4

import httpx

from ..ports.model import ModelAuditSink
from .config import ModelSettings
from .errors import ProviderHTTPError
from .retry import (
    run_with_empty_response_retries,
    run_with_empty_response_retries_async,
    run_with_retries,
    run_with_retries_async,
)

EMPTY_RESPONSE_FALLBACK = (
    "Model returned an empty terminal response after bounded infrastructure retries."
)


@dataclass(frozen=True, slots=True)
class ProviderHTTPResponse:
    data: dict[str, Any]
    raw_body: bytes
    raw_capture_ref: str | None
    status_code: int
    latency_ms: float


class HTTPModelAdapter:
    def __init__(
        self,
        settings: ModelSettings,
        *,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=settings.timeout_seconds)
        self._owns_async_client = async_client is None
        self._async_client = async_client
        self.audit_sink: ModelAuditSink | None = None

    def set_audit_sink(self, sink: ModelAuditSink | None) -> None:
        self.audit_sink = sink

    @staticmethod
    def new_call_id() -> str:
        return f"model_{uuid4().hex}"

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    async def aclose(self) -> None:
        if self._owns_async_client and self._async_client is not None:
            await self._async_client.aclose()

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self.settings.timeout_seconds
            )
        return self._async_client

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

    def _post_json(
        self, url: str, payload: Mapping[str, Any], *, call_id: str
    ) -> ProviderHTTPResponse:
        started = time.perf_counter()
        try:
            response = self.client.post(url, headers=self._headers(), json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as error:
            self._record_attempt(
                call_id=call_id,
                endpoint=url,
                payload=payload,
                response_body=None,
                status_code=None,
                started=started,
                error=f"{type(error).__name__}: {error}",
            )
            raise ConnectionError(str(error)) from error
        capture_ref = self._record_attempt(
            call_id=call_id,
            endpoint=url,
            payload=payload,
            response_body=response.content,
            status_code=response.status_code,
            started=started,
        )
        try:
            body = response.json()
        except ValueError as error:
            body = None
            if response.is_error:
                raise ProviderHTTPError(
                    f"provider HTTP {response.status_code} returned non-JSON body",
                    status_code=response.status_code,
                    body=response.text,
                ) from error
            raise ProviderHTTPError(
                "provider returned invalid JSON",
                status_code=response.status_code,
                body=response.text,
            ) from error
        if response.is_error:
            message = _error_message(body)
            raise ProviderHTTPError(
                f"provider HTTP {response.status_code}: {message}",
                status_code=response.status_code,
                body=body,
            )
        if not isinstance(body, dict):
            raise ProviderHTTPError(
                "provider response must be a JSON object",
                status_code=response.status_code,
                body=body,
            )
        return ProviderHTTPResponse(
            body,
            response.content,
            capture_ref,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )

    async def _apost_json(
        self, url: str, payload: Mapping[str, Any], *, call_id: str
    ) -> ProviderHTTPResponse:
        started = time.perf_counter()
        try:
            response = await self.async_client.post(
                url, headers=self._headers(), json=payload
            )
        except (httpx.TimeoutException, httpx.TransportError) as error:
            self._record_attempt(
                call_id=call_id,
                endpoint=url,
                payload=payload,
                response_body=None,
                status_code=None,
                started=started,
                error=f"{type(error).__name__}: {error}",
            )
            raise ConnectionError(str(error)) from error
        capture_ref = self._record_attempt(
            call_id=call_id,
            endpoint=url,
            payload=payload,
            response_body=response.content,
            status_code=response.status_code,
            started=started,
        )
        try:
            body = response.json()
        except ValueError as error:
            body = None
            if response.is_error:
                raise ProviderHTTPError(
                    f"provider HTTP {response.status_code} returned non-JSON body",
                    status_code=response.status_code,
                    body=response.text,
                ) from error
            raise ProviderHTTPError(
                "provider returned invalid JSON",
                status_code=response.status_code,
                body=response.text,
            ) from error
        if response.is_error:
            message = _error_message(body)
            raise ProviderHTTPError(
                f"provider HTTP {response.status_code}: {message}",
                status_code=response.status_code,
                body=body,
            )
        if not isinstance(body, dict):
            raise ProviderHTTPError(
                "provider response must be a JSON object",
                status_code=response.status_code,
                body=body,
            )
        return ProviderHTTPResponse(
            body,
            response.content,
            capture_ref,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )

    def _record_attempt(
        self,
        *,
        call_id: str,
        endpoint: str,
        payload: Mapping[str, Any],
        response_body: bytes | None,
        status_code: int | None,
        started: float,
        error: str | None = None,
    ) -> str | None:
        if self.audit_sink is None:
            return None
        result = self.audit_sink.record_attempt(
            call_id=call_id,
            provider=self.settings.provider,
            model=self.settings.model,
            endpoint=endpoint,
            request_payload=dict(payload),
            response_body=response_body,
            status_code=status_code,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=error,
        )
        return getattr(result, "response_ref", None)

    def _complete_audit(
        self,
        *,
        call_id: str,
        started: float,
        response: Any | None = None,
        error: BaseException | None = None,
    ) -> None:
        if self.audit_sink is not None:
            self.audit_sink.complete_call(
                call_id=call_id,
                provider=self.settings.provider,
                model=self.settings.model,
                latency_ms=(time.perf_counter() - started) * 1000,
                response=response,
                error=error,
            )

    @staticmethod
    def _annotate_call(response: Any, call_id: str) -> Any:
        """Link normalized responses to raw audit records without conversation leakage."""

        from dataclasses import replace

        return replace(
            response,
            provider_metadata={
                **getattr(response, "provider_metadata", {}),
                "model_call_id": call_id,
            },
        )

    def _with_retries(self, operation):
        return run_with_retries(
            operation,
            model_label=self.settings.model,
            max_retries=self.settings.max_retries,
            base_seconds=self.settings.retry_base_seconds,
            max_seconds=self.settings.retry_max_seconds,
        )

    async def _awith_retries(self, operation):
        return await run_with_retries_async(
            operation,
            model_label=self.settings.model,
            max_retries=self.settings.max_retries,
            base_seconds=self.settings.retry_base_seconds,
            max_seconds=self.settings.retry_max_seconds,
        )

    def _with_empty_retries(self, operation):
        return run_with_empty_response_retries(
            operation,
            is_empty=_empty_terminal,
            on_exhausted=_replace_empty,
            on_success=_mark_empty_retries,
            model_label=self.settings.model,
            max_retries=self.settings.empty_response_retries,
            base_seconds=self.settings.empty_response_base_seconds,
            retry_transport=self._with_retries,
        )

    async def _awith_empty_retries(self, operation):
        return await run_with_empty_response_retries_async(
            operation,
            is_empty=_empty_terminal,
            on_exhausted=_replace_empty,
            on_success=_mark_empty_retries,
            model_label=self.settings.model,
            max_retries=self.settings.empty_response_retries,
            base_seconds=self.settings.empty_response_base_seconds,
            retry_transport=self._awith_retries,
        )


def _error_message(body: object) -> str:
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            for key in ("message", "detail", "code"):
                if error.get(key) is not None:
                    return str(error[key])
        if body.get("detail") is not None:
            return str(body["detail"])
    return json.dumps(body, ensure_ascii=False, default=str)


def _empty_terminal(response: object) -> bool:
    if not hasattr(response, "text") or not hasattr(response, "finish_reason"):
        return False
    finish = response.finish_reason
    return (
        finish in {"stop", "completed", "end_turn"}
        and not str(getattr(response, "text", "") or "").strip()
        and not getattr(response, "tool_calls", ())
    )


def _replace_empty(response: object, retry_count: int) -> object:
    # ModelResponse is frozen; import locally to keep this base independent of
    # the concrete adapter modules during import.
    from dataclasses import replace

    return replace(
        response,
        text=EMPTY_RESPONSE_FALLBACK,
        provider_metadata={
            **getattr(response, "provider_metadata", {}),
            "empty_response_retries": retry_count,
        },
    )


def _mark_empty_retries(response: object, retry_count: int) -> object:
    if retry_count == 0:
        return response
    from dataclasses import replace

    return replace(
        response,
        provider_metadata={
            **getattr(response, "provider_metadata", {}),
            "empty_response_retries": retry_count,
        },
    )
