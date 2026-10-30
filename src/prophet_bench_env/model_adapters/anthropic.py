"""Anthropic Messages adapter with thinking-block replay."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from prophet_bench_env.ports.model import ModelRequest, ModelResponse

from .capture import capture_provider_response, parse_json_arguments
from .errors import ProviderInvocationError
from .http import HTTPModelAdapter
from .multimodal import anthropic_content


class AnthropicMessagesAdapter(HTTPModelAdapter):
    """Call Anthropic-compatible ``/v1/messages`` routes.

    The thinking request intentionally follows the hv3.8 CommonStack Messages
    profile (``type=enabled`` plus a bounded budget).  A future adaptive profile
    can be added without changing the core or this profile's replay semantics.
    """

    def generate(self, request: ModelRequest) -> ModelResponse:
        call_id = self.new_call_id()
        started = time.perf_counter()
        payload = self._payload(request)
        try:
            response = self._with_empty_retries(
                lambda: self._generate_once(payload, call_id)
            )
            response = self._annotate_call(response, call_id)
        except BaseException as error:
            self._complete_audit(call_id=call_id, started=started, error=error)
            raise
        self._complete_audit(call_id=call_id, started=started, response=response)
        return response

    async def agenerate(self, request: ModelRequest) -> ModelResponse:
        call_id = self.new_call_id()
        started = time.perf_counter()
        payload = self._payload(request)
        try:
            response = await self._awith_empty_retries(
                lambda: self._agenerate_once(payload, call_id)
            )
            response = self._annotate_call(response, call_id)
        except BaseException as error:
            self._complete_audit(call_id=call_id, started=started, error=error)
            raise
        self._complete_audit(call_id=call_id, started=started, response=response)
        return response

    def _generate_once(self, payload: dict[str, Any], call_id: str) -> ModelResponse:
        result = self._post_json(self._endpoint(), payload, call_id=call_id)
        return self._parse(result)

    async def _agenerate_once(
        self, payload: dict[str, Any], call_id: str
    ) -> ModelResponse:
        result = await self._apost_json(self._endpoint(), payload, call_id=call_id)
        return self._parse(result)

    def _endpoint(self) -> str:
        base = self.settings.base_url or "https://api.anthropic.com"
        base = base.rstrip("/")
        base = base.removesuffix("/v1")
        return base + "/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.settings.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        system: list[str] = []
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            role = str(message.get("role") or "user")
            if role == "system":
                content = message.get("content", "")
                if content:
                    system.append(str(content))
                continue
            if role == "tool":
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": str(
                                    message.get("tool_call_id")
                                    or message.get("call_id")
                                    or ""
                                ),
                                "content": str(message.get("content") or ""),
                            }
                        ],
                    }
                )
                continue
            if role == "assistant":
                replay = message.get("provider_replay")
                if isinstance(replay, Mapping) and isinstance(
                    replay.get("content_blocks"), list
                ):
                    content = [
                        dict(item)
                        for item in replay["content_blocks"]
                        if isinstance(item, Mapping)
                    ]
                else:
                    content = self._assistant_content(message)
                messages.append({"role": "assistant", "content": content})
                continue
            messages.append(
                {
                    "role": "user",
                    "content": anthropic_content(
                        message.get("content", ""),
                        supports_images=self.settings.supports_images,
                    ),
                }
            )

        max_output_tokens = (
            request.max_output_tokens
            if request.max_output_tokens is not None
            else self.settings.max_output_tokens
        )
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "max_tokens": max_output_tokens,
        }
        if system:
            payload["system"] = "\n\n".join(system)
        if request.tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in request.tools
            ]
        temperature = (
            request.temperature
            if request.temperature is not None
            else self.settings.temperature
        )
        if temperature is not None:
            payload["temperature"] = temperature
        effort_value = (
            request.reasoning_effort
            if request.reasoning_effort is not None
            else self.settings.reasoning_effort
        )
        if effort_value:
            effort = effort_value.casefold()
            if effort != "none":
                # CommonStack requires max_tokens > budget_tokens. Reserve
                # output room for the final answer/tool call as hv3.8 did.
                budget_tokens = min(
                    max(max_output_tokens - 8_000, 1_024),
                    16_000,
                )
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
        return payload

    @staticmethod
    def _assistant_content(message: Mapping[str, Any]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        text = message.get("content")
        if text:
            content.append({"type": "text", "text": str(text)})
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, Mapping):
                    continue
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id", call.get("call_id", "")),
                        "name": call.get("name", ""),
                        "input": call.get("arguments", call.get("args", {})),
                    }
                )
        return content or [{"type": "text", "text": ""}]

    def _parse(self, result) -> ModelResponse:
        data = result.data
        content = data.get("content")
        if not isinstance(content, list):
            raise ProviderInvocationError("Anthropic response has no content blocks")
        text_parts: list[str] = []
        calls: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, Mapping):
                continue
            kind = block.get("type")
            if kind == "text":
                text_parts.append(str(block.get("text") or ""))
            elif kind == "tool_use":
                calls.append(
                    {
                        "id": block.get("id"),
                        "name": block.get("name", ""),
                        "arguments": parse_json_arguments(block.get("input", {})),
                    }
                )
        capture = capture_provider_response(data, raw_body=result.raw_body)
        capture["http_status"] = result.status_code
        capture["provider_latency_ms"] = round(result.latency_ms, 3)
        usage = (
            dict(data.get("usage")) if isinstance(data.get("usage"), Mapping) else {}
        )
        if capture.get("reasoning_tokens") is not None:
            usage.setdefault("reasoning_tokens", capture["reasoning_tokens"])
        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tuple(calls),
            finish_reason=str(data.get("stop_reason"))
            if data.get("stop_reason")
            else None,
            usage=usage,
            provider=self.settings.provider,
            raw_capture_ref=result.raw_capture_ref,
            response_sha256=str(capture["provider_response_sha256"]),
            provider_replay={
                "content_blocks": [
                    dict(item) for item in content if isinstance(item, Mapping)
                ]
            },
            provider_metadata=capture,
        )
