"""OpenAI-compatible Chat Completions adapters.

The wire shape is shared by many providers, but reasoning fields are selected
by an explicit protocol profile.  This preserves the hv3.8 GLM/DeepSeek/
StepFun/Ling/Kimi/CommonStack behavior without guessing from the hostname.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from prophet_bench_env.ports.model import ModelRequest, ModelResponse

from .capture import capture_provider_response, parse_json_arguments, visible_text
from .errors import ProviderInvocationError
from .http import HTTPModelAdapter
from .multimodal import openai_chat_content


class OpenAICompatibleChatAdapter(HTTPModelAdapter):
    """Call a provider's ``/chat/completions`` endpoint."""

    def generate(self, request: ModelRequest) -> ModelResponse:
        call_id = self.new_call_id()
        started = time.perf_counter()
        payload = self._payload(request)
        try:
            response = self._with_empty_retries(
                lambda: self._generate_once(payload, request, call_id)
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

    def _generate_once(
        self,
        payload: dict[str, Any],
        request: ModelRequest,
        call_id: str,
    ) -> ModelResponse:
        del request
        result = self._post_json(
            self._endpoint("/chat/completions"), payload, call_id=call_id
        )
        return self._parse(result)

    async def _agenerate_once(
        self, payload: dict[str, Any], call_id: str
    ) -> ModelResponse:
        result = await self._apost_json(
            self._endpoint("/chat/completions"), payload, call_id=call_id
        )
        return self._parse(result)

    def _endpoint(self, suffix: str) -> str:
        base = self.settings.base_url
        if not base:
            raise ValueError("openai-compatible adapter requires MODEL_BASE_URL")
        return base.rstrip("/") + suffix

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [self._message(message) for message in request.messages],
        }
        if request.tools:
            payload["tools"] = [tool.as_provider_schema() for tool in request.tools]
        payload.update(self._reasoning_parameters(request))
        temperature = (
            request.temperature
            if request.temperature is not None
            else self.settings.temperature
        )
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

    def _message(self, message: Mapping[str, Any]) -> dict[str, Any]:
        role = str(message.get("role") or "user")
        if role == "tool":
            return {
                "role": "tool",
                "tool_call_id": str(
                    message.get("tool_call_id") or message.get("call_id") or ""
                ),
                "content": str(message.get("content") or ""),
            }
        encoded: dict[str, Any] = {
            "role": role,
            "content": openai_chat_content(
                message.get("content", ""),
                supports_images=self.settings.supports_images,
            ),
        }
        calls = message.get("tool_calls")
        if isinstance(calls, list) and calls:
            encoded["tool_calls"] = [self._tool_call(call) for call in calls]
        # The provider fields are returned by the model and replayed unchanged;
        # they never come from model-facing tool arguments.
        replay = message.get("provider_replay")
        if isinstance(replay, Mapping):
            fields = replay.get("message_fields")
            if isinstance(fields, Mapping):
                encoded.update(dict(fields))
        return encoded

    @staticmethod
    def _tool_call(call: object) -> dict[str, Any]:
        if not isinstance(call, Mapping):
            return {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        arguments = call.get("arguments", call.get("args", {}))
        import json

        return {
            "id": call.get("id", call.get("call_id", "")),
            "type": "function",
            "function": {
                "name": call.get("name", ""),
                "arguments": arguments
                if isinstance(arguments, str)
                else json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            },
        }

    def _reasoning_parameters(self, request: ModelRequest) -> dict[str, Any]:
        protocol = self.settings.reasoning_protocol.casefold()
        effort = (
            request.reasoning_effort
            if request.reasoning_effort is not None
            else self.settings.reasoning_effort
        )
        max_output_tokens = (
            request.max_output_tokens
            if request.max_output_tokens is not None
            else self.settings.max_output_tokens
        )
        if effort and effort.casefold() == "none":
            if protocol == "deepseek":
                return {
                    "thinking": {"type": "disabled"},
                    "max_tokens": max_output_tokens,
                }
            return {"max_tokens": max_output_tokens}

        commonstack_chat = protocol == "commonstack" or (
            self.settings.base_url
            and "commonstack.ai" in self.settings.base_url.casefold()
            and protocol != "commonstack_responses"
        )
        output_field = "max_output_tokens" if commonstack_chat else "max_tokens"
        result: dict[str, Any] = {output_field: max_output_tokens}
        if not effort:
            return result
        if protocol == "commonstack":
            result["reasoning"] = {"effort": effort, "summary": "auto"}
        elif protocol == "deepseek":
            result["reasoning_effort"] = effort
            result["thinking"] = {"type": "enabled"}
        elif protocol == "glm":
            result["reasoning_effort"] = effort
            result["thinking"] = {"type": "enabled", "clear_thinking": False}
        elif protocol == "stepfun":
            result["reasoning_effort"] = effort
            result["reasoning_format"] = "general"
        elif protocol == "bailing":
            result["thinking"] = {"type": "enabled"}
        elif protocol == "commonstack_responses":
            raise ValueError(
                "commonstack_responses requires OpenAIResponsesAdapter, not Chat Completions"
            )
        else:
            result["reasoning_effort"] = effort
        return result

    def _parse(self, result) -> ModelResponse:
        data = result.data
        choices = data.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], Mapping)
        ):
            raise ProviderInvocationError("provider response has no choices")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise ProviderInvocationError("provider response has no assistant message")
        raw_calls = message.get("tool_calls")
        calls: list[dict[str, Any]] = []
        if isinstance(raw_calls, list):
            for call in raw_calls:
                if not isinstance(call, Mapping):
                    calls.append({"id": None, "name": "", "arguments": call})
                    continue
                function = call.get("function")
                if isinstance(function, Mapping):
                    name = function.get("name", "")
                    arguments = function.get("arguments", "")
                else:
                    name = call.get("name", "")
                    arguments = call.get("arguments", call.get("args", ""))
                calls.append(
                    {
                        "id": call.get("id", call.get("call_id")),
                        "name": name,
                        "arguments": parse_json_arguments(arguments),
                    }
                )
        provider_fields = {
            key: message[key]
            for key in (
                "reasoning_content",
                "reasoning",
                "reasoning_details",
                "thinking",
                "raw_text",
                "encrypted_content",
                "signature",
            )
            if message.get(key) is not None
        }
        capture = capture_provider_response(data, raw_body=result.raw_body)
        capture["http_status"] = result.status_code
        capture["provider_latency_ms"] = round(result.latency_ms, 3)
        usage = (
            dict(data.get("usage")) if isinstance(data.get("usage"), Mapping) else {}
        )
        if capture.get("reasoning_tokens") is not None:
            usage.setdefault("reasoning_tokens", capture["reasoning_tokens"])
        return ModelResponse(
            text=visible_text(message.get("content")),
            tool_calls=tuple(calls),
            finish_reason=str(choice.get("finish_reason"))
            if choice.get("finish_reason")
            else None,
            usage=usage,
            provider=self.settings.provider,
            raw_capture_ref=result.raw_capture_ref,
            response_sha256=str(capture["provider_response_sha256"]),
            provider_replay={"message_fields": provider_fields},
            provider_metadata=capture,
        )
