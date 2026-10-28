"""Native OpenAI Responses adapter with exact reasoning-item replay."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from prophet_bench_env.ports.model import ModelRequest, ModelResponse

from .capture import capture_provider_response, parse_json_arguments, visible_text
from .errors import ProviderInvocationError
from .http import HTTPModelAdapter
from .multimodal import openai_responses_content


class OpenAIResponsesAdapter(HTTPModelAdapter):
    """Call a native ``/responses`` route (OpenAI or a compatible gateway)."""

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
        result = self._post_json(self._endpoint("/responses"), payload, call_id=call_id)
        return self._parse(result)

    async def _agenerate_once(
        self, payload: dict[str, Any], call_id: str
    ) -> ModelResponse:
        result = await self._apost_json(
            self._endpoint("/responses"), payload, call_id=call_id
        )
        return self._parse(result)

    def _endpoint(self, suffix: str) -> str:
        base = self.settings.base_url or "https://api.openai.com/v1"
        return base.rstrip("/") + suffix

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "input": self._input(request.messages),
        }
        if request.tools:
            payload["tools"] = [self._tool(tool) for tool in request.tools]
        payload["max_output_tokens"] = (
            request.max_output_tokens
            if request.max_output_tokens is not None
            else self.settings.max_output_tokens
        )
        temperature = (
            request.temperature
            if request.temperature is not None
            else self.settings.temperature
        )
        if temperature is not None:
            payload["temperature"] = temperature
        effort = (
            request.reasoning_effort
            if request.reasoning_effort is not None
            else self.settings.reasoning_effort
        )
        if effort and effort.casefold() != "none":
            payload["reasoning"] = {
                "effort": effort,
                "summary": "auto",
            }
            payload["include"] = ["reasoning.encrypted_content"]
        if self.settings.store is not None:
            payload["store"] = self.settings.store
        elif effort:
            # hv3.8 used stateless Responses calls so opaque reasoning state is
            # carried explicitly in the next input.
            payload["store"] = False
        return payload

    @staticmethod
    def _tool(tool) -> dict[str, Any]:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }

    def _input(self, messages: tuple[Mapping[str, Any], ...]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            if role == "tool":
                result.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(
                            message.get("tool_call_id") or message.get("call_id") or ""
                        ),
                        "output": str(message.get("content") or ""),
                    }
                )
                continue
            if role == "assistant":
                replay = message.get("provider_replay")
                if isinstance(replay, Mapping) and isinstance(
                    replay.get("output_items"), list
                ):
                    result.extend(
                        dict(item)
                        for item in replay["output_items"]
                        if isinstance(item, Mapping)
                    )
                    continue
                # Fallback for a conversation originating from another adapter.
                if message.get("content"):
                    result.append(
                        {"role": "assistant", "content": message.get("content")}
                    )
                for call in message.get("tool_calls", []):
                    if not isinstance(call, Mapping):
                        continue
                    arguments = call.get("arguments", call.get("args", {}))
                    result.append(
                        {
                            "type": "function_call",
                            "call_id": call.get("id", call.get("call_id", "")),
                            "name": call.get("name", ""),
                            "arguments": arguments
                            if isinstance(arguments, str)
                            else json.dumps(
                                arguments, ensure_ascii=False, separators=(",", ":")
                            ),
                        }
                    )
                continue
            result.append(
                {
                    "role": role,
                    "content": openai_responses_content(
                        message.get("content", ""),
                        supports_images=self.settings.supports_images,
                    ),
                }
            )
        return result

    def _parse(self, result) -> ModelResponse:
        data = result.data
        output = data.get("output")
        if not isinstance(output, list):
            raise ProviderInvocationError("Responses response has no output")
        calls: list[dict[str, Any]] = []
        visible_parts: list[str] = []
        for item in output:
            if not isinstance(item, Mapping):
                continue
            kind = item.get("type")
            if kind == "function_call":
                calls.append(
                    {
                        "id": item.get("call_id", item.get("id")),
                        "name": item.get("name", ""),
                        "arguments": parse_json_arguments(item.get("arguments", "")),
                    }
                )
            elif kind == "message":
                visible_parts.append(visible_text(item.get("content")))
            elif kind == "output_text":
                visible_parts.append(str(item.get("text") or ""))
        capture = capture_provider_response(data, raw_body=result.raw_body)
        capture["http_status"] = result.status_code
        capture["provider_latency_ms"] = round(result.latency_ms, 3)
        usage = (
            dict(data.get("usage")) if isinstance(data.get("usage"), Mapping) else {}
        )
        if capture.get("reasoning_tokens") is not None:
            usage.setdefault("reasoning_tokens", capture["reasoning_tokens"])
        return ModelResponse(
            text="".join(visible_parts),
            tool_calls=tuple(calls),
            finish_reason=str(data.get("status")) if data.get("status") else None,
            usage=usage,
            provider=self.settings.provider,
            raw_capture_ref=result.raw_capture_ref,
            response_sha256=str(capture["provider_response_sha256"]),
            provider_replay={
                "output_items": [
                    dict(item) for item in output if isinstance(item, Mapping)
                ]
            },
            provider_metadata=capture,
        )
