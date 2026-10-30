"""Explicit provider protocols, arbitrary model IDs, optional SSE and extra parameters."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import replace

from prophet_bench_env.model_adapters import (
    AnthropicMessagesAdapter,
    OpenAICompatibleChatAdapter,
    OpenAIResponsesAdapter,
)
from prophet_bench_env.model_adapters.errors import EmptyModelResponseError, ProviderHTTPError
from prophet_bench_env.model_adapters.http import (
    ProviderHTTPResponse,
    _empty_terminal,
    _mark_empty_retries,
)
from prophet_bench_env.model_adapters.retry import (
    run_with_empty_response_retries,
    run_with_empty_response_retries_async,
)

from .config import Route
from .streams import AnthropicStreamAssembly, StreamAssembly


class Configurable:
    @staticmethod
    def _empty_exhausted(response, count):
        raise EmptyModelResponseError("Provider returned no answer after bounded retries")

    def _with_empty_retries(self, operation):
        return run_with_empty_response_retries(
            operation,
            is_empty=_empty_terminal,
            on_exhausted=self._empty_exhausted,
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
            on_exhausted=self._empty_exhausted,
            on_success=_mark_empty_retries,
            model_label=self.settings.model,
            max_retries=self.settings.empty_response_retries,
            base_seconds=self.settings.empty_response_base_seconds,
            retry_transport=self._awith_retries,
        )

    def _payload(self, request):
        if not any(m.get("role") in ("system", "developer") for m in request.messages):
            request = replace(
                request,
                messages=(
                    {
                        "role": "system",
                        "content": "Complete the requested task using only the supplied context.",
                    },
                    *request.messages,
                ),
            )
        body = super()._payload(request)
        if self.settings.provider == "openai_responses":
            system = [m for m in body["input"] if m.get("role") in ("system", "developer")]
            if system:
                body["input"] = [m for m in body["input"] if m not in system]
                body["instructions"] = "\n\n".join(
                    m["content"]
                    if isinstance(m.get("content"), str)
                    else "".join(c.get("text", "") for c in m.get("content", []))
                    for m in system
                )
        body.update(deepcopy(self.route.extra_body))
        if request.metadata.get("output_schema"):
            schema = request.metadata["output_schema"]
            if self.settings.provider == "openai_responses":
                body["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "grade",
                        "strict": True,
                        "schema": schema,
                    }
                }
            elif self.settings.provider == "openai_compatible":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "grade", "strict": True, "schema": schema},
                }
        if self.route.stream:
            body["stream"] = True
            if self.settings.provider == "openai_compatible":
                body.setdefault("stream_options", {"include_usage": True})
        return body

    def _headers(self):
        return {**super()._headers(), **self.route.extra_headers}

    def _post_json(self, endpoint, payload, *, call_id):
        if not self.route.stream:
            return super()._post_json(endpoint, payload, call_id=call_id)
        start = time.perf_counter()
        raw = bytearray()
        capture = None
        assembly = (
            AnthropicStreamAssembly()
            if self.settings.provider == "anthropic"
            else StreamAssembly(self.settings.provider == "openai_responses")
        )
        with self.client.stream(
            "POST", endpoint, headers=self._headers(), json=payload
        ) as response:
            if response.is_error:
                body = response.read()
                self._record_attempt(
                    call_id=call_id,
                    endpoint=endpoint,
                    payload=payload,
                    response_body=body,
                    status_code=response.status_code,
                    started=start,
                )
                raise ProviderHTTPError(
                    "Provider request failed",
                    status_code=response.status_code,
                    body=body.decode(errors="replace"),
                )
            if "text/event-stream" not in response.headers.get("content-type", ""):
                body = response.read()
                data = json.loads(body)
                capture = self._record_attempt(
                    call_id=call_id,
                    endpoint=endpoint,
                    payload=payload,
                    response_body=body,
                    status_code=response.status_code,
                    started=start,
                )
                return ProviderHTTPResponse(
                    data, body, capture, response.status_code, (time.perf_counter() - start) * 1000
                )
            event = []
            for line in response.iter_lines():
                raw.extend((line + "\n").encode())
                if len(raw) > 64 * 1024 * 1024:
                    raise ValueError("Provider stream exceeded 64 MiB")
                if time.perf_counter() - start > self.settings.timeout_seconds:
                    raise TimeoutError("Provider stream exceeded request deadline")
                if line.startswith("data:"):
                    event.append(line[5:].lstrip())
                elif not line and event:
                    assembly.add("\n".join(event))
                    event = []
            if event:
                assembly.add("\n".join(event))
            capture = self._record_attempt(
                call_id=call_id,
                endpoint=endpoint,
                payload=payload,
                response_body=bytes(raw),
                status_code=response.status_code,
                started=start,
            )
        return ProviderHTTPResponse(
            assembly.result(), bytes(raw), capture, 200, (time.perf_counter() - start) * 1000
        )


class Chat(Configurable, OpenAICompatibleChatAdapter):
    pass


class Responses(Configurable, OpenAIResponsesAdapter):
    pass


class Messages(Configurable, AnthropicMessagesAdapter):
    pass


def model(route: Route, **kwargs):
    kind = {"openai_compatible": Chat, "openai_responses": Responses, "anthropic": Messages}[
        route.settings.provider
    ]
    adapter = kind(route.settings, **kwargs)
    adapter.route = route
    return adapter
