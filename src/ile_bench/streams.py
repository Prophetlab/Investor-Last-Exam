import json
from copy import deepcopy

from prophet_bench_env.model_adapters.errors import ProviderHTTPError, ProviderInvocationError


class AnthropicStreamAssembly:
    """Native content blocks, including untouched thinking/signature replay."""

    responses = False

    def __init__(self):
        self.body, self.blocks = {}, {}
        self.done = self.terminal = False

    def add(self, event):
        if event == "[DONE]":
            return
        obj = json.loads(event)
        kind = obj.get("type")
        if kind == "error":
            raise ProviderHTTPError("Anthropic stream error", status_code=200, body=obj)
        if kind == "message_start":
            self.body = deepcopy(obj["message"])
        elif kind == "content_block_start":
            self.blocks[obj["index"]] = deepcopy(obj["content_block"])
        elif kind == "content_block_delta":
            block, delta = self.blocks[obj["index"]], obj["delta"]
            field = {
                "text_delta": "text",
                "thinking_delta": "thinking",
                "signature_delta": "signature",
                "input_json_delta": "partial_json",
            }.get(delta["type"])
            if field == "signature":
                # Anthropic's SDK treats this as a full signature update, not
                # a text fragment. Concatenating repeated/replacement signatures
                # would corrupt the opaque value on the next tool-result turn.
                block[field] = delta.get(field, "")
            elif field:
                block[field] = block.get(field, "") + delta.get(field, "")
            else:
                block.setdefault("provider_deltas", []).append(deepcopy(delta))
        elif kind == "message_delta":
            self.body.update(deepcopy(obj.get("delta", {})))
            self.body.setdefault("usage", {}).update(deepcopy(obj.get("usage", {})))
        elif kind == "message_stop":
            self.done = self.terminal = True

    def result(self):
        if not self.done or not self.body:
            raise ConnectionError("Anthropic stream ended without message_stop")
        blocks = []
        for index in sorted(self.blocks):
            block = deepcopy(self.blocks[index])
            if "partial_json" in block:
                raw = block.pop("partial_json")
                # Some Messages routes emit input={} then an empty JSON delta
                # for a no-argument tool. No delta means keep the START object,
                # not replace a legitimate {} with the invalid string "".
                if raw.strip():
                    try:
                        block["input"] = json.loads(raw)
                    except ValueError:
                        block["input"] = raw  # Preserve genuinely invalid JSON for loop correction.
            blocks.append(block)
        return {**self.body, "content": blocks}


class StreamAssembly:
    def __init__(self, responses):
        self.responses = responses
        self.terminal = False
        self.done = False
        self.body = {}
        self.message = {"role": "assistant", "content": ""}
        self.calls = {}
        self.finish = None

    def add(self, event):
        if event == "[DONE]":
            self.done = True
            return
        try:
            obj = json.loads(event)
        except (TypeError, ValueError) as error:
            raise ConnectionError("Provider emitted an invalid SSE JSON event") from error
        if not isinstance(obj, dict):
            raise ConnectionError("Invalid stream event")
        if obj.get("error") or obj.get("type") in {
            "error",
            "response.failed",
            "response.cancelled",
        }:
            details = obj.get("response", obj)
            if not details.get("error"):
                details = {"error": details}
            raise ProviderHTTPError("Provider stream error", status_code=200, body=details)
        if self.responses:
            if obj.get("type") in {"response.completed", "response.incomplete"}:
                self.body = obj["response"]
                self.terminal = True
            return
        for key in (
            "id",
            "model",
            "created",
            "system_fingerprint",
            "service_tier",
            "usage",
        ):
            if obj.get(key) is not None:
                self.body[key] = obj[key]
        for choice in obj.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ProviderInvocationError("Only one completion per ILE request is allowed")
            if choice.get("finish_reason") is not None:
                self.finish = choice["finish_reason"]
                self.terminal = True
            if isinstance(choice.get("message"), dict):
                # Some approved Chat-compatible routes emit complete messages
                # inside SSE. The final native message supersedes its deltas.
                self.message = deepcopy(choice["message"])
                self.calls = {}
            delta = choice.get("delta") or {}
            for key, val in delta.items():
                if val is None or key == "role":
                    continue
                if key == "tool_calls":
                    for part in val:
                        ix = part.get("index", 0)
                        call = self.calls.setdefault(
                            ix,
                            {
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if part.get("id"):
                            if call.get("id") and call["id"] != part["id"]:
                                raise ConnectionError("Stream changed tool call identity")
                            call["id"] = part["id"]
                        for field, fragment in part.get("function", {}).items():
                            if fragment is not None:
                                if field == "name" and call["function"].get(field) == fragment:
                                    continue  # Some compatible endpoints repeat the whole name.
                                call["function"][field] = call["function"].get(field, "") + fragment
                elif isinstance(val, str):
                    self.message[key] = self.message.get(key, "") + val
                elif isinstance(val, list):
                    if self.message.get(key) == "":
                        self.message[key] = []
                    self.message.setdefault(key, []).extend(deepcopy(val))
                else:
                    self.message[key] = deepcopy(val)

    def result(self):
        if not self.terminal:
            raise ConnectionError("Provider stream ended without terminal response")
        if self.responses:
            if not isinstance(self.body.get("output"), list):
                raise ConnectionError("Terminal Responses event omitted native output items")
            return self.body
        if self.calls:
            self.message["tool_calls"] = [self.calls[k] for k in sorted(self.calls)]
        return {
            **self.body,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": self.message, "finish_reason": self.finish}],
        }
