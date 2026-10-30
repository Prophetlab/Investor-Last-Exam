"""A provider-neutral port of the BenchResearch hv3.8 Compact ReAct loop.

The loop preserves the useful hv3.8 semantics while using ProphetBenchEnv's canonical model and
tool contracts. Research rounds are not capped, while one malformed or unbounded assistant batch
cannot expand into an arbitrary number of tool executions.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from prophet_bench_env.core.audit_context import subagent_scope
from prophet_bench_env.core.contracts import (
    SubmissionPolicy,
    ToolCall,
    ToolError,
    ToolResult,
)
from prophet_bench_env.core.submission import received
from prophet_bench_env.core.trajectory import JsonlTrajectorySink
from prophet_bench_env.ports.data_specs import DATA_TOOL_NAMES
from prophet_bench_env.ports.model import ModelAdapter, ModelRequest, ModelResponse
from prophet_bench_env.ports.tooling import ToolGateway


class CompactReactError(RuntimeError):
    pass


class MissingSubmissionError(CompactReactError):
    pass


class SubmissionRetriesExceededError(CompactReactError):
    pass


class InvalidModelResponseError(CompactReactError):
    pass


class CompactReactTimeoutError(CompactReactError):
    pass


class UndeclaredToolError(CompactReactError):
    pass


@dataclass(frozen=True, slots=True)
class CompactReactConfig:
    context_window_characters: int | None = None
    compaction_ratio: float = 0.70
    recent_messages_after_compaction: int = 8
    max_submit_retries: int = 3
    max_malformed_corrections: int = 3
    max_tool_calls_per_response: int = 16
    wall_clock_seconds: float | None = None
    enforce_tool_allowlist: bool = False
    allowed_data_tools: tuple[str, ...] = DATA_TOOL_NAMES

    def __post_init__(self) -> None:
        if (
            self.context_window_characters is not None
            and self.context_window_characters < 1
        ):
            raise ValueError("context_window_characters must be positive")
        if not 0 < self.compaction_ratio < 1:
            raise ValueError("compaction_ratio must be between 0 and 1")
        if self.recent_messages_after_compaction < 1:
            raise ValueError("recent_messages_after_compaction must be positive")
        if self.max_submit_retries < 1 or self.max_malformed_corrections < 1:
            raise ValueError("retry limits must be positive")
        if self.max_tool_calls_per_response < 1:
            raise ValueError("max_tool_calls_per_response must be positive")
        if self.wall_clock_seconds is not None and self.wall_clock_seconds <= 0:
            raise ValueError("wall_clock_seconds must be positive")


@dataclass(frozen=True, slots=True)
class CompactReactResult:
    submission: Any
    messages: tuple[dict[str, Any], ...]
    raw_output: str | None
    malformed_corrections: int
    submit_attempts: int


MALFORMED_CORRECTION = (
    "Runtime correction: the previous assistant response contained malformed tool calls and was "
    "discarded. Send one corrected response with valid JSON tool calls."
)


class CompactReactLoop:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolGateway,
        trajectory: JsonlTrajectorySink | None = None,
        config: CompactReactConfig | None = None,
        cancel_requested=None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.trajectory = trajectory
        self.config = config or CompactReactConfig()
        if not tools.submission.explicit_policy:
            tools.submission.configure(
                SubmissionPolicy(max_attempts=self.config.max_submit_retries),
                explicit=False,
            )
        self.cancel_requested = cancel_requested
        if self.config.enforce_tool_allowlist:
            self._check_tool_allowlist()

    def _check_tool_allowlist(self) -> None:
        extras = [
            definition.name
            for definition in self.tools.definitions()
            if definition.name not in self.config.allowed_data_tools
            and definition.side_effect != "submission"
        ]
        if extras:
            raise UndeclaredToolError(
                "managed harness registered tools outside the environment capability set: "
                + ", ".join(sorted(extras))
            )

    def run(
        self, *, task_prompt: str, system_prompt: str | None = None
    ) -> CompactReactResult:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": task_prompt})
        self._event("task_presented", {"has_system_prompt": bool(system_prompt)})
        deadline = (
            time.monotonic() + self.config.wall_clock_seconds
            if self.config.wall_clock_seconds is not None
            else None
        )
        return self.resume(messages, deadline=deadline)

    def resume(
        self,
        messages: list[dict[str, Any]],
        *,
        deadline: float | None = None,
    ) -> CompactReactResult:
        messages = list(messages)
        malformed_corrections = 0
        raw_output: str | None = None
        while True:
            self._check_deadline(deadline)
            messages = self._maybe_compact(messages)
            request = ModelRequest(
                messages=tuple(messages),
                tools=self.tools.definitions(),
            )
            self._event(
                "model_request",
                {
                    "messages": list(request.messages),
                    "tools": [item.as_provider_schema() for item in request.tools],
                },
                actor="runtime",
            )
            response = self.model.generate(request)
            tool_call_count = len(response.tool_calls)
            if tool_call_count > self.config.max_tool_calls_per_response:
                self._event(
                    "invalid_model_response",
                    {
                        "reason": "tool_call_limit_exceeded",
                        "tool_call_count": tool_call_count,
                        "max_tool_calls": self.config.max_tool_calls_per_response,
                        "provider_response_sha256": response.response_sha256,
                        "discarded_response": {
                            **_model_response_event(response, []),
                            "raw_tool_calls": list(response.tool_calls),
                        },
                    },
                    actor="runtime",
                )
                raise InvalidModelResponseError(
                    f"assistant returned {tool_call_count} tool calls; maximum is "
                    f"{self.config.max_tool_calls_per_response}"
                )
            calls, malformed_reason = _parse_tool_calls(
                response,
                submission_names={
                    d.name
                    for d in self.tools.definitions()
                    if d.side_effect == "submission"
                },
            )
            self._check_deadline(deadline)
            if malformed_reason is not None:
                malformed_corrections += 1
                self._event(
                    "malformed_tool_batch",
                    {
                        "reason": malformed_reason,
                        "correction_attempt": malformed_corrections,
                        "discarded_response": {
                            **_model_response_event(response, []),
                            "raw_tool_calls": list(response.tool_calls),
                        },
                    },
                    actor="runtime",
                )
                if malformed_corrections > self.config.max_malformed_corrections:
                    raise InvalidModelResponseError(
                        "assistant repeatedly returned malformed tool calls"
                    )
                correction = {"role": "user", "content": MALFORMED_CORRECTION}
                messages.append(correction)
                self._event("correction_message", correction, actor="runtime")
                continue

            assistant_message = _assistant_message(response, calls)
            messages.append(assistant_message)
            self._event(
                "model_response",
                _model_response_event(response, calls),
                actor="model",
            )
            for call in calls:
                self._event(
                    "tool_request",
                    {"name": call.name, "arguments": call.arguments},
                    actor="model",
                    parent_id=call.call_id,
                )
            raw_output = response.text.strip() or raw_output
            if not calls:
                raise MissingSubmissionError(
                    "model ended without calling the task submit tool"
                )

            submission_indices = [
                index
                for index, call in enumerate(calls)
                if (definition := self.tools.definition(call.name)) is not None
                and definition.side_effect == "submission"
            ]
            if submission_indices and len(calls) != 1:
                ordinary_calls = [
                    call
                    for index, call in enumerate(calls)
                    if index not in submission_indices
                ]
                ordinary_results = self._execute_calls(ordinary_calls)
                self._check_deadline(deadline)
                ordinary_by_id = {result.call_id: result for result in ordinary_results}
                results = tuple(
                    ordinary_by_id[call.call_id]
                    if call.call_id in ordinary_by_id
                    else self.tools.execute(
                        call,
                        precheck=ToolError(
                            "exclusive_tool_required",
                            "The task submit tool must be called alone.",
                        ),
                    )
                    for call in calls
                )
            else:
                results = self._execute_calls(calls)
                self._check_deadline(deadline)
            for call, result in zip(calls, results, strict=True):
                payload = result.as_dict()
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "name": call.name,
                        "content": json.dumps(
                            payload, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                )
                self._event(
                    "tool_result", payload, actor="tool", parent_id=call.call_id
                )

            if not submission_indices:
                continue
            if len(calls) != 1:
                self._event(
                    "submission_rejected",
                    {
                        "reason": "submission_must_be_exclusive",
                        "call_count": len(calls),
                    },
                    actor="runtime",
                )
            submit_result = next(
                (results[i] for i in submission_indices if received(results[i])),
                results[submission_indices[-1]],
            )
            if received(submit_result):
                payload = submit_result.output
                self._event(
                    "submission_received",
                    {"payload": payload, **submit_result.metadata["submission"]},
                    actor="runtime",
                )
                return CompactReactResult(
                    submission=payload,
                    messages=tuple(messages),
                    raw_output=raw_output,
                    malformed_corrections=malformed_corrections,
                    submit_attempts=self.tools.submission.attempts,
                )
            if submit_result.metadata.get("submission", {}).get("terminal"):
                raise SubmissionRetriesExceededError(
                    f"task submit tool closed after {self.tools.submission.attempts} attempts"
                )

    def _check_deadline(self, deadline: float | None) -> None:
        if self.cancel_requested is not None and self.cancel_requested():
            import asyncio

            raise asyncio.CancelledError("trial cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            self._event(
                "episode_truncated",
                {"reason": "wall_clock_timeout"},
                actor="runtime",
            )
            raise CompactReactTimeoutError("Compact ReAct wall-clock limit reached")

    def _execute_calls(self, calls: list[ToolCall]) -> tuple[ToolResult, ...]:
        parallel = len(calls) > 1 and all(
            (definition := self.tools.definition(call.name)) is not None
            and definition.parallel_safe
            for call in calls
        )
        if not parallel:
            return tuple(self.tools.execute(call) for call in calls)
        with ThreadPoolExecutor(
            max_workers=len(calls), thread_name_prefix="prophet-tools"
        ) as pool:
            futures = [pool.submit(self.tools.execute, call) for call in calls]
            return tuple(future.result() for future in futures)

    def _maybe_compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        limit = self.config.context_window_characters
        if limit is None:
            return messages
        encoded = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))
        trigger = int(limit * self.config.compaction_ratio)
        if (
            encoded < trigger
            or len(messages) <= self.config.recent_messages_after_compaction + 2
        ):
            return messages
        prefix = messages[:2]
        recent = messages[-self.config.recent_messages_after_compaction :]
        removed = messages[len(prefix) : -self.config.recent_messages_after_compaction]
        if not removed:
            return messages
        compaction_request = ModelRequest(
            messages=(
                {
                    "role": "user",
                    "content": "Summarize this completed research for continuation. Preserve facts, units, periods, calculations, document references, and unresolved questions; do not add analysis.\n\n"
                    + json.dumps(removed, ensure_ascii=False, separators=(",", ":")),
                },
            ),
            tools=(),
        )
        compaction_id = "compaction-" + uuid4().hex
        audit_link = {
            "compaction_id": compaction_id,
            "model_name": getattr(getattr(self.model, "settings", None), "model", None),
        }
        self._event(
            "compaction_model_request",
            {
                "messages": list(compaction_request.messages),
                **audit_link,
            },
            actor="compactor",
        )
        try:
            with subagent_scope(compaction_id):
                summary_response = self.model.generate(compaction_request)
        except Exception as error:
            self._event(
                "compaction_model_error",
                {**audit_link, "error_type": type(error).__name__},
                actor="compactor",
            )
            raise
        self._event(
            "compaction_model_response",
            {
                **_model_response_event(summary_response, []),
                **audit_link,
            },
            actor="compactor",
        )
        summary = summary_response.text.strip()
        compacted = [
            *prefix,
            {"role": "system", "content": f"Earlier research summary:\n{summary}"},
            *recent,
        ]
        self._event(
            "context_compaction",
            {
                "compaction_id": compaction_id,
                "messages_after_compaction": compacted,
                "copied_message_indices": list(range(len(prefix)))
                + list(range(len(prefix) + 1, len(compacted))),
                "summary_message_index": len(prefix),
                "messages_before": len(messages),
                "messages_after": len(compacted),
                "characters_before": encoded,
                "characters_after": len(json.dumps(compacted, ensure_ascii=False)),
                "removed_messages": removed,
                "summary": summary,
                "raw_capture_ref": summary_response.raw_capture_ref,
                "usage": summary_response.usage,
            },
        )
        return compacted

    def _event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        actor: str = "runtime",
        parent_id: str | None = None,
    ) -> None:
        if self.trajectory is not None:
            self.trajectory.append(
                event_type,
                payload,
                actor=actor,
                parent_id=parent_id,
            )


def _parse_tool_calls(
    response: ModelResponse, *, submission_names=()
) -> tuple[list[ToolCall], str | None]:
    calls: list[ToolCall] = []
    seen: set[str] = set()
    for raw in response.tool_calls:
        if not isinstance(raw, Mapping):
            return [], "tool call must be an object"
        call_id = raw.get("id", raw.get("call_id"))
        name = raw.get("name")
        arguments = raw.get("arguments", raw.get("args", {}))
        if not isinstance(call_id, str) or not call_id:
            return [], "missing tool call id"
        if call_id in seen:
            return [], "duplicate tool call id"
        if not isinstance(name, str) or not name:
            return [], "missing tool name"
        if not isinstance(arguments, Mapping) and name not in submission_names:
            return [], "tool arguments must be a JSON object"
        seen.add(call_id)
        calls.append(
            ToolCall(
                call_id,
                name,
                dict(arguments) if isinstance(arguments, Mapping) else arguments,
            )
        )
    return calls, None


def _assistant_message(
    response: ModelResponse, calls: list[ToolCall]
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response.text,
        "tool_calls": [
            {"id": call.call_id, "name": call.name, "arguments": call.arguments}
            for call in calls
        ],
    }
    # Keep the opaque provider state needed to continue reasoning across tool
    # turns.  The adapter, rather than the harness, decides how to replay it.
    if response.provider_replay:
        message["provider_replay"] = response.provider_replay
    return message


def _model_response_event(
    response: ModelResponse, calls: list[ToolCall]
) -> dict[str, Any]:
    return {
        "content": response.text,
        "tool_calls": [
            {"id": call.call_id, "name": call.name, "arguments": call.arguments}
            for call in calls
        ],
        "finish_reason": response.finish_reason,
        "usage": response.usage,
        "provider": response.provider,
        "provider_metadata": response.provider_metadata,
        "raw_capture_ref": response.raw_capture_ref,
        "response_sha256": response.response_sha256,
    }
