"""Main ReAct with the established ILE budgets and repeat/compaction guards."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from plugins.harness.compact_react import CompactReactConfig, CompactReactLoop
from prophet_bench_env import ToolError, ToolResult
from prophet_bench_env.ports.plugins import AgentResult, PluginManifest


class ILELoop(CompactReactLoop):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_batch, self.repeat_count = None, 0

    def _maybe_compact(self, messages):
        config = self.config
        boundary = max(2, len(messages) - config.recent_messages_after_compaction)
        while boundary > 2 and messages[boundary].get("role") == "tool":
            boundary -= 1
        self.config = replace(
            config, recent_messages_after_compaction=max(1, len(messages) - boundary)
        )
        try:
            return super()._maybe_compact(messages)
        finally:
            self.config = config

    def _execute_calls(self, calls):
        submits = [
            c
            for c in calls
            if (d := self.tools.definition(c.name)) and d.side_effect == "submission"
        ]
        if not calls or submits:
            self.last_batch, self.repeat_count = None, 0
            return super()._execute_calls(calls)
        batch = json.dumps(
            [(c.name, c.arguments) for c in calls],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.repeat_count = self.repeat_count + 1 if batch == self.last_batch else 1
        self.last_batch = batch
        if self.repeat_count < 4:
            return super()._execute_calls(calls)
        self._event(
            "repeated_tool_batch",
            {
                "consecutive_repetitions": self.repeat_count,
                "warning_after": 4,
                "abort_after": 8,
                "calls": [{"name": c.name, "arguments": c.arguments} for c in calls],
            },
        )
        if self.repeat_count >= 8:
            raise RuntimeError(
                "solver repeated an identical tool batch eight times without progress"
            )
        message = (
            "This exact tool call has already returned the same research path repeatedly. "
            "Do not retry it unchanged. Use a different query or document, calculate from "
            "evidence already collected, or submit the best supported answer."
        )
        return tuple(
            ToolResult(
                c.call_id,
                c.name,
                False,
                output={
                    "ok": False,
                    "error": "repeated_identical_tool_call",
                    "message": message,
                    "consecutive_repetitions": self.repeat_count,
                },
                error=ToolError("repeated_identical_tool_call", message),
            )
            for c in calls
        )


@dataclass
class ILEAgent:
    model: object
    system_prompt: str

    def manifest(self):
        return PluginManifest("ile-v1-compact-react", "1.0", "harness.v1")

    def run(self, *, task, context, tools, trajectory):
        window = getattr(self.model.settings, "context_window_tokens", None)
        config = CompactReactConfig(
            context_window_characters=window * 4 if window else None,
            compaction_ratio=0.85,
            recent_messages_after_compaction=12,
            max_submit_retries=3,
            max_malformed_corrections=5,
            max_tool_calls_per_response=16,
            wall_clock_seconds=7200,
            enforce_tool_allowlist=True,
        )
        result = ILELoop(
            model=self.model,
            tools=tools,
            trajectory=trajectory,
            config=config,
            cancel_requested=context.cancel_requested,
        ).run(task_prompt=task.prompt, system_prompt=self.system_prompt)
        return AgentResult(
            result.submission,
            result.raw_output,
            {
                "submit_attempts": result.submit_attempts,
                "malformed_corrections": result.malformed_corrections,
            },
        )
