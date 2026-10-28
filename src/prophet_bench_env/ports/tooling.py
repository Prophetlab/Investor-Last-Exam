"""Model-facing tool definitions and a bounded-by-capability tool gateway."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from ..core.audit_context import tool_call_scope
from ..core.contracts import (
    RegisteredTool,
    SubmissionPolicy,
    ToolCall,
    ToolDefinition,
    ToolError,
    ToolResult,
)
from ..core.submission import SubmissionController
from ..core.validation import schema_issues
from .data_specs import (  # noqa: F401 - stable public schema imports
    BASH_DESCRIPTION,
    DATASETS,
    DOWNLOAD_DESCRIPTION,
    READ_DESCRIPTION,
    SEARCH_DESCRIPTION,
    bash_definition,
    default_data_tool_definitions,
    download_definition,
    read_definition,
    search_definition,
)


@dataclass(slots=True)
class ToolGateway:
    """Execute only registered tools; harnesses own per-response batch policy."""

    _tools: dict[str, RegisteredTool]
    submission: SubmissionController = field(default_factory=SubmissionController)

    @classmethod
    def from_tools(
        cls,
        tools: Iterable[RegisteredTool],
        *,
        submission_policy: SubmissionPolicy | None = None,
    ) -> ToolGateway:
        registered: dict[str, RegisteredTool] = {}
        submission_names: list[str] = []
        for tool in tools:
            name = tool.definition.name
            if name in registered:
                raise ValueError(f"duplicate registered tool name: {name}")
            try:
                Draft202012Validator.check_schema(tool.definition.input_schema)
            except SchemaError as error:
                raise ValueError(
                    f"invalid JSON Schema for tool {name}: {error.message}"
                ) from error
            registered[name] = tool
            if tool.definition.side_effect == "submission":
                submission_names.append(name)
        if not registered:
            raise ValueError("at least one tool must be registered")
        if len(submission_names) > 1:
            raise ValueError(
                "only one task submission tool may be registered: "
                + ", ".join(submission_names)
            )
        return cls(registered, SubmissionController(submission_policy))

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._tools.values())

    def definition(self, name: str) -> ToolDefinition | None:
        registered = self._tools.get(name)
        return registered.definition if registered is not None else None

    def execute(
        self, call: ToolCall, *, precheck: ToolError | None = None
    ) -> ToolResult:
        registered = self._tools.get(call.name)
        if registered and registered.definition.side_effect == "submission":
            return self.submission.execute(call, registered, self._execute, precheck)
        return self._execute(call)

    def _execute(self, call: ToolCall) -> ToolResult:
        registered = self._tools.get(call.name)
        if registered is None:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error=ToolError(
                    "unknown_tool",
                    f"Tool {call.name!r} is not enabled for this episode.",
                ),
            )
        if not isinstance(call.arguments, dict):
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error=ToolError(
                    "invalid_arguments", "Tool arguments must be a JSON object."
                ),
            )
        issues, truncated = schema_issues(
            registered.definition.input_schema, call.arguments, limit=8
        )
        if issues:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error=ToolError(
                    "invalid_arguments",
                    "Invalid arguments at "
                    + issues[0]["pointer"]
                    + ": "
                    + issues[0]["message"],
                    details={"issues": issues, "issues_truncated": truncated},
                ),
            )
        try:
            with tool_call_scope(call.call_id):
                result = registered.handler(call)
        except Exception as error:  # noqa: BLE001 - tool errors must not escape the gateway
            code = str(getattr(error, "code", "tool_exception"))
            retryable = bool(getattr(error, "retryable", False))
            trace_id = getattr(error, "trace_id", None)
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error=ToolError(
                    code,
                    str(error) or type(error).__name__,
                    retryable=retryable,
                    trace_id=trace_id,
                ),
                metadata={
                    "failure_classification": "tool_execution_error",
                    "exception_type": type(error).__name__,
                },
            )
        if result.call_id != call.call_id:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error=ToolError(
                    "invalid_tool_result", "Tool returned a mismatched call ID."
                ),
            )
        return result

    def execute_batch(self, calls: Iterable[ToolCall]) -> tuple[ToolResult, ...]:
        """Execute every call independently; batch limits belong to the harness."""

        return tuple(self.execute(call) for call in calls)


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    """Tools plus an optional lifecycle closer for per-rollout resources."""

    tools: tuple[RegisteredTool, ...]
    close: Callable[[], None] | None = None
