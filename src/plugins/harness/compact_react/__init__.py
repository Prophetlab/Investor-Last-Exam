"""BenchResearch hv3.8-compatible Compact ReAct harness plugin."""

from .agent import (
    CompactReactAgent,
    CompactReactHarnessPlugin,
)
from .loop import (
    CompactReactConfig,
    CompactReactError,
    CompactReactLoop,
    CompactReactResult,
    CompactReactTimeoutError,
    InvalidModelResponseError,
    MissingSubmissionError,
    SubmissionRetriesExceededError,
    UndeclaredToolError,
)

__all__ = [
    "CompactReactAgent",
    "CompactReactConfig",
    "CompactReactError",
    "CompactReactHarnessPlugin",
    "CompactReactLoop",
    "CompactReactResult",
    "CompactReactTimeoutError",
    "InvalidModelResponseError",
    "MissingSubmissionError",
    "SubmissionRetriesExceededError",
    "UndeclaredToolError",
]
