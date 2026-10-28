"""Trusted, pluggable per-trial composition, independent of customer delivery."""

from collections.abc import Callable
from dataclasses import dataclass

from ...ports.plugins import Agent
from ...ports.tooling import ToolRegistration


@dataclass
class TrialRuntime:
    agent: Agent
    tools: ToolRegistration
    close_model: Callable[[], None] | None = None

    def close(self) -> None:
        try:
            if self.tools.close:
                self.tools.close()
        finally:
            if self.close_model:
                self.close_model()
