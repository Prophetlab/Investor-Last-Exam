"""Optional Gymnasium wrapper over :class:`EpisodeEngine`."""

from __future__ import annotations

import json
import string
from typing import Any, ClassVar

import gymnasium as gym
from gymnasium import spaces

from ..core.contracts import Assignment
from ..core.engine import EpisodeEngine


class ProphetBenchGymEnv(gym.Env[dict[str, Any], str]):
    """Generic wrapper using JSON text for dynamic assistant actions."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self, engine: EpisodeEngine, assignment: Assignment) -> None:
        super().__init__()
        self.engine = engine
        self.assignment = assignment
        self.action_space = spaces.Text(
            max_length=1_000_000, min_length=0, charset=string.printable
        )
        self.observation_space = spaces.Dict(
            {
                "step_id": spaces.Discrete(1_000_000),
                "kind": spaces.Text(max_length=128, charset=string.printable),
                "json": spaces.Text(max_length=2_000_000, charset=string.printable),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        episode_seed = (
            seed if seed is not None else int(self.np_random.integers(0, 2**31))
        )
        if options and set(options) - {"assignment"}:
            raise ValueError("unknown reset options")
        assignment = self.assignment
        if options and "assignment" in options:
            assignment = Assignment(**options["assignment"])
        observation, info = self.engine.reset(assignment, seed=episode_seed)
        return _observation_space_value(observation.as_dict()), info

    def step(self, action: str):
        try:
            payload = json.loads(action)
        except json.JSONDecodeError as error:
            payload = {
                "type": "assistant_response",
                "tool_calls": "invalid",
                "text": str(error),
            }
        result = self.engine.step(payload)
        scored = (result.info.get("submission_state") or {}).get(
            "grading_status"
        ) == "completed"
        return (
            _observation_space_value(result.observation.as_dict()),
            result.reward if result.reward is not None else 0.0,
            result.terminated,
            result.truncated,
            {
                **result.info,
                "reward_available": scored,
                "evaluation_reward": result.reward if scored else None,
            },
        )

    def close(self) -> None:
        self.engine.close()


def _observation_space_value(value: dict[str, Any]) -> dict[str, Any]:
    # Runtime identities belong in reset info, not the learned observation.
    value = {key: item for key, item in value.items() if key != "episode_id"}
    return {
        "step_id": int(value.get("step_id", 0)),
        "kind": str(value.get("kind", "")),
        "json": json.dumps(value, ensure_ascii=True, separators=(",", ":")),
    }
