"""stdin/stdout JSONL transport using the same EpisodeEngine as HTTP."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, TextIO

from ..core.contracts import Assignment
from ..core.engine import EpisodeEngine
from .server import CreateEpisodeRequest


@dataclass(slots=True)
class JsonlRunner:
    engine_factory: Callable[[CreateEpisodeRequest], EpisodeEngine]
    records: dict[str, EpisodeEngine] = field(default_factory=dict)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        if operation == "create":
            config = CreateEpisodeRequest.model_validate(request)
            engine = self.engine_factory(config)
            observation, info = engine.reset(
                Assignment(
                    benchmark_id=config.benchmark_id,
                    task_id=config.task_id,
                    task_type=config.task_type,
                    pipeline_id=config.pipeline_id,
                    seed=config.seed,
                    metadata=config.metadata,
                ),
                seed=config.seed,
            )
            episode_id = str(info["episode_id"])
            self.records[episode_id] = engine
            return {
                "episode_id": episode_id,
                "observation": observation.as_dict(),
                "info": info,
            }
        episode_id = str(request.get("episode_id", ""))
        engine = self.records.get(episode_id)
        if engine is None:
            raise ValueError("unknown episode")
        if operation == "step":
            result = engine.step(request.get("action", {}))
            return {"episode_id": episode_id, **result.as_dict()}
        if operation == "close":
            engine.close()
            del self.records[episode_id]
            return {"episode_id": episode_id, "closed": True}
        raise ValueError("op must be create, step, or close")

    def run(self, source: Iterable[str], target: TextIO = sys.stdout) -> None:
        for line in source:
            if not line.strip():
                continue
            try:
                response = {"ok": True, **self.handle(json.loads(line))}
            except Exception as error:  # noqa: BLE001 - transport must continue after one bad line
                response = {
                    "ok": False,
                    "error": {"code": "invalid_request", "message": str(error)},
                }
            target.write(
                json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            target.flush()


def run_jsonl(
    engine_factory: Callable[[CreateEpisodeRequest], EpisodeEngine],
    source: Iterable[str] = sys.stdin,
    target: TextIO = sys.stdout,
) -> None:
    JsonlRunner(engine_factory).run(source, target)
