"""HTTP facade for a caller-supplied EpisodeEngine factory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..core.contracts import Assignment
from ..core.engine import EpisodeEngine


class CreateEpisodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_id: str = Field(min_length=1)
    task_id: str | None = None
    task_type: str | None = None
    pipeline_id: str | None = None
    seed: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: int = Field(ge=0)
    idempotency_key: str = Field(min_length=1, max_length=200)
    action: dict[str, Any]


@dataclass(slots=True)
class _EpisodeRecord:
    engine: EpisodeEngine
    episode_id: str
    step_id: int = 0
    idempotent: dict[str, dict[str, Any]] = field(default_factory=dict)


EngineFactory = Callable[[CreateEpisodeRequest], EpisodeEngine]


def create_app(engine_factory: EngineFactory | None = None) -> FastAPI:
    app = FastAPI(title="ProphetBenchEnv", version="0.1.0a0")
    records: dict[str, _EpisodeRecord] = {}

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "environment_api_version": "v1",
            "episode_api": ["create", "reset", "step", "close"],
            "tool_batch_limit": None,
            "data_lake_api": "v2",
            "note": "Concrete tools and pipeline IDs are supplied by the configured plugin registry.",
        }

    @app.post("/v1/episodes")
    def create_episode(request: CreateEpisodeRequest) -> dict[str, Any]:
        if engine_factory is None:
            raise HTTPException(
                status_code=503, detail="no pipeline registry is configured"
            )
        engine = engine_factory(request)
        assignment = Assignment(
            benchmark_id=request.benchmark_id,
            task_id=request.task_id,
            task_type=request.task_type,
            pipeline_id=request.pipeline_id,
            seed=request.seed,
            metadata=request.metadata,
        )
        observation, info = engine.reset(assignment, seed=request.seed)
        episode_id = str(info["episode_id"])
        records[episode_id] = _EpisodeRecord(engine=engine, episode_id=episode_id)
        return {
            "episode_id": episode_id,
            "observation": observation.as_dict(),
            "info": info,
        }

    @app.post("/v1/episodes/{episode_id}/reset")
    def reset_episode(episode_id: str, request: CreateEpisodeRequest) -> dict[str, Any]:
        record = records.get(episode_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown episode")
        observation, info = record.engine.reset(
            Assignment(
                benchmark_id=request.benchmark_id,
                task_id=request.task_id,
                task_type=request.task_type,
                pipeline_id=request.pipeline_id,
                seed=request.seed,
                metadata=request.metadata,
            ),
            seed=request.seed,
            episode_id=episode_id,
        )
        record.step_id = observation.step_id
        record.idempotent.clear()
        return {
            "episode_id": episode_id,
            "observation": observation.as_dict(),
            "info": info,
        }

    @app.post("/v1/episodes/{episode_id}/steps")
    def step_episode(episode_id: str, request: StepRequest) -> dict[str, Any]:
        record = records.get(episode_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown episode")
        cached = record.idempotent.get(request.idempotency_key)
        if cached is not None:
            return cached
        if request.step_id != record.step_id:
            raise HTTPException(
                status_code=409, detail=f"expected step_id {record.step_id}"
            )
        result = record.engine.step(request.action)
        record.step_id = result.observation.step_id
        payload = {"episode_id": episode_id, **result.as_dict()}
        record.idempotent[request.idempotency_key] = payload
        return payload

    @app.post("/v1/episodes/{episode_id}/close")
    def close_episode(episode_id: str) -> dict[str, Any]:
        record = records.pop(episode_id, None)
        if record is None:
            return {"episode_id": episode_id, "closed": True, "already_closed": True}
        record.engine.close()
        return {"episode_id": episode_id, "closed": True}

    return app


app = create_app()
