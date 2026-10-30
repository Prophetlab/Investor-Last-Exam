"""Append-only JSONL trajectory sink with a lightweight hash chain."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class JsonlTrajectorySink:
    path: Path
    episode_id: str
    schema_version: str = "trajectory.v1"
    _sequence: int = field(default=0, init=False)
    _previous_hash: str = field(default="", init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = self.path.expanduser().resolve()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        step_id: int | None = None,
        actor: str = "environment",
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            event: dict[str, Any] = {
                "event_id": f"evt_{uuid4().hex}",
                "sequence": self._sequence,
                "episode_id": self.episode_id,
                "step_id": step_id,
                "actor": actor,
                "event_type": event_type,
                "parent_id": parent_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "schema_version": self.schema_version,
                "payload": dict(payload),
                "previous_hash": self._previous_hash,
            }
            encoded = json.dumps(
                event, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            event_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            event["event_hash"] = event_hash
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            self._previous_hash = event_hash
            return event

    def close(self) -> None:
        return None
