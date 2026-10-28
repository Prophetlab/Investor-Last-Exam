"""Customer-independent export port over frozen tasks and collected artifacts."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class DeliverySource:
    tasks_dir: Path
    job_dirs: tuple[Path, ...]
    environment_files: dict[str, Path] = field(default_factory=dict)
    documentation_files: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class DeliveryReceipt:
    path: Path
    sha256: str
    task_count: int
    suggested_upload_path: str


class DeliveryExporter(Protocol):
    def export(self, source: DeliverySource, destination: Path) -> DeliveryReceipt: ...
