"""Workspace backends used by download and future bash tools."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import uuid4


class SandboxBackend(Protocol):
    def put_file(self, relative_path: str, source: BinaryIO) -> str: ...

    def close(self, reason: str = "normal") -> None: ...


@dataclass(slots=True)
class LocalSandbox:
    root: Path
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.root = self.root.expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def put_file(self, relative_path: str, source: BinaryIO) -> str:
        if self._closed:
            raise RuntimeError("sandbox is closed")
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("sandbox path must remain inside the workspace")
        target = self.root / path
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            source.seek(0)
            with temporary.open("wb") as handle:
                while chunk := source.read(1024 * 1024):
                    handle.write(chunk)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return str(target)

    def close(self, reason: str = "normal") -> None:
        del reason
        self._closed = True


@dataclass(slots=True)
class StreamingSandboxAdapter:
    """Generic adapter for E2B or another remote backend.

    ``writer`` receives a relative path and an iterator of byte chunks. Keeping this callback based
    avoids a hard dependency on one E2B SDK version in the environment core.
    """

    writer: Callable[[str, Iterable[bytes]], None]
    closer: Callable[[str], None] | None = None
    _closed: bool = field(default=False, init=False)

    def put_file(self, relative_path: str, source: BinaryIO) -> str:
        if self._closed:
            raise RuntimeError("sandbox is closed")
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("sandbox path must be relative")
        source.seek(0)
        self.writer(str(path), iter(lambda: source.read(1024 * 1024), b""))
        return str(path)

    def close(self, reason: str = "normal") -> None:
        if not self._closed and self.closer is not None:
            self.closer(reason)
        self._closed = True
