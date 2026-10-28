"""Synchronous financial tools over Harbor's one async trial workspace."""

from __future__ import annotations

import asyncio
import shlex
import tempfile
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path, PurePosixPath
from threading import Event
from typing import BinaryIO

from harbor.environments.base import BaseEnvironment

from ...adapters.e2b_backend import CommandResult


class HarborBackend:
    """Does not create/kill sandboxes. Harbor alone owns those operations.

    Synchronous tools run in a worker thread, with all async sandbox I/O
    scheduled onto Harbor's running event loop. Cancellation closes this port
    so a delayed model response cannot issue new commands after trial timeout.
    """

    def __init__(self, environment: BaseEnvironment, loop: asyncio.AbstractEventLoop):
        self.environment = environment
        self.loop = loop
        self.cancelled = Event()

    def _call(self, coroutine, timeout: float):
        if self.cancelled.is_set():
            coroutine.close()
            raise RuntimeError("trial workspace is closed")
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout)
        except FutureTimeout:
            future.cancel()
            raise TimeoutError("sandbox operation timed out") from None

    def execute(self, command: str, *, timeout: int = 300) -> CommandResult:
        if not 1 <= timeout <= 3600:
            raise ValueError("command timeout must be between 1 and 3600 seconds")
        try:
            result = self._call(
                self.environment.exec(
                    command, cwd="/app/workspace", timeout_sec=timeout
                ),
                timeout + 10,
            )
        except Exception as error:
            if (
                isinstance(error, TimeoutError)
                or type(error).__name__ == "TimeoutException"
            ):
                return CommandResult("", "Command timed out.", 124, timed_out=True)
            raise
        return CommandResult(
            result.stdout or "", result.stderr or "", result.return_code
        )

    def put_file(self, relative_path: str, source: BinaryIO) -> str:
        path = PurePosixPath(relative_path)
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError("download path must remain inside the workspace")
        target = str(PurePosixPath("/app/workspace") / path)
        self._call(
            self.environment.exec(
                "mkdir -p " + shlex.quote(str(PurePosixPath(target).parent)),
                timeout_sec=10,
            ),
            20,
        )
        with tempfile.TemporaryDirectory(prefix="prophet-harbor-upload-") as tmp:
            local = Path(tmp) / "payload"
            with local.open("wb") as output:
                source.seek(0)
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            self._call(self.environment.upload_file(local, target), 190)
        return target

    def close(self, reason: str = "normal") -> None:
        del reason
        self.cancelled.set()
