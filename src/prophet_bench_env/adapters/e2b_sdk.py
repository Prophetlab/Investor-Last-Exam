"""E2B SDK 2.x hooks for a network-disabled command backend."""

from __future__ import annotations

import io
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from prophet_bench_env import CommandResult


@dataclass(slots=True)
class E2BSdkHooks:
    """Bind the generic E2B lifecycle port to the installed E2B SDK."""

    api_key: str
    pipeline_id: str | None = None
    sandboxes: dict[str, object] = field(default_factory=dict, init=False)

    @classmethod
    def from_env(
        cls,
        env_file: str | Path = ".env",
        *,
        pipeline_id: str | None = None,
    ) -> E2BSdkHooks:
        values: dict[str, str] = {}
        path = Path(env_file)
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key, value = stripped.split("=", 1)
                    values[key.strip()] = value.strip().strip("'\"")
        values.update(os.environ)
        api_key = values.get("E2B_API_KEY", "").strip()
        if not api_key:
            raise ValueError("E2B_API_KEY is required")
        return cls(api_key, pipeline_id)

    def list_pipeline(self) -> Iterable[str]:
        """List sandboxes left by this pipeline, used once before a launch."""

        if not self.pipeline_id:
            return ()
        from e2b import Sandbox, SandboxQuery

        paginator = Sandbox.list(
            query=SandboxQuery(metadata={"prophetbench_pipeline": self.pipeline_id}),
            api_key=self.api_key,
        )
        result: list[str] = []
        while paginator.has_next:
            result.extend(str(item.sandbox_id) for item in paginator.next_items())
        return tuple(result)

    def list_tagged(self, tag: str) -> Iterable[str]:
        from e2b import Sandbox, SandboxQuery

        paginator = Sandbox.list(
            query=SandboxQuery(metadata={"prophetbench_tag": tag}),
            api_key=self.api_key,
        )
        while paginator.has_next:
            for item in paginator.next_items():
                yield str(item.sandbox_id)

    def create(
        self,
        *,
        tag: str,
        sandbox_timeout: int,
        command_timeout: int,
        allow_internet_access: bool,
        template: str | None = None,
        python_packages: tuple[str, ...] = (),
    ) -> str:
        del command_timeout, python_packages
        from e2b import ALL_TRAFFIC, Sandbox

        metadata = {"prophetbench_tag": tag}
        if self.pipeline_id:
            metadata["prophetbench_pipeline"] = self.pipeline_id
        sandbox = Sandbox.create(
            template=template or "base",
            timeout=sandbox_timeout,
            api_key=self.api_key,
            secure=True,
            allow_internet_access=allow_internet_access,
            network={"deny_out": [ALL_TRAFFIC]} if not allow_internet_access else None,
            metadata=metadata,
        )
        self.sandboxes[str(sandbox.sandbox_id)] = sandbox
        return str(sandbox.sandbox_id)

    def _sandbox(self, sandbox_id: str):
        sandbox = self.sandboxes.get(sandbox_id)
        if sandbox is not None:
            return sandbox
        from e2b import Sandbox

        sandbox = Sandbox.connect(sandbox_id, api_key=self.api_key)
        self.sandboxes[sandbox_id] = sandbox
        return sandbox

    def write_file(
        self, sandbox_id: str, relative_path: str, chunks: Iterable[bytes]
    ) -> None:
        self._sandbox(sandbox_id).files.write(
            relative_path, io.BytesIO(b"".join(chunks))
        )

    def execute(self, sandbox_id: str, command: str, timeout: int) -> CommandResult:
        try:
            result = self._sandbox(sandbox_id).commands.run(command, timeout=timeout)
        except Exception as error:
            if not _is_timeout_error(error):
                raise
            return CommandResult(
                stdout="",
                stderr=f"E2B command timed out after {timeout} seconds.",
                exit_code=124,
                timed_out=True,
            )
        return CommandResult(
            stdout=str(result.stdout or ""),
            stderr=str(result.stderr or ""),
            exit_code=int(result.exit_code),
            timed_out=False,
        )

    def probe(self, sandbox_id: str) -> bool:
        return bool(self._sandbox(sandbox_id).is_running())

    def kill(self, sandbox_id: str) -> bool:
        try:
            return bool(self._sandbox(sandbox_id).kill())
        except Exception:  # noqa: BLE001 - kill is best effort during cleanup
            return False

    def cleanup_pipeline(self) -> tuple[str, ...]:
        """Kill only stale sandboxes bearing this pipeline's explicit tag."""

        killed: list[str] = []
        for sandbox_id in self.list_pipeline():
            if self.kill(sandbox_id):
                killed.append(sandbox_id)
        return tuple(killed)


def _is_timeout_error(error: BaseException) -> bool:
    """Recognize SDK and ConnectRPC deadline errors without pinning internals."""

    if type(error).__name__ == "TimeoutException":
        return True
    if type(error).__name__ != "ConnectError":
        return False
    details = f"{getattr(error, 'code', '')} {error}".casefold()
    return "deadline_exceeded" in details or "deadline exceeded" in details
