"""E2B workspace/command backend with an SDK-independent lifecycle.

The hooks adapter is deliberately tiny so the RL environment owns safety and cleanup while a
caller can bind the exact E2B SDK version used by its deployment.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Protocol

from ..core.trajectory import JsonlTrajectorySink

# The exact hv3.8 bash environment. These are baked into the E2B template and
# verified after every create/recycle; runtime commands never pip-install them.
DEFAULT_PYTHON_PACKAGES = ("numpy", "scipy", "pandas", "openpyxl")
PACKAGE_PROBE_TIMEOUT_SECONDS = 60


@dataclass(frozen=True, slots=True)
class CommandResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False


class E2BHooks(Protocol):
    def list_tagged(self, tag: str) -> Iterable[str]: ...

    def create(
        self,
        *,
        tag: str,
        sandbox_timeout: int,
        command_timeout: int,
        allow_internet_access: bool,
        template: str | None = None,
        python_packages: tuple[str, ...] = DEFAULT_PYTHON_PACKAGES,
    ) -> str: ...

    def write_file(
        self, sandbox_id: str, relative_path: str, chunks: Iterable[bytes]
    ) -> None: ...

    def execute(self, sandbox_id: str, command: str, timeout: int) -> CommandResult: ...

    def probe(self, sandbox_id: str) -> bool: ...

    def kill(self, sandbox_id: str) -> bool: ...


LifecycleWriter = Callable[[str, dict[str, object]], None]


@dataclass(slots=True)
class E2BBackend:
    hooks: E2BHooks
    sandbox_id: str
    tag: str
    sandbox_timeout: int = 1800
    command_timeout: int = 300
    lifecycle: LifecycleWriter | None = None
    trajectory: JsonlTrajectorySink | None = None
    template: str | None = None
    python_packages: tuple[str, ...] = DEFAULT_PYTHON_PACKAGES
    kill_attempts: int = 2
    recycle_attempts: int = 3
    _closed: bool = field(default=False, init=False)
    _resource_lease: AbstractContextManager[None] | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def create(
        cls,
        hooks: E2BHooks,
        *,
        tag: str,
        sandbox_timeout: int = 1800,
        command_timeout: int = 300,
        template: str | None = None,
        python_packages: tuple[str, ...] = DEFAULT_PYTHON_PACKAGES,
        lifecycle: LifecycleWriter | None = None,
        trajectory: JsonlTrajectorySink | None = None,
        resource_guard: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> E2BBackend:
        if sandbox_timeout < 1 or command_timeout < 1:
            raise ValueError("E2B timeouts must be positive")
        _validate_python_packages(python_packages)
        lease = resource_guard() if resource_guard is not None else nullcontext()
        lease.__enter__()
        try:
            stale = tuple(str(item) for item in hooks.list_tagged(tag))
            for stale_id in stale:
                _kill_with_retry(
                    hooks,
                    stale_id,
                    attempts=2,
                    emit=lambda action, payload: _emit(
                        lifecycle, trajectory, action, payload
                    ),
                )
            sandbox_id = hooks.create(
                tag=tag,
                sandbox_timeout=sandbox_timeout,
                command_timeout=command_timeout,
                allow_internet_access=False,
                template=template,
                python_packages=python_packages,
            )
        except BaseException:
            lease.__exit__(None, None, None)
            raise
        _emit(
            lifecycle,
            trajectory,
            "sandbox_create",
            {
                "sandbox_id": sandbox_id,
                "tag": tag,
                "sandbox_timeout": sandbox_timeout,
                "command_timeout": command_timeout,
                "allow_internet_access": False,
                "template": template,
                "python_packages": list(python_packages),
            },
        )
        try:
            _verify_python_packages(
                hooks,
                sandbox_id,
                python_packages,
                timeout=min(command_timeout, PACKAGE_PROBE_TIMEOUT_SECONDS),
                emit=lambda action, payload: _emit(
                    lifecycle, trajectory, action, payload
                ),
            )
        except BaseException:
            lease.__exit__(None, None, None)
            raise
        result = cls(
            hooks=hooks,
            sandbox_id=sandbox_id,
            tag=tag,
            sandbox_timeout=sandbox_timeout,
            command_timeout=command_timeout,
            lifecycle=lifecycle,
            trajectory=trajectory,
            template=template,
            python_packages=python_packages,
        )
        result._resource_lease = lease
        return result

    def put_file(self, relative_path: str, source: BinaryIO) -> str:
        path = _safe_path(relative_path)
        if self._closed:
            raise RuntimeError("E2B sandbox is closed")
        source.seek(0)
        self.hooks.write_file(
            self.sandbox_id,
            path,
            iter(lambda: source.read(1024 * 1024), b""),
        )
        _emit(
            self.lifecycle,
            self.trajectory,
            "sandbox_write",
            {"sandbox_id": self.sandbox_id, "path": path},
        )
        return path

    def execute(self, command: str, *, timeout: int | None = None) -> CommandResult:
        if self._closed:
            raise RuntimeError("E2B sandbox is closed")
        command_timeout = self.command_timeout if timeout is None else timeout
        if not 1 <= command_timeout <= self.command_timeout:
            raise ValueError(
                f"command timeout must be between 1 and {self.command_timeout} seconds"
            )
        result = self.hooks.execute(self.sandbox_id, command, command_timeout)
        if result.timed_out or result.exit_code == 124:
            _emit(
                self.lifecycle,
                self.trajectory,
                "sandbox_timeout",
                {
                    "sandbox_id": self.sandbox_id,
                    "timeout_seconds": command_timeout,
                },
            )
            try:
                healthy = bool(self.hooks.probe(self.sandbox_id))
            except Exception:  # noqa: BLE001 - probe failure means the sandbox is unhealthy
                healthy = False
            if not healthy:
                _emit(
                    self.lifecycle,
                    self.trajectory,
                    "sandbox_probe",
                    {"sandbox_id": self.sandbox_id, "success": False},
                )
                result = self._recycle_and_retry(command, command_timeout)
            else:
                _emit(
                    self.lifecycle,
                    self.trajectory,
                    "sandbox_probe",
                    {"sandbox_id": self.sandbox_id, "success": True},
                )
        return result

    def close(self, reason: str = "normal") -> None:
        if self._closed:
            return
        cleanup_error: Exception | None = None
        try:
            _kill_with_retry(
                self.hooks,
                self.sandbox_id,
                attempts=self.kill_attempts,
                emit=lambda action, payload: _emit(
                    self.lifecycle, self.trajectory, action, payload
                ),
                reason=reason,
            )
        except Exception as error:  # noqa: BLE001 - tagged cleanup still runs
            cleanup_error = error
        try:
            try:
                tagged = tuple(str(item) for item in self.hooks.list_tagged(self.tag))
                for sandbox_id in tagged:
                    if sandbox_id != self.sandbox_id:
                        _kill_with_retry(
                            self.hooks,
                            sandbox_id,
                            attempts=self.kill_attempts,
                            emit=lambda action, payload: _emit(
                                self.lifecycle, self.trajectory, action, payload
                            ),
                            reason="final_tagged_cleanup",
                        )
            except Exception as error:  # noqa: BLE001 - lease release is mandatory
                if cleanup_error is None:
                    cleanup_error = error
            _emit(
                self.lifecycle,
                self.trajectory,
                "sandbox_cleanup",
                {"sandbox_id": self.sandbox_id, "tag": self.tag, "reason": reason},
            )
        finally:
            self._closed = True
            if self._resource_lease is not None:
                self._resource_lease.__exit__(None, None, None)
                self._resource_lease = None
        if cleanup_error is not None:
            raise cleanup_error

    def _recycle_and_retry(self, command: str, timeout: int) -> CommandResult:
        old_id = self.sandbox_id
        _kill_with_retry(
            self.hooks,
            old_id,
            attempts=self.kill_attempts,
            emit=lambda action, payload: _emit(
                self.lifecycle, self.trajectory, action, payload
            ),
            reason="timeout_recycle",
        )
        last_error: Exception | None = None
        for attempt in range(1, self.recycle_attempts + 1):
            try:
                replacement = self.hooks.create(
                    tag=self.tag,
                    sandbox_timeout=self.sandbox_timeout,
                    command_timeout=self.command_timeout,
                    allow_internet_access=False,
                    template=self.template,
                    python_packages=self.python_packages,
                )
                self.sandbox_id = replacement
                _emit(
                    self.lifecycle,
                    self.trajectory,
                    "sandbox_recycle",
                    {
                        "previous_sandbox_id": old_id,
                        "sandbox_id": replacement,
                        "attempt": attempt,
                        "allow_internet_access": False,
                        "template": self.template,
                        "python_packages": list(self.python_packages),
                    },
                )
                _verify_python_packages(
                    self.hooks,
                    replacement,
                    self.python_packages,
                    timeout=min(self.command_timeout, PACKAGE_PROBE_TIMEOUT_SECONDS),
                    emit=lambda action, payload: _emit(
                        self.lifecycle, self.trajectory, action, payload
                    ),
                )
                return self.hooks.execute(replacement, command, timeout)
            except Exception as error:  # noqa: BLE001 - bounded recycle loop
                last_error = error
                _emit(
                    self.lifecycle,
                    self.trajectory,
                    "sandbox_recycle_failed",
                    {"previous_sandbox_id": old_id, "attempt": attempt},
                )
        raise RuntimeError(
            "E2B sandbox recycle failed after bounded attempts"
        ) from last_error


def _kill_with_retry(
    hooks: E2BHooks,
    sandbox_id: str,
    *,
    attempts: int,
    emit: LifecycleWriter,
    reason: str = "stale_cleanup",
) -> None:
    for attempt in range(1, attempts + 1):
        try:
            success = bool(hooks.kill(sandbox_id))
        except Exception:  # noqa: BLE001 - kill is retried and reported
            success = False
        emit(
            "sandbox_kill",
            {
                "sandbox_id": sandbox_id,
                "attempt": attempt,
                "success": success,
                "reason": reason,
            },
        )
        if success:
            return
    raise RuntimeError(
        f"unable to kill E2B sandbox {sandbox_id} after {attempts} attempts"
    )


def _emit(
    lifecycle: LifecycleWriter | None,
    trajectory: JsonlTrajectorySink | None,
    action: str,
    payload: dict[str, object],
) -> None:
    if lifecycle is not None:
        lifecycle(action, payload)
    if trajectory is not None:
        trajectory.append(action, payload, actor="resource_manager")


def _safe_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(
            "E2B workspace path must be relative and stay inside the workspace"
        )
    return str(path)


def _validate_python_packages(packages: tuple[str, ...]) -> None:
    if not packages:
        raise ValueError("at least one preinstalled Python package is required")
    for package in packages:
        if not package or not package.replace("_", "").isalnum():
            raise ValueError(f"invalid Python import name: {package!r}")


def _verify_python_packages(
    hooks: E2BHooks,
    sandbox_id: str,
    packages: tuple[str, ...],
    *,
    timeout: int,
    emit: LifecycleWriter,
) -> None:
    imports = "; ".join(f"import {package}" for package in packages)
    command = f"python3 -c \"{imports}; print('packages-ready')\""
    try:
        result = hooks.execute(sandbox_id, command, timeout)
        success = result.exit_code == 0 and not result.timed_out
        error = result.stderr if not success else None
    except Exception as exc:  # noqa: BLE001 - package probe is a hard admission gate
        success = False
        error = f"{type(exc).__name__}: {exc}"
    emit(
        "sandbox_package_probe",
        {
            "sandbox_id": sandbox_id,
            "packages": list(packages),
            "timeout_seconds": timeout,
            "success": success,
            "error": error,
        },
    )
    if success:
        return
    try:
        _kill_with_retry(
            hooks,
            sandbox_id,
            attempts=2,
            emit=emit,
            reason="package_probe_failed",
        )
    finally:
        raise RuntimeError(
            "E2B template is missing required hv3.8 Python packages: "
            + ", ".join(packages)
        )
