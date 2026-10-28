"""Harbor Docker with an offline, self-cleaning kernel probe.

The upstream probe only reads /proc; it does not need a bridge or outbound
network. All task/container lifecycle operations remain upstream Harbor.
"""

import asyncio
import base64
import functools
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import yaml
from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment

from . import docker_guard as guard


class _OfflineKernelProbeDockerEnvironment(DockerEnvironment):
    @classmethod
    @functools.cache
    def _egress_control_kernel_support(cls) -> bool:
        name = "prophet-kernel-probe-" + uuid4().hex
        try:
            result = subprocess.run(
                cls._engine_cmd(
                    "container",
                    "run",
                    "--rm",
                    "--network=none",
                    "--name",
                    name,
                    "--label",
                    "prophetbench.role=kernel-probe",
                    cls._EGRESS_CONTROL_KERNEL_PROBE_IMAGE,
                    "sh",
                    "-c",
                    cls._EGRESS_CONTROL_KERNEL_PROBE_SCRIPT,
                ),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
        finally:
            # `docker run --rm` can leave a Created container if its CLI times
            # out before start. This UUID names only the probe we just created.
            try:
                subprocess.run(
                    cls._engine_cmd("container", "rm", "--force", name),
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass


class OutputBudgetExceeded(RuntimeError):
    code = "output_budget_exceeded"


def managed_docker_config(lease_seconds=1800):
    return {
        "type": "docker",
        "import_path": "prophet_bench_env.integrations.harbor.docker:ManagedDockerEnvironment",
        "delete": True,
        "kwargs": {"lease_seconds": lease_seconds},
    }


class ManagedDockerEnvironment(_OfflineKernelProbeDockerEnvironment):
    """Harbor lifecycle with owned leases, bounded I/O and verified teardown.

    Does not create a second sandbox, alter task CPU/RAM or change networking.
    Python 3/Linux is required for the trusted command supervisor.
    """

    def __init__(
        self,
        *args,
        lease_seconds=1800,
        output_bytes=None,
        artifact_bytes=None,
        min_free_bytes=None,
        **kwargs,
    ):
        self.output_bytes = int(
            output_bytes or os.getenv("PROPHET_DOCKER_OUTPUT_BYTES", str(32 * 1024**2))
        )
        self.artifact_bytes = int(
            artifact_bytes
            or os.getenv("PROPHET_DOCKER_ARTIFACT_BYTES", str(4 * 1024**3))
        )
        self.min_free_bytes = int(
            min_free_bytes
            if min_free_bytes is not None
            else os.getenv("PROPHET_DOCKER_MIN_FREE_BYTES", str(2 * 1024**3))
        )
        if (
            not 30 <= lease_seconds <= 86400
            or min(self.output_bytes, self.artifact_bytes) < 1024
            or self.min_free_bytes < 0
        ):
            raise ValueError("invalid Docker lease/output/artifact policy")
        if kwargs.get("keep_containers"):
            raise ValueError(
                "managed Docker does not support keep_containers; resources have a finite lease"
            )
        self.lease_seconds = lease_seconds
        self._guard_path = None
        self._safety_compose = None
        self._guard_process = None
        self._heartbeat_task = None
        self._command_ready = False
        self._stopped = False
        self._token = uuid4().hex
        self._control_dir = "/tmp/.prophet-control-" + self._token
        super().__init__(*args, **kwargs)
        if self._is_windows_container:
            raise ValueError("managed Docker supervision currently requires Linux")

    @property
    def _docker_compose_paths(self):
        paths = super()._docker_compose_paths
        if self._safety_compose:
            paths.append(self._safety_compose)
        return paths

    async def _setup_guard(self):
        root = Path(
            os.getenv(
                "PROPHET_DOCKER_STATE_DIR",
                str(Path.home() / ".local/state/prophetbench/docker"),
            )
        ).absolute()
        if root.is_symlink():
            raise ValueError("Docker lease root must not be a symlink")
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
            raise ValueError(
                "Docker lease root must be private (0700) and owned by the controller"
            )
        if shutil.disk_usage(root).free < self.min_free_bytes:
            raise RuntimeError("insufficient local free space for a new Docker trial")
        # Reconcile old, dead-owner receipts on the next launch after a reboot.
        for old in root.glob("*/lease.json"):
            if old.parent.is_symlink():
                continue
            state = guard.read_state(old)
            if state["status"] != "clean" and guard.expired(state):
                await asyncio.to_thread(guard.finish, old, "stale_owner_on_start")
        await asyncio.to_thread(guard.maintain_cache, root)
        private = root / self._token
        private.mkdir(mode=0o700)
        self._guard_path = private / "lease.json"
        from harbor.environments.docker.docker import (
            _sanitize_docker_compose_project_name,
        )

        state = {
            "token": self._token,
            "uid": os.getuid(),
            "pid": os.getpid(),
            "process_start": guard.process_identity(os.getpid()),
            "boot_id": guard.boot_id(),
            "project": _sanitize_docker_compose_project_name(self.session_id),
            "daemon_id": (
                await asyncio.to_thread(guard.docker, "info", "--format", "{{.ID}}")
            ).strip(),
            "deadline": time.monotonic() + self.lease_seconds,
            "status": "active",
            "artifact_bytes": self.artifact_bytes,
            "artifact_roots": [str(self.trial_paths.trial_dir.absolute())],
        }
        guard.write_state(self._guard_path, state)
        labels = {guard.LABEL: self._token, guard.OWNER: str(os.getuid())}
        services = {"main": {"labels": labels}}
        if not self.task_env_config.docker_image and self._dockerfile_path.exists():
            services["main"]["build"] = {
                "labels": {guard.IMAGE_OWNER: "v1", guard.OWNER: str(os.getuid())}
            }
        networks = {"default": {"labels": labels}}
        volumes = {}
        authored = [
            self._environment_docker_compose_path,
            *self.extra_docker_compose_paths,
        ]
        for file in authored:
            if not file.exists():
                continue
            document = yaml.safe_load(file.read_text()) or {}
            for name in document.get("services", {}):
                services[name] = {"labels": labels}
            for kind, target in (("networks", networks), ("volumes", volumes)):
                for name, value in (document.get(kind) or {}).items():
                    if not (value or {}).get("external"):
                        target[name] = {"labels": labels}
                    elif name in target:
                        target.pop(name)
        if self._enable_egress_control:
            services[self._EGRESS_CONTROL_SERVICE_NAME] = {"labels": labels}
        self._safety_compose = private / "ownership.json"
        self._safety_compose.write_text(
            json.dumps({"services": services, "networks": networks, "volumes": volumes})
        )
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            in {
                "PATH",
                "HOME",
                "LANG",
                "DOCKER_HOST",
                "DOCKER_CONTEXT",
                "DOCKER_CONFIG",
                "DOCKER_CERT_PATH",
                "DOCKER_TLS_VERIFY",
                "XDG_RUNTIME_DIR",
            }
        }
        self._guard_process = await asyncio.to_thread(
            subprocess.Popen,
            [
                sys.executable,
                str(Path(guard.__file__).resolve()),
                str(self._guard_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
        self._heartbeat_task = asyncio.create_task(self._heartbeat())

    async def _heartbeat(self):
        while True:
            await asyncio.sleep(min(10, self.lease_seconds / 3))
            with guard.locked(self._guard_path):
                state = guard.read_state(self._guard_path)
                if state["status"] != "active":
                    return
                if self._guard_process.poll() is not None:
                    state.update(status="cleanup_failed", reason="lease_guard_exited")
                else:
                    state["deadline"] = time.monotonic() + self.lease_seconds
                guard.write_state(self._guard_path, state)
            if state["status"] != "active":
                await asyncio.to_thread(
                    guard.finish, self._guard_path, "lease_guard_exited"
                )
                return

    async def start(self, force_build):
        try:
            await self._setup_guard()
            await super().start(force_build)
            # Private root-owned control directory is not writable by agent.
            await super()._compose_exec(
                "install -d -m 700 " + shlex.quote(self._control_dir),
                service="main",
                cwd=None,
                env=None,
                timeout_sec=15,
                user="root",
            )
            await super().upload_file(
                Path(__file__).with_name("docker_command.py"),
                self._control_dir + "/runner.py",
            )
            self._command_ready = True
        except BaseException:
            await asyncio.shield(self.stop(True))
            raise

    async def _raw_control(self, action, token, spec=None):
        command = (
            "python3 "
            + shlex.quote(self._control_dir + "/runner.py")
            + " "
            + action
            + " "
            + shlex.quote(self._control_dir)
            + " "
            + token
        )
        if spec is not None:
            command += " " + base64.b64encode(json.dumps(spec).encode()).decode()
        return await super()._compose_exec(
            command,
            service="main",
            cwd=None,
            env=None,
            timeout_sec=(
                spec.get("timeout") + 5 if spec and spec.get("timeout") else None
            )
            if action == "run"
            else 10,
            user="root",
        )

    async def _compose_exec(self, command, *, service, cwd, env, timeout_sec, user):
        if not self._command_ready or service != "main":
            return await super()._compose_exec(
                command,
                service=service,
                cwd=cwd,
                env=env,
                timeout_sec=timeout_sec,
                user=user,
            )
        token = uuid4().hex
        spec = {
            "command": command,
            "cwd": cwd,
            "env": env or {},
            "timeout": timeout_sec,
            "user": user,
            "max_bytes": self.output_bytes,
        }
        try:
            result = await self._raw_control("run", token, spec)
        except BaseException:
            await asyncio.shield(self._cancel_and_confirm(token))
            raise
        status = await self._raw_control("status", token)
        if status.return_code:
            await self.stop(True)
            raise RuntimeError(
                "command supervisor receipt is missing; environment stopped"
            )
        try:
            receipt = json.loads(status.stdout)
            if (
                receipt["reason"]
                not in {
                    "completed",
                    "timeout",
                    "output_limit",
                    "cancelled",
                    "reap_failed",
                }
                or type(receipt["exit_code"]) is not int
            ):
                raise ValueError("invalid command receipt")
        except (ValueError, KeyError, TypeError):
            await self.stop(True)
            raise RuntimeError(
                "invalid command supervisor receipt; environment stopped"
            ) from None
        if receipt["reason"] == "reap_failed":
            await self.stop(True)
            raise RuntimeError(
                "command cleanup failed; entire owned environment was stopped"
            )
        if receipt["reason"] == "timeout":
            raise TimeoutError("container command timed out; descendants reaped")
        if receipt["reason"] == "output_limit":
            raise OutputBudgetExceeded(
                "container command output exceeded its budget; descendants reaped"
            )
        if receipt["reason"] == "cancelled":
            raise asyncio.CancelledError
        result.return_code = receipt["exit_code"]
        return result

    async def _cancel_and_confirm(self, token):
        try:
            async with asyncio.timeout(5):
                await self._raw_control("cancel", token)
                while True:
                    status = await self._raw_control("status", token)
                    if not status.return_code:
                        if json.loads(status.stdout)["reason"] != "reap_failed":
                            return
                        break
                    await asyncio.sleep(0.1)
        except (OSError, RuntimeError, ValueError, TimeoutError):
            pass  # The fallback below verifies the entire environment is gone.
        await self.stop(True)

    async def _collect(self, process, *, timeout_sec, stdin_data=None, on_output=None):
        chunks, size = [], 0
        if self._guard_path:
            with guard.locked(self._guard_path):
                state = guard.read_state(self._guard_path)
                state.setdefault("clients", []).append(
                    {"pid": process.pid, "start": guard.process_identity(process.pid)}
                )
                guard.write_state(self._guard_path, state)

        async def read():
            nonlocal size
            while data := await process.stdout.read(65536):
                size += len(data)
                # A small framing margin lets the in-container supervisor write
                # its trusted receipt before the host needs to abort transport.
                if size > self.output_bytes + 65536:
                    raise OutputBudgetExceeded(
                        "Docker transport output exceeded its budget"
                    )
                chunks.append(data)
                if on_output:
                    await on_output(data.decode(errors="replace"), "stdout")

        async def work():
            async def write():
                if stdin_data is not None:
                    process.stdin.write(stdin_data)
                    await process.stdin.drain()
                    process.stdin.close()

            async with asyncio.TaskGroup() as group:
                group.create_task(read())
                group.create_task(write())
            await process.wait()

        try:
            await asyncio.wait_for(work(), timeout=timeout_sec)
        except BaseException as exc:
            if process.returncode is None:
                await self._terminate_process(process)
            if isinstance(exc, BaseExceptionGroup) and exc.subgroup(
                OutputBudgetExceeded
            ):
                raise OutputBudgetExceeded(
                    "Docker transport output exceeded its budget"
                ) from None
            raise
        finally:
            if self._guard_path:
                with guard.locked(self._guard_path):
                    state = guard.read_state(self._guard_path)
                    state["clients"] = [
                        c for c in state.get("clients", []) if c["pid"] != process.pid
                    ]
                    guard.write_state(self._guard_path, state)
        return ExecResult(
            stdout=b"".join(chunks).decode(errors="replace"),
            stderr=None,
            return_code=process.returncode,
        )

    async def _collect_buffered_output(self, process, *, timeout_sec, stdin_data=None):
        return await self._collect(
            process, timeout_sec=timeout_sec, stdin_data=stdin_data
        )

    async def _collect_streamed_output(
        self, process, *, timeout_sec, stdin_data=None, on_output
    ):
        return await self._collect(
            process, timeout_sec=timeout_sec, stdin_data=stdin_data, on_output=on_output
        )

    async def _check_download_size(self, source_path):
        if not self._command_ready:
            return
        result = await self.exec(
            "du -sb -- " + shlex.quote(str(source_path)), timeout_sec=15, user="root"
        )
        if result.return_code:
            raise RuntimeError("cannot inspect artifact size before collection")
        if int(result.stdout.split()[0]) > self.artifact_bytes:
            raise RuntimeError("artifact exceeds configured collection budget")

    async def download_dir(self, source_dir, target_dir):
        await self._check_download_size(source_dir)
        return await super().download_dir(source_dir, target_dir)

    async def download_file(self, source_path, target_path):
        await self._check_download_size(source_path)
        return await super().download_file(source_path, target_path)

    async def stop(self, delete):
        if self._stopped:
            return
        self._command_ready = False
        try:
            if self._guard_path and await asyncio.to_thread(
                guard.project_is_owned, guard.read_state(self._guard_path)
            ):
                await asyncio.wait_for(self.prepare_logs_for_host(), timeout=15)
                # Unlike upstream, never ask compose to delete unproven volumes
                # or image tags. The ownership reconciler handles only our labels.
                await asyncio.wait_for(
                    self._run_docker_compose_command(["down", "--remove-orphans"]),
                    timeout=30,
                )
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 - reconciliation below is authoritative
            # Upstream cleanup is best effort; the labeled reconciliation below
            # is authoritative and the independent guard survives cancellation.
            self.logger.warning(
                "Harbor stop needed labeled reconciliation: %s", type(exc).__name__
            )
        finally:
            self._cleanup_mounts_compose_file()
            self._cleanup_resources_compose_file()
            self._cleanup_env_compose_file()
            self._cleanup_egress_control_services_compose_file()
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._heartbeat_task
        if self._guard_path:
            for attempt in range(3):
                try:
                    await asyncio.to_thread(
                        guard.finish, self._guard_path, "harbor_stop"
                    )
                    break
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                    if attempt == 2:
                        raise RuntimeError(
                            "Docker cleanup could not be verified; lease guard will retry"
                        ) from None
                    await asyncio.sleep(1)
        self._stopped = True
        if self._guard_process:
            with suppress(subprocess.TimeoutExpired):
                await asyncio.to_thread(self._guard_process.wait, timeout=7)


class OfflineProbeDockerEnvironment(ManagedDockerEnvironment):
    """Compatibility entrypoint for the existing Docker none-network profile."""
