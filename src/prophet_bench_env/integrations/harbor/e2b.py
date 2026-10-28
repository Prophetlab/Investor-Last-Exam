"""Harbor E2B with a bounded crash lease, not a second lifecycle manager."""

import asyncio
import hashlib
import logging
from contextlib import suppress

from harbor.environments.e2b import E2BEnvironment

_BUILD_LOCKS: dict[str, asyncio.Lock] = {}


class LeasedE2BEnvironment(E2BEnvironment):
    def _sandbox_create_network_options(self, network_policy=None):
        policy = network_policy or self.network_policy
        if policy.network_mode.value == "no-network":
            from e2b import ALL_TRAFFIC

            return {"deny_out": [ALL_TRAFFIC]}
        return super()._sandbox_create_network_options(policy)

    def _sandbox_network_update(self, network_policy=None):
        policy = network_policy or self.network_policy
        if policy.network_mode.value == "no-network":
            from e2b import ALL_TRAFFIC

            return {"allow_internet_access": False, "deny_out": [ALL_TRAFFIC]}
        return super()._sandbox_network_update(policy)

    def __init__(
        self, *args, lease_seconds: int = 1800, renew_lease: bool = False, **kwargs
    ):
        if not 60 <= lease_seconds <= 14400:
            raise ValueError("E2B lease_seconds must be between 60 and 14400")
        self.lease_seconds = lease_seconds
        self.renew_lease = renew_lease
        self._lease_task = None
        super().__init__(*args, **kwargs)
        identity = (
            f"{self.environment_id}:{self._effective_cpus}:{self._effective_memory_mb}"
        )
        self._template_name = (
            "prophet-harbor-" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        )

    async def start(self, force_build: bool):
        # Identical workspace recipes share a template across task IDs. Only
        # template preparation is serialized, never agent execution.
        lock_key = f"{id(asyncio.get_running_loop())}:{self._template_name}"
        lock = _BUILD_LOCKS.setdefault(lock_key, asyncio.Lock())
        async with lock:
            await super().start(force_build)

    async def _create_sandbox(self):
        # Version-pinned override: upstream 0.23.0 otherwise requests 86400s.
        # This is the same Harbor-owned handle used by exec/upload/stop.
        from e2b import AsyncSandbox

        self._sandbox = await AsyncSandbox.create(
            template=self._template_name,
            metadata={
                "environment_name": self.environment_name,
                "session_id": self.session_id,
                "prophetbench": "harbor",
            },
            envs=self._startup_env(),
            timeout=self.lease_seconds,
            allow_internet_access=(
                self.network_policy.network_mode.value != "no-network"
            ),
            network=self._sandbox_create_network_options(),
        )
        if self.renew_lease:
            self._lease_task = asyncio.create_task(self._maintain_lease())

    async def _maintain_lease(self):
        """Only a live Harbor owner extends the crash lease; stop cancels renewal."""
        delay = min(300, self.lease_seconds / 3)
        while True:
            await asyncio.sleep(delay)
            if self._sandbox is None:
                return
            try:
                await self._sandbox.set_timeout(self.lease_seconds)
                delay = min(300, self.lease_seconds / 3)
            except Exception as error:  # noqa: BLE001 - retry lease renewal without logging credentials
                logging.getLogger(__name__).warning(
                    "E2B lease renewal failed: %s", type(error).__name__
                )
                delay = min(15, self.lease_seconds / 6)

    async def stop(self, delete: bool):
        if self._lease_task is not None:
            self._lease_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._lease_task
            self._lease_task = None
        await super().stop(delete)
