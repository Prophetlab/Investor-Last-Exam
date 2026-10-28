"""Fine-grained shared resource guards used by environment-owned adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import BoundedSemaphore
from typing import Any

from ..ports.model import ModelAdapter, ModelAuditSink, ModelRequest, ModelResponse


@dataclass(frozen=True, slots=True)
class ResourceGuards:
    """Shared limits whose leases cover only actual resource use."""

    datalake_semaphore: BoundedSemaphore
    model_semaphore: BoundedSemaphore
    e2b_semaphore: BoundedSemaphore | None

    @contextmanager
    def datalake(self) -> Iterator[None]:
        with _semaphore_slot(self.datalake_semaphore):
            yield

    @contextmanager
    def model(self, per_model: BoundedSemaphore | None = None) -> Iterator[None]:
        with _semaphore_slot(self.model_semaphore), _semaphore_slot(per_model):
            yield

    @contextmanager
    def e2b(self) -> Iterator[None]:
        with _semaphore_slot(self.e2b_semaphore):
            yield


class LimitedModelAdapter:
    """Apply request concurrency without hiding provider adapter capabilities."""

    def __init__(
        self,
        inner: ModelAdapter,
        resources: ResourceGuards,
        per_model_semaphore: BoundedSemaphore | None = None,
    ) -> None:
        self.inner = inner
        self.resources = resources
        self.per_model_semaphore = per_model_semaphore

    def generate(self, request: ModelRequest) -> ModelResponse:
        with self.resources.model(self.per_model_semaphore):
            return self.inner.generate(request)

    async def agenerate(self, request: ModelRequest) -> ModelResponse:
        operation = getattr(self.inner, "agenerate", None)
        if not callable(operation):
            return await asyncio.to_thread(self.generate, request)
        await asyncio.to_thread(self.resources.model_semaphore.acquire)
        if self.per_model_semaphore is not None:
            await asyncio.to_thread(self.per_model_semaphore.acquire)
        try:
            return await operation(request)
        finally:
            if self.per_model_semaphore is not None:
                self.per_model_semaphore.release()
            self.resources.model_semaphore.release()

    def set_audit_sink(self, sink: ModelAuditSink | None) -> None:
        setter = getattr(self.inner, "set_audit_sink", None)
        if callable(setter):
            setter(sink)

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@contextmanager
def _semaphore_slot(semaphore: BoundedSemaphore | None) -> Iterator[None]:
    if semaphore is None:
        yield
        return
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()
