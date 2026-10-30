"""Scoped SDK 1 client; the SDK owns sessions, integrity and contract validation."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

import httpx
from financial_data_lake import DataLakeClient as DataCollectionClient
from financial_data_lake.client import ApiError, DataLakeClientEnvironment
from pydantic import ValidationError

from .data_lake import DEFAULT_TIMEOUT_SECONDS, DataLakeError
from .sdk_transport import acquire_client, release_client


@dataclass(frozen=True, slots=True)
class FileReceipt:
    document_id: str
    format: str
    content_type: str
    size_bytes: int
    sha256: str
    trace_id: str

    def as_dict(self):
        from dataclasses import asdict

        return asdict(self)


@dataclass(slots=True)
class SDKDataLakeClient:
    client: Any
    cutoff: datetime | None
    datasets: tuple[str, ...]
    project: str = "default"
    purpose: str = "research"
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    http_client: httpx.Client | None = field(default=None, repr=False)
    timezone: str = "UTC"
    _closed: bool = field(default=False, init=False, repr=False)
    _capabilities: dict | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_task(cls, task, *, run_id: str):
        return cls.from_scope(
            cutoff=task.cutoff,
            datasets=tuple(task.capability_set.get("datasets", ())),
            run_id=run_id,
            timezone=task.timezone,
        )

    @classmethod
    def from_scope(
        cls,
        *,
        cutoff: datetime | None,
        datasets: tuple[str, ...],
        run_id: str,
        timeout_seconds: float | None = None,
        http_max_connections: int | None = None,
        http_max_keepalive_connections: int | None = None,
        env_file: str | Path = ".env",
        base_url: str | None = None,
        timezone: str = "UTC",
    ):
        from financial_data_lake.taxonomy import DATASETS as SDK_DATASETS

        from ..ports.data_specs import DATASETS

        # SQL has no per-request dataset filter. Its actual session grant, not
        # merely the displayed catalog, must enforce the v1 product boundary.
        # Empty scope is used by trusted preparation to discover the key's grant.
        # Do not demand every product from a narrower key. TableTools separately
        # refuses a broad/incompatible grant before exposing any SQL operation.
        datasets = tuple(datasets)
        if set(datasets) - set(DATASETS):
            raise DataLakeError("Dataset is unavailable in v1", code="not_authorized")
        if set(datasets) - set(SDK_DATASETS):
            raise DataLakeError(
                "Dataset requires the published table SDK", code="sdk_upgrade_required"
            )
        try:
            settings = DataLakeClientEnvironment(
                _env_file=env_file,
                **({"DATALAKE_BASE_URL": base_url} if base_url else {}),
            )
        except ValidationError:
            # Settings errors may otherwise echo unrelated secrets from .env.
            raise ValueError(
                "Invalid Data Lake configuration; check DATALAKE_BASE_URL/API_KEY/PROJECT/PURPOSE"
            ) from None
        timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv("DATALAKE_TIMEOUT_SECONDS", "190")
        )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if cutoff is not None and cutoff.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware")
        connections = int(
            http_max_connections
            if http_max_connections is not None
            else os.getenv("DATALAKE_HTTP_MAX_CONNECTIONS", "64")
        )
        keepalive = int(
            http_max_keepalive_connections
            if http_max_keepalive_connections is not None
            else os.getenv("DATALAKE_HTTP_MAX_KEEPALIVE_CONNECTIONS", "32")
        )
        transport = acquire_client(timeout, connections, keepalive)
        try:
            client = DataCollectionClient(
                base_url or settings.base_url,
                api_key=settings.api_key.get_secret_value(),
                project=settings.project,
                purpose=settings.purpose,
                run_id=run_id,
                requested_as_of=cutoff.isoformat() if cutoff else None,
                requested_datasets=datasets,
                timezone=timezone,
                timeout_seconds=timeout,
                http_client=transport,
                max_document_bytes=int(
                    os.getenv("DATALAKE_MAX_DOCUMENT_BYTES", str(2 * 1024**3))
                ),
            )
        except BaseException:
            release_client(transport)
            raise
        return cls(
            client,
            cutoff,
            datasets,
            settings.project,
            settings.purpose,
            timeout,
            transport,
            timezone,
        )

    def capabilities(self) -> dict:
        if self._capabilities is None:
            self._capabilities = self._call(
                self.client.preflight,
                datasets=self.datasets,
                cutoff=self.cutoff,
                timezone=self.timezone,
                required_capabilities=("lookup", "search", "read_window", "download"),
            )
        return dict(self._capabilities)

    def preflight_scope(self) -> dict:
        caps = self.capabilities()
        return {
            "api_contract": caps["api_contract"],
            "project": self.project,
            "purpose": self.purpose,
            "requested_datasets": list(self.datasets),
            "authorized_datasets": caps["datasets"],
            "requested_as_of": self.cutoff.isoformat() if self.cutoff else None,
            "authorized_as_of": caps["authorized_as_of"],
        }

    def _scope(self, datasets=()):
        return {
            "requested_as_of": self.cutoff.isoformat() if self.cutoff else None,
            "timezone": self.timezone,
            "datasets": datasets or self.datasets,
        }

    def lookup(self, **kwargs):
        return self._call(self.client.lookup, **self._discovery_scope(kwargs))

    def search(self, **kwargs):
        return self._call(self.client.search, **self._discovery_scope(kwargs))

    def _discovery_scope(self, kwargs):
        if set(kwargs) & {"requested_as_of", "timezone"}:
            raise ValueError(
                "discovery cannot override the fixed client cutoff/timezone"
            )
        datasets = kwargs.get("datasets") or self.datasets
        if self.datasets and set(datasets) - set(self.datasets):
            raise DataLakeError(
                "datasets exceed the fixed client scope", code="not_authorized"
            )
        return {**kwargs, **self._scope(datasets)}

    def read_content(
        self,
        document_id: str,
        *,
        format: str | None = None,
        unit: str | None = None,
        start: int | None = None,
        count: int | None = None,
        cursor: str | None = None,
        max_bytes: int = 256 * 1024,
        datasets=(),
    ):
        return self._call(
            self.client.read,
            document_id,
            format=format,
            unit=unit,
            start=start,
            count=count,
            cursor=cursor,
            max_bytes=max_bytes,
            **self._scope(datasets),
        )

    def get_document(self, document_id: str):
        return self._call(self.client.get_document, document_id, **self._scope())

    def page_image(self, document_id: str, page_number: int) -> bytes:
        image = self._call(
            self.client.page_image,
            document_id,
            page_number=page_number,
            **self._scope(),
        )
        return image.data

    def download_document(
        self,
        document_id: str,
        destination: BinaryIO,
        *,
        format: str | None = None,
        datasets=(),
        page_number: int | None = None,
    ):
        # SDK checks headers, SHA and size before any bytes reach the workspace.
        with tempfile.TemporaryDirectory(prefix="prophet-download-") as directory:
            receipt = self._call(
                self.client.download,
                document_id,
                Path(directory) / "file",
                format=format,
                page_number=page_number,
                **self._scope(datasets),
            )
            with receipt.path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
        return FileReceipt(
            str(receipt.document_id),
            receipt.format,
            receipt.content_type,
            receipt.size_bytes,
            receipt.sha256,
            receipt.trace_id,
        )

    def _call(self, operation, *args, **kwargs):
        if self._closed:
            raise DataLakeError("Data Lake client is closed", code="client_closed")
        try:
            result = operation(*args, **kwargs)
            return (
                result.model_dump(mode="json")
                if hasattr(result, "model_dump")
                else result
            )
        except Exception as error:
            raise self._translate_error(error) from error

    @staticmethod
    def _translate_error(error: BaseException):
        if isinstance(error, DataLakeError):
            return error
        if isinstance(error, ApiError):
            return DataLakeError(
                str(error),
                code=error.code,
                status=error.status_code,
                retryable=error.status_code in {429, 502, 503, 504},
                details={"errors": list(error.details)} if error.details else None,
            )
        if isinstance(error, (TimeoutError, httpx.TimeoutException)):
            return DataLakeError(
                "Data Lake request timed out", code="timeout", retryable=True
            )
        if isinstance(error, (ConnectionError, OSError, httpx.TransportError)):
            return DataLakeError(
                "Data Lake connection failed",
                code="network_unavailable",
                retryable=True,
            )
        return DataLakeError(
            str(error),
            code="invalid_request"
            if isinstance(error, ValueError)
            else "data_lake_error",
        )

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self.client.close()
            finally:
                if self.http_client is not None:
                    release_client(self.http_client)
