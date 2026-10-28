"""Shared Data Lake error contract and the public SDK-backed client facade."""

from collections.abc import Mapping
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 190.0


class DataLakeError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str = "data_lake_error",
        retryable: bool = False,
        trace_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.retryable = retryable
        self.trace_id = trace_id
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": str(self),
            "retryable": self.retryable,
        }
        if self.status is not None:
            result["status"] = self.status
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.details:
            result["details"] = self.details
        return result


def __getattr__(name):
    # Lazy import avoids a cycle: the SDK bridge uses this module's error type.
    if name in {"DataLakeClient", "DownloadReceipt"}:
        from .sdk_data_lake import FileReceipt, SDKDataLakeClient

        return SDKDataLakeClient if name == "DataLakeClient" else FileReceipt
    raise AttributeError(name)
