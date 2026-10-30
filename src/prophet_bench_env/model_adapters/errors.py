"""Errors shared by managed model adapters."""

from __future__ import annotations

from typing import Any


class ModelAdapterError(RuntimeError):
    """Base class for provider adapter failures."""


class ProviderInvocationError(ModelAdapterError):
    """The provider returned an unusable response or transport failed."""


class EmptyModelResponseError(ModelAdapterError):
    """Reserved for callers that want to distinguish persistent empty output."""


class ProviderHTTPError(ProviderInvocationError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
