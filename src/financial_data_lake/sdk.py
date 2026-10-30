"""SDK 1 candidate: one final /v2 contract, no agent/LLM facade or field fallback."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import UUID

import httpx
from pydantic import AliasChoices, BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .api.contract import (
    Capabilities,
    ContentResponse,
    ContractModel,
    DiscoveryResponse,
    DocumentDetail,
    ReadRequest,
)
from .api.control_contract import (
    ApiKeyIssueRequest,
    ApiKeyIssueResponse,
    DataScope,
    TaskTokenIssueRequest,
    TaskTokenIssueResponse,
)
from .api.document_contract import DocumentLookupRequest as LookupRequest
from .api.document_contract import DocumentSearchRequest as SearchRequest
from .api.session_contract import SessionCreateRequest, SessionCreateResponse
from .api.table_contract import (
    TableAccepted,
    TableCancellation,
    TableCatalog,
    TableIdentities,
    TableQueryRequest,
    TableResult,
    TableSchema,
)
from .errors import ConfigurationError, DataLakeClientError
from .taxonomy import API_CONTRACT, DATASETS, DatasetName, Format
from .time import DEFAULT_TIMEZONE, parse_as_of


class DataLakeClientEnvironment(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    base_url: str = Field(
        validation_alias=AliasChoices("DATALAKE_BASE_URL", "PROPHET_DATA_LAKE_BASE_URL")
    )
    api_key: SecretStr = Field(validation_alias="DATALAKE_API_KEY")
    project: str = Field(default="default", validation_alias="DATALAKE_PROJECT")
    run_id: str = Field(default="api-key-client", validation_alias="DATALAKE_RUN_ID")
    purpose: str = Field(default="research", validation_alias="DATALAKE_PURPOSE")


class ApiError(DataLakeClientError):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: tuple[dict[str, Any], ...] = (),
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.status_code, self.code, self.details = status_code, code, details


@dataclass(frozen=True)
class SessionInfo:
    api_contract: str
    authorized_as_of: datetime
    datasets: tuple[DatasetName, ...]
    scopes: tuple[str, ...]
    expires_at: datetime | None = None
    scope_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if (
            self.authorized_as_of.tzinfo is None
            or self.expires_at is not None
            and self.expires_at.tzinfo is None
            or not set(self.datasets).issubset(DATASETS)
            or not self.datasets
            or not self.scopes
        ):
            raise ApiError(
                502,
                "invalid_grant",
                "session contains invalid time or dataset metadata",
            )


@dataclass(frozen=True)
class DownloadReceipt:
    document_id: UUID
    path: Path
    format: Format
    content_type: str
    size_bytes: int
    sha256: str
    trace_id: str


@dataclass(frozen=True)
class TableDownloadReceipt:
    result_id: UUID
    path: Path
    format: Literal["json", "parquet"]
    content_type: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ImageContent:
    document_id: UUID
    page_number: int
    data: bytes = field(repr=False)
    sha256: str
    trace_id: str
    format: Literal["png"] = "png"


def _base_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigurationError(
            "base URL must be HTTP(S), without credentials, query or fragment"
        )
    if parsed.scheme == "http" and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ConfigurationError(
            "plaintext HTTP is only allowed through a local tunnel"
        )
    return value.strip().rstrip("/")


def _error(response: httpx.Response, data: bytes) -> ApiError:
    try:
        value = json.loads(data)
        error = value["error"]
        if (
            not isinstance(error["code"], str)
            or not isinstance(error["message"], str)
            or not isinstance(error.get("errors", []), list)
            or any(not isinstance(item, dict) for item in error.get("errors", []))
        ):
            raise ValueError("invalid error fields")
        return ApiError(
            response.status_code,
            str(error["code"]),
            str(error["message"])[:500],
            details=tuple(error.get("errors", ())),
        )
    except (ValueError, KeyError, TypeError, AttributeError):
        return ApiError(
            response.status_code,
            "invalid_server_response",
            "server did not return the final error contract",
        )


def _bounded_bytes(response: httpx.Response, maximum: int) -> bytes:
    result = bytearray()
    for chunk in response.iter_bytes(chunk_size=65536):
        if len(result) + len(chunk) > maximum:
            raise ApiError(
                413, "response_too_large", "response exceeds its byte budget"
            )
        result.extend(chunk)
    return bytes(result)


def _control_json(
    client: httpx.Client,
    base_url: str,
    path: str,
    *,
    key: str | None = None,
    payload: Mapping[str, Any] | None = None,
    issuer_key: str | None = None,
) -> dict[str, Any]:
    try:
        with client.stream(
            "POST" if payload is not None else "GET",
            base_url + path,
            json=payload,
            auth=None,
            follow_redirects=False,
            headers={
                "Authorization": f"Bearer {key}" if key else "",
                "X-Prophet-Token-Issuer-Key": issuer_key or "",
                "X-DataLake-Contract": API_CONTRACT,
                "Accept-Encoding": "identity",
            },
        ) as response:
            data = _bounded_bytes(response, 65536)
            if response.status_code >= 400:
                raise _error(response, data)
            if (
                response.status_code != 200
                or response.headers.get("X-DataLake-Contract") != API_CONTRACT
            ):
                raise ApiError(
                    409,
                    "client_upgrade_required",
                    "unexpected server contract or redirect",
                )
            try:
                value = json.loads(data)
                if not isinstance(value, dict):
                    raise ValueError("not an object")
                return value
            except ValueError as error:
                raise ApiError(
                    502, "invalid_server_response", "invalid control-plane JSON"
                ) from error
    except httpx.HTTPError as error:
        raise ApiError(
            503, "transport_error", "control-plane connection failed"
        ) from error


@contextmanager
def _transport_stream(
    client: httpx.Client, *args: Any, **kwargs: Any
) -> Iterator[httpx.Response]:
    try:
        with client.stream(*args, **kwargs) as response:
            yield response
    except httpx.HTTPError as error:
        raise ApiError(503, "transport_error", "data connection failed") from error


class ApiKeySessionProvider:
    """Public session provider; grants are inspectable without exposing the token."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        http_client: httpx.Client,
        project: str,
        run_id: str,
        purpose: str,
        requested_datasets: tuple[DatasetName, ...],
        requested_as_of: str | None,
        timezone: str,
    ) -> None:
        self.base_url, self.http_client = _base_url(base_url), http_client
        self._key = api_key
        self._lock = threading.RLock()
        self._access_token: str | None = None
        self._previous_session: str | None = None
        self._protocol_checked = False
        self.session_info: SessionInfo | None = None
        self.parameters = SessionCreateRequest(
            project=project,
            run_id=run_id,
            purpose=purpose,
            requested_datasets=requested_datasets,
            requested_as_of=requested_as_of,
            timezone=timezone,
        ).model_dump(mode="json", exclude_none=True)

    def refresh(self) -> None:
        with self._lock:
            self._access_token = None
            # Retain the signed previous scope. A refresh is not a new episode.

    def __call__(self) -> str:
        with self._lock:
            if not self._protocol_checked:
                try:
                    info = _control_json(
                        self.http_client, self.base_url, "/v2/service-info"
                    )
                except ApiError as error:
                    if error.code == "transport_error":
                        raise
                    raise ApiError(
                        409,
                        "client_upgrade_required",
                        "session service has not switched",
                    ) from error
                if info.get("api_contract") != API_CONTRACT:
                    raise ApiError(
                        409,
                        "client_upgrade_required",
                        "session service has not switched",
                    )
                self._protocol_checked = True
            if (
                self._access_token is not None
                and self.session_info is not None
                and self.session_info.expires_at is not None
                and self.session_info.expires_at
                > datetime.now(UTC) + timedelta(seconds=30)
            ):
                return self._access_token
            payload = _control_json(
                self.http_client,
                self.base_url,
                "/v2/sessions",
                key=self._key,
                payload={
                    key: value
                    for key, value in {
                        **self.parameters,
                        "previous_session": self._previous_session,
                    }.items()
                    if value is not None
                },
            )
            try:
                grant = SessionCreateResponse.model_validate(payload)
            except ValueError as error:
                raise ApiError(
                    502, "invalid_grant", "invalid session response"
                ) from error
            current = SessionInfo(
                api_contract=API_CONTRACT,
                authorized_as_of=grant.authorized_as_of,
                expires_at=grant.expires_at,
                datasets=grant.datasets,
                scopes=grant.scopes,
                scope_fingerprint=grant.scope_fingerprint,
            )
            if (
                grant.project != self.parameters["project"]
                or grant.run_id != self.parameters["run_id"]
                or grant.expires_at <= datetime.now(UTC)
                or self.parameters["requested_datasets"]
                and set(grant.datasets) != set(self.parameters["requested_datasets"])
                or self.parameters.get("requested_as_of") is not None
                and grant.authorized_as_of
                > parse_as_of(
                    self.parameters["requested_as_of"],
                    default_tz=self.parameters["timezone"],
                )
            ):
                raise ApiError(
                    502, "invalid_grant", "session differs from requested task"
                )
            prior = self.session_info
            if prior is not None and (
                current.scope_fingerprint != prior.scope_fingerprint
                or current.authorized_as_of != prior.authorized_as_of
                or set(current.datasets) != set(prior.datasets)
                or set(current.scopes) != set(prior.scopes)
            ):
                raise ApiError(
                    403,
                    "session_scope_changed",
                    "renewal changed the fixed episode scope; "
                    "start a new episode explicitly",
                )
            self.parameters["requested_as_of"] = current.authorized_as_of.isoformat()
            self.parameters["requested_datasets"] = list(current.datasets)
            self.session_info = current
            self._access_token = grant.access_token
            self._previous_session = grant.access_token
            return self._access_token


class TrustedTaskTokenIssuer:
    """Trusted host issuer. Never put its key or this object into an Agent tool.

    Supplied HTTP clients remain caller-owned. Issuance is never automatically
    retried after an HTTP failure; a retry could mint a second credential.
    """

    def __init__(
        self,
        base_url: str,
        issuer_key: str,
        *,
        http_client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 60,
    ) -> None:
        self.base_url = _base_url(base_url)
        if not issuer_key.strip():
            raise ConfigurationError("issuer_key is required")
        if http_client is not None and transport is not None:
            raise ConfigurationError("http_client and transport are mutually exclusive")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ConfigurationError("timeout must be finite and positive")
        self._issuer_key = issuer_key
        self.http_client = http_client or httpx.Client(
            transport=transport,
            timeout=timeout_seconds,
            follow_redirects=False,
        )
        self._owns_http_client = http_client is None
        self._protocol_checked = False
        self._lock = threading.RLock()

    def close(self) -> None:
        if self._owns_http_client:
            self.http_client.close()

    def __enter__(self) -> TrustedTaskTokenIssuer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _issue(self, path: str, request: BaseModel) -> dict[str, Any]:
        with self._lock:
            if not self._protocol_checked:
                try:
                    info = _control_json(
                        self.http_client, self.base_url, "/v2/service-info"
                    )
                except ApiError as error:
                    if error.code == "transport_error":
                        raise
                    raise ApiError(
                        409,
                        "client_upgrade_required",
                        "issuer service has not switched",
                    ) from error
                if info.get("api_contract") != API_CONTRACT:
                    raise ApiError(
                        409,
                        "client_upgrade_required",
                        "issuer service has not switched",
                    )
                self._protocol_checked = True
            return _control_json(
                self.http_client,
                self.base_url,
                path,
                issuer_key=self._issuer_key,
                payload=request.model_dump(mode="json", exclude_none=True),
            )

    def issue(
        self,
        *,
        tenant_slug: str,
        principal: str,
        run_id: str,
        purpose: str,
        datasets: tuple[DatasetName, ...],
        scopes: tuple[DataScope, ...],
        authorized_as_of: str,
        timezone: str = DEFAULT_TIMEZONE,
        ttl_minutes: int = 15,
        previous_session: str | None = None,
    ) -> TaskTokenIssueResponse:
        request = TaskTokenIssueRequest(
            tenant_slug=tenant_slug,
            principal=principal,
            run_id=run_id,
            purpose=purpose,
            datasets=datasets,
            scopes=scopes,
            authorized_as_of=authorized_as_of,
            timezone=timezone,
            ttl_minutes=ttl_minutes,
            previous_session=previous_session,
        )
        grant = _validated_response(
            TaskTokenIssueResponse,
            self._issue("/internal/v1/task-tokens", request),
            code="invalid_grant",
        )
        if (
            grant.tenant_slug != request.tenant_slug
            or grant.principal != request.principal
            or grant.run_id != request.run_id
            or grant.purpose != request.purpose
            or set(grant.datasets) != set(request.datasets)
            or set(grant.scopes) != set(request.scopes)
            or grant.authorized_as_of
            != parse_as_of(authorized_as_of, default_tz=timezone)
            or grant.expires_at <= datetime.now(UTC)
        ):
            raise ApiError(
                502, "invalid_grant", "issuer response differs from requested task"
            )
        return grant

    def provider(
        self,
        *,
        tenant_slug: str,
        principal: str,
        run_id: str,
        purpose: str,
        datasets: tuple[DatasetName, ...],
        scopes: tuple[DataScope, ...],
        authorized_as_of: str,
        timezone: str = DEFAULT_TIMEZONE,
        ttl_minutes: int = 15,
    ) -> TrustedTaskTokenProvider:
        return TrustedTaskTokenProvider(
            self,
            TaskTokenIssueRequest(
                tenant_slug=tenant_slug,
                principal=principal,
                run_id=run_id,
                purpose=purpose,
                datasets=datasets,
                scopes=scopes,
                authorized_as_of=authorized_as_of,
                timezone=timezone,
                ttl_minutes=ttl_minutes,
            ),
        )

    def issue_api_key(self, request: ApiKeyIssueRequest) -> ApiKeyIssueResponse:
        """Create one key using explicit canonical scope and existing source grants."""
        grant = _validated_response(
            ApiKeyIssueResponse,
            self._issue("/internal/v1/api-keys", request),
            code="invalid_grant",
        )
        if (
            grant.principal != request.principal
            or grant.purpose != request.purpose
            or grant.dataset_access != request.dataset_access
            or set(grant.scopes) != set(request.scopes)
            or request.dataset_access == "selected"
            and set(grant.datasets) != set(request.datasets)
        ):
            raise ApiError(
                502, "invalid_grant", "key response differs from requested scope"
            )
        return grant


class TrustedTaskTokenProvider:
    """Callable fixed episode provider with public, credential-free session_info."""

    def __init__(
        self, issuer: TrustedTaskTokenIssuer, request: TaskTokenIssueRequest
    ) -> None:
        if request.previous_session is not None:
            raise ConfigurationError(
                "new provider cannot take an unverified prior session"
            )
        self.issuer = issuer
        self._request = request.model_copy(deep=True)
        self._lock = threading.RLock()
        self._access_token: str | None = None
        self._previous_session: str | None = None
        self.session_info: SessionInfo | None = None

    def refresh(self) -> None:
        with self._lock:
            self._access_token = None

    def __call__(self) -> str:
        with self._lock:
            if (
                self._access_token is not None
                and self.session_info is not None
                and self.session_info.expires_at is not None
                and self.session_info.expires_at
                > datetime.now(UTC) + timedelta(seconds=30)
            ):
                return self._access_token
            grant = self.issuer.issue(
                **{
                    **self._request.model_dump(exclude_none=True),
                    "previous_session": self._previous_session,
                }
            )
            current = SessionInfo(
                API_CONTRACT,
                grant.authorized_as_of,
                grant.datasets,
                grant.scopes,
                grant.expires_at,
                grant.scope_fingerprint,
            )
            prior = self.session_info
            if prior is not None and (
                current.scope_fingerprint != prior.scope_fingerprint
                or current.authorized_as_of != prior.authorized_as_of
                or set(current.datasets) != set(prior.datasets)
                or set(current.scopes) != set(prior.scopes)
            ):
                raise ApiError(
                    403,
                    "session_scope_changed",
                    "renewal changed the fixed episode scope",
                )
            self.session_info = current
            self._previous_session = self._access_token = grant.access_token
            return grant.access_token


def _validated_response[ResponseModel: BaseModel](
    model: type[ResponseModel],
    value: Any,
    *,
    code: str = "invalid_server_response",
) -> ResponseModel:
    try:
        return model.model_validate(value)
    except ValueError as error:
        raise ApiError(
            502, code, "response does not match the final contract"
        ) from error


class _ResourceScope(ContractModel):
    requested_as_of: str | None = None
    timezone: str = DEFAULT_TIMEZONE
    datasets: tuple[DatasetName, ...] = ()


class _DownloadParameters(_ResourceScope):
    format: Format | None = None
    page_number: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def page_format(self) -> _DownloadParameters:
        if self.page_number is not None and self.format != "png":
            raise ValueError("page_number is only valid with format=png")
        return self


class DataLakeClient:
    def __init__(
        self,
        base_url: str,
        token_provider: Callable[[], str] | None = None,
        *,
        api_key: str | None = None,
        project: str = "default",
        run_id: str = "api-key-client",
        purpose: str = "research",
        requested_datasets: tuple[DatasetName, ...] = (),
        requested_as_of: str | None = None,
        timezone: str = DEFAULT_TIMEZONE,
        timeout_seconds: float = 60,
        max_document_bytes: int = 2 * 1024 * 1024 * 1024,
        http_client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = _base_url(base_url)
        if (api_key is None) == (token_provider is None):
            raise ConfigurationError("provide exactly one of api_key or token_provider")
        if http_client is not None and transport is not None:
            raise ConfigurationError("http_client and transport are mutually exclusive")
        if (
            not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or isinstance(max_document_bytes, bool)
            or not isinstance(max_document_bytes, int)
            or max_document_bytes <= 0
        ):
            raise ConfigurationError("timeout and byte limit must be positive")
        self.http_client = http_client or httpx.Client(
            transport=transport, timeout=timeout_seconds, follow_redirects=False
        )
        self._owns_http_client = http_client is None
        self.max_document_bytes = max_document_bytes
        self.credentials: Callable[[], str] = token_provider or ApiKeySessionProvider(
            self.base_url,
            api_key or "",
            http_client=self.http_client,
            project=project,
            run_id=run_id,
            purpose=purpose,
            requested_datasets=requested_datasets,
            requested_as_of=requested_as_of,
            timezone=timezone,
        )
        self._scope: SessionInfo | None = None
        self._protocol_checked = False
        self._protocol_lock = threading.Lock()

    @classmethod
    def from_env(cls, env_file: str | Path = ".env", **kwargs: Any) -> DataLakeClient:
        environment = DataLakeClientEnvironment(_env_file=env_file)
        return cls(
            environment.base_url,
            api_key=environment.api_key.get_secret_value(),
            project=environment.project,
            run_id=environment.run_id,
            purpose=environment.purpose,
            **kwargs,
        )

    @property
    def session_info(self) -> SessionInfo | None:
        return (
            self.credentials.session_info
            if isinstance(
                self.credentials, (ApiKeySessionProvider, TrustedTaskTokenProvider)
            )
            else self._scope
        )

    def close(self) -> None:
        if self._owns_http_client:
            self.http_client.close()

    def __enter__(self) -> DataLakeClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def preflight(
        self,
        *,
        required_capabilities: tuple[str, ...] = (),
        datasets: tuple[DatasetName, ...] = (),
        cutoff: str | datetime | None = None,
        timezone: str = DEFAULT_TIMEZONE,
    ) -> Capabilities:
        self._check_protocol(force=True)
        capabilities = self.capabilities()
        if not set(required_capabilities).issubset(
            capabilities.capabilities
        ) or not set(datasets).issubset(capabilities.datasets):
            raise ApiError(
                403,
                "scope_unavailable",
                "requested capabilities/datasets are outside this session",
            )
        expected = (
            parse_as_of(cutoff, default_tz=timezone) if cutoff is not None else None
        )
        if expected is not None and expected != capabilities.authorized_as_of:
            raise ApiError(
                403,
                "cutoff_mismatch",
                "service cutoff differs from the fixed task cutoff",
            )
        self._scope = SessionInfo(
            API_CONTRACT,
            capabilities.authorized_as_of,
            capabilities.datasets,
            capabilities.scopes,
        )
        return capabilities

    def _check_protocol(self, *, force: bool = False) -> None:
        with self._protocol_lock:
            if self._protocol_checked and not force:
                return
            try:
                info = _control_json(
                    self.http_client, self.base_url, "/v2/service-info"
                )
                compatible = info.get("api_contract") == API_CONTRACT
            except ApiError as error:
                if error.code == "transport_error":
                    raise
                compatible = False
            if not compatible:
                raise ApiError(
                    409,
                    "client_upgrade_required",
                    "service has not switched to phase4-tables-1",
                )
            self._protocol_checked = True

    def capabilities(self) -> Capabilities:
        return _validated_response(
            Capabilities, self._json("GET", "/v2/catalog/capabilities")
        )

    def tables(self, *, catalog_snapshot_id: UUID | str | None = None) -> TableCatalog:
        return _validated_response(
            TableCatalog,
            self._json(
                "GET",
                "/v2/tables",
                parameters={
                    "catalog_snapshot_id": str(catalog_snapshot_id)
                    if catalog_snapshot_id
                    else None
                },
            ),
        )

    def table_schema(
        self, table_id: str, *, catalog_snapshot_id: UUID | str | None = None
    ) -> TableSchema:
        if (
            not all(part.replace("_", "").isalnum() for part in table_id.split("."))
            or len(table_id.split(".")) != 2
        ):
            raise ValueError("table_id requires a registered schema.table name")
        return _validated_response(
            TableSchema,
            self._json(
                "GET",
                f"/v2/tables/{table_id}/schema",
                parameters={
                    "catalog_snapshot_id": str(catalog_snapshot_id)
                    if catalog_snapshot_id
                    else None
                },
            ),
        )

    def table_identities(
        self,
        query: str,
        *,
        dataset: Literal["financial_statement", "analyst_estimate"],
        market: str | None = None,
        requested_as_of: datetime | None = None,
        limit: int = 50,
    ) -> TableIdentities:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be 1..100")
        if requested_as_of is not None and requested_as_of.tzinfo is None:
            raise ValueError("requested_as_of requires a timezone")
        return _validated_response(
            TableIdentities,
            self._json(
                "GET",
                "/v2/tables/identities",
                parameters={
                    "query": query,
                    "dataset": dataset,
                    "market": market,
                    "requested_as_of": requested_as_of.isoformat()
                    if requested_as_of
                    else None,
                    "limit": limit,
                },
            ),
        )

    def query_table(
        self,
        sql: str,
        *,
        catalog_snapshot_id: UUID | str | None = None,
        parameters: Mapping[str, Any] | None = None,
        requested_as_of: datetime | None = None,
        page_size: int = 200,
    ) -> TableAccepted:
        snapshot = (
            UUID(str(catalog_snapshot_id))
            if catalog_snapshot_id
            else self.tables().catalog_snapshot_id
        )
        payload = TableQueryRequest(
            catalog_snapshot_id=snapshot,
            sql=sql,
            parameters=dict(parameters or {}),
            requested_as_of=requested_as_of,
            page_size=page_size,
        )
        return _validated_response(
            TableAccepted,
            self._json(
                "POST",
                "/v2/tables/query",
                payload=payload.model_dump(mode="json", exclude_none=True),
            ),
        )

    def table_result(
        self, result_id: UUID | str, *, cursor: str | None = None, count: int = 200
    ) -> TableResult:
        if type(count) is not int or not 1 <= count <= 200:
            raise ValueError("count must be 1..200; use next_cursor for more rows")
        return _validated_response(
            TableResult,
            self._json(
                "GET",
                f"/v2/table-results/{UUID(str(result_id))}/rows",
                parameters={"cursor": cursor, "count": count},
            ),
        )

    def wait_table(
        self,
        result_id: UUID | str,
        *,
        timeout: float = 120,
        poll_interval: float = 0.5,
        count: int = 200,
    ) -> TableResult:
        if (
            not math.isfinite(timeout)
            or not math.isfinite(poll_interval)
            or timeout <= 0
            or not 0.1 <= poll_interval <= 10
        ):
            raise ValueError(
                "timeout must be positive and poll_interval must be .1..10 seconds"
            )
        deadline = time.monotonic() + timeout
        while True:
            result = self.table_result(result_id, count=count)
            if result.status == "ready":
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "query is still running; keep the result_id to read or cancel it"
                )
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))

    def iter_table_rows(
        self, result_id: UUID | str, *, count: int = 200, timeout: float = 120
    ) -> Iterator[dict[str, Any]]:
        page = self.wait_table(result_id, timeout=timeout, count=count)
        while True:
            names = [column.name for column in page.columns]
            yield from (dict(zip(names, row, strict=True)) for row in page.rows)
            if page.next_cursor is None:
                return
            page = self.table_result(result_id, cursor=page.next_cursor, count=count)

    def cancel_table(self, query_id: UUID | str) -> TableCancellation:
        return _validated_response(
            TableCancellation,
            self._json("DELETE", f"/v2/table-queries/{UUID(str(query_id))}"),
        )

    def download_table(
        self,
        result_id: UUID | str,
        destination: str | Path,
        *,
        format: Literal["json", "parquet"] = "parquet",
        overwrite: bool = False,
    ) -> TableDownloadReceipt:
        if format not in {"json", "parquet"}:
            raise ValueError("format must be json or parquet")
        identity = UUID(str(result_id))
        receipt = self._download_file(
            identity,
            Path(destination),
            f"/v2/table-results/{identity}/download",
            {"format": format},
            format,
            overwrite,
            identity_header="X-Result-Id",
        )
        return TableDownloadReceipt(
            identity,
            receipt.path,
            format,
            receipt.content_type,
            receipt.size_bytes,
            receipt.sha256,
        )

    def lookup(self, query: str = "", **filters: Any) -> DiscoveryResponse:
        return self._discover("lookup", LookupRequest(query=query, **filters))

    def search(self, query: str, **filters: Any) -> DiscoveryResponse:
        return self._discover("search", SearchRequest(query=query, **filters))

    def _discover(
        self, operation: str, request: LookupRequest | SearchRequest
    ) -> DiscoveryResponse:
        result = _validated_response(
            DiscoveryResponse,
            self._json(
                "POST",
                f"/v2/{operation}",
                payload=request.model_dump(mode="json", exclude_none=True),
            ),
        )
        if result.returned_count != len(result.items):
            raise ApiError(
                502, "invalid_server_response", "inconsistent discovery count"
            )
        return result

    def iter_pages(
        self,
        operation: Literal["lookup", "search"],
        *,
        max_pages: int = 100,
        **parameters: Any,
    ) -> Iterator[DiscoveryResponse]:
        if operation not in {"lookup", "search"}:
            raise ValueError("operation must be lookup or search")
        if (
            isinstance(max_pages, bool)
            or not isinstance(max_pages, int)
            or not 1 <= max_pages <= 1000
        ):
            raise ValueError("max_pages must be 1..1000")
        seen = set()
        for _ in range(max_pages):
            if operation == "lookup":
                page = self.lookup(**parameters)
            else:
                page = self.search(**parameters)
            yield page
            if not page.has_more:
                return
            if not page.next_cursor or page.next_cursor in seen:
                raise ApiError(
                    502, "invalid_cursor", "server repeated or omitted a continuation"
                )
            seen.add(page.next_cursor)
            parameters["cursor"] = page.next_cursor

    def get_document(self, document_id: UUID | str, **scope: Any) -> DocumentDetail:
        parameters = _ResourceScope(**scope).model_dump(mode="json", exclude_none=True)
        identity = UUID(str(document_id))
        result = _validated_response(
            DocumentDetail,
            self._json("GET", f"/v2/documents/{identity}", parameters=parameters),
        )
        if result.document_id != identity:
            raise ApiError(502, "invalid_server_response", "document identity mismatch")
        return result

    def read(
        self,
        document_id: UUID | str,
        *,
        format: Format | None = None,
        page_number: int | None = None,
        **parameters: Any,
    ) -> ContentResponse | ImageContent:
        parameters = {
            key: value for key, value in parameters.items() if value is not None
        }
        if format == "png":
            if (
                set(parameters) - {"requested_as_of", "timezone", "datasets", "count"}
                or parameters.pop("count", 1) != 1
            ):
                raise ValueError("PNG read takes one page, not a text window/cursor")
            if page_number is None:
                metadata = self.get_document(document_id, **parameters)
                if metadata.source != "png":
                    raise ValueError("PDF PNG read requires an explicit page_number")
                page_number = 1
            return self.page_image(document_id, page_number=page_number, **parameters)
        if page_number is not None:
            raise ValueError("page_number is only valid with format=png")
        request = ReadRequest(format=format, **parameters)
        result = _validated_response(
            ContentResponse,
            self._json(
                "GET",
                f"/v2/documents/{UUID(str(document_id))}/content",
                parameters=request.model_dump(mode="json", exclude_none=True),
            ),
        )
        if result.document_id != UUID(str(document_id)):
            raise ApiError(502, "invalid_server_response", "content identity mismatch")
        return result

    def page_image(
        self, document_id: UUID | str, *, page_number: int, **scope: Any
    ) -> ImageContent:
        request = _DownloadParameters(format="png", page_number=page_number, **scope)
        identity = UUID(str(document_id))
        with self._stream(
            "GET",
            f"/v2/documents/{identity}/pages/{page_number}/image",
            parameters=request.model_dump(
                mode="json", exclude={"format", "page_number"}, exclude_none=True
            ),
        ) as response:
            data = self._bounded_bytes(response, 32 * 1024 * 1024)
            sha = self._verify_download(response, data, identity, "png")
            return ImageContent(
                identity, page_number, data, sha, response.headers.get("X-Trace-Id", "")
            )

    def download(
        self,
        document_id: UUID | str,
        destination: str | Path,
        *,
        format: Format | None = None,
        page_number: int | None = None,
        overwrite: bool = False,
        **scope: Any,
    ) -> DownloadReceipt:
        identity = UUID(str(document_id))
        target = Path(destination)
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        parameters = _DownloadParameters(
            **scope,
            format=format,
            page_number=page_number,
        ).model_dump(mode="json", exclude_none=True)
        return self._download_file(
            identity,
            target,
            f"/v2/documents/{identity}/download",
            parameters,
            format,
            overwrite,
        )

    def _download_file(
        self,
        identity: UUID,
        target: Path,
        path: str,
        parameters: Mapping[str, Any],
        format: Format | None,
        overwrite: bool,
        *,
        identity_header: str = "X-Document-Id",
    ) -> DownloadReceipt:
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        temporary: Path | None = None
        try:
            with self._stream("GET", path, parameters=parameters) as response:
                expected_size, expected_sha, actual_format = self._download_headers(
                    response, identity, format, identity_header=identity_header
                )
                if expected_size > self.max_document_bytes:
                    raise ApiError(
                        413,
                        "download_too_large",
                        "download exceeds the configured byte limit",
                    )
                with NamedTemporaryFile(
                    mode="wb", dir=target.parent, prefix=".datalake-", delete=False
                ) as output:
                    temporary = Path(output.name)
                    digest, size = hashlib.sha256(), 0
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        size += len(chunk)
                        if size > expected_size:
                            raise ApiError(
                                502,
                                "integrity_mismatch",
                                "download exceeds declared length",
                            )
                        output.write(chunk)
                        digest.update(chunk)
                    if size != expected_size or digest.hexdigest() != expected_sha:
                        raise ApiError(
                            502, "integrity_mismatch", "download length/SHA256 mismatch"
                        )
                    output.flush()
                    os.fsync(output.fileno())
                if overwrite:
                    os.replace(temporary, target)
                else:
                    os.link(temporary, target)
                    temporary.unlink()
                temporary = None
                return DownloadReceipt(
                    identity,
                    target,
                    actual_format,
                    response.headers["Content-Type"],
                    size,
                    expected_sha,
                    response.headers.get("X-Trace-Id", ""),
                )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _download_headers(
        self,
        response: httpx.Response,
        identity: UUID,
        format: Format | None,
        *,
        identity_header: str = "X-Document-Id",
    ) -> tuple[int, str, Format]:
        from .taxonomy import FORMATS

        sha = response.headers.get("X-Content-SHA256", "")
        encoding = response.headers.get("Content-Encoding", "identity")
        actual = response.headers.get("X-Content-Format")
        size = response.headers.get("Content-Length", "")
        if (
            not size.isdigit()
            or len(sha) != 64
            or any(char not in "0123456789abcdef" for char in sha)
            or response.headers.get(identity_header) != str(identity)
            or actual not in FORMATS
            or format is not None
            and actual != format
            or encoding != "identity"
            or not response.headers.get("Content-Type", "").strip()
            or actual == "png"
            and response.headers.get("Content-Type") != "image/png"
        ):
            raise ApiError(
                502,
                "integrity_metadata_missing",
                "download integrity/format headers are invalid",
            )
        from typing import cast

        return int(size), sha, cast(Format, actual)

    def _verify_download(
        self, response: httpx.Response, data: bytes, identity: UUID, format: Format
    ) -> str:
        size, sha, _ = self._download_headers(response, identity, format)
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            raise ApiError(502, "integrity_mismatch", "download length/SHA256 mismatch")
        return sha

    def _json(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> Any:
        with self._stream(
            method, path, payload=payload, parameters=parameters
        ) as response:
            try:
                return json.loads(self._bounded_bytes(response, 4 * 1024 * 1024))
            except ValueError as error:
                raise ApiError(
                    502, "invalid_server_response", "invalid response JSON"
                ) from error

    @staticmethod
    def _bounded_bytes(response: httpx.Response, maximum: int) -> bytes:
        return _bounded_bytes(response, maximum)

    def _stream(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[httpx.Response]:
        @contextmanager
        def response_stream() -> Iterator[httpx.Response]:
            self._check_protocol()
            query: list[tuple[str, str]] = []
            for key, value in (parameters or {}).items():
                if value is None:
                    continue
                query.extend(
                    (key, str(item))
                    for item in (value if isinstance(value, (tuple, list)) else [value])
                )
            for attempt in range(2):
                token = self.credentials()
                if not isinstance(token, str) or not token.strip():
                    raise ConfigurationError("token provider returned no credential")
                with _transport_stream(
                    self.http_client,
                    method,
                    self.base_url + path,
                    json=payload,
                    params=tuple(query),
                    auth=None,
                    follow_redirects=False,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "X-Prophet-Token-Issuer-Key": "",
                        "X-DataLake-Contract": API_CONTRACT,
                        "Accept-Encoding": "identity",
                    },
                ) as response:
                    if response.status_code >= 400:
                        error = _error(response, self._bounded_bytes(response, 65536))
                        if (
                            attempt == 0
                            and error.status_code == 401
                            and error.code
                            in {"session_refresh_required", "authentication_required"}
                            and (
                                method == "GET" or path in {"/v2/lookup", "/v2/search"}
                            )
                            and isinstance(
                                self.credentials,
                                (ApiKeySessionProvider, TrustedTaskTokenProvider),
                            )
                        ):
                            self.credentials.refresh()
                            continue
                        raise error
                    if (
                        response.status_code
                        != (
                            202
                            if method == "POST" and path == "/v2/tables/query"
                            else 200
                        )
                        or response.headers.get("X-DataLake-Contract") != API_CONTRACT
                    ):
                        raise ApiError(
                            409,
                            "client_upgrade_required",
                            "unexpected server contract or redirect",
                        )
                    yield response
                    return

        return response_stream()
