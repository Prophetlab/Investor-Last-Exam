"""Trusted issuer contracts. Display categories never replace source grants."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, field_validator, model_validator

from ..taxonomy import DatasetAccessPolicy, DatasetName
from ..time import DEFAULT_TIMEZONE
from .contract import ContractModel

DataScope = Literal["lookup", "search", "content", "download"]


class PrincipalRequest(ContractModel):
    tenant_slug: str = Field(min_length=1, max_length=100)
    principal: str = Field(min_length=1, max_length=255)
    purpose: str = Field(min_length=1, max_length=100)
    scopes: tuple[DataScope, ...] = Field(min_length=1, max_length=4)

    @field_validator("tenant_slug", "principal", "purpose")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identity fields cannot be blank")
        return value.strip()

    @field_validator("scopes")
    @classmethod
    def unique_scopes(cls, value: tuple[DataScope, ...]) -> tuple[DataScope, ...]:
        if len(set(value)) != len(value):
            raise ValueError("scopes must be unique")
        return value


class TaskTokenIssueRequest(PrincipalRequest):
    run_id: str = Field(min_length=1, max_length=255)
    datasets: tuple[DatasetName, ...] = Field(min_length=1, max_length=11)
    authorized_as_of: str = Field(min_length=1, max_length=255)
    timezone: str = Field(default=DEFAULT_TIMEZONE, min_length=1, max_length=100)
    ttl_minutes: int = Field(default=15, ge=1, le=1440, strict=True)
    previous_session: str | None = Field(
        default=None, min_length=1, max_length=16384, repr=False
    )

    @field_validator("run_id")
    @classmethod
    def nonblank_run(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("run_id cannot be blank")
        return value.strip()


class TaskTokenIssueResponse(ContractModel):
    api_contract: Literal["phase4-tables-1"] = "phase4-tables-1"
    access_token: str = Field(min_length=1, max_length=16384, repr=False)
    token_type: Literal["bearer"] = "bearer"
    expires_at: AwareDatetime
    authorized_as_of: AwareDatetime
    datasets: tuple[DatasetName, ...] = Field(min_length=1)
    scopes: tuple[DataScope, ...] = Field(min_length=1)
    tenant_slug: str
    principal: str
    purpose: str
    run_id: str
    scope_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class ApiKeyIssueRequest(PrincipalRequest):
    name: str = Field(min_length=1, max_length=100)
    datasets: tuple[DatasetName, ...] = Field(default=(), max_length=11)
    dataset_access: DatasetAccessPolicy = "selected"
    expires_in_days: int | None = Field(default=None, ge=1, le=3650, strict=True)
    requests_per_minute: int | None = Field(default=None, ge=1, le=100_000, strict=True)
    concurrency_limit: int | None = Field(default=None, ge=1, le=256, strict=True)

    @model_validator(mode="after")
    def explicit_selection(self) -> Self:
        if not self.name.strip():
            raise ValueError("name cannot be blank")
        if (self.dataset_access == "selected") != bool(self.datasets):
            raise ValueError("selected requires datasets; all forbids datasets")
        return self


class ApiKeyIssueResponse(ContractModel):
    api_contract: Literal["phase4-tables-1"] = "phase4-tables-1"
    api_key: str = Field(repr=False)
    key_id: UUID
    client_id: UUID
    name: str
    masked_key: str
    principal: str
    purpose: str
    datasets: tuple[DatasetName, ...]
    dataset_access: DatasetAccessPolicy
    scopes: tuple[DataScope, ...]
    expires_at: datetime | None
    requests_per_minute: int
    concurrency_limit: int
