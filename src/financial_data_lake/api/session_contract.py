"""Final control-plane models; canonical categories outside, source grants inside."""

from typing import Literal

from pydantic import AwareDatetime, Field

from ..taxonomy import DatasetName
from ..time import DEFAULT_TIMEZONE
from .contract import ContractModel


class SessionCreateRequest(ContractModel):
    project: str = Field(default="default", min_length=1, max_length=255)
    run_id: str = Field(min_length=1, max_length=255)
    purpose: str = Field(default="research", min_length=1, max_length=100)
    requested_datasets: tuple[DatasetName, ...] = ()
    requested_as_of: str | None = Field(default=None, max_length=255)
    timezone: str = Field(default=DEFAULT_TIMEZONE, min_length=1, max_length=100)
    ttl_minutes: int | None = Field(default=None, ge=1, le=1440)
    previous_session: str | None = Field(
        default=None, min_length=1, max_length=16384, repr=False
    )


class SessionCreateResponse(ContractModel):
    api_contract: Literal["phase4-tables-1"] = "phase4-tables-1"
    access_token: str = Field(min_length=1, max_length=16384, repr=False)
    token_type: Literal["bearer"] = "bearer"
    expires_at: AwareDatetime
    authorized_as_of: AwareDatetime
    datasets: tuple[DatasetName, ...]
    project: str
    run_id: str
    scopes: tuple[str, ...]
    scope_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
