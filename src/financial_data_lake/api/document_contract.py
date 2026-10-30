"""Document discovery inputs; numerical datasets use the native table API."""

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from ..taxonomy import DatasetName, Market, validate_subtypes
from ..time import DEFAULT_TIMEZONE
from .contract import ContractModel, Period


class DocumentLookupRequest(ContractModel):
    query: str = Field(default="", max_length=1000)
    requested_as_of: str | None = None
    published_from: str | None = None
    timezone: str = DEFAULT_TIMEZONE
    datasets: tuple[DatasetName, ...] = ()
    subtypes: tuple[str, ...] = Field(default=(), max_length=20)
    markets: tuple[Market, ...] = ()
    filing_forms: tuple[Annotated[str, Field(min_length=1, max_length=30)], ...] = (
        Field(default=(), max_length=20)
    )
    period: Period | None = None
    version_policy: Literal["latest", "all"] = "latest"
    limit: int = Field(default=20, ge=1, le=100, strict=True)
    cursor: str | None = Field(default=None, min_length=1, max_length=8192)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def valid_scope(self) -> Self:
        validate_subtypes(self.datasets, self.subtypes)
        if not self.query and not any(
            (self.datasets, self.markets, self.filing_forms, self.period)
        ):
            raise ValueError("empty query requires an explicit metadata scope")
        return self


class DocumentSearchRequest(DocumentLookupRequest):
    query: str = Field(min_length=1, max_length=1000)

    @field_validator("query")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be empty")
        return value.strip()


TABLE_DATASETS = frozenset(
    {"market_data", "financial_statement", "analyst_estimate", "institutional_holdings"}
)
