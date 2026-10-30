"""The final /v2 wire contract. No legacy request aliases or silent extra fields."""

from datetime import date, datetime
from typing import Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..taxonomy import (
    DatasetName,
    Format,
    Market,
    PeriodKind,
    ReadUnit,
    validate_subtypes,
)
from ..time import DEFAULT_TIMEZONE


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Period(ContractModel):
    kind: PeriodKind | None = None
    fiscal_year: int | None = Field(default=None, ge=1900, le=2200)
    fiscal_period: str | None = Field(default=None, max_length=30)
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.start and self.end and self.start > self.end:
            raise ValueError("period.start must not exceed period.end")
        return self


class TableFilters(ContractModel):
    product: Literal["forecast", "price_target", "recommendation"] | None = None
    metric: str | None = Field(default=None, min_length=1, max_length=30)
    forecast_period_end: date | None = None
    frequency: Literal["year", "quarter", "long_term"] | None = None
    basis: Literal["adjusted", "unadjusted"] | None = None
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    state: Literal["events"] = "events"

    @model_validator(mode="after")
    def product_dimensions(self) -> Self:
        if self.product in {"price_target", "recommendation"} and any(
            value is not None
            for value in (self.metric, self.forecast_period_end, self.frequency)
        ):
            raise ValueError("forecast dimensions require product=forecast")
        return self


class DiscoveryRequest(ContractModel):
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
    table_filters: TableFilters | None = None
    version_policy: Literal["latest", "all"] = Field(
        default="latest",
        description=(
            "Select the latest visible revision of each document, or all visible "
            "revisions. Does not collapse table rows, IBES events, or separate reports."
        ),
    )
    limit: int = Field(default=20, ge=1, le=100, strict=True)
    cursor: str | None = Field(default=None, min_length=1, max_length=8192)

    @model_validator(mode="after")
    def scoped_dimensions(self) -> Self:
        validate_subtypes(self.datasets, self.subtypes)
        if self.table_filters is not None and self.datasets != ("analyst_estimate",):
            raise ValueError("table_filters requires datasets=[analyst_estimate]")
        return self


class LookupRequest(DiscoveryRequest):
    query: str = Field(default="", max_length=1000)

    @field_validator("query")
    @classmethod
    def trim_query(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def bounded_scope(self) -> Self:
        if not self.query and not any(
            (self.datasets, self.markets, self.filing_forms, self.period)
        ):
            raise ValueError("empty query requires an explicit metadata scope")
        return self


class SearchRequest(DiscoveryRequest):
    query: str = Field(min_length=1, max_length=1000)

    @field_validator("query")
    @classmethod
    def valid_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("search query must not be blank")
        return value.strip()


class Issuer(ContractModel):
    id: UUID | None = None
    name: str | None = None


class Security(ContractModel):
    id: UUID | None = None
    symbol: str | None = None
    market: Market | None = None


class Filing(ContractModel):
    form: str
    accession: str | None = None
    amendment: bool = False
    amendment_type: Literal["restatement", "new_holdings"] | None = None


class Availability(ContractModel):
    public_at: datetime
    precision: str
    timezone: str


class Provenance(ContractModel):
    provider: str
    record_id: str | None = None
    url: str | None = None


class TableSummary(ContractModel):
    schema_name: str
    schema_version: str
    row_count: int | None = Field(default=None, ge=0)
    input_snapshot_at: datetime | None = Field(
        default=None,
        description=(
            "Committed source-input snapshot, not the event publication cutoff."
        ),
    )
    state: Literal["events"] | None = None
    as_of: datetime | None = None
    published_from: datetime | None = None
    metric: str | None = None
    basis: Literal["adjusted", "unadjusted"] | None = None
    currency: str | None = None


class Snippet(ContractModel):
    field: Literal["metadata", "body"]
    text: str
    page_number: int | None = None
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)


class Match(ContractModel):
    score: float = Field(ge=0)
    fields: tuple[Literal["metadata", "body"], ...]
    snippets: tuple[Snippet, ...] = ()


class Hit(ContractModel):
    document_id: UUID
    lineage_id: UUID
    title: str
    dataset: DatasetName
    subtype: str | None = None
    issuer: Issuer
    security: Security
    period: Period | None = None
    filing: Filing | None = None
    source: Format | None = None
    format: tuple[Format, ...]
    readable_format: tuple[Format, ...]
    availability: Availability
    provenance: Provenance | None = None
    table: TableSummary | None = None
    match: Match | None = None

    @model_validator(mode="after")
    def valid_formats(self) -> Self:
        if len(set(self.format)) != len(self.format) or len(
            set(self.readable_format)
        ) != len(self.readable_format):
            raise ValueError("format lists must be unique")
        if not set(self.readable_format).issubset(self.format):
            raise ValueError("readable_format must be a subset of format")
        if self.subtype is not None:
            validate_subtypes((self.dataset,), (self.subtype,))
        return self


class DiscoveryResponse(ContractModel):
    items: list[Hit]
    returned_count: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    has_more: bool
    next_cursor: str | None = None
    total_count: int | None = Field(default=None, ge=0)
    total_count_exact: bool = False
    partial: bool = False
    partial_reason: str | None = None
    snapshot_at: datetime
    effective_as_of: datetime
    effective_published_from: datetime | None = None
    timezone: str
    trace_id: str
    query_interpretation: dict[str, Any] | None = None


class ReadRequest(ContractModel):
    format: Format | None = None
    unit: ReadUnit | None = None
    start: int | None = Field(default=None, ge=1, strict=True)
    count: int | None = Field(default=None, ge=1, le=200, strict=True)
    cursor: str | None = Field(default=None, min_length=1, max_length=8192)
    max_bytes: int = Field(default=256 * 1024, ge=1024, le=1024 * 1024)
    requested_as_of: str | None = None
    timezone: str = DEFAULT_TIMEZONE
    datasets: tuple[DatasetName, ...] = ()

    @model_validator(mode="after")
    def finite_window(self) -> Self:
        if self.cursor and self.start is not None:
            raise ValueError("cursor and start are mutually exclusive")
        if self.unit and self.count is not None:
            maximum = {"page": 10, "block": 100, "chunk": 10, "row": 200}[self.unit]
            if self.count > maximum:
                raise ValueError(f"count exceeds {self.unit} limit {maximum}")
        return self


class ContentRange(ContractModel):
    unit: ReadUnit
    start: int = Field(ge=1)
    end: int = Field(ge=0)


class Locator(ContractModel):
    unit: ReadUnit
    index: int = Field(ge=1)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    page_number: int | None = Field(default=None, ge=1)


class ContentPart(ContractModel):
    locator: Locator
    text: str | None = None
    rows: list[dict[str, Any]] | None = None


class ContentResponse(ContractModel):
    document_id: UUID
    format: Format
    requested_range: ContentRange
    returned_range: ContentRange
    parts: list[ContentPart]
    table_schema: dict[str, Any] | None = None
    fragment: bool
    window_complete: bool
    truncated: bool
    has_more: bool
    next_cursor: str | None
    next_start: int | None
    effective_as_of: datetime
    trace_id: str


class Error(ContractModel):
    code: str
    message: str
    trace_id: str
    errors: list[dict[str, Any]] = Field(default_factory=list)


class Capabilities(ContractModel):
    api_contract: Literal["phase4-tables-1"] = "phase4-tables-1"
    taxonomy_version: str
    datasets: tuple[DatasetName, ...]
    subtypes: dict[str, tuple[str, ...]]
    formats: tuple[Format, ...]
    capabilities: tuple[str, ...]
    document_datasets: tuple[DatasetName, ...] = ()
    table_datasets: tuple[DatasetName, ...] = ()
    table_api: str = "/v2/tables"
    limits: dict[str, int]
    authorized_as_of: datetime
    scopes: tuple[str, ...]
    dataset_access: Literal["selected", "all"] = "selected"
    markets: tuple[Market, ...] = ("a_share", "hk", "us", "china", "global")


class ServiceInfo(ContractModel):
    api_contract: Literal["phase4-tables-1"] = "phase4-tables-1"


class RepresentationDetail(ContractModel):
    format: Format
    representation_id: UUID
    mime_type: str
    size_bytes: int | None = None
    sha256: str | None = None
    producer_version: str
    is_original: bool
    recipe: dict[str, Any]


class DocumentDetail(Hit):
    representations: tuple[RepresentationDetail, ...]
    selection: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Frozen table selection plan; detailed audit only, not a user query input."
        ),
    )
    trace_id: str


class ErrorResponse(ContractModel):
    error: Error
