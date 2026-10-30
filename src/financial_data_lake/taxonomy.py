"""Versioned public taxonomy; storage memberships remain authorization facts."""

from typing import Literal, cast, get_args

API_CONTRACT = "phase4-tables-1"
TAXONOMY_VERSION = "taxonomy-5"

type DatasetName = Literal[
    "financial_report",
    "financial_statement",
    "research_report",
    "earnings_call",
    "news",
    "institutional_holdings",
    "investor_material",
    "analyst_estimate",
    "issuer_event",
    "market_data",
]
type Format = Literal[
    "parquet",
    "json",
    "markdown",
    "plain_text",
    "html",
    "xml",
    "png",
    "pdf",
    "doc",
    "docx",
    "ppt",
    "pptx",
    "xls",
    "xlsx",
    "binary",
]
type Market = Literal["us", "hk", "a_share", "china", "global"]
type DatasetAccessPolicy = Literal["selected", "all"]
type PeriodKind = Literal["year", "half_year", "quarter", "unclassified"]
type ReadUnit = Literal["page", "block", "chunk", "row"]

DATASETS: tuple[DatasetName, ...] = get_args(DatasetName.__value__)
FORMATS: tuple[Format, ...] = get_args(Format.__value__)
SUBTYPES: dict[DatasetName, tuple[str, ...]] = {
    "financial_report": ("annual", "interim", "quarterly"),
    "financial_statement": ("annual", "interim", "quarterly"),
    "research_report": ("company", "industry", "macro", "strategy"),
    "earnings_call": ("transcript",),
    "news": (),
    "institutional_holdings": ("13f",),
    "investor_material": (
        "presentation",
        "investor_day_transcript",
        "conference_transcript",
        "activity_record",
    ),
    "analyst_estimate": ("forecast", "price_target", "recommendation"),
    "issuer_event": (),
    "market_data": ("daily",),
}

# Offline/catalog interpretation, NOT HTTP aliases and NOT permission unions.
MEMBERSHIP_TAXONOMY: dict[str, str] = {
    "annual-reports": "financial_report",
    "fundamentals": "financial_statement",
    "research-reports": "research_report",
    "transcripts": "earnings_call",
    "news": "news",
    "institutional-holdings": "institutional_holdings",
    "investor-materials": "investor_material",
    "consensus": "analyst_estimate",
    "issuer-event-materials": "issuer_event",
    # Internal intake classifications are retained, not advertised as released
    # API/SDK choices. A licensed membership alone does not publish a product.
    "issuer_event": "issuer_event",
    "valuation_model": "valuation_model",
    "other": "other",
    **{name: name for name in DATASETS},
}


def canonical_dataset(membership: str) -> DatasetName | None:
    """One public release boundary for tokens, capabilities and discovery."""
    if membership == "issuer-event-materials":
        # Legacy's empty reserved membership is not the new release's grant.
        return None
    value = MEMBERSHIP_TAXONOMY.get(membership)
    return cast(DatasetName, value) if value in DATASETS else None


def public_memberships(memberships: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(slug for slug in memberships if canonical_dataset(slug) is not None)


MIME_FORMATS: dict[str, Format] = {
    "application/pdf": "pdf",
    "text/html": "html",
    "application/xhtml+xml": "html",
    "application/xml": "xml",
    "text/xml": "xml",
    "application/json": "json",
    "text/json": "json",
    "text/plain": "plain_text",
    "text/markdown": "markdown",
    "image/png": "png",
    "application/vnd.apache.parquet": "parquet",
    "application/x-parquet": "parquet",
    "application/msword": "doc",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
}


def source_format(mime: str | None) -> Format | None:
    if mime is None:
        return None
    return MIME_FORMATS.get(mime.partition(";")[0].strip().lower(), "binary")


def statement_subtype(period_kind: str | None) -> str | None:
    """Use the same report-frequency labels for statements and report documents.

    The frequency must come from the source period, never inferred from an end
    date: a fourth-quarter statement is not an annual statement.
    """
    return {"year": "annual", "half_year": "interim", "quarter": "quarterly"}.get(
        period_kind or ""
    )


def validate_subtypes(
    datasets: tuple[DatasetName, ...], values: tuple[str, ...]
) -> None:
    if values and not datasets:
        raise ValueError("subtypes requires an explicit dataset scope")
    permitted = {value for dataset in datasets for value in SUBTYPES[dataset]}
    if any(value not in permitted for value in values):
        raise ValueError("subtype is not registered for the requested datasets")
