"""Agent-facing financial tools. Schema generation has no service side effects."""

from ..core.contracts import ToolDefinition

DATASETS = (
    "financial_report",
    "financial_statement",
    "earnings_call",
    "news",
    "investor_material",
    "issuer_event",
    "analyst_estimate",
    "market_data",
)
MARKETS = ("us", "hk", "a_share", "china", "global")
TABLE_DATASETS = ("financial_statement", "analyst_estimate", "market_data")
DOCUMENT_DATASETS = tuple(d for d in DATASETS if d not in TABLE_DATASETS)
TABLE_TOOL_NAMES = (
    "tables",
    "table_lookup",
    "table_query",
    "table_read",
    "table_cancel",
    "table_download",
)
DATA_TOOL_NAMES = ("search", "read", "download", "bash", *TABLE_TOOL_NAMES)
READ_FORMATS = ("markdown", "plain_text", "json", "png")
FORMATS = (
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
)
SEARCH_DESCRIPTION = (
    "Find financial documents by company, security code or symbol, title, or keywords. Use metadata mode "
    "for names, symbols, and titles; use text mode for a two-to-ten-word phrase in the content. "
    "Results include document_ref, available formats, readable formats, snippets, and a recommended action."
)
READ_DESCRIPTION = (
    "Read documents to extract, explain, compare, or calculate information from their contents. "
    "Use document_ref for one document, or sources for several documents or separate ranges to read together. "
    "Omit ranges to read complete documents within the available reading budget; results identify "
    "the covered ranges and where to continue. Use start/count/unit for specific pages, blocks, "
    "chunks, or rows; a search snippet's page_number can be used as start with unit=page. "
    "Use a source's remaining.resume fields with query to continue reading. PNG reads one page separately."
)
DOWNLOAD_DESCRIPTION = (
    "Download a document returned by search into the working directory. Prefer read for large "
    "unstructured documents, and Parquet downloads with Python or SQL for tabular analysis; "
    "use download when no suitable readable format is available."
)
BASH_DESCRIPTION = (
    "Run a shell command in the isolated working directory for file inspection, Python analysis, "
    "arithmetic, or deterministic checks. Python 3, numpy, scipy, pandas, openpyxl, pyarrow, and "
    "duckdb are available. Network access is disabled."
)


def array(values, description):
    return {
        "type": "array",
        "items": {"type": "string", "enum": list(values)},
        "minItems": 1,
        "uniqueItems": True,
        "description": description,
    }


def obj(properties, required):
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def cursor():
    return {
        "type": "string",
        "minLength": 1,
        "maxLength": 8192,
        "description": "Return the cursor unchanged with the same query, filters, format, and window settings.",
    }


def search_definition(*, datasets=DATASETS, subtypes=None):
    datasets = tuple(d for d in datasets if d not in TABLE_DATASETS)
    mapping = subtypes or {}
    pairs = "; ".join(
        f"{name}: {', '.join(mapping[name])}" for name in datasets if mapping.get(name)
    )
    properties = {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1000,
            "description": "Company, security code or symbol, or title in metadata mode; a two-to-ten-word phrase in text mode. Use ticker:CODE followed by keywords to restrict results to a security.",
        },
        "mode": {"type": "string", "enum": ["text", "metadata"], "default": "text"},
        "datasets": array(datasets, "Optional document categories."),
        "subtypes": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 20,
            "uniqueItems": True,
            "description": "Optional subcategories; specify their datasets. " + pairs,
        },
        "markets": array(MARKETS, "Optional market filter."),
        "filing_forms": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string", "minLength": 1, "maxLength": 30},
            "description": "Exact filing forms, such as 10-K, 10-Q, or 8-K.",
        },
        "period": obj(
            {
                "kind": {
                    "type": "string",
                    "enum": ["year", "half_year", "quarter", "unclassified"],
                },
                "fiscal_year": {"type": "integer", "minimum": 1900, "maximum": 2200},
                "fiscal_period": {"type": "string", "maxLength": 30},
                "start": {"type": "string", "format": "date"},
                "end": {"type": "string", "format": "date"},
            },
            [],
        ),
        "published_from": {
            "type": "string",
            "description": "Optional inclusive publication lower bound: ISO date or datetime with offset. Not the reporting period.",
        },
        "version_policy": {
            "type": "string",
            "enum": ["latest", "all"],
            "default": "latest",
            "description": "latest selects each document's latest visible revision; all includes its historical revisions. Separate reports and forecast events remain separate.",
        },
        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
        "cursor": cursor(),
    }
    properties["period"]["description"] = (
        "Exact reporting-period fields; not publication dates or overlapping date ranges."
    )
    return ToolDefinition("search", SEARCH_DESCRIPTION, obj(properties, ["query"]))


def read_source_properties():
    return {
        "document_ref": {"type": "string", "minLength": 5, "maxLength": 64},
        "format": {
            "type": "string",
            "enum": list(READ_FORMATS),
            "description": "Choose from readable_format. If omitted, prefer markdown, plain_text, then JSON. An explicit format is never substituted.",
        },
        "unit": {
            "type": "string",
            "enum": ["page", "block", "chunk", "row"],
            "description": "Location unit; omit to use the document's default. PNG uses page.",
        },
        "start": {
            "type": "integer",
            "minimum": 1,
            "description": "One-based first location, default 1; omit when using cursor.",
        },
        "count": {
            "type": "integer",
            "minimum": 1,
            "maximum": 200,
            "description": "Continuous locations for an explicit range, default 10 for text/JSON. Maximum: page/chunk 10, block 100, row 200; PNG 1. Omit all range fields for budgeted full-document reading.",
        },
        "cursor": cursor(),
    }


def read_definition():
    source = read_source_properties()
    schema = obj(
        {
            **source,
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2000,
                "description": "What to read for: extract facts or passages, summarize, explain, compare, reconcile, or calculate using the supplied documents. Specify the desired detail and output format when useful.",
            },
            "sources": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": obj(read_source_properties(), ["document_ref"]),
                "description": "Documents or ranges to read together under one shared budget. Repeat a document_ref for separate ranges. Use this instead of top-level document_ref, format, and range fields; PNG must be read separately.",
            },
        },
        ["query"],
    )
    schema["oneOf"] = [
        {"required": ["document_ref"], "not": {"required": ["sources"]}},
        {
            "required": ["sources"],
            "not": {"anyOf": [{"required": [key]} for key in source]},
        },
    ]
    return ToolDefinition("read", READ_DESCRIPTION, schema)


def download_definition():
    return ToolDefinition(
        "download",
        DOWNLOAD_DESCRIPTION,
        obj(
            {
                "document_ref": {"type": "string", "minLength": 5, "maxLength": 64},
                "format": {
                    "type": "string",
                    "enum": list(FORMATS),
                    "description": "Choose an available format; default Parquet for tables, otherwise the source format.",
                },
                "page_number": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Required for a PDF page downloaded as PNG; otherwise omit.",
                },
                "filename": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 180,
                    "description": "Optional filename; directory components are ignored.",
                },
            },
            ["document_ref"],
        ),
        side_effect="workspace_write",
        parallel_safe=False,
    )


def bash_definition():
    return ToolDefinition(
        "bash",
        BASH_DESCRIPTION,
        obj(
            {
                "command": {"type": "string", "minLength": 1, "maxLength": 16000},
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                    "default": 300,
                },
            },
            ["command"],
        ),
        side_effect="workspace_write",
        parallel_safe=False,
    )


def default_data_tool_definitions(*, datasets=DATASETS, subtypes=None):
    return (
        search_definition(datasets=datasets, subtypes=subtypes),
        read_definition(),
        download_definition(),
    )
