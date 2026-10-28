"""Wire models for the native table service; no data interpretation or execution."""

from datetime import datetime
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from .contract import ContractModel


class TableColumn(ContractModel):
    name: str
    data_type: dict[str, Any]
    description: str
    unit: str | None
    nullable: bool


class TableSchema(ContractModel):
    table_id: str
    dataset: str
    title: str
    columns: tuple[TableColumn, ...]
    key_columns: tuple[str, ...]
    revision_policy: Literal["latest", "events"]


class TableCatalog(ContractModel):
    catalog_snapshot_id: UUID
    contract: Literal["phase4-tables-1"]
    dataset_labels: dict[str, str]
    tables: tuple[TableSchema, ...]
    result_formats: tuple[Literal["json", "parquet"], ...]
    query_mode: Literal["async"]


class TableQueryRequest(ContractModel):
    catalog_snapshot_id: UUID
    sql: str = Field(min_length=1, max_length=16384)
    parameters: dict[str, Any] = Field(default_factory=dict)
    requested_as_of: datetime | None = None
    page_size: int = Field(default=200, ge=1, le=200, strict=True)

    @model_validator(mode="after")
    def aware_cutoff(self) -> Self:
        if self.requested_as_of is not None and self.requested_as_of.tzinfo is None:
            raise ValueError("requested_as_of requires an explicit timezone")
        return self


class TableAccepted(ContractModel):
    query_id: UUID
    result_id: UUID
    status: Literal["queued"]


class ResultColumn(ContractModel):
    name: str
    data_type: str
    json_encoding: Literal["native", "decimal_string", "integer_string", "iso8601"]


class TableResult(ContractModel):
    query_id: UUID
    status: Literal["queued", "running", "ready"]
    result_id: UUID | None = None
    catalog_snapshot_id: UUID | None = None
    effective_as_of: datetime | None = None
    columns: tuple[ResultColumn, ...] = ()
    rows: list[list[Any]] = Field(default_factory=list)
    returned_count: int | None = None
    total_rows: int | None = None
    total_count_exact: bool | None = None
    next_cursor: str | None = None
    coverage: str | None = None
    source_completeness: str | None = None

    @model_validator(mode="after")
    def complete_window(self) -> Self:
        if self.status == "ready" and (
            self.result_id != self.query_id
            or self.returned_count != len(self.rows)
            or any(len(row) != len(self.columns) for row in self.rows)
            or self.total_rows is None
            or self.total_rows < len(self.rows)
            or self.effective_as_of is None
        ):
            raise ValueError("inconsistent table result window")
        return self


class TableCancellation(ContractModel):
    query_id: UUID
    status: Literal["finished", "cancel_requested"]


class TableIdentities(ContractModel):
    dataset: str
    table_id: str
    query_column: str
    effective_as_of: datetime
    items: list[dict[str, Any]]
    has_more: bool
    matching: Literal["exact_registered_alias"]
    interval_semantics: str


def extend_openapi(schema: dict[str, Any]) -> None:
    """Advertise native routes at the unified origin, without Python handlers."""
    models = (
        TableCatalog,
        TableSchema,
        TableQueryRequest,
        TableAccepted,
        TableResult,
        TableCancellation,
        TableIdentities,
    )
    definitions = schema["components"]["schemas"]
    for model in models:
        value = model.model_json_schema(ref_template="#/components/schemas/{model}")
        definitions.update(value.pop("$defs", {}))
        definitions[model.__name__] = value
    for path, method, output, status in (
        ("/v2/tables", "get", "TableCatalog", "200"),
        ("/v2/tables/{table_id}/schema", "get", "TableSchema", "200"),
        ("/v2/tables/query", "post", "TableAccepted", "202"),
        ("/v2/table-results/{result_id}", "get", "TableResult", "200"),
        ("/v2/table-results/{result_id}/rows", "get", "TableResult", "200"),
        ("/v2/table-queries/{query_id}", "delete", "TableCancellation", "200"),
    ):
        operation: dict[str, Any] = {
            "tags": ["tables"],
            "security": [{"HTTPBearer": []}],
            "responses": {
                status: {
                    "description": "Native table response",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": f"#/components/schemas/{output}"}
                        }
                    },
                }
            },
            "parameters": [],
        }
        for key in ("table_id", "result_id", "query_id"):
            if "{" + key + "}" in path:
                operation["parameters"].append(
                    {
                        "name": key,
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                )
        if method == "post":
            operation["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/TableQueryRequest"}
                    }
                },
            }
        elif method == "get":
            parameters = (
                (("cursor", "string"), ("count", "integer"))
                if "table-results" in path
                else (("catalog_snapshot_id", "string"),)
            )
            operation["parameters"].extend(
                {"name": name, "in": "query", "schema": {"type": kind}}
                for name, kind in parameters
            )
        schema["paths"][path] = {method: operation}
    schema["paths"]["/v2/table-results/{result_id}/download"] = {
        "get": {
            "tags": ["tables"],
            "security": [{"HTTPBearer": []}],
            "parameters": [
                {
                    "name": "result_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string", "format": "uuid"},
                },
                {
                    "name": "format",
                    "in": "query",
                    "schema": {
                        "type": "string",
                        "enum": ["json", "parquet"],
                        "default": "json",
                    },
                },
            ],
            "responses": {
                "200": {
                    "description": "Frozen result with SHA256 and byte length",
                    "content": {
                        "application/json": {"schema": {"type": "object"}},
                        "application/vnd.apache.parquet": {
                            "schema": {"type": "string", "format": "binary"}
                        },
                    },
                }
            },
        }
    }
    schema["paths"]["/v2/tables/identities"] = {
        "get": {
            "tags": ["tables"],
            "security": [{"HTTPBearer": []}],
            "parameters": [
                {
                    "name": "dataset",
                    "in": "query",
                    "required": True,
                    "schema": {
                        "type": "string",
                        "enum": ["financial_statement", "analyst_estimate"],
                    },
                },
                {
                    "name": "query",
                    "in": "query",
                    "required": True,
                    "schema": {"type": "string", "maxLength": 255},
                },
                {"name": "market", "in": "query", "schema": {"type": "string"}},
                {
                    "name": "requested_as_of",
                    "in": "query",
                    "schema": {"type": "string", "format": "date-time"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 50,
                    },
                },
            ],
            "responses": {
                "200": {
                    "description": "Dated registered provider identities",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/TableIdentities"}
                        }
                    },
                }
            },
        }
    }
