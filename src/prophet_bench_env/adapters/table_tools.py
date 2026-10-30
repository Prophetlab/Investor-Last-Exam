"""Small financial table surface over public SDK 1.1.0, never a SQL engine.

The service validates SQL and enforces the session grant before executing it.
Catalog/result UUIDs and remote cursors stay on the trusted controller.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from ..core.contracts import RegisteredTool, ToolDefinition
from ..ports.data_specs import MARKETS, obj
from .data_lake import DataLakeError

CONTRACT = "phase4-tables-1"
MARKET_NOTE = (
    "US common stocks/ADRs, daily unadjusted OHLC, prev_close, volume and amount; "
    "mv/float_mv are optional. Three years is the target coverage, not a completeness guarantee. "
    "For historical backtests use the task cutoff; the service determines visible history. "
    "Missing history must not be filled or queried with a later cutoff."
)
METHODS = (
    "tables",
    "table_schema",
    "table_identities",
    "query_table",
    "table_result",
    "cancel_table",
    "download_table",
)


def supported(client, caps):
    native = getattr(client, "client", None)
    return (
        bool(caps.get("table_datasets"))
        and caps.get("api_contract") == CONTRACT
        and all(callable(getattr(native, name, None)) for name in METHODS)
    )


def definitions():
    text = {"type": "string", "minLength": 1, "maxLength": 160}
    count = {"type": "integer", "minimum": 1, "maximum": 200}
    ref = {
        "table_ref": {
            **text,
            "description": "Use table_ref returned by table_query in this run.",
        }
    }
    return (
        ToolDefinition(
            "tables",
            "List currently available data tables. Supply a returned table_id for its fields, units and revision meaning before writing SQL. New queries use the current catalog; existing results keep their original snapshot.",
            obj(
                {
                    "table_id": {
                        **text,
                        "description": "Optional schema.table name copied from tables; omit to list tables.",
                    }
                },
                [],
            ),
        ),
        ToolDefinition(
            "table_lookup",
            "Find registered company names or codes for financial statements or analyst events. Keep all dated candidates; provider identifiers are not exchange symbols.",
            obj(
                {
                    "query": {
                        **text,
                        "maxLength": 255,
                        "description": "A registered company name or code, not a financial question or keyword sentence.",
                    },
                    "dataset": {
                        "type": "string",
                        "enum": ["financial_statement", "analyst_estimate"],
                    },
                    "market": {"type": "string", "enum": list(MARKETS)},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 50,
                    },
                },
                ["query", "dataset"],
            ),
        ),
        ToolDefinition(
            "table_query",
            "Query discovered tables using one read-only SELECT and named value parameters. Supports screening, joins, aggregates and windows. Returns table_ref; use table_read until ready. Preserve source units and periods; event averages are not consensus and historical trade dates do not prove historical availability.",
            obj(
                {
                    "sql": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 16384,
                        "description": "One SELECT using discovered table/column names. Named value placeholders use $name.",
                    },
                    "parameters": {
                        "type": "object",
                        "maxProperties": 100,
                        "description": 'Optional values for $name placeholders, without $ in the keys. Example: amount > $minimum uses {"minimum": 0}. Do not bind table or column names.',
                    },
                    "count": {
                        **count,
                        "default": 50,
                        "description": "Rows per preview page, not a limit on the complete query result.",
                    },
                },
                ["sql"],
            ),
        ),
        ToolDefinition(
            "table_read",
            "Read one page of a frozen table result, or check its status. Continue with its cursor unchanged. Decimal and large integers remain strings; total_rows counts this query, not source coverage.",
            obj(
                {
                    **ref,
                    "cursor": {
                        **text,
                        "description": "Copy next_cursor unchanged to read the next page of this table_ref.",
                    },
                    "count": {
                        **count,
                        "description": "Optional first-page size; defaults to the query's count. Omit when passing a cursor to keep its page size.",
                    },
                },
                ["table_ref"],
            ),
        ),
        ToolDefinition(
            "table_cancel",
            "Cancel unfinished table computation in this run.",
            obj(ref, ["table_ref"]),
        ),
        ToolDefinition(
            "table_download",
            "Download the complete frozen result as Parquet or JSON into the same working directory used by bash, with size and SHA-256 verification. This is not just the displayed page.",
            obj(
                {
                    **ref,
                    "format": {
                        "type": "string",
                        "enum": ["parquet", "json"],
                        "default": "parquet",
                    },
                    "filename": {
                        **text,
                        "description": "Optional output filename; directory components are ignored.",
                    },
                },
                ["table_ref"],
            ),
            side_effect="workspace_write",
            parallel_safe=False,
        ),
    )


class TableTools:
    def __init__(self, owner, caps):
        self.owner = owner
        self.native = owner.client.client
        self.lock = RLock()
        self.results = {}
        self.cursors = {}
        self.allowed = set(owner.allowed_datasets)
        # A filtered menu is not an authorization boundary for arbitrary SELECT.
        if not set(caps["datasets"]).issubset(self.allowed):
            raise DataLakeError(
                "Table session exceeds the task dataset scope", code="not_authorized"
            )
        self.cutoff = owner.client.cutoff
        if self.cutoff is None:
            self.cutoff = datetime.fromisoformat(caps["authorized_as_of"])

    def _request(self, method, *args, **kwargs):
        with self.owner._resource_slot():
            value = self.owner.client._call(
                getattr(self.native, method), *args, **kwargs
            )
        audit_value = asdict(value) if is_dataclass(value) else value
        self.owner._audit(
            method,
            json.loads(json.dumps({"args": args, **kwargs}, default=str)),
            json.loads(json.dumps(audit_value, default=str)),
        )
        return value

    def _catalog(self):
        # Discover or submit against today's published catalog, within the same
        # task grant/cutoff. Each accepted query keeps its own immutable snapshot.
        raw = self._request("tables")
        if raw.get("contract") != CONTRACT or any(
            t["dataset"] not in self.allowed for t in raw["tables"]
        ):
            raise DataLakeError("Unexpected table scope", code="scope_mismatch")
        return raw

    def registered_tools(self):
        from .data_tools import _handler_for

        return tuple(
            RegisteredTool(d, _handler_for(lambda args, d=d: self.call(d, args)))
            for d in definitions()
        )

    def call(self, definition, args):
        try:
            Draft202012Validator(definition.input_schema).validate(args)
            with self.lock:
                result = {"ok": True, **self._dispatch(definition.name, args)}
                if len(json.dumps(result, ensure_ascii=False).encode()) > 256 * 1024:
                    raise DataLakeError(
                        "Use fewer rows or download", code="result_too_large"
                    )
                return result
        except Exception as error:  # noqa: BLE001 - sanitize untrusted SDK errors
            code = (
                error.code
                if isinstance(error, DataLakeError)
                else "invalid_arguments"
                if isinstance(error, (ValueError, ValidationError))
                else "table_request_failed"
            )
            return {
                "ok": False,
                "error": {
                    "code": code,
                    "message": _error_hint(code, error, definition),
                    "retryable": bool(getattr(error, "retryable", False)),
                },
            }

    def _dispatch(self, name, args):
        if name == "tables":
            catalog = self._catalog()
            if not args.get("table_id"):
                return {
                    "market_data_note": MARKET_NOTE
                    if any(t["dataset"] == "market_data" for t in catalog["tables"])
                    else None,
                    "items": [
                        {k: t[k] for k in ("table_id", "dataset", "title")}
                        for t in catalog["tables"]
                    ],
                }
            table_id = args["table_id"]
            if table_id not in {t["table_id"] for t in catalog["tables"]}:
                raise DataLakeError("Unknown table", code="not_authorized")
            raw = self._request(
                "table_schema",
                table_id,
                catalog_snapshot_id=catalog["catalog_snapshot_id"],
            )
            if raw["table_id"] != table_id or raw["dataset"] not in self.allowed:
                raise DataLakeError("Unexpected schema", code="scope_mismatch")
            public = {
                k: raw[k]
                for k in (
                    "table_id",
                    "dataset",
                    "title",
                    "columns",
                    "key_columns",
                    "revision_policy",
                )
            }
            if raw["dataset"] == "market_data":
                public["market_data_note"] = MARKET_NOTE
            return public
        if name == "table_lookup":
            if args["dataset"] not in self.allowed:
                raise DataLakeError("Dataset unavailable", code="not_authorized")
            if len(args["query"].encode("utf-8")) > 255:
                raise DataLakeError(
                    "Use a shorter company name or code", code="identity_query_too_long"
                )
            raw = self._request("table_identities", **args, requested_as_of=self.cutoff)
            if raw["dataset"] != args["dataset"]:
                raise DataLakeError("Identity scope differs", code="scope_mismatch")
            self._check_time(raw["effective_as_of"])
            public = {
                k: raw[k]
                for k in (
                    "dataset",
                    "table_id",
                    "query_column",
                    "has_more",
                    "matching",
                    "interval_semantics",
                    "effective_as_of",
                )
            }
            public["items"] = _identity_items(raw)
            return public
        if name == "table_query":
            if len(self.results) >= 1000:
                raise DataLakeError("Too many queries", code="quota_exceeded")
            catalog = self._catalog()
            raw = self._request(
                "query_table",
                args["sql"],
                parameters=args.get("parameters", {}),
                requested_as_of=self.cutoff,
                catalog_snapshot_id=catalog["catalog_snapshot_id"],
                page_size=args.get("count", 50),
            )
            if raw["query_id"] != raw["result_id"] or raw["status"] != "queued":
                raise DataLakeError("Invalid accepted query", code="invalid_response")
            ref = "table-" + uuid4().hex
            self.results[ref] = {
                **raw,
                "count": args.get("count", 50),
                "snapshot": catalog["catalog_snapshot_id"],
            }
            return {"table_ref": ref, "status": raw["status"]}
        ref = args["table_ref"]
        if ref not in self.results:
            raise DataLakeError("Use a result from this run", code="unknown_table_ref")
        binding = self.results[ref]
        if name == "table_cancel":
            raw = self._request("cancel_table", binding["query_id"])
            if raw["query_id"] != binding["query_id"]:
                raise DataLakeError("Query differs", code="scope_mismatch")
            return {"table_ref": ref, "status": raw["status"]}
        params = {"count": args.get("count", binding["count"])}
        if args.get("cursor"):
            saved = self.cursors.get(args["cursor"])
            if (
                not saved
                or saved["ref"] != ref
                or ("count" in args and args["count"] != saved["count"])
            ):
                raise DataLakeError("Invalid continuation", code="invalid_cursor")
            params = {"cursor": saved["cursor"], "count": saved["count"]}
        raw = self._request("table_result", binding["result_id"], **params)
        if raw["query_id"] != binding["query_id"] or raw["status"] not in {
            "queued",
            "running",
            "ready",
        }:
            raise DataLakeError("Result differs", code="scope_mismatch")
        if raw["status"] != "ready":
            if name == "table_download":
                raise DataLakeError("Result is not ready", code="result_not_ready")
            return {"table_ref": ref, "status": raw["status"]}
        self._check_time(raw["effective_as_of"])
        if (
            raw["result_id"] != binding["result_id"]
            or raw["catalog_snapshot_id"] != binding["snapshot"]
        ):
            raise DataLakeError("Snapshot differs", code="scope_mismatch")
        if name == "table_download":
            return self._download(ref, binding, args)
        if (
            raw["returned_count"] != len(raw["rows"])
            or len(raw["rows"]) > params["count"]
            or raw["total_rows"] < len(raw["rows"])
            or any(len(r) != len(raw["columns"]) for r in raw["rows"])
        ):
            raise DataLakeError("Invalid window", code="invalid_response")
        cursor = None
        if raw.get("next_cursor"):
            if len(self.cursors) >= 10000:
                raise DataLakeError("Too many pages", code="quota_exceeded")
            cursor = "page-" + uuid4().hex
            self.cursors[cursor] = {
                "ref": ref,
                "cursor": raw["next_cursor"],
                "count": params["count"],
            }
        return {
            "table_ref": ref,
            **{
                k: raw.get(k)
                for k in (
                    "status",
                    "effective_as_of",
                    "columns",
                    "rows",
                    "returned_count",
                    "total_rows",
                    "total_count_exact",
                    "coverage",
                    "source_completeness",
                )
            },
            "next_cursor": cursor,
        }

    def _check_time(self, value):
        if datetime.fromisoformat(value) != self.cutoff:
            raise DataLakeError("Unexpected cutoff", code="scope_mismatch")

    def _download(self, ref, binding, args):
        from .data_tools import _safe_filename

        fmt = args.get("format", "parquet")
        filename = _safe_filename(args.get("filename") or f"result.{fmt}")
        with tempfile.TemporaryDirectory(prefix="prophet-table-") as directory:
            target = Path(directory) / "result"
            receipt = self._request(
                "download_table", binding["result_id"], target, format=fmt
            )
            if (
                str(receipt.result_id) != binding["result_id"]
                or receipt.format != fmt
                or Path(receipt.path) != target
            ):
                raise DataLakeError("Invalid download", code="integrity_mismatch")
            with target.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if (
                    target.stat().st_size != receipt.size_bytes
                    or actual != receipt.sha256
                ):
                    raise DataLakeError("Invalid file", code="integrity_mismatch")
                stream.seek(0)
                path = self.owner.backend.put_file(
                    f"downloads/{ref}-{uuid4().hex[:8]}-{filename}", stream
                )
        return {
            "table_ref": ref,
            "path": path,
            "filename": filename,
            "format": fmt,
            "size_bytes": receipt.size_bytes,
            "sha256": receipt.sha256,
        }


def _identity_items(raw):
    column = {
        "financial_statement": "provider_company_id",
        "analyst_estimate": "provider_ticker",
    }[raw["dataset"]]
    if raw["query_column"] != column:
        raise DataLakeError("Unexpected identity column", code="scope_mismatch")
    items = []
    for item in raw["items"]:
        value = item.get("provider_identity", item.get(column))
        if (
            not isinstance(value, str)
            or not value
            or (column in item and item[column] != value)
        ):
            raise DataLakeError("Invalid identity", code="invalid_response")
        items.append(
            {
                column: value,
                **{
                    k: item[k]
                    for k in (
                        "name",
                        "alias",
                        "alias_kind",
                        "alias_role",
                        "market",
                        "valid_from",
                        "valid_to",
                        "public_available_at",
                    )
                    if k in item
                },
            }
        )
    return items


def _error_hint(code, error, definition):
    """Actionable local hints, never raw backend messages or model argument values."""
    if isinstance(error, ValidationError):
        if error.validator == "required" and isinstance(error.instance, dict):
            missing = [k for k in error.validator_value if k not in error.instance]
            return "Provide the required fields: " + ", ".join(missing) + "."
        if error.validator == "additionalProperties":
            return (
                "Use only these fields: "
                + ", ".join(definition.input_schema["properties"])
                + "."
            )
        path = list(error.absolute_path)
        field = (
            path[0]
            if path and path[0] in definition.input_schema["properties"]
            else "arguments"
        )
        return f"Check {field}: use the declared type, choices and size limits."
    hints = {
        "invalid_query": "Check SQL against tables(table_id=...). Use one SELECT; bind $name values with parameters keys without $. No file or URL queries.",
        "invalid_arguments": "Use the tool's declared fields and types; count must be an integer from 1 to 200.",
        "invalid_request": "Check the declared fields and parameter values; use tables(table_id=...) for column names.",
        "identity_query_too_long": "Use a shorter registered company name or a security code.",
        "unknown_table_ref": "Use a table_ref returned by table_query in this run; submit a new query if needed.",
        "invalid_cursor": "Copy next_cursor unchanged with its original table_ref; omit count when continuing a page.",
        "result_not_ready": "Call table_read with this table_ref until ready, then download.",
        "result_expired": "Submit the SQL again with table_query; the old result can no longer be read.",
        "not_found": "Discover current tables or submit the query again; the requested object is unavailable.",
        "not_authorized": "Use only the tables and categories listed for this task.",
        "result_too_large": "Read fewer rows or columns, or use table_download for the complete result.",
        "quota_exceeded": "This run has reached its query/page budget; reuse existing results where possible.",
        "integrity_mismatch": "The download failed verification and was not delivered; retry the download if the result is still available.",
        "scope_mismatch": "The response did not match this task/query and was rejected; do not use it.",
    }
    return hints.get(
        code,
        "Table request failed. Retry transient errors; check available tables, arguments and result expiry.",
    )
