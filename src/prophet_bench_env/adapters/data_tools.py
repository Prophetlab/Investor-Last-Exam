"""Financial document tools, independent of tasks, model providers and sandboxes."""

from __future__ import annotations

import re
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, BinaryIO, Protocol
from uuid import uuid4

from jsonschema import Draft202012Validator

from ..core.audit_context import tool_call_scope
from ..core.contracts import Reader, RegisteredTool, ToolCall, ToolError, ToolResult
from ..core.trajectory import JsonlTrajectorySink
from ..ports.data_specs import (
    DATASETS,
    READ_FORMATS,
    TABLE_DATASETS,
    bash_definition,
    default_data_tool_definitions,
)
from .data_lake import DataLakeError
from .read_context import (
    ReadSelection,
    collect_read_context,
    reading_metadata,
    window_source_result,
)

MAX_READER_CHARACTERS = 200_000
READ_MAX_BYTES = 1024 * 1024


class WorkspaceBackend(Protocol):
    def put_file(self, relative_path: str, source: BinaryIO) -> str: ...


class CommandBackend(Protocol):
    def execute(self, command: str, *, timeout: int) -> Any: ...


@dataclass(slots=True)
class LocalWorkspaceBackend:
    root: Path

    def __post_init__(self):
        self.root = self.root.expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def put_file(self, relative_path: str, source: BinaryIO) -> str:
        target = self.root / _safe_relative_path(relative_path)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not target.resolve().is_relative_to(self.root):
            raise ValueError("workspace path escapes root")
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as output:
                source.seek(0)
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return str(target)


@dataclass(slots=True)
class CallbackWorkspaceBackend:
    writer: Callable[[str, Iterable[bytes]], None]

    def put_file(self, relative_path: str, source: BinaryIO) -> str:
        safe = _safe_relative_path(relative_path)
        source.seek(0)
        self.writer(safe, iter(lambda: source.read(1024 * 1024), b""))
        return safe


@dataclass(frozen=True, slots=True)
class PassthroughReader:
    """Deterministic reader for tests or caller-owned summarization."""

    supports_images = False

    def read(self, *, query, content, metadata, image=None):
        return content.strip()


@dataclass(slots=True)
class DataLakeTools:
    client: Any
    backend: WorkspaceBackend
    reader: Reader = field(default_factory=PassthroughReader)
    command_backend: CommandBackend | None = None
    resource_guard: Callable[[], AbstractContextManager[None]] | None = None
    allowed_datasets: tuple[str, ...] = DATASETS
    trajectory: JsonlTrajectorySink | None = None
    reader_characters: int | None = None
    read_max_bytes: int = READ_MAX_BYTES
    read_max_requests: int = 64
    refs: dict[str, dict] = field(default_factory=dict, init=False)
    _ids: dict[str, str] = field(default_factory=dict, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _subtypes: dict = field(default_factory=dict, init=False)
    _definitions: tuple = field(default=(), init=False)
    _read_cursors: dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _table_tools: Any = field(default=None, init=False, repr=False)
    _document_datasets: tuple = field(default=(), init=False)

    def __post_init__(self):
        if self.reader_characters is not None and self.reader_characters < 1024:
            raise ValueError("reader_characters must be at least 1024")
        if not 1024 <= self.read_max_bytes <= 1024 * 1024:
            raise ValueError("read_max_bytes must be between 1024 and 1048576")
        if self.read_max_requests < 1:
            raise ValueError("read_max_requests must be positive")
        with self._resource_slot():
            caps = self.client.capabilities()
        self._audit("capabilities", {}, caps)
        self.allowed_datasets = tuple(
            d
            for d in self.allowed_datasets
            if d in caps["datasets"]
            and d not in {"research_report", "institutional_holdings"}
        )
        if not self.allowed_datasets:
            raise DataLakeError(
                "No enabled document categories are authorized", code="not_authorized"
            )
        self._subtypes = {
            d: caps.get("subtypes", {}).get(d, []) for d in self.allowed_datasets
        }
        from .table_tools import TableTools, supported

        self._document_datasets = tuple(
            d
            for d in self.allowed_datasets
            if d not in TABLE_DATASETS
            and d in caps.get("document_datasets", self.allowed_datasets)
        )
        if supported(self.client, caps):
            self._table_tools = TableTools(self, caps)
        self._definitions = tuple(
            d
            for d in default_data_tool_definitions(
                datasets=self._document_datasets, subtypes=self._subtypes
            )
            if d.name != "search" or self._document_datasets
        )

    def _audit(self, operation, parameters, response):
        if self.trajectory is not None:
            self.trajectory.append(
                "datalake_response",
                {
                    "operation": operation,
                    "parameters": parameters,
                    "response": response,
                },
                actor="data_lake",
            )

    def _register(self, raw):
        if (
            not isinstance(raw, Mapping)
            or not raw.get("document_id")
            or raw.get("dataset") not in self.allowed_datasets
        ):
            raise DataLakeError(
                "Discovery returned an invalid or out-of-scope document",
                code="invalid_response",
            )
        identity = str(raw["document_id"])
        with self._lock:
            ref = self._ids.get(identity)
            if ref is None:
                ref = f"doc-{len(self._ids) + 1:04d}"
                self._ids[identity] = ref
            self.refs[ref] = {**raw, "document_ref": ref}
        return _model_item(self.refs[ref])

    def seed_document(self, raw):
        """Resolve task-supplied IDs through the same authorized metadata API."""
        with self._resource_slot():
            metadata = self.client.get_document(str(raw["document_id"]))
        self._audit("get_document", {"document_id": str(raw["document_id"])}, metadata)
        return self._register(metadata)

    def lookup(self, args):
        """Python convenience only; no second model-visible discovery tool."""
        return self.search({**args, "mode": "metadata"})

    def search(self, args):
        return self._invoke("search", args, self._search)

    def _search(self, args):
        datasets = tuple(args.get("datasets") or self._document_datasets)
        if not datasets:
            raise DataLakeError(
                "Use tables for structured data", code="use_table_query"
            )
        subtypes = args.get("subtypes", [])
        if subtypes and not args.get("datasets"):
            raise ValueError("subtypes requires explicit datasets")
        if set(subtypes) - {s for d in datasets for s in self._subtypes.get(d, [])}:
            raise ValueError("subtypes must belong to the selected datasets")
        parameters = {k: v for k, v in args.items() if k != "mode"}
        parameters.update(datasets=datasets)
        parameters.setdefault("limit", 20)
        mode = args.get("mode", "text")
        operation = self.client.lookup if mode == "metadata" else self.client.search
        with self._resource_slot():
            raw = operation(**parameters)
        self._audit("lookup" if mode == "metadata" else "search", parameters, raw)
        if not isinstance(raw.get("items"), list):
            raise DataLakeError(
                "Discovery response is missing items", code="invalid_response"
            )
        if any(item.get("dataset") not in datasets for item in raw["items"]):
            raise DataLakeError(
                "Discovery returned a non-document category", code="invalid_response"
            )
        result = {
            k: raw[k]
            for k in (
                "returned_count",
                "limit",
                "has_more",
                "next_cursor",
                "total_count",
                "total_count_exact",
                "partial",
                "partial_reason",
                "effective_as_of",
                "effective_published_from",
            )
            if k in raw
        }
        result.update(ok=True, items=[self._register(item) for item in raw["items"]])
        result.setdefault("returned_count", len(result["items"]))
        if mode == "text" and not 2 <= len(args["query"].split()) <= 10:
            result["query_warning"] = (
                "Text search works best with two to ten words; this query was accepted unchanged."
            )
        return result

    def read(self, args):
        return self._invoke("read", args, self._read)

    def _read(self, args):
        specs = args.get("sources") or [{k: v for k, v in args.items() if k != "query"}]
        # Validate every reference and range before fetching any source. A later
        # invalid source must not partially execute an otherwise invalid call.
        selections = [self._read_selection(spec) for spec in specs]
        if any(s.format == "png" for s in selections):
            if len(selections) != 1:
                raise ValueError(
                    "Read PNG pages separately; sources supports joint text/JSON reading"
                )
            return self._read_image({**specs[0], "query": args["query"]})
        return self._read_group(args["query"], selections)

    def _read_selection(self, spec):
        hit = self._resolve(spec["document_ref"])
        formats = hit.get("readable_format", [])
        chosen = spec.get("format") or next(
            (f for f in READ_FORMATS if f in formats), None
        )
        if chosen not in READ_FORMATS or chosen not in formats:
            raise DataLakeError(
                "Requested format is not readable; select a readable_format or use download",
                code="format_unavailable",
            )
        if "cursor" in spec and "start" in spec:
            raise ValueError("cursor and start are mutually exclusive")
        unit, count = spec.get("unit"), spec.get("count")
        if (
            unit
            and count
            and count > {"page": 10, "block": 100, "chunk": 10, "row": 200}[unit]
        ):
            raise ValueError(f"count exceeds the limit for unit={unit}")
        parameters = {
            k: spec[k] for k in ("unit", "count", "cursor", "start") if k in spec
        }
        automatic = not parameters
        token = str(spec.get("cursor", ""))
        if token.startswith("read-"):
            with self._lock:
                saved = self._read_cursors.get(token)
            if saved is None or saved["document_ref"] != hit["document_ref"]:
                raise ValueError(
                    "Read cursor is unknown or belongs to another document"
                )
            if "format" not in spec:
                chosen = saved["format"]
            if saved["format"] != chosen or any(
                k in spec and spec[k] != saved["parameters"].get(k)
                for k in ("unit", "count")
            ):
                raise ValueError(
                    "Read cursor format or range changed; pass the resume fields unchanged"
                )
            parameters = dict(saved["parameters"])
            automatic = saved["automatic"]
        elif "cursor" in parameters:
            # Public API cursors emitted by older explicit reads used this fixed
            # byte budget. Do not inject count=10 into a cursor for another count.
            parameters["max_bytes"] = self.read_max_bytes
        else:
            parameters.setdefault("start", 1)
            parameters.setdefault("count", 10)
        return ReadSelection(
            hit,
            chosen,
            parameters,
            automatic,
            "cursor" not in parameters and parameters.get("start", 1) == 1,
        )

    def _remember_read(self, selection, parameters):
        token = f"read-{uuid4().hex}"
        with self._lock:
            self._read_cursors[token] = {
                "document_ref": selection.hit["document_ref"],
                "format": selection.format,
                "parameters": dict(parameters),
                "automatic": selection.automatic,
            }
        return token

    def _fetch_read(self, selection, parameters):
        with self._resource_slot():
            raw = self.client.read_content(
                str(selection.hit["document_id"]),
                format=selection.format,
                datasets=(selection.hit["dataset"],),
                **parameters,
            )
        self._audit(
            "read",
            {
                "document_id": selection.hit["document_id"],
                "format": selection.format,
                **parameters,
            },
            raw,
        )
        if raw.get("format") != selection.format or not isinstance(
            raw.get("parts"), list
        ):
            raise DataLakeError(
                "Read response does not match the selected format",
                code="invalid_response",
            )
        return raw

    def _read_group(self, query, selections):
        metadata = reading_metadata(selections)
        get_budget = getattr(self.reader, "content_budget", None)
        budget = get_budget(query=query, metadata=metadata) if get_budget else None
        collection = collect_read_context(
            selections,
            fetch=self._fetch_read,
            remember=self._remember_read,
            budget=budget
            if budget is not None
            else self.reader_characters or MAX_READER_CHARACTERS,
            measure=self.reader.measure_content if budget is not None else len,
            character_limit=self.reader_characters,
            max_bytes=self.read_max_bytes,
            max_requests=self.read_max_requests,
        )
        sources = collection["sources"]
        errors = [s["error"] for s in sources if s.get("error")]
        result = {
            "ok": not errors,
            "sources": sources,
            "reader_text": "" if errors else "No content in the supplied ranges.",
        }
        if errors:
            result["error"] = {
                "code": "read_incomplete",
                "message": "Some sources could not be read; inspect sources for errors",
                "retryable": any(e.get("retryable") for e in errors),
            }
        if collection["has_content"]:
            try:
                result["reader_text"] = self.reader.read(
                    query=query,
                    content=collection["content"],
                    metadata=metadata,
                    image=None,
                )
            except Exception as error:  # noqa: BLE001 - preserve coverage on model failure
                result.update(
                    ok=False,
                    reader_text="",
                    error={"code": "reader_failed", "message": str(error)},
                )
        elif any(s["coverage"] == "partial" for s in sources) and not errors:
            result.update(
                ok=False,
                reader_text="",
                error={
                    "code": "reader_budget_exhausted",
                    "message": "No content fits this call; request fewer sources or a smaller range",
                },
            )
        return result

    def _read_image(self, args):
        hit = self._resolve(args["document_ref"])
        if not getattr(self.reader, "supports_images", False):
            raise DataLakeError(
                "The selected reader does not accept images",
                code="reader_image_unsupported",
            )
        if (
            "cursor" in args
            or args.get("unit", "page") != "page"
            or args.get("count", 1) != 1
        ):
            raise ValueError("PNG read requires one page, not a cursor or another unit")
        page = args.get("start", 1)
        with self._resource_slot():
            image = self.client.page_image(str(hit["document_id"]), page)
        import hashlib

        self._audit(
            "page_image",
            {"document_id": hit["document_id"], "page_number": page},
            {"sha256": hashlib.sha256(image).hexdigest(), "size_bytes": len(image)},
        )
        location = {"unit": "page", "start": page, "end": page}
        raw = {
            "format": "png",
            "requested_range": location,
            "returned_range": location,
            "window_complete": True,
            "has_more": None,
            "parts": [{"locator": {"unit": "page", "index": page}}],
        }
        source = window_source_result(
            ReadSelection(
                hit, "png", {"unit": "page", "start": page, "count": 1}, False, False
            ),
            raw,
            self._remember_read,
        )
        try:
            answer = self.reader.read(
                query=args["query"],
                content="",
                image=image,
                metadata={
                    "title": hit.get("title"),
                    "format": "png",
                    "navigation": {"returned_range": location},
                },
            )
        except Exception as error:  # noqa: BLE001 - reader boundary preserves coverage
            return {
                "ok": False,
                "sources": [source],
                "reader_text": "",
                "error": {"code": "reader_failed", "message": str(error)},
            }
        return {"ok": True, "sources": [source], "reader_text": answer}

    def download(self, args):
        return self._invoke("download", args, self._download)

    def _download(self, args):
        hit = self._resolve(args["document_ref"])
        chosen = args.get("format") or (
            "parquet" if "parquet" in hit["format"] else hit.get("source")
        )
        if not chosen and hit["format"]:
            chosen = hit["format"][0]
        if chosen not in hit["format"]:
            raise DataLakeError(
                "Choose a format listed for this document", code="format_unavailable"
            )
        page = args.get("page_number")
        if page is not None and chosen != "png":
            raise ValueError("page_number is only used for PNG")
        if chosen == "png" and hit.get("source") == "pdf" and page is None:
            raise ValueError("PDF PNG download requires page_number")
        suffix = {"markdown": "md", "plain_text": "txt", "binary": "bin"}.get(
            chosen, chosen
        )
        filename = _safe_filename(args.get("filename") or f"document.{suffix}")
        relative = f"downloads/{hit['document_ref']}-{uuid4().hex[:8]}-{filename}"
        with tempfile.SpooledTemporaryFile(
            max_size=16 * 1024**2, mode="w+b"
        ) as temporary:
            with self._resource_slot():
                receipt = self.client.download_document(
                    str(hit["document_id"]),
                    temporary,
                    format=chosen,
                    datasets=(hit["dataset"],),
                    page_number=page,
                )
            self._audit(
                "download",
                {
                    "document_id": hit["document_id"],
                    "format": chosen,
                    "page_number": page,
                },
                receipt.as_dict(),
            )
            path = self.backend.put_file(relative, temporary)
        return {
            "ok": True,
            "document_ref": hit["document_ref"],
            "path": path,
            "filename": filename,
            "format": receipt.format,
            "content_type": receipt.content_type,
            "size_bytes": receipt.size_bytes,
            "sha256": receipt.sha256,
        }

    def bash(self, args):
        if self.command_backend is None:
            return _error("bash_unavailable", "No isolated command backend is enabled.")
        try:
            Draft202012Validator(bash_definition().input_schema).validate(args)
            result = self.command_backend.execute(
                args["command"], timeout=args.get("timeout", 300)
            )
            return {
                "ok": result.exit_code == 0,
                "output": getattr(result, "stdout", ""),
                "stderr": getattr(result, "stderr", ""),
                "exit_code": result.exit_code,
                "timed_out": bool(result.timed_out),
            }
        except Exception as error:  # noqa: BLE001 - errors return to the tool caller
            return _error("bash_failed", str(error))

    def _invoke(self, name, args, operation):
        started = time.perf_counter()
        try:
            definition = next(d for d in self._definitions if d.name == name)
            Draft202012Validator(definition.input_schema).validate(args)
            result = operation(args)
        except DataLakeError as error:
            result = {"ok": False, "error": error.as_dict()}
        except (ValueError, TypeError) as error:
            result = _error("invalid_arguments", str(error))
        except Exception as error:  # noqa: BLE001 - errors return to the tool caller
            from jsonschema.exceptions import ValidationError

            result = _error(
                "invalid_arguments"
                if isinstance(error, ValidationError)
                else "tool_exception",
                str(error),
            )
        if self.trajectory is not None:
            self.trajectory.append(
                "data_tool_completed",
                {
                    "name": name,
                    "ok": result["ok"],
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "error_code": (result.get("error") or {}).get("code"),
                },
                actor="data_tools",
            )
        return result

    def _resolve(self, ref):
        with self._lock:
            if ref not in self.refs:
                raise DataLakeError(
                    "document_ref must come from search in this run",
                    code="unknown_document_ref",
                )
            return dict(self.refs[ref])

    def resolve_document_ref(self, ref):
        return str(self._resolve(ref)["document_id"])

    def _resource_slot(self):
        return self.resource_guard() if self.resource_guard else nullcontext()

    def registered_tools(self):
        definitions = self._definitions + (
            (bash_definition(),) if self.command_backend is not None else ()
        )
        return tuple(
            RegisteredTool(d, _handler_for(getattr(self, d.name))) for d in definitions
        ) + (self._table_tools.registered_tools() if self._table_tools else ())

    def close(self):
        try:
            if callable(getattr(self.backend, "close", None)):
                self.backend.close("normal")
        finally:
            self.client.close()


def _model_item(raw):
    """Explicit projection: no UUIDs, recipe, provider, grant, or license metadata."""
    readable = [f for f in raw.get("readable_format", []) if f in READ_FORMATS]
    return {
        "document_ref": raw["document_ref"],
        "title": raw.get("title"),
        "dataset": raw["dataset"],
        "subtype": raw.get("subtype"),
        "issuer": {"name": (raw.get("issuer") or {}).get("name")},
        "security": {
            k: (raw.get("security") or {}).get(k) for k in ("symbol", "market")
        },
        "period": raw.get("period"),
        "filing": raw.get("filing"),
        "availability": raw.get("availability"),
        "source": raw.get("source"),
        "format": list(raw.get("format", [])),
        "readable_format": readable,
        "recommended_action": "download"
        if "parquet" in raw.get("format", []) or not readable
        else "read",
        "snippets": (raw.get("match") or {}).get("snippets", []),
        "table": {
            k: raw["table"].get(k)
            for k in ("row_count", "schema_name", "metric", "basis", "currency")
        }
        if raw.get("table")
        else None,
    }


def _handler_for(handler):
    def execute(call: ToolCall):
        with tool_call_scope(call.call_id):
            output = handler(call.arguments)
        error = output.get("error") or {}
        return ToolResult(
            call.call_id,
            call.name,
            output.get("ok", True),
            output=output,
            error=ToolError(
                error.get("code", "tool_error"),
                error.get("message", "tool failed"),
                error.get("retryable", False),
            )
            if not output.get("ok", True)
            else None,
        )

    return execute


def _error(code, message):
    return {"ok": False, "error": {"code": code, "message": message}}


def _safe_relative_path(value):
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("workspace path must be relative without dot components")
    return str(path)


def _safe_filename(value):
    return (
        re.sub(r"[^A-Za-z0-9._-]+", "_", Path(value).name).strip("._")[:180]
        or "document.bin"
    )
