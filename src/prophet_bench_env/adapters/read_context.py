"""Collect authorized document windows into one bounded, jointly readable context.

No model calls or credentials live here. A caller supplies scoped fetching and
session-local continuation storage. Only material that fits is marked covered.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .data_lake import DataLakeError


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@dataclass
class ReadSelection:
    hit: dict
    format: str
    parameters: dict
    automatic: bool
    from_beginning: bool


@dataclass
class _Source:
    selection: ReadSelection
    index: int
    pending: dict | None = None
    continuation: dict | None = None
    windows: list[dict] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    signatures: set[str] = field(default_factory=set)
    reason: str = "context_budget"
    complete: bool = False
    document_end: bool = False
    error: dict | None = None
    minimum_cost: int = 0

    def __post_init__(self):
        self.pending = dict(self.selection.parameters)
        self.continuation = dict(self.pending)

    def coverage(self):
        if self.selection.from_beginning and self.document_end:
            return "full_document"
        return "requested_range" if self.complete else "partial"

    def render(self):
        return (
            f"Source {self.index}: {self.selection.hit['document_ref']}\n"
            f"Coverage: {self.coverage()}\n" + "\n\n".join(self.texts)
        )


def reading_metadata(selections):
    return {
        "sources": [
            {
                "source_index": i,
                "document_ref": s.hit["document_ref"],
                "title": s.hit.get("title"),
                "format": s.format,
                "scope": "document" if s.automatic else "range",
            }
            for i, s in enumerate(selections, 1)
        ]
    }


def collect_read_context(
    selections,
    *,
    fetch,
    remember,
    budget,
    measure=len,
    character_limit=None,
    max_bytes=1024 * 1024,
    max_requests=64,
):
    """Read in rounds, sharing capacity and redistributing unused document shares.

    Service has_more describes a *finite window*, not the remaining document.
    Signed cursors bind max_bytes/count/unit, so continuations retain all three.
    There is no local slicing: an unaccepted response is retried from its original
    request on the next tool call, never resumed after unseen material.
    """
    sources = [_Source(s, i) for i, s in enumerate(selections, 1)]

    def render():
        return "\n\n".join(s.render() for s in sources)

    requests = 0
    while requests < max_requests:
        active = [s for s in sources if s.pending is not None and s.error is None]
        if not active:
            break
        progress = False
        for position, source in enumerate(active):
            if requests >= max_requests:
                break
            current = render()
            remaining = budget - measure(current) - 512
            if character_limit is not None:
                remaining = min(remaining, character_limit - len(current) - 512)
            allowance = remaining // (len(active) - position)
            # The service needs room for its envelope, locators and cursor in
            # addition to text. Stop before requesting an unusably small window.
            if allowance < max(4096, source.minimum_cost):
                continue
            parameters = dict(source.pending)
            parameters.setdefault("max_bytes", min(max_bytes, allowance))
            requests += 1
            try:
                raw = fetch(source.selection, parameters)
                next_parameters = _next_parameters(raw, parameters)
                if (
                    source.selection.automatic
                    and next_parameters
                    and "start" in next_parameters
                ):
                    # Once the real coordinate unit is known, use its allowed
                    # window size. Never alter a signed fragment continuation.
                    next_parameters["count"] = {
                        "page": 10,
                        "block": 100,
                        "chunk": 10,
                        "row": 200,
                    }[next_parameters["unit"]]
                signature = hashlib.sha256(
                    _json([raw.get("returned_range"), raw["parts"]]).encode()
                ).hexdigest()
                if signature in source.signatures or (
                    not raw["parts"] and next_parameters is not None
                ):
                    raise DataLakeError(
                        "Reading did not advance; retry the indicated range",
                        code="read_no_progress",
                    )
                window = _window(raw)
                source.texts.append(_window_text(raw))
                candidate = render()
                cost = measure(candidate) - measure(current)
                if (
                    cost > allowance
                    or measure(candidate) > budget - 512
                    or (
                        character_limit is not None
                        and len(candidate) > character_limit - 512
                    )
                ):
                    source.texts.pop()
                    source.minimum_cost = cost
                    continue
                source.minimum_cost = 0
                source.signatures.add(signature)
                source.windows.append(window)
                source.continuation = next_parameters
                source.document_end = (
                    next_parameters is None
                    and raw.get("window_complete") is True
                    and raw.get("has_more") is False
                )
                complete_window = raw.get("window_complete") is True
                source.complete = source.document_end or (
                    not source.selection.automatic and complete_window
                )
                if source.complete:
                    source.pending = None
                    source.reason = (
                        "document_end" if source.document_end else "range_end"
                    )
                elif next_parameters is None:
                    raise DataLakeError(
                        "Reading returned no reliable continuation or document end",
                        code="invalid_response",
                    )
                else:
                    source.pending = dict(next_parameters)
                progress = True
            except DataLakeError as error:
                source.reason = "read_error"
                source.error = error.as_dict()
        if not progress:
            break

    for source in sources:
        if source.pending is not None and source.error is None:
            source.reason = (
                "request_limit" if requests >= max_requests else "context_budget"
            )
    content = render()
    if measure(content) > budget or (
        character_limit is not None and len(content) > character_limit
    ):
        raise ValueError(
            "Reader budget cannot fit source descriptions; request fewer sources"
        )
    return {
        "content": content,
        "has_content": any(
            s.texts and any(w["locators"] for w in s.windows) for s in sources
        ),
        "sources": [_result(s, remember) for s in sources],
    }


def _next_parameters(raw, parameters):
    requested = raw.get("requested_range") or {}
    result = {
        k: parameters[k] for k in ("unit", "count", "max_bytes") if k in parameters
    }
    if requested.get("unit"):
        result["unit"] = requested["unit"]
    if requested.get("start") is not None and requested.get("end") is not None:
        result["count"] = requested["end"] - requested["start"] + 1
    if raw.get("next_cursor"):
        return {**result, "cursor": raw["next_cursor"]}
    if raw.get("truncated") or raw.get("window_complete") is False:
        raise DataLakeError(
            "Partial content is missing its cursor", code="invalid_response"
        )
    if raw.get("next_start") is not None:
        start = raw["next_start"]
        end = (raw.get("returned_range") or {}).get("end")
        if end is not None and start <= end:
            raise DataLakeError("Reading did not advance", code="read_no_progress")
        # A new finite window can use a smaller byte budget. It is not a cursor.
        result.pop("max_bytes", None)
        return {**result, "start": start}
    return None


def _window(raw):
    return {
        k: raw.get(k)
        for k in (
            "requested_range",
            "returned_range",
            "fragment",
            "window_complete",
            "truncated",
        )
    } | {"locators": [p["locator"] for p in raw["parts"]]}


def _window_text(raw):
    values = [f"Returned range: {_json(raw.get('returned_range'))}"]
    if raw.get("table_schema"):
        values.append(f"Table schema: {_json(raw['table_schema'])}")
    for part in raw["parts"]:
        text = part.get("text")
        if text is None:
            text = _json(part.get("rows") or [])
        values.append(f"Locator: {_json(part['locator'])}\n{text}")
    return "\n".join(values)


def _result(source, remember):
    selection = source.selection
    continuation = source.continuation
    remaining: dict[str, Any] | None = None
    if continuation is not None:
        cursor = remember(selection, continuation)
        last = source.windows[-1] if source.windows else {}
        locators = last.get("locators") or []
        locator = locators[-1] if locators else {}
        remaining = {
            "resume": {
                "document_ref": selection.hit["document_ref"],
                "format": selection.format,
                "cursor": cursor,
            },
        }
        if "cursor" in continuation:
            # The API's opaque cursor may continue the same page or the next
            # row/unit. Describe the last covered position, not a guessed start.
            remaining["after"] = {
                k: locator[k] for k in ("unit", "index", "char_end") if k in locator
            } or None
        else:
            remaining.update(
                unit=continuation.get("unit"), start=continuation.get("start")
            )
    result = {
        "document_ref": selection.hit["document_ref"],
        "title": selection.hit.get("title"),
        "format": selection.format,
        "coverage": source.coverage(),
        "covered_ranges": _covered_ranges(source.windows),
        "remaining": remaining,
    }
    if not source.complete:
        result["stop_reason"] = source.reason
    if source.error is not None:
        result["error"] = {
            k: source.error[k]
            for k in ("code", "message", "retryable")
            if k in source.error
        }
    return result


def _covered_ranges(windows):
    """Compact actual locators into adjacent ranges; keep partial character bounds."""
    ranges = []
    for window in windows:
        for locator in window["locators"]:
            current = {
                "unit": locator["unit"],
                "start": locator["index"],
                "end": locator["index"],
            }
            current.update(
                {
                    k: locator[k]
                    for k in ("char_start", "char_end")
                    if locator.get(k) is not None
                }
            )
            previous = ranges[-1] if ranges else None
            if (
                previous
                and previous["unit"] == current["unit"]
                and (
                    previous["end"] + 1 == current["start"]
                    or (
                        previous["end"] == current["start"]
                        and previous.get("char_end") is not None
                        and previous["char_end"] == current.get("char_start")
                    )
                )
            ):
                previous["end"] = current["end"]
                previous.pop("char_end", None)
                if "char_end" in current:
                    previous["char_end"] = current["char_end"]
            else:
                ranges.append(current)
    return ranges


def window_source_result(selection, raw, remember):
    """Use the same concise output for explicit single windows and page images."""
    source = _Source(selection, 1)
    source.windows = [_window(raw)]
    source.continuation = _next_parameters(raw, selection.parameters)
    source.complete = raw.get("window_complete") is True
    source.document_end = (
        source.complete and raw.get("has_more") is False and source.continuation is None
    )
    source.reason = "range_end" if source.complete else "window_limit"
    return _result(source, remember)
