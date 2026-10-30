"""Typed validation paths and receipts; no benchmark fields or scoring rules."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


class Wildcard(Enum):
    ANY = "any_segment"
    INDEX = "array_index"


ANY = Wildcard.ANY
ANY_INDEX = Wildcard.INDEX


def path_matches(path, pattern, *, descendants=False):
    """Match segments, not string prefixes; integer 0 differs from object key '0'."""
    if len(path) < len(pattern) or (not descendants and len(path) != len(pattern)):
        return False
    for actual, expected in zip(path, pattern):
        if expected is ANY:
            continue
        if expected is ANY_INDEX:
            if type(actual) is not int:
                return False
        elif type(actual) is not type(expected) or actual != expected:
            return False
    return True


def json_pointer(path):
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in path)


def resolve_pointer(document, pointer):
    """Resolve a JSON pointer into an actual typed path and its value."""
    if pointer == "":
        return (), document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with /")
    current, path = document, []
    for encoded in pointer[1:].split("/"):
        if re.search(r"~(?![01])", encoded):
            raise ValueError("invalid JSON pointer escape")
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise ValueError("array pointer needs a nonnegative canonical index")
            token = int(token)
        elif not isinstance(current, dict):
            raise TypeError("pointer traverses a scalar")
        current = current[token]
        path.append(token)
    return tuple(path), current


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: tuple[str | int, ...] = ()
    schema_path: tuple[str | int, ...] = ()
    message: str = ""
    source: str = "plugin"
    validator: str | None = None

    def __post_init__(self):
        if not self.code or not isinstance(self.code, str):
            raise ValueError("issue code is required")
        for key in ("path", "schema_path"):
            tokens = getattr(self, key)
            if not isinstance(tokens, (tuple, list)) or any(
                type(t) not in {str, int} or (type(t) is int and t < 0) for t in tokens
            ):
                raise ValueError(
                    "validation paths require string keys and nonnegative integer indices"
                )
            object.__setattr__(self, key, tuple(tokens))

    def as_dict(self):
        value = asdict(self)
        value.update(
            path=list(self.path),
            schema_path=list(self.schema_path),
            pointer=json_pointer(self.path),
        )
        return value


def schema_issues(schema, value, *, limit=4096):
    """Required/additional-property issues name actual fields, including array indices."""
    found = []

    def expand(error):
        path = tuple(error.absolute_path)
        paths = [path]
        if error.validator == "required" and isinstance(error.instance, dict):
            paths = [
                path + (key,)
                for key in error.validator_value
                if key not in error.instance
            ]
        elif error.validator == "additionalProperties" and isinstance(
            error.instance, dict
        ):
            properties = error.schema.get("properties", {})
            patterns = error.schema.get("patternProperties", {})
            paths = [
                path + (key,)
                for key in error.instance
                if key not in properties
                and not any(re.search(p, key) for p in patterns)
            ] or [path]
        for actual in paths:
            issue = ValidationIssue(
                "schema." + str(error.validator),
                actual,
                tuple(error.absolute_schema_path),
                error.message,
                "schema",
                str(error.validator),
            ).as_dict()
            if isinstance(error.schema, dict) and "properties" in error.schema:
                issue["allowed_properties"] = sorted(error.schema["properties"])
                issue["required_properties"] = list(error.schema.get("required", []))
            yield issue
        for child in error.context:
            yield from expand(child)

    seen = set()
    for error in Draft202012Validator(schema).iter_errors(value):
        for issue in expand(error):
            key = (issue["code"], tuple(issue["path"]), tuple(issue["schema_path"]))
            if key in seen:
                continue
            seen.add(key)
            if len(found) == limit:
                return found, True
            found.append(issue)
    return found, False


def check_attachment(
    root: Path, relative: str, *, field_path=(), max_bytes=64 * 1024**2
):
    """Check a task-local regular file; receipts do not embed host paths or file contents."""
    root = root.resolve()
    receipt: dict[str, Any] = {
        "version": "attachment-check.v1",
        "path": relative,
        "status": "invalid",
    }
    code = None
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or any(p in {"", ".", ".."} for p in relative.split("/"))
        or "\\" in relative
        or "\x00" in relative
    ):
        code = "attachment.unsafe_path"
    else:
        import errno

        descriptors = []
        try:
            descriptors.append(
                os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            )
            for part in Path(relative).parts[:-1]:
                descriptors.append(
                    os.open(
                        part,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=descriptors[-1],
                    )
                )
            fd = os.open(
                Path(relative).name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=descriptors[-1],
            )
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise OSError(errno.ENOTDIR, "attachment must be a regular file")
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    code = "attachment.unsafe_path"
                elif before.st_size > max_bytes:
                    code = "attachment.too_large"
                else:
                    h, count = hashlib.sha256(), 0
                    while block := stream.read(min(1024**2, max_bytes - count + 1)):
                        count += len(block)
                        if count > max_bytes:
                            code = "attachment.too_large"
                            break
                        h.update(block)
                    after = os.fstat(stream.fileno())
                    if (before.st_size, before.st_mtime_ns) != (
                        after.st_size,
                        after.st_mtime_ns,
                    ):
                        raise OSError("attachment changed during checking")
                    if code is None:
                        receipt.update(
                            status="valid", size_bytes=count, sha256=h.hexdigest()
                        )
        except OSError as error:
            if error.errno in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}:
                code = (
                    "attachment.missing"
                    if error.errno == errno.ENOENT
                    else "attachment.unsafe_path"
                )
            else:
                raise
        finally:
            for fd in reversed(descriptors):
                os.close(fd)
    # Unexpected I/O errors propagate: they are infrastructure errors, not bad answers.
    issues = (
        (ValidationIssue(code, tuple(field_path), message=code, source="attachment"),)
        if code
        else ()
    )
    return receipt, issues
