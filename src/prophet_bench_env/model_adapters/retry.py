"""Bounded provider retry semantics carried over from BenchResearch hv3.8."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


def is_transient_model_error(error: BaseException) -> bool:
    """Classify infrastructure/transient failures without retrying quota errors."""

    message = str(error).casefold()
    permanent_quota_markers = (
        "1113",
        "余额不足",
        "无可用资源包",
        "insufficient balance",
        "insufficient quota",
        "quota exhausted",
        "access key max cost limit exceeded",
        "max cost limit exceeded",
    )
    if any(marker in message for marker in permanent_quota_markers):
        return False
    status_code = getattr(error, "status_code", None)
    if status_code in {408, 409, 425, 429}:
        return True
    if isinstance(status_code, int) and status_code >= 500:
        return True
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True

    invalid_encrypted_content = "invalid_encrypted_content" in message or (
        "encrypted content for item" in message
        and (
            "could not be verified" in message
            or "could not be decrypted or parsed" in message
        )
    )
    if invalid_encrypted_content:
        return True
    commonstack_anthropic_route_mismatch = (
        "thinking.type.enabled" in message
        and "thinking.type.adaptive" in message
        and "output_config.effort" in message
    )
    if commonstack_anthropic_route_mismatch:
        return True
    transient_markers = (
        "request timed out",
        "timed out",
        "timeout",
        "connection error",
        "connection reset",
        "upstream request failed",
        "upstream error",
        "upstream_error",
        "bad gateway",
        "gateway timeout",
        "temporarily unavailable",
        "service unavailable",
        "rate limit",
        "rate_limit",
    )
    return any(marker in message for marker in transient_markers) or any(
        marker in type(error).__name__.casefold() for marker in transient_markers
    )


def retry_delay(retry_number: int, *, base: float, maximum: float) -> float:
    return min(base * (2 ** (retry_number - 1)), maximum)


def run_with_retries[T](
    operation: Callable[[], T],
    *,
    model_label: str,
    max_retries: int,
    base_seconds: float,
    max_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    for retry_number in range(max_retries + 1):
        try:
            return operation()
        except Exception as error:
            if retry_number >= max_retries or not is_transient_model_error(error):
                raise
            sleep(retry_delay(retry_number + 1, base=base_seconds, maximum=max_seconds))
    raise AssertionError(f"model retry loop exhausted for {model_label}")


async def run_with_retries_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    model_label: str,
    max_retries: int,
    base_seconds: float,
    max_seconds: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    for retry_number in range(max_retries + 1):
        try:
            return await operation()
        except Exception as error:
            if retry_number >= max_retries or not is_transient_model_error(error):
                raise
            await sleep(
                retry_delay(retry_number + 1, base=base_seconds, maximum=max_seconds)
            )
    raise AssertionError(f"model retry loop exhausted for {model_label}")


async def run_with_empty_response_retries_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    is_empty: Callable[[T], bool],
    on_exhausted: Callable[[T, int], T],
    on_success: Callable[[T, int], T],
    model_label: str,
    max_retries: int,
    base_seconds: float,
    retry_transport: Callable[[Callable[[], Awaitable[T]]], Awaitable[T]],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    for retry_number in range(max_retries + 1):
        result = await retry_transport(operation)
        if not is_empty(result):
            return on_success(result, retry_number)
        if retry_number >= max_retries:
            return on_exhausted(result, retry_number)  # type: ignore[return-value]
        await sleep(
            retry_delay(
                retry_number + 1,
                base=base_seconds,
                maximum=base_seconds * 4,
            )
        )
    raise AssertionError(f"empty response retry loop exhausted for {model_label}")


def run_with_empty_response_retries[T](
    operation: Callable[[], T],
    *,
    is_empty: Callable[[T], bool],
    on_exhausted: Callable[[T, int], T],
    on_success: Callable[[T, int], T],
    model_label: str,
    max_retries: int,
    base_seconds: float,
    retry_transport: Callable[[Callable[[], T]], T],
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    for retry_number in range(max_retries + 1):
        result = retry_transport(operation)
        if not is_empty(result):
            return on_success(result, retry_number)
        if retry_number >= max_retries:
            return on_exhausted(result, retry_number)
        sleep(
            retry_delay(retry_number + 1, base=base_seconds, maximum=base_seconds * 4)
        )
    raise AssertionError(f"empty response retry loop exhausted for {model_label}")
