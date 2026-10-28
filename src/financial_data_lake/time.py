from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zoneinfo import TZPATH, ZoneInfo, ZoneInfoNotFoundError

from .errors import TimeNormalizationError

DEFAULT_TIMEZONE = "Asia/Shanghai"
TIME_PARSER_VERSION = "source-time-parser-v2"
ROUNDING_POLICY_VERSION = "public-as-of-ceiling-v1"


class TimePrecision(StrEnum):
    NANOSECOND = "nanosecond"
    MICROSECOND = "microsecond"
    MILLISECOND = "millisecond"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DATE = "date"
    MONTH = "month"
    INFERRED = "inferred"


class TimeTrustLevel(StrEnum):
    OFFICIAL = "official"
    OBSERVED = "observed"
    CUSTOMER_ASSERTED = "customer_asserted"
    INFERRED = "inferred"
    UNTRUSTED = "untrusted"


class NormalizationStatus(StrEnum):
    NORMALIZED = "normalized"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class NormalizedSourceTime:
    raw_value: str
    normalization_input: str
    source_timezone: str
    precision: TimePrecision
    source_published_at: datetime
    source_time_lower_bound: datetime
    source_time_upper_bound: datetime
    public_available_at: datetime
    parser_version: str
    rounding_policy_version: str
    tzdb_version: str
    trust_level: TimeTrustLevel
    status: NormalizationStatus = NormalizationStatus.NORMALIZED


def require_aware(value: datetime, *, field: str = "datetime") -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise TimeNormalizationError(f"{field} must include a timezone")
    return value.astimezone(UTC)


def parse_as_of(
    value: str | date | datetime,
    *,
    default_tz: str = DEFAULT_TIMEZONE,
    strict_naive: bool = False,
) -> datetime:
    """Return an exclusive UTC cutoff for a user-facing as-of value."""
    if isinstance(value, datetime):
        parsed: date | datetime = value
    elif isinstance(value, date):
        parsed = value
    else:
        raw = value.strip()
        if not raw:
            raise TimeNormalizationError("as_of cannot be empty")
        try:
            parsed = (
                date.fromisoformat(raw) if _is_iso_date(raw) else _parse_datetime(raw)
            )
        except ValueError as error:
            raise TimeNormalizationError(f"invalid as_of value: {value!r}") from error

    zone = _load_zone(default_tz)
    if isinstance(parsed, datetime):
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            if strict_naive:
                raise TimeNormalizationError("naive as_of datetime is not allowed")
            parsed = _localize(parsed, zone)
        return parsed.astimezone(UTC)

    next_midnight = datetime.combine(parsed + timedelta(days=1), time.min)
    return _localize(next_midnight, zone).astimezone(UTC)


def parse_published_from(
    value: str | date | datetime,
    *,
    default_tz: str = DEFAULT_TIMEZONE,
) -> datetime:
    """Inclusive lower bound; dates mean start of day, never next midnight."""
    zone = _load_zone(default_tz)
    if isinstance(value, datetime):
        return require_aware(value, field="published_from")
    if isinstance(value, date):
        return _localize(datetime.combine(value, time.min), zone).astimezone(UTC)
    raw = value.strip()
    try:
        if _is_iso_date(raw):
            return parse_published_from(date.fromisoformat(raw), default_tz=default_tz)
        parsed = _parse_datetime(raw)
        return require_aware(parsed, field="published_from")
    except ValueError as error:
        raise TimeNormalizationError("invalid published_from") from error


def normalize_source_time(
    value: str | int | date | datetime,
    *,
    source_tz: str,
    precision: TimePrecision | str,
    raw_value: str | None = None,
    fold: int | None = None,
    parser_version: str = TIME_PARSER_VERSION,
    rounding_policy_version: str = ROUNDING_POLICY_VERSION,
    trust_level: TimeTrustLevel | str = TimeTrustLevel.OFFICIAL,
) -> NormalizedSourceTime:
    """Normalize a trusted source time to a conservative visibility interval."""
    try:
        precision = TimePrecision(precision)
        trust_level = TimeTrustLevel(trust_level)
    except ValueError as error:
        raise TimeNormalizationError("unsupported time normalization policy") from error
    if precision is TimePrecision.INFERRED or trust_level in {
        TimeTrustLevel.INFERRED,
        TimeTrustLevel.UNTRUSTED,
    }:
        raise TimeNormalizationError("untrusted or inferred source time is quarantined")
    if not parser_version or not rounding_policy_version:
        raise TimeNormalizationError("normalization versions cannot be empty")

    zone = _load_zone(source_tz)
    normalized_input = _normalization_input(value)
    evidence_raw = raw_value if raw_value is not None else normalized_input
    if not evidence_raw:
        raise TimeNormalizationError("raw source time cannot be empty")

    lower, upper = _source_interval(value, precision, zone, fold=fold)
    lower_utc = lower.astimezone(UTC)
    upper_utc = upper.astimezone(UTC)
    if upper_utc <= lower_utc:
        raise TimeNormalizationError("source time interval must be non-empty")
    return NormalizedSourceTime(
        raw_value=evidence_raw,
        normalization_input=normalized_input,
        source_timezone=source_tz,
        precision=precision,
        source_published_at=lower_utc,
        source_time_lower_bound=lower_utc,
        source_time_upper_bound=upper_utc,
        public_available_at=upper_utc,
        parser_version=parser_version,
        rounding_policy_version=rounding_policy_version,
        tzdb_version=tzdb_version(),
        trust_level=trust_level,
    )


def replay_source_time(evidence: NormalizedSourceTime) -> NormalizedSourceTime:
    """Recompute persisted evidence and fail if the result is not reproducible."""
    if evidence.parser_version != TIME_PARSER_VERSION:
        raise TimeNormalizationError(
            f"unsupported parser version: {evidence.parser_version!r}"
        )
    if evidence.rounding_policy_version != ROUNDING_POLICY_VERSION:
        raise TimeNormalizationError(
            f"unsupported rounding policy version: {evidence.rounding_policy_version!r}"
        )
    current_tzdb = tzdb_version()
    if evidence.tzdb_version != current_tzdb:
        raise TimeNormalizationError(
            f"tzdb version mismatch: stored={evidence.tzdb_version!r}, "
            f"current={current_tzdb!r}"
        )
    value: str | int = evidence.normalization_input
    if (
        evidence.precision
        in {
            TimePrecision.NANOSECOND,
            TimePrecision.MILLISECOND,
        }
        and isinstance(value, str)
        and value.removeprefix("-").isdigit()
    ):
        value = int(value)
    replayed = normalize_source_time(
        value,
        source_tz=evidence.source_timezone,
        precision=evidence.precision,
        raw_value=evidence.raw_value,
        parser_version=evidence.parser_version,
        rounding_policy_version=evidence.rounding_policy_version,
        trust_level=evidence.trust_level,
    )
    comparable_fields = (
        "source_time_lower_bound",
        "source_time_upper_bound",
        "public_available_at",
        "raw_value",
        "normalization_input",
    )
    if any(
        getattr(replayed, field) != getattr(evidence, field)
        for field in comparable_fields
    ):
        raise TimeNormalizationError("persisted source time evidence does not replay")
    return replayed


def tzdb_version() -> str:
    """Return a persistable tzdb release identifier for normalization evidence."""
    try:
        return f"tzdata:{version('tzdata')}"
    except PackageNotFoundError:
        pass
    for root in TZPATH:
        candidate = Path(root) / "tzdata.zi"
        try:
            first_line = candidate.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError, UnicodeError):
            continue
        marker = "# version "
        if first_line.startswith(marker):
            return f"system:{first_line.removeprefix(marker).strip()}"
    return "system:unknown"


def format_local(value: datetime, *, target_tz: str = DEFAULT_TIMEZONE) -> str:
    return require_aware(value).astimezone(_load_zone(target_tz)).isoformat()


def effective_cutoff(requested: datetime | None, authorized: datetime) -> datetime:
    authorized_utc = require_aware(authorized, field="authorized_as_of")
    if requested is None:
        return authorized_utc
    return min(require_aware(requested, field="requested_as_of"), authorized_utc)


def _source_interval(
    value: str | int | date | datetime,
    precision: TimePrecision,
    zone: ZoneInfo,
    *,
    fold: int | None,
) -> tuple[datetime, datetime]:
    if precision is TimePrecision.NANOSECOND:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TimeNormalizationError("nanosecond epoch must be an integer")
        try:
            epoch_ns = int(value)
        except (TypeError, ValueError) as error:
            raise TimeNormalizationError(
                "nanosecond precision requires an integer Unix epoch"
            ) from error
        seconds, nanoseconds = divmod(epoch_ns, 1_000_000_000)
        lower = datetime.fromtimestamp(seconds, UTC) + timedelta(
            microseconds=nanoseconds // 1_000
        )
        return lower, lower + timedelta(microseconds=1)

    if precision is TimePrecision.MILLISECOND and isinstance(value, int):
        seconds, milliseconds = divmod(value, 1_000)
        lower = datetime.fromtimestamp(seconds, UTC) + timedelta(
            milliseconds=milliseconds
        )
        return lower, lower + timedelta(milliseconds=1)

    if precision is TimePrecision.DATE:
        source_date = _coerce_date(value)
        lower = _localize(datetime.combine(source_date, time.min), zone, fold=fold)
        upper = _localize(
            datetime.combine(source_date + timedelta(days=1), time.min), zone, fold=fold
        )
        return lower, upper

    if precision is TimePrecision.MONTH:
        year, month = _coerce_month(value)
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        lower = _localize(datetime(year, month, 1), zone, fold=fold)
        upper = _localize(datetime(next_year, next_month, 1), zone, fold=fold)
        return lower, upper

    parsed = _coerce_datetime(value)
    local = (
        _localize(parsed, zone, fold=fold)
        if parsed.tzinfo is None or parsed.utcoffset() is None
        else parsed.astimezone(zone)
    )
    _validate_precision(local, precision)
    delta = {
        TimePrecision.MICROSECOND: timedelta(microseconds=1),
        TimePrecision.MILLISECOND: timedelta(milliseconds=1),
        TimePrecision.SECOND: timedelta(seconds=1),
        TimePrecision.MINUTE: timedelta(minutes=1),
        TimePrecision.HOUR: timedelta(hours=1),
    }.get(precision)
    if delta is None:
        raise TimeNormalizationError(f"unsupported time precision: {precision.value!r}")
    lower_utc = local.astimezone(UTC)
    return lower_utc, lower_utc + delta


def _validate_precision(value: datetime, precision: TimePrecision) -> None:
    invalid = False
    if precision is TimePrecision.MICROSECOND:
        return
    if precision is TimePrecision.MILLISECOND:
        invalid = value.microsecond % 1_000 != 0
    elif precision is TimePrecision.SECOND:
        invalid = value.microsecond != 0
    elif precision is TimePrecision.MINUTE:
        invalid = value.second != 0 or value.microsecond != 0
    elif precision is TimePrecision.HOUR:
        invalid = value.minute != 0 or value.second != 0 or value.microsecond != 0
    if invalid:
        raise TimeNormalizationError(
            f"source value contains more precision than declared {precision.value}"
        )


def _normalization_input(value: str | int | date | datetime) -> str:
    if isinstance(value, bool):
        raise TimeNormalizationError("boolean is not a source timestamp")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value.strip()


def _is_iso_date(value: str) -> bool:
    return len(value) == 10 and value[4] == "-" and value[7] == "-"


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(
        value[:-1] + "+00:00" if value.endswith("Z") else value
    )


def _coerce_date(value: str | int | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise TimeNormalizationError("date precision requires an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise TimeNormalizationError(f"expected an ISO date, got {value!r}") from error


def _coerce_month(value: str | int | date | datetime) -> tuple[int, int]:
    if isinstance(value, (date, datetime)):
        return value.year, value.month
    if not isinstance(value, str) or len(value) != 7 or value[4] != "-":
        raise TimeNormalizationError("month precision requires YYYY-MM")
    try:
        parsed = date.fromisoformat(f"{value}-01")
    except ValueError as error:
        raise TimeNormalizationError(f"invalid source month: {value!r}") from error
    return parsed.year, parsed.month


def _coerce_datetime(value: str | int | date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if not isinstance(value, str):
        raise TimeNormalizationError("datetime precision requires an ISO datetime")
    try:
        return _parse_datetime(value)
    except ValueError as error:
        raise TimeNormalizationError(
            f"expected an ISO datetime, got {value!r}"
        ) from error


def _load_zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as error:
        raise TimeNormalizationError(f"unknown IANA timezone: {name!r}") from error


def _localize(value: datetime, zone: ZoneInfo, *, fold: int | None = None) -> datetime:
    if value.tzinfo is not None and value.utcoffset() is not None:
        return value.astimezone(zone)
    candidates: list[datetime] = []
    folds = (fold,) if fold is not None else (0, 1)
    for candidate_fold in folds:
        candidate = value.replace(tzinfo=zone, fold=candidate_fold)
        round_trip = candidate.astimezone(UTC).astimezone(zone)
        if round_trip.replace(tzinfo=None) == value:
            candidates.append(candidate)
    if not candidates:
        raise TimeNormalizationError(
            f"nonexistent local time {value.isoformat()} in {zone.key}"
        )
    offsets = {candidate.utcoffset() for candidate in candidates}
    if len(offsets) > 1 and fold is None:
        raise TimeNormalizationError(
            f"ambiguous local time {value.isoformat()} in {zone.key}; fold is required"
        )
    return candidates[0]
