from __future__ import annotations

import datetime as dt
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from .registry import register_converter

try:  # py3.9+
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


__all__ = [
    "CURRENT_TZINFO",
    "normalize_fractional_seconds",
    "normalize_datetime_string",
    "str_to_date",
    "str_to_time",
    "str_to_datetime",
    "str_to_timedelta",
    "str_to_tzinfo",
    "datetime_to_date",
    "datetime_to_time",
    "date_to_datetime",
    "time_to_datetime",
    "datetime_to_datetime",
    "int_to_datetime",
    "float_to_datetime",
    "int_to_date",
    "float_to_date",
    "int_to_timedelta",
    "float_to_timedelta",
    "timedelta_to_tzinfo",
    "tzinfo_to_timedelta",
    "any_to_date",
    "any_to_time",
    "any_to_datetime",
    "any_to_timedelta",
    "any_to_tzinfo",
    "truncate_datetime",
    "truncate_datetime_value",
    "iter_datetime_ranges",
]


_UTC = dt.timezone.utc
_DATETIME = dt.datetime
_DATE = dt.date
_TIME = dt.time
_TIMEDELTA = dt.timedelta
_FROMTIMESTAMP = _DATETIME.fromtimestamp
_NOW_TS = time.time
_ISFINITE = math.isfinite

_RE_FRACTIONAL_SECONDS = re.compile(r"(\.)(\d+)(?=(?:[+-]\d{2}:?\d{2})?$)")
_RE_DATE_SLASH = re.compile(r"^(\d{4})/(\d{2})/(\d{2})")
_RE_TZ_NO_COLON = re.compile(r"([+-]\d{2})(\d{2})$")
_RE_COMPACT_DATETIME = re.compile(
    r"(\d{4})(\d{2})(\d{2})"
    r"(?:[T\s]?"
    r"(\d{2})(\d{2})(\d{2})"
    r"(?:\.(\d{1,6}))?"
    r")?"
    r"(?:(Z)|([+-]\d{2}:?\d{2}))?$"
)
_RE_TIMEDELTA_HMS = re.compile(
    r"(?:(?P<days>-?\d+)d\s+)?"
    r"(?P<hours>\d{1,2}):(?P<minutes>\d{1,2})"
    r"(?::(?P<seconds>\d{1,2})(?:\.(?P<fraction>\d{1,6}))?)?$"
)
_RE_TIMEDELTA_UNIT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.IGNORECASE)
_RE_TZ_OFFSET = re.compile(r"([+-])(\d{2})(?::?(\d{2}))?$")
_RE_ISO_DURATION = re.compile(
    r"^(?P<sign>[+-])?P"
    r"(?:(?P<years>\d+(?:\.\d+)?)Y)?"
    r"(?:(?P<months>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<weeks>\d+(?:\.\d+)?)W)?"
    r"(?:(?P<days>\d+(?:\.\d+)?)D)?"
    r"(?:T"
    r"(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
    r")?$",
    re.IGNORECASE,
)

_STRPTIME_FORMATS = (
    "%Y-%m-%d %H:%M%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%Y-%m-%d",
)

CURRENT_TZINFO: dt.tzinfo = _UTC


@dataclass(frozen=True, slots=True)
class _IntervalSpec:
    raw: str
    months: int = 0
    days: int = 0
    seconds: int = 0
    microseconds: int = 0

    @property
    def is_calendar(self) -> bool:
        return self.months != 0

    @property
    def is_fixed(self) -> bool:
        return self.months == 0 and (
            self.days != 0 or self.seconds != 0 or self.microseconds != 0
        )

    @property
    def fixed_delta(self) -> dt.timedelta:
        if self.months:
            raise ValueError("Calendar interval cannot be represented as timedelta.")
        return _TIMEDELTA(
            days=self.days,
            seconds=self.seconds,
            microseconds=self.microseconds,
        )


def _coerce_target_tzinfo(tz: Any = None, opts: Any = None) -> dt.tzinfo | None:
    candidate = tz if tz is not None else opts
    if candidate is None:
        return None
    if isinstance(candidate, dt.tzinfo):
        return candidate
    if isinstance(candidate, str):
        return str_to_tzinfo(candidate, opts)
    if isinstance(candidate, _TIMEDELTA):
        return timedelta_to_tzinfo(candidate, opts)
    raise TypeError(f"Cannot interpret timezone from {type(candidate).__name__}")


def _apply_target_tz(value: dt.datetime, tz: Any = None, opts: Any = None) -> dt.datetime:
    target_tz = _coerce_target_tzinfo(tz=tz, opts=opts)
    if target_tz is None:
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=_UTC)
    return value.astimezone(target_tz)


def normalize_fractional_seconds(value: str) -> str:
    match = _RE_FRACTIONAL_SECONDS.search(value)
    if not match:
        return value
    start, end = match.span(2)
    frac = match.group(2)[:6].ljust(6, "0")
    return value[:start] + frac + value[end:]


def normalize_datetime_string(value: str) -> str:
    s = value.strip()

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    s = _RE_DATE_SLASH.sub(r"\1-\2-\3", s)
    s = _RE_TZ_NO_COLON.sub(r"\1:\2", s)
    s = normalize_fractional_seconds(s)

    m = _RE_COMPACT_DATETIME.fullmatch(s)
    if not m:
        return s

    yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
    HH, MM, SS = m.group(4), m.group(5), m.group(6)
    frac = m.group(7)
    z = m.group(8)
    off = m.group(9)

    date_part = f"{yyyy}-{mm}-{dd}"
    if HH is None:
        return date_part

    frac_part = "" if frac is None else "." + frac[:6].ljust(6, "0")

    if z:
        tz_part = "+00:00"
    elif off:
        tz_part = off if len(off) != 5 else off[:3] + ":" + off[3:]
    else:
        tz_part = ""

    sep = "T" if "T" in s else " "
    return f"{date_part}{sep}{HH}:{MM}:{SS}{frac_part}{tz_part}"


@register_converter(str, dt.date)
def str_to_date(value: str, opts: Any = None) -> dt.date:
    return str_to_datetime(value, opts=opts).date()


@register_converter(str, dt.datetime)
def str_to_datetime(value: str, opts: Any = None, tz: Any = None) -> dt.datetime:
    s = value.strip()

    if s == "utcnow":
        return _apply_target_tz(_DATETIME.now(tz=_UTC), tz=tz, opts=opts)
    if s == "now":
        return _apply_target_tz(_DATETIME.now(tz=CURRENT_TZINFO), tz=tz, opts=opts)

    s = normalize_datetime_string(s)

    try:
        parsed = _DATETIME.fromisoformat(s)
    except ValueError:
        last: Optional[ValueError] = None
        for fmt in _STRPTIME_FORMATS:
            try:
                parsed = _DATETIME.strptime(s, fmt)
                break
            except ValueError as e:
                last = e
        else:
            raise last or ValueError(f"Cannot parse datetime from {value!r}")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_UTC)

    return _apply_target_tz(parsed, tz=tz, opts=opts)


@register_converter(str, dt.time)
def str_to_time(value: str, opts: Any = None) -> dt.time:
    return _TIME.fromisoformat(value)


@register_converter(str, dt.timedelta)
def str_to_timedelta(value: str, opts: Any = None) -> dt.timedelta:
    s = value.strip()

    m = _RE_TIMEDELTA_HMS.fullmatch(s)
    if m:
        days = int(m.group("days")) if m.group("days") else 0
        hours = int(m.group("hours"))
        minutes = int(m.group("minutes"))
        seconds = int(m.group("seconds")) if m.group("seconds") else 0
        micro = int((m.group("fraction") or "0").ljust(6, "0"))
        return _TIMEDELTA(
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=micro,
        )

    m = _RE_TIMEDELTA_UNIT.fullmatch(s)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "s":
            return _TIMEDELTA(seconds=val)
        if unit == "m":
            return _TIMEDELTA(minutes=val)
        if unit == "h":
            return _TIMEDELTA(hours=val)
        if unit == "d":
            return _TIMEDELTA(days=val)
        if unit == "w":
            return _TIMEDELTA(weeks=val)

    iso = _parse_iso_duration(s)
    if iso is not None:
        if iso.months:
            raise ValueError(
                f"Cannot convert calendar duration {value!r} to timedelta without a reference date."
            )
        return iso.fixed_delta

    try:
        return _TIMEDELTA(seconds=float(s))
    except ValueError as e:
        raise ValueError(f"Cannot parse timedelta from {value!r}") from e


@register_converter(str, dt.tzinfo)
def str_to_tzinfo(value: str, opts: Any = None) -> dt.tzinfo:
    s = value.strip()
    u = s.upper()

    if u in {"UTC", "Z"}:
        return _UTC
    if u in {"LOCAL", "CURRENT", "NOW"}:
        return CURRENT_TZINFO

    m = _RE_TZ_OFFSET.fullmatch(s)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        hh = int(m.group(2))
        mm = int(m.group(3) or "0")
        off = _TIMEDELTA(hours=hh, minutes=mm) * sign
        return timedelta_to_tzinfo(off, opts)

    if ZoneInfo is not None:
        try:
            return ZoneInfo(s)
        except Exception as e:
            raise ValueError(f"Cannot parse tzinfo from {value!r}") from e

    raise ValueError(f"Cannot parse tzinfo from {value!r} (zoneinfo unavailable)")


@register_converter(dt.datetime, dt.date)
def datetime_to_date(value: dt.datetime, opts: Any = None) -> dt.date:
    return value.date()


@register_converter(dt.datetime, dt.time)
def datetime_to_time(value: dt.datetime, opts: Any = None) -> dt.time:
    return value.timetz() if value.tzinfo else value.time()


@register_converter(dt.date, dt.datetime)
def date_to_datetime(value: dt.date, opts: Any = None, tz: Any = None) -> dt.datetime:
    return _apply_target_tz(
        _DATETIME(value.year, value.month, value.day, tzinfo=CURRENT_TZINFO),
        tz=tz,
        opts=opts,
    )


@register_converter(dt.time, dt.datetime)
def time_to_datetime(value: dt.time, opts: Any = None, tz: Any = None) -> dt.datetime:
    src_tz = value.tzinfo if value.tzinfo is not None else CURRENT_TZINFO
    return _apply_target_tz(
        _DATETIME(
            1970,
            1,
            1,
            value.hour,
            value.minute,
            value.second,
            value.microsecond,
            tzinfo=src_tz,
        ),
        tz=tz,
        opts=opts,
    )


@register_converter(dt.datetime, dt.datetime)
def datetime_to_datetime(value: dt.datetime, opts: Any = None, tz: Any = None) -> dt.datetime:
    return _apply_target_tz(value, tz=tz, opts=opts)


def _numeric_timestamp_to_seconds(value: int | float) -> float:
    v = float(value)
    if not _ISFINITE(v):
        raise ValueError(f"Cannot convert non-finite timestamp: {value!r}")

    x = -v if v < 0.0 else v
    now_s = _NOW_TS()

    if x < now_s * 100.0:
        return v
    if x < now_s * 100_000.0:
        return v * 1e-3
    return v * 1e-6


def _numeric_to_datetime(value: int | float, opts: Any = None, tz: Any = None) -> dt.datetime:
    return _apply_target_tz(
        _FROMTIMESTAMP(_numeric_timestamp_to_seconds(value), tz=_UTC),
        tz=tz,
        opts=opts,
    )


@register_converter(int, dt.datetime)
def int_to_datetime(value: int, opts: Any = None, tz: Any = None) -> dt.datetime:
    return _numeric_to_datetime(value, opts=opts, tz=tz)


@register_converter(float, dt.datetime)
def float_to_datetime(value: float, opts: Any = None, tz: Any = None) -> dt.datetime:
    return _numeric_to_datetime(value, opts=opts, tz=tz)


@register_converter(int, dt.date)
def int_to_date(value: int, opts: Any = None) -> dt.date:
    return _FROMTIMESTAMP(_numeric_timestamp_to_seconds(value), tz=_UTC).date()


@register_converter(float, dt.date)
def float_to_date(value: float, opts: Any = None) -> dt.date:
    return _FROMTIMESTAMP(_numeric_timestamp_to_seconds(value), tz=_UTC).date()


@register_converter(int, dt.timedelta)
def int_to_timedelta(value: int, opts: Any = None) -> dt.timedelta:
    return _TIMEDELTA(seconds=float(value))


@register_converter(float, dt.timedelta)
def float_to_timedelta(value: float, opts: Any = None) -> dt.timedelta:
    return _TIMEDELTA(seconds=value)


@register_converter(dt.timedelta, dt.tzinfo)
def timedelta_to_tzinfo(value: dt.timedelta, opts: Any = None) -> dt.tzinfo:
    if value <= _TIMEDELTA(hours=-24) or value >= _TIMEDELTA(hours=24):
        raise ValueError("tz offset must be strictly between -24h and +24h")
    return dt.timezone(value)


@register_converter(dt.tzinfo, dt.timedelta)
def tzinfo_to_timedelta(value: dt.tzinfo, opts: Any = None) -> dt.timedelta:
    off = _DATETIME.now(tz=value).utcoffset()
    return off if off is not None else _TIMEDELTA(0)


@register_converter(Any, dt.datetime)
def any_to_datetime(value: Any, opts: Any = None, tz: Any = None) -> dt.datetime:
    if value is None:
        raise TypeError("Cannot convert None.")
    if isinstance(value, _DATETIME):
        return datetime_to_datetime(value, opts=opts, tz=tz)
    if isinstance(value, str):
        return str_to_datetime(value, opts=opts, tz=tz)
    if isinstance(value, bool):
        raise TypeError("Refusing to treat bool as epoch seconds for datetime conversion.")
    if isinstance(value, int):
        return int_to_datetime(value, opts=opts, tz=tz)
    if isinstance(value, float):
        return float_to_datetime(value, opts=opts, tz=tz)
    if isinstance(value, _DATE):
        return date_to_datetime(value, opts=opts, tz=tz)
    if isinstance(value, _TIME):
        return time_to_datetime(value, opts=opts, tz=tz)
    raise TypeError(f"No conversion path for {type(value).__name__} -> datetime")


@register_converter(Any, dt.date)
def any_to_date(value: Any, opts: Any = None) -> dt.date:
    if value is None:
        raise TypeError("Cannot convert None.")
    if isinstance(value, _DATE) and not isinstance(value, _DATETIME):
        return value
    if isinstance(value, _DATETIME):
        return datetime_to_date(value, opts)
    if isinstance(value, str):
        return str_to_date(value, opts)
    if isinstance(value, bool):
        raise TypeError("Refusing to treat bool as epoch seconds for date conversion.")
    if isinstance(value, int):
        return int_to_date(value, opts)
    if isinstance(value, float):
        return float_to_date(value, opts)
    raise TypeError(f"No conversion path for {type(value).__name__} -> date")


@register_converter(Any, dt.time)
def any_to_time(value: Any, opts: Any = None) -> dt.time:
    if value is None:
        raise TypeError("Cannot convert None.")
    if isinstance(value, _TIME):
        return value
    if isinstance(value, _DATETIME):
        return datetime_to_time(value, opts)
    if isinstance(value, str):
        return str_to_time(value, opts)
    raise TypeError(f"No conversion path for {type(value).__name__} -> time")


@register_converter(Any, dt.timedelta)
def any_to_timedelta(value: Any, opts: Any = None) -> dt.timedelta:
    if value is None:
        raise TypeError("Cannot convert None.")
    if isinstance(value, _TIMEDELTA):
        return value
    if isinstance(value, str):
        return str_to_timedelta(value, opts)
    if isinstance(value, bool):
        raise TypeError("Refusing to treat bool as seconds for timedelta conversion.")
    if isinstance(value, int):
        return int_to_timedelta(value, opts)
    if isinstance(value, float):
        return float_to_timedelta(value, opts)
    if isinstance(value, dt.tzinfo):
        return tzinfo_to_timedelta(value, opts)
    raise TypeError(f"No conversion path for {type(value).__name__} -> timedelta")


@register_converter(Any, dt.tzinfo)
def any_to_tzinfo(value: Any, opts: Any = None) -> dt.tzinfo:
    if value is None:
        raise TypeError("Cannot convert None.")
    if isinstance(value, dt.tzinfo):
        return value
    if isinstance(value, str):
        return str_to_tzinfo(value, opts)
    if isinstance(value, _TIMEDELTA):
        return timedelta_to_tzinfo(value, opts)
    if isinstance(value, bool):
        raise TypeError("Refusing to treat bool as offset seconds for tzinfo conversion.")
    if isinstance(value, int):
        return timedelta_to_tzinfo(int_to_timedelta(value, opts), opts)
    if isinstance(value, float):
        return timedelta_to_tzinfo(float_to_timedelta(value, opts), opts)
    raise TypeError(f"No conversion path for {type(value).__name__} -> tzinfo")


def truncate_datetime(
    value: Any,
    interval: str | dt.timedelta,
    tz: str | dt.tzinfo | dt.timedelta | None = None,
    add_interval: bool = False,
) -> dt.datetime:
    """
    Truncate a datetime-like value to the boundary defined by `interval`.

    Supported interval examples:
        - dt.timedelta(seconds=15)
        - dt.timedelta(minutes=5)
        - dt.timedelta(hours=4)
        - dt.timedelta(days=7)
        - "PT15S", "PT15M", "PT4H"
        - "P1D", "P1W", "P1M", "P3M", "P1Y"

    Rules:
        - timedelta and fixed-width ISO intervals truncate from Unix epoch.
        - Calendar intervals (months/years) truncate from calendar boundaries.
        - If add_interval=True and value is not already aligned, return the next boundary.
    """
    dt_value = any_to_datetime(value, tz=tz)
    spec = _coerce_interval(interval)
    truncated = _truncate_datetime_value(dt_value, spec)

    if add_interval and truncated != dt_value:
        return _add_interval(truncated, spec)

    return truncated


def truncate_datetime_value(
    value: Any,
    spec: Any,
    tz: str | dt.tzinfo | dt.timedelta | None = None,
) -> dt.datetime:
    dt_value = any_to_datetime(value, tz=tz)
    interval_spec = spec if isinstance(spec, _IntervalSpec) else _coerce_interval(spec)
    return _truncate_datetime_value(dt_value, interval_spec)


def iter_datetime_ranges(
    start: Any,
    end: Any,
    interval: str | dt.timedelta,
    tz: str | dt.tzinfo | dt.timedelta | None = None,
) -> Iterator[tuple[dt.datetime, dt.datetime]]:
    """
    Iterate over aligned datetime ranges between `start` and `end`.

    Supported interval examples:
        - dt.timedelta(seconds=30)
        - dt.timedelta(minutes=15)
        - dt.timedelta(hours=4)
        - dt.timedelta(days=1)
        - "PT1S", "PT30S"
        - "PT1M", "PT15M"
        - "PT1H", "PT4H"
        - "P1D", "P7D"
        - "P1W"
        - "P1M", "P3M"
        - "P1Y"
    """
    start_dt = any_to_datetime(start, tz=tz)
    end_dt = any_to_datetime(end, tz=tz)

    if start_dt >= end_dt:
        return

    spec = _coerce_interval(interval)
    current = _truncate_datetime_value(start_dt, spec)

    while current < end_dt:
        nxt = _add_interval(current, spec)
        yield current, nxt
        current = nxt


def _coerce_interval(interval: str | dt.timedelta) -> _IntervalSpec:
    if isinstance(interval, _TIMEDELTA):
        return _interval_spec_from_timedelta(interval)
    if isinstance(interval, str):
        return _parse_interval(interval)
    raise TypeError(
        f"Unsupported interval type {type(interval).__name__}. "
        "Expected str or datetime.timedelta."
    )


def _interval_spec_from_timedelta(value: dt.timedelta) -> _IntervalSpec:
    if value <= _TIMEDELTA(0):
        raise ValueError(f"Interval must be positive: {value!r}")

    total_days = value.days
    total_seconds = value.seconds
    micros = value.microseconds

    return _IntervalSpec(
        raw=repr(value),
        days=total_days,
        seconds=total_seconds,
        microseconds=micros,
    )


def _parse_iso_duration(value: str) -> _IntervalSpec | None:
    m = _RE_ISO_DURATION.fullmatch(value.strip())
    if not m:
        return None

    sign = -1 if m.group("sign") == "-" else 1

    years = float(m.group("years") or 0.0)
    months = float(m.group("months") or 0.0)
    weeks = float(m.group("weeks") or 0.0)
    days = float(m.group("days") or 0.0)
    hours = float(m.group("hours") or 0.0)
    minutes = float(m.group("minutes") or 0.0)
    seconds = float(m.group("seconds") or 0.0)

    if not years.is_integer():
        raise ValueError(f"Fractional years are not supported: {value!r}")
    if not months.is_integer():
        raise ValueError(f"Fractional months are not supported: {value!r}")

    total_months = int(years) * 12 + int(months)

    total_seconds = (
        weeks * 7.0 * 86400.0
        + days * 86400.0
        + hours * 3600.0
        + minutes * 60.0
        + seconds
    )

    whole_seconds = math.floor(abs(total_seconds))
    micros = round((abs(total_seconds) - whole_seconds) * 1_000_000)

    if micros == 1_000_000:
        whole_seconds += 1
        micros = 0

    total_days, rem_seconds = divmod(whole_seconds, 86400)

    return _IntervalSpec(
        raw=value,
        months=sign * total_months,
        days=sign * int(total_days),
        seconds=sign * int(rem_seconds),
        microseconds=sign * micros,
    )


def _parse_interval(interval: str) -> _IntervalSpec:
    spec = _parse_iso_duration(interval)
    if spec is None:
        raise ValueError(
            f"Unsupported interval {interval!r}. "
            "Examples: 'PT1S', 'PT15M', 'PT4H', 'P1D', 'P1W', 'P1M', 'P1Y'."
        )

    if spec.months == 0 and spec.days == 0 and spec.seconds == 0 and spec.microseconds == 0:
        raise ValueError(f"Interval must be non-zero: {interval!r}")

    if spec.months != 0 and (spec.days != 0 or spec.seconds != 0 or spec.microseconds != 0):
        raise ValueError(
            f"Mixed calendar/fixed interval not supported for truncation/iteration: {interval!r}"
        )

    if spec.months < 0 or spec.days < 0 or spec.seconds < 0 or spec.microseconds < 0:
        raise ValueError(f"Negative intervals are not supported: {interval!r}")

    return spec


def _truncate_datetime_value(value: dt.datetime, spec: _IntervalSpec) -> dt.datetime:
    if spec.is_calendar:
        months = spec.months
        if months % 12 == 0:
            years = months // 12
            return _truncate_to_year_boundary(value, years)
        return _truncate_to_month_boundary(value, months)

    return _truncate_from_epoch(value, spec.fixed_delta)


def _add_interval(value: dt.datetime, spec: _IntervalSpec) -> dt.datetime:
    if spec.is_calendar:
        return _add_months(value, spec.months)
    return value + spec.fixed_delta


def _truncate_to_year_boundary(value: dt.datetime, years: int) -> dt.datetime:
    if years <= 0:
        raise ValueError(f"Invalid year interval: {years}")

    year = (value.year // years) * years
    if year < 1:
        year = 1

    return value.replace(
        year=year,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )


def _truncate_to_month_boundary(value: dt.datetime, months: int) -> dt.datetime:
    if months <= 0:
        raise ValueError(f"Invalid month interval: {months}")

    total_months = value.year * 12 + (value.month - 1)
    truncated_total = (total_months // months) * months
    year = truncated_total // 12
    month = (truncated_total % 12) + 1

    return value.replace(
        year=year,
        month=month,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )


def _truncate_from_epoch(value: dt.datetime, step: dt.timedelta) -> dt.datetime:
    step_us = _timedelta_to_microseconds(step)
    if step_us <= 0:
        raise ValueError(f"Invalid fixed interval: {step!r}")

    if value.tzinfo is None:
        epoch = _DATETIME(1970, 1, 1)
        delta = value - epoch
        delta_us = _timedelta_to_microseconds(delta)
        truncated_us = (delta_us // step_us) * step_us
        return epoch + _TIMEDELTA(microseconds=truncated_us)

    epoch_utc = _DATETIME(1970, 1, 1, tzinfo=_UTC)
    value_utc = value.astimezone(_UTC)
    delta = value_utc - epoch_utc
    delta_us = _timedelta_to_microseconds(delta)
    truncated_us = (delta_us // step_us) * step_us
    truncated_utc = epoch_utc + _TIMEDELTA(microseconds=truncated_us)
    return truncated_utc.astimezone(value.tzinfo)


def _timedelta_to_microseconds(value: dt.timedelta) -> int:
    return (
        (value.days * 86400 + value.seconds) * 1_000_000
        + value.microseconds
    )


def _days_in_month(year: int, month: int) -> int:
    if month == 2:
        is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if is_leap else 28
    if month in (4, 6, 9, 11):
        return 30
    return 31


def _add_months(value: dt.datetime, months: int) -> dt.datetime:
    total_months = value.year * 12 + (value.month - 1) + months
    year = total_months // 12
    month = (total_months % 12) + 1
    day = min(value.day, _days_in_month(year, month))

    return value.replace(year=year, month=month, day=day)
