# yggdrasil/data/cast/datetime.py
from __future__ import annotations

import datetime as dt
import math
import re
import time
from typing import Any, Optional

from .registry import register_converter

try:  # py3.9+
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


__all__ = [
    "CURRENT_TZINFO",
    "resolve_current_tzinfo",
    "normalize_fractional_seconds",
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
]


# ---- cached globals / compiled regexes ----

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
    r"(?:(\d+)d\s+)?"
    r"(\d{1,2}):(\d{1,2})"
    r"(?::(\d{1,2})(?:\.(\d{1,6}))?)?"
)
_RE_TIMEDELTA_UNIT = re.compile(r"(-?\d+(?:\.\d+)?)([smhd])")
_RE_TZ_OFFSET = re.compile(r"([+-])(\d{2})(?::?(\d{2}))?")

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


def resolve_current_tzinfo() -> dt.tzinfo:
    try:
        return _DATETIME.now().astimezone().tzinfo or _UTC
    except Exception:  # pragma: no cover
        return _UTC


CURRENT_TZINFO: dt.tzinfo = resolve_current_tzinfo()


def normalize_fractional_seconds(value: str) -> str:
    """Normalize fractional seconds to microsecond precision for fromisoformat()."""
    match = _RE_FRACTIONAL_SECONDS.search(value)
    if not match:
        return value
    start, end = match.span(2)
    frac = match.group(2)[:6].ljust(6, "0")
    return value[:start] + frac + value[end:]


def normalize_datetime_string(value: str) -> str:
    """
    Normalize common datetime string variants into something fromisoformat() likes.
    """
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

    if frac is None:
        frac_part = ""
    else:
        frac_part = "." + frac[:6].ljust(6, "0")

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
    return str_to_datetime(value, opts).date()


@register_converter(str, dt.datetime)
def str_to_datetime(value: str, opts: Any = None) -> dt.datetime:
    s = value.strip()

    if s == "utcnow":
        return _DATETIME.now(tz=_UTC)
    if s == "now":
        return _DATETIME.now(tz=CURRENT_TZINFO)

    s = normalize_datetime_string(s)

    try:
        parsed = _DATETIME.fromisoformat(s)
    except ValueError:
        last: Optional[ValueError] = None
        for fmt in _STRPTIME_FORMATS:
            try:
                parsed = _DATETIME.strptime(s, fmt)
                break
            except ValueError as e:  # pragma: no cover
                last = e
        else:
            raise last or ValueError(f"Cannot parse datetime from {value!r}")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_UTC)

    return parsed


@register_converter(str, dt.time)
def str_to_time(value: str, opts: Any = None) -> dt.time:
    return _TIME.fromisoformat(value)


@register_converter(str, dt.timedelta)
def str_to_timedelta(value: str, opts: Any = None) -> dt.timedelta:
    s = value.strip()

    m = _RE_TIMEDELTA_HMS.fullmatch(s)
    if m:
        days = int(m.group(1)) if m.group(1) else 0
        hours = int(m.group(2))
        minutes = int(m.group(3))
        seconds = int(m.group(4)) if m.group(4) else 0
        micro = int((m.group(5) or "0").ljust(6, "0"))
        return _TIMEDELTA(days=days, hours=hours, minutes=minutes, seconds=seconds, microseconds=micro)

    m = _RE_TIMEDELTA_UNIT.fullmatch(s)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit == "s":
            return _TIMEDELTA(seconds=val)
        if unit == "m":
            return _TIMEDELTA(seconds=val * 60.0)
        if unit == "h":
            return _TIMEDELTA(seconds=val * 3600.0)
        return _TIMEDELTA(seconds=val * 86400.0)

    try:
        return _TIMEDELTA(seconds=float(s))
    except ValueError as e:
        raise ValueError(f"Cannot parse timedelta from {value!r}") from e


@register_converter(str, dt.tzinfo)
def str_to_tzinfo(value: str, opts: Any = None) -> dt.tzinfo:
    s = value.strip()
    u = s.upper()

    if u == "UTC" or u == "Z":
        return _UTC
    if u == "LOCAL" or u == "CURRENT" or u == "NOW":
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
def date_to_datetime(value: dt.date, opts: Any = None) -> dt.datetime:
    return _DATETIME(value.year, value.month, value.day, tzinfo=CURRENT_TZINFO)


@register_converter(dt.time, dt.datetime)
def time_to_datetime(value: dt.time, opts: Any = None) -> dt.datetime:
    tz = value.tzinfo if value.tzinfo is not None else CURRENT_TZINFO
    return _DATETIME(1970, 1, 1, value.hour, value.minute, value.second, value.microsecond, tzinfo=tz)


@register_converter(dt.datetime, dt.datetime)
def datetime_to_datetime(value: dt.datetime, opts: Any = None) -> dt.datetime:
    return value


def _numeric_timestamp_to_seconds(value: int | float) -> float:
    v = float(value)
    if not _ISFINITE(v):
        raise ValueError(f"Cannot convert non-finite timestamp: {value!r}")

    x = -v if v < 0.0 else v
    now_s = _NOW_TS()

    # Heuristic:
    #   seconds      ~ 1e9
    #   milliseconds ~ 1e12
    #   microseconds ~ 1e15
    #
    # Use current epoch magnitude with cheap branch-only inference.
    if x < now_s * 100.0:
        return v
    if x < now_s * 100_000.0:
        return v * 1e-3
    return v * 1e-6


def _numeric_to_datetime(value: int | float) -> dt.datetime:
    return _FROMTIMESTAMP(_numeric_timestamp_to_seconds(value), tz=_UTC)


@register_converter(int, dt.datetime)
def int_to_datetime(value: int, opts: Any = None) -> dt.datetime:
    return _numeric_to_datetime(value)


@register_converter(float, dt.datetime)
def float_to_datetime(value: float, opts: Any = None) -> dt.datetime:
    return _numeric_to_datetime(value)


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
def any_to_datetime(value: Any, opts: Any = None) -> dt.datetime:
    if value is None:
        raise TypeError("Cannot convert None.")
    if isinstance(value, _DATETIME):
        return value
    if isinstance(value, str):
        return str_to_datetime(value, opts)
    if isinstance(value, bool):
        raise TypeError("Refusing to treat bool as epoch seconds for datetime conversion.")
    if isinstance(value, int):
        return int_to_datetime(value, opts)
    if isinstance(value, float):
        return float_to_datetime(value, opts)
    if isinstance(value, _DATE):
        return date_to_datetime(value, opts)
    if isinstance(value, _TIME):
        return time_to_datetime(value, opts)
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