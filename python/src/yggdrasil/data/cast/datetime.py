# yggdrasil/data/cast/datetime.py
from __future__ import annotations

import datetime as dt
import re
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


def resolve_current_tzinfo() -> dt.tzinfo:
    try:
        return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc
    except Exception:  # pragma: no cover
        return dt.timezone.utc


# persisted at import time
CURRENT_TZINFO: dt.tzinfo = resolve_current_tzinfo()


def normalize_fractional_seconds(value: str) -> str:
    """Normalize fractional seconds to microsecond precision for fromisoformat()."""
    match = re.search(r"(\.)(\d+)(?=(?:[+-]\d{2}:?\d{2})?$)", value)
    if not match:
        return value
    start, end = match.span(2)
    frac = match.group(2)[:6].ljust(6, "0")
    return value[:start] + frac + value[end:]


def normalize_datetime_string(value: str) -> str:
    """
    Normalize common datetime string variants into something fromisoformat() likes.

    Supported normalizations:
      - trailing "Z" -> "+00:00"
      - timezone offsets without colon: +HHMM / -HHMM -> +HH:MM / -HH:MM
      - "YYYY/MM/DD" -> "YYYY-MM-DD"
      - "YYYY-MM-DD HH:MM:SS" stays (fromisoformat accepts space)
      - compact date/time:
          * YYYYMMDD -> YYYY-MM-DD
          * YYYYMMDDHHMMSS -> YYYY-MM-DD HH:MM:SS
          * YYYYMMDDTHHMMSS -> YYYY-MM-DDTHH:MM:SS
        and with optional fractional seconds + optional timezone offset
    """
    s = value.strip()

    # Zulu suffix
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    # swap date separators
    # only apply to the leading date chunk
    s = re.sub(r"^(\d{4})/(\d{2})/(\d{2})", r"\1-\2-\3", s)

    # offsets like +0200 / -0530 -> +02:00 / -05:30
    s = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", s)

    # normalize fractional seconds (microseconds max)
    s = normalize_fractional_seconds(s)

    # Compact forms: YYYYMMDD..., optionally with 'T' or space, optionally with fractions and tz
    # Examples:
    #   20240131
    #   20240131T235959
    #   20240131235959
    #   20240131T235959.123Z
    #   20240131T235959.123456+0200
    m = re.fullmatch(
        r"(\d{4})(\d{2})(\d{2})"                          # date
        r"(?:[T\s]?"                                      # optional separator
        r"(\d{2})(\d{2})(\d{2})"                          # time HHMMSS
        r"(?:\.(\d{1,6}))?"                               # optional fraction
        r")?"
        r"(?:(Z)|([+-]\d{2}:?\d{2}))?$",                  # optional tz (Z or +HH:MM/+HHMM)
        s,
    )
    if m:
        yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
        HH, MM, SS = m.group(4), m.group(5), m.group(6)
        frac = m.group(7)
        z = m.group(8)
        off = m.group(9)

        date_part = f"{yyyy}-{mm}-{dd}"
        if HH is None:
            # date-only compact
            return date_part

        frac_part = ""
        if frac is not None:
            frac_part = "." + frac[:6].ljust(6, "0")

        tz_part = ""
        if z:
            tz_part = "+00:00"
        elif off:
            # ensure colon
            if len(off) == 5:  # +HHMM
                off = off[:3] + ":" + off[3:]
            tz_part = off

        # preserve 'T' if original had it, else use space
        sep = "T" if "T" in s else " "
        return f"{date_part}{sep}{HH}:{MM}:{SS}{frac_part}{tz_part}"

    return s


@register_converter(str, dt.date)
def str_to_date(value: str, opts: Any = None) -> dt.date:
    return str_to_datetime(value, opts).date()


@register_converter(str, dt.datetime)
def str_to_datetime(value: str, opts: Any = None) -> dt.datetime:
    s = value.strip()

    if s == "utcnow":
        return dt.datetime.now(tz=dt.timezone.utc)
    if s == "now":
        return dt.datetime.now(tz=CURRENT_TZINFO)

    s = normalize_datetime_string(s)

    try:
        parsed = dt.datetime.fromisoformat(s)
    except ValueError:
        last: Optional[ValueError] = None

        # Extra patterns (most common “real-world” garbage)
        for fmt in (
            # ISO-ish with space
            "%Y-%m-%d %H:%M%z",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S.%f%z",
            # ISO-ish without tz
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            # Slash date variants (post-normalization may already handle some)
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
            # Date-only
            "%Y-%m-%d",
        ):
            try:
                parsed = dt.datetime.strptime(s, fmt)
                break
            except ValueError as e:  # pragma: no cover
                last = e
        else:
            raise last or ValueError(f"Cannot parse datetime from {value!r}")

    # IMPORTANT: only strings get default tzinfo
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=CURRENT_TZINFO)

    return parsed


@register_converter(str, dt.time)
def str_to_time(value: str, opts: Any = None) -> dt.time:
    # supports "HH:MM[:SS[.ffffff]][+HH:MM]" per stdlib
    return dt.time.fromisoformat(value)


@register_converter(str, dt.timedelta)
def str_to_timedelta(value: str, opts: Any = None) -> dt.timedelta:
    s = value.strip()

    m = re.fullmatch(
        r"(?:(\d+)d\s+)?"
        r"(\d{1,2}):(\d{1,2})"
        r"(?::(\d{1,2})(?:\.(\d{1,6}))?)?",
        s,
    )
    if m:
        days = int(m.group(1)) if m.group(1) else 0
        hours = int(m.group(2))
        minutes = int(m.group(3))
        seconds = int(m.group(4)) if m.group(4) else 0
        micro = int((m.group(5) or "0").ljust(6, "0"))
        return dt.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds, microseconds=micro)

    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)([smhd])", s)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        return dt.timedelta(seconds=val * scale)

    try:
        return dt.timedelta(seconds=float(s))
    except ValueError as e:
        raise ValueError(f"Cannot parse timedelta from {value!r}") from e


@register_converter(str, dt.tzinfo)
def str_to_tzinfo(value: str, opts: Any = None) -> dt.tzinfo:
    s = value.strip()
    u = s.upper()

    if u in {"UTC", "Z"}:
        return dt.timezone.utc
    if u in {"LOCAL", "CURRENT", "NOW"}:
        return CURRENT_TZINFO

    m = re.fullmatch(r"([+-])(\d{2})(?::?(\d{2}))?", s)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        hh = int(m.group(2))
        mm = int(m.group(3) or "0")
        off = dt.timedelta(hours=hh, minutes=mm) * sign
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
    # date->datetime needs a tz; keep CURRENT_TZINFO
    return dt.datetime(value.year, value.month, value.day, tzinfo=CURRENT_TZINFO)


@register_converter(dt.time, dt.datetime)
def time_to_datetime(value: dt.time, opts: Any = None) -> dt.datetime:
    anchor = dt.date(1970, 1, 1)
    tz = value.tzinfo if value.tzinfo is not None else CURRENT_TZINFO
    return dt.datetime(
        anchor.year,
        anchor.month,
        anchor.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        tzinfo=tz,
    )


@register_converter(dt.datetime, dt.datetime)
def datetime_to_datetime(value: dt.datetime, opts: Any = None) -> dt.datetime:
    # IMPORTANT: do NOT default tzinfo here. Only str parsing does that.
    return value


@register_converter(int, dt.datetime)
def int_to_datetime(value: int, opts: Any = None) -> dt.datetime:
    return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)


@register_converter(float, dt.datetime)
def float_to_datetime(value: float, opts: Any = None) -> dt.datetime:
    return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc)


@register_converter(int, dt.date)
def int_to_date(value: int, opts: Any = None) -> dt.date:
    return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc).date()


@register_converter(float, dt.date)
def float_to_date(value: float, opts: Any = None) -> dt.date:
    return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).date()


@register_converter(int, dt.timedelta)
def int_to_timedelta(value: int, opts: Any = None) -> dt.timedelta:
    return dt.timedelta(seconds=float(value))


@register_converter(float, dt.timedelta)
def float_to_timedelta(value: float, opts: Any = None) -> dt.timedelta:
    return dt.timedelta(seconds=value)


@register_converter(dt.timedelta, dt.tzinfo)
def timedelta_to_tzinfo(value: dt.timedelta, opts: Any = None) -> dt.tzinfo:
    if value <= dt.timedelta(hours=-24) or value >= dt.timedelta(hours=24):
        raise ValueError("tz offset must be strictly between -24h and +24h")
    return dt.timezone(value)


@register_converter(dt.tzinfo, dt.timedelta)
def tzinfo_to_timedelta(value: dt.tzinfo, opts: Any = None) -> dt.timedelta:
    off = dt.datetime.now(tz=value).utcoffset()
    return off if off is not None else dt.timedelta(0)


@register_converter(Any, dt.datetime)
def any_to_datetime(value: Any, opts: Any = None) -> dt.datetime:
    if value is None:
        raise TypeError("Cannot convert None.")

    if isinstance(value, dt.datetime):
        return datetime_to_datetime(value, opts)

    if isinstance(value, str):
        return str_to_datetime(value, opts)

    if isinstance(value, bool):
        raise TypeError("Refusing to treat bool as epoch seconds for datetime conversion.")

    if isinstance(value, int):
        return int_to_datetime(value, opts)

    if isinstance(value, float):
        return float_to_datetime(value, opts)

    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return date_to_datetime(value, opts)

    if isinstance(value, dt.time):
        return time_to_datetime(value, opts)

    raise TypeError(f"No conversion path for {type(value).__name__} -> datetime")


@register_converter(Any, dt.date)
def any_to_date(value: Any, opts: Any = None) -> dt.date:
    if value is None:
        raise TypeError("Cannot convert None.")

    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value

    if isinstance(value, dt.datetime):
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

    if isinstance(value, dt.time):
        return value

    if isinstance(value, dt.datetime):
        return datetime_to_time(value, opts)

    if isinstance(value, str):
        return str_to_time(value, opts)

    raise TypeError(f"No conversion path for {type(value).__name__} -> time")


@register_converter(Any, dt.timedelta)
def any_to_timedelta(value: Any, opts: Any = None) -> dt.timedelta:
    if value is None:
        raise TypeError("Cannot convert None.")

    if isinstance(value, dt.timedelta):
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

    if isinstance(value, dt.timedelta):
        return timedelta_to_tzinfo(value, opts)

    if isinstance(value, bool):
        raise TypeError("Refusing to treat bool as offset seconds for tzinfo conversion.")

    if isinstance(value, int):
        return timedelta_to_tzinfo(int_to_timedelta(value, opts), opts)

    if isinstance(value, float):
        return timedelta_to_tzinfo(float_to_timedelta(value, opts), opts)

    raise TypeError(f"No conversion path for {type(value).__name__} -> tzinfo")