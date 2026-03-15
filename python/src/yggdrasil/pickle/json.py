# yggdrasil/pickle/json.py
from __future__ import annotations

import base64
import json as _json
import re
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time, timezone, tzinfo
from functools import lru_cache
from typing import Any, IO, Iterable, overload

__all__ = ["load", "loads", "dump", "dumps"]


# ---------------------------------------------------------------------------
# Datetime parsing / encoding
# ---------------------------------------------------------------------------

_NULL_STR_VALUES = frozenset({"", "null", "None", "NaN"})
_RE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RE_TIME = re.compile(r"^\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?$")
_RE_RFC1123 = re.compile(r"^[A-Za-z]{3}, \d{2} [A-Za-z]{3} \d{4} \d{2}:\d{2}:\d{2} GMT$")

_RE_DT_SEC = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?$"
)
_RE_DT_MIN = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?:Z|[+-]\d{2}:?\d{2})?$")


def _normalize_tz(s: str) -> str:
    # Fast normalization for:
    #   - trailing Z -> +00:00
    #   - +0100 -> +01:00
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if (
        len(s) >= 5
        and (s[-5] in "+-")
        and s[-4:-2].isdigit()
        and s[-2:].isdigit()
        and s[-3] != ":"
    ):
        s = s[:-2] + ":" + s[-2:]
    return s


def _parse_dt_minutes_to_seconds(s: str) -> str:
    # Convert "YYYY-MM-DDTHH:MM(+TZ)" -> "YYYY-MM-DDTHH:MM:00(+TZ)"
    plus = s.rfind("+")
    minus = s.rfind("-")
    tz_pos = max(plus, minus)
    if tz_pos > 10:
        base, tz = s[:tz_pos], s[tz_pos:]
        return base + ":00" + tz
    return s + ":00"


def _apply_default_tz(x: datetime | time, default_tz: tzinfo | None) -> datetime | time:
    if default_tz is None:
        return x
    if isinstance(x, datetime):
        return x if x.tzinfo is not None else x.replace(tzinfo=default_tz)
    return x if x.tzinfo is not None else x.replace(tzinfo=default_tz)


def _reduce_zoneoffset_to_utc(x: datetime | time | date) -> datetime | time | date:
    """
    Reduce fixed-offset tzinfo (datetime.timezone(...), aka ZoneOffset style) to UTC.
    Keep "named" zones (e.g. zoneinfo.ZoneInfo) as-is.

    - datetime with timezone(+HH:MM) -> astimezone(UTC)
    - time with timezone(+HH:MM) -> shift clock and set tzinfo=UTC (date boundary ignored)
    - date unchanged
    """
    if isinstance(x, datetime):
        tz = x.tzinfo
        if tz is None:
            return x
        # Only fixed offsets: datetime.timezone (including timezone(timedelta(...)))
        if isinstance(tz, timezone) and tz is not timezone.utc:
            return x.astimezone(timezone.utc)
        return x

    if isinstance(x, time):
        tz = x.tzinfo
        if tz is None:
            return x
        if isinstance(tz, timezone) and tz is not timezone.utc:
            dt = datetime(1970, 1, 1, x.hour, x.minute, x.second, x.microsecond, tzinfo=tz)
            dt_utc = dt.astimezone(timezone.utc)
            return dt_utc.timetz().replace(tzinfo=timezone.utc)
        return x

    return x


# ---- Caching hot parsing paths ------------------------------------------------
#
# Big win: repeated timestamp strings are super common in trading/log pipelines.
# We cache the "raw parse" result that is independent of default_tz.
#
# Then default_tz is applied after cache-hit, so you still get correct tz attach.
#
@lru_cache(maxsize=8192)
def _try_parse_datetime_cached(s: str) -> datetime | date | time | None:
    """
    Parse supported datetime/date/time strings.
    IMPORTANT: Does NOT apply default_tz. That is applied by caller.
    IMPORTANT: Does perform "zoneoffset -> UTC" reduction for fixed offsets.
    """
    s = s.strip()
    if not s:
        return None

    # RFC1123
    if _RE_RFC1123.match(s):
        try:
            dt = datetime.strptime(s, "%a, %d %b %Y %H:%M:%S GMT")
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    # Datetimes with seconds
    if _RE_DT_SEC.match(s):
        s2 = _normalize_tz(s.replace(" ", "T", 1))
        try:
            dt = datetime.fromisoformat(s2)
            return _reduce_zoneoffset_to_utc(dt)
        except ValueError:
            return None

    # Datetimes with minutes (no seconds)
    if _RE_DT_MIN.match(s):
        s2 = _normalize_tz(s.replace(" ", "T", 1))
        s2 = _parse_dt_minutes_to_seconds(s2)
        try:
            dt = datetime.fromisoformat(s2)
            return _reduce_zoneoffset_to_utc(dt)
        except ValueError:
            return None

    # Date-only
    if _RE_DATE.match(s):
        try:
            return date.fromisoformat(s)
        except ValueError:
            return None

    # Time-only
    if _RE_TIME.match(s):
        try:
            # normalize fractional seconds to microseconds (6 digits)
            if "." in s:
                base, frac = s.split(".", 1)
                frac = (frac + "000000")[:6]
                s = f"{base}.{frac}"
            t = time.fromisoformat(s)
            return _reduce_zoneoffset_to_utc(t)
        except ValueError:
            return None

    return None


def _try_parse_datetime(s: str, default_tz: tzinfo | None) -> datetime | date | time | None:
    hit = _try_parse_datetime_cached(s)
    if hit is None:
        return None

    # Apply default_tz only for naive datetime/time.
    if default_tz is None:
        return hit

    if isinstance(hit, datetime):
        if hit.tzinfo is None:
            return hit.replace(tzinfo=default_tz)
        return hit

    if isinstance(hit, time):
        if hit.tzinfo is None:
            return hit.replace(tzinfo=default_tz)
        return hit

    return hit


# ---- Null normalization caching ------------------------------------------------

def _walk_parse_values(
    x: Any,
    *,
    default_tz: tzinfo | None,
    nulls: set[str],
    parse_datetimes: bool,
) -> Any:
    """
    Recursively:
      1) convert "null-ish" strings to None (case-insensitive, trimmed)
      2) optionally parse datetime/date/time strings
    """
    # Localize hot lookups (tiny win, but free)
    _walk = _walk_parse_values
    _parse = _try_parse_datetime

    if isinstance(x, str):
        if nulls and x in nulls:
            return None

        if parse_datetimes:
            hit = _parse(x, default_tz)

            return hit if hit is not None else x

        return x

    if isinstance(x, list):
        return [_walk(v, default_tz=default_tz, nulls=nulls, parse_datetimes=parse_datetimes) for v in x]

    if isinstance(x, tuple):
        return tuple(_walk(v, default_tz=default_tz, nulls=nulls, parse_datetimes=parse_datetimes) for v in x)

    if isinstance(x, dict):
        return {k: _walk(v, default_tz=default_tz, nulls=nulls, parse_datetimes=parse_datetimes) for k, v in x.items()}

    return x

def _is_json_serializable(obj: Any) -> bool:
    if isinstance(obj, (str, int, float, bool)):
        return True
    if isinstance(obj, (datetime, date, time)):
        return True
    if is_dataclass(obj):
        return True
    if isinstance(obj, bytes):
        return True
    if obj is None:
        return True
    return False

def _default_encoder(obj: Any) -> Any:
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if is_dataclass(obj):
        return {
            k: v
            for k, v in asdict(obj).items()
            if _is_json_serializable(v)
        }
    if isinstance(obj, bytes):
        return obj.decode("latin-1")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@overload
def loads(
    s: str | bytes | bytearray | memoryview,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    parse_datetimes: bool = True,
    default_tz: tzinfo | None = None,
    null_str_values: Iterable[str] | None = _NULL_STR_VALUES,
) -> Any: ...


def loads(
    s: str | bytes | bytearray | memoryview,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    parse_datetimes: bool = True,
    default_tz: tzinfo | None = None,
    null_str_values: Iterable[str] | bool | None = True,
) -> Any:
    """
    Parse JSON from str/bytes.

    parse_datetimes:
      Parse supported datetime/date/time strings recursively.

    default_tz:
      If provided, attach tzinfo to parsed naive datetime/time values.

    null_str_values:
      Iterable of strings treated as JSON null when found as string values
      (case-insensitive, trimmed). Example: {"", "null", "none", "na", "n/a"}.
    """
    if null_str_values is True:
        null_str_values = _NULL_STR_VALUES
    elif null_str_values is False:
        null_str_values = None

    if isinstance(s, (bytes, bytearray, memoryview)):
        text = bytes(s).decode(encoding, errors=errors)
    else:
        text = s

    obj = _json.loads(text)

    # ---- Fast paths: avoid walking when nothing to do -----------------------
    if not parse_datetimes and not null_str_values:
        return obj

    if not parse_datetimes and not null_str_values:
        return obj

    # If parse_datetimes is False but we have nulls, we still need a walk.
    # If parse_datetimes is True but nulls empty, still need parse walk.
    return _walk_parse_values(
        obj,
        default_tz=default_tz,
        nulls=null_str_values, parse_datetimes=parse_datetimes
    )


def dumps(
    obj: Any,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    default: Any = _default_encoder,
) -> bytes:
    # default compact output (no whitespace)
    if separators is None and indent is None:
        separators = (",", ":")

    if is_dataclass(obj):
        obj = {
            k: v
            for k, v in asdict(obj)
            if _is_json_serializable(v)
        }

    text = _json.dumps(
        obj,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        indent=indent,
        separators=separators,
        default=default,
    )
    return text.encode(encoding, errors=errors)


def load(
    fp: IO[Any],
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    parse_datetimes: bool = True,
    default_tz: tzinfo | None = None,
    null_str_values: Iterable[str] | bool | None = True,
) -> Any:
    data = fp.read()
    return loads(
        data,
        encoding=encoding,
        errors=errors,
        parse_datetimes=parse_datetimes,
        default_tz=default_tz,
        null_str_values=null_str_values,
    )


def dump(
    obj: Any,
    fp: IO[Any],
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    default: Any = _default_encoder,
) -> None:
    b = dumps(
        obj,
        encoding=encoding,
        errors=errors,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        indent=indent,
        separators=separators,
        default=default,
    )
    try:
        fp.write(b)
    except TypeError:
        fp.write(b.decode(encoding, errors=errors))  # type: ignore[arg-type]