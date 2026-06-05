"""Centralized time-unit enum shared by every temporal ``DataType``.

Temporal types (``DateType`` / ``TimeType`` / ``TimestampType`` /
``DurationType``) carry a ``unit`` field — a short string token like
``"us"`` or ``"ns"`` — that decides Arrow precision, Polars dtype
mapping, Spark unit conversion, and scalar epoch math. Before this
module the canonical token list lived inline in ``temporal.py`` as a
loose dict, and aliases (``"microsecond"``, ``"nanoseconds"``,
``"millis"``, …) coming in from user dicts had no one place to be
normalized.

:class:`TimeUnit` provides:

* canonical members for the eight units the temporal types actually
  understand (``s``, ``ms``, ``us``, ``ns``, ``d``, plus the interval
  forms ``year_month`` / ``day_time`` / ``month_day_nano``);
* :meth:`parse` for forgiving string / ``TimeUnit`` / ``None`` input
  with alias support;
* :meth:`is_valid` for boolean checks without raising;
* :attr:`seconds` / :attr:`order` for scalar conversion and
  precision-rank comparisons.

The enum subclasses :class:`str` so a ``TimeUnit`` instance is
interchangeable with the string token everywhere it lands —
``unit: str = "us"`` field declarations don't need to change.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

__all__ = [
    "TimeUnit",
]


# ---------------------------------------------------------------------------
# Alias table — accepted spellings that normalize to a canonical token.
# Kept intentionally small: long forms (``microsecond``), plurals
# (``microseconds``), and a couple of common short variants
# (``ms`` / ``millis``). Anything else raises a clear error.
# ---------------------------------------------------------------------------

_TIMEUNIT_ALIASES: dict[str, str] = {
    "s":            "s",
    "sec":          "s",
    "second":       "s",
    "seconds":      "s",
    "secs":         "s",
    "ms":           "ms",
    "milli":        "ms",
    "millis":       "ms",
    "millisecond":  "ms",
    "milliseconds": "ms",
    "us":           "us",
    "µs":           "us",
    "micro":        "us",
    "micros":       "us",
    "microsecond":  "us",
    "microseconds": "us",
    "ns":           "ns",
    "nano":         "ns",
    "nanos":        "ns",
    "nanosecond":   "ns",
    "nanoseconds":  "ns",
    "d":            "d",
    "day":          "d",
    "days":         "d",
    "year_month":   "year_month",
    "yearmonth":    "year_month",
    "month_year":   "year_month",
    "day_time":     "day_time",
    "daytime":      "day_time",
    "month_day_nano": "month_day_nano",
    "monthdaynano":   "month_day_nano",
}


class TimeUnit(str, Enum):
    """Canonical time-unit token for temporal ``DataType`` instances.

    Members carry the lowercase short form so the enum is a drop-in
    string replacement::

        TimeType(unit=TimeUnit.MICROSECOND)
        TimeType(unit="us")  # equivalent — both store ``"us"``

    Use :meth:`parse` when accepting external input — it canonicalizes
    aliases (``"microseconds"``, ``"micros"``, ``"µs"``) to a member
    and raises :class:`ValueError` for unknown tokens.
    """

    SECOND = "s"
    MILLISECOND = "ms"
    MICROSECOND = "us"
    NANOSECOND = "ns"
    DAY = "d"
    YEAR_MONTH = "year_month"
    DAY_TIME = "day_time"
    MONTH_DAY_NANO = "month_day_nano"

    # ── Convenience aliases for callers that prefer the short name ──
    S: ClassVar["TimeUnit"]
    MS: ClassVar["TimeUnit"]
    US: ClassVar["TimeUnit"]
    NS: ClassVar["TimeUnit"]
    D: ClassVar["TimeUnit"]

    # ── Coercion ────────────────────────────────────────────────────────────

    @classmethod
    def from_(cls, value: Any, *, default: Any = ...) -> "TimeUnit":
        """Coerce any Python value into a :class:`TimeUnit`.

        Accepts:

        * :class:`TimeUnit` (returned as-is);
        * any string the alias table or canonical member values know —
          ``s`` / ``ms`` / ``us`` / ``ns`` / ``d`` / interval forms,
          plurals (``microseconds``), long forms (``millisecond``),
          mixed case, ``µs``, hyphens / spaces;
        * objects exposing a ``time_unit`` / ``unit`` attribute (Polars
          ``Datetime`` / ``Duration``, PyArrow ``TimestampType`` /
          ``DurationType`` / ``Time32Type`` / ``Time64Type``); the
          attribute is re-funneled through ``from_``;
        * ``None`` — returns *default* if supplied, else raises.

        ``default`` swallows unknown / unparseable input. Without it,
        unknown tokens raise :class:`ValueError` and unsupported types
        raise :class:`TypeError`.
        """
        if isinstance(value, cls):
            return value

        if value is None:
            if default is not ...:
                return default
            raise ValueError("TimeUnit cannot be derived from None")

        if isinstance(value, str):
            return cls._from_str(value, default=default)

        # Engine dtypes — pull out ``time_unit`` / ``unit`` and recurse.
        for attr in ("time_unit", "unit"):
            inner = getattr(value, attr, None)
            if inner is not None and inner is not value:
                return cls.from_(inner, default=default)

        if default is not ...:
            return default
        raise TypeError(
            f"Cannot derive TimeUnit from {type(value).__name__}: {value!r}"
        )

    @classmethod
    def _from_str(cls, value: str, *, default: Any = ...) -> "TimeUnit":
        # Fast path: already-canonical token (``"us"`` / ``"ms"`` /
        # ``"day_time"`` / ``"US"`` / ``"microsecond"``). A single
        # ``dict.get`` resolves them without paying any string
        # normalisation cost.
        hit = _TIMEUNIT_LOOKUP.get(value)
        if hit is not None:
            return hit

        token = value.strip().lower().replace("-", "_").replace(" ", "_")
        if not token:
            if default is not ...:
                return default
            raise ValueError("TimeUnit string cannot be empty")

        hit = _TIMEUNIT_LOOKUP.get(token)
        if hit is not None:
            return hit

        if default is not ...:
            return default
        raise ValueError(
            f"Unknown time unit: {value!r}. "
            f"Valid units are: {sorted({m.value for m in cls})!r}"
        )

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        """Return ``True`` when :meth:`from_` would succeed for *value*."""
        try:
            cls.from_(value)
            return True
        except (TypeError, ValueError):
            return False

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def seconds(self) -> float:
        """Seconds per one of this unit (used for scalar epoch math).

        Calendar-style interval units (``year_month`` / ``day_time`` /
        ``month_day_nano``) have no fixed second-count and return
        ``float('nan')`` so comparisons surface the mismatch instead of
        silently truncating to zero.
        """
        return _UNIT_SECONDS[self.value]

    @property
    def order(self) -> int:
        """Precision rank — higher = finer.

        Used by ``TemporalType._merge_with_same_id`` to pick the wider
        of two units. Calendar interval units sit at rank ``-1`` so
        they don't outrank fixed-precision ones in normal merges.
        """
        return _UNIT_ORDER[self.value]

    @property
    def is_subsecond(self) -> bool:
        """``True`` for ``ms`` / ``us`` / ``ns``."""
        return self in (TimeUnit.MILLISECOND, TimeUnit.MICROSECOND, TimeUnit.NANOSECOND)

    @property
    def is_calendar(self) -> bool:
        """``True`` for the variable-length interval forms."""
        return self in (
            TimeUnit.YEAR_MONTH,
            TimeUnit.DAY_TIME,
            TimeUnit.MONTH_DAY_NANO,
        )

    # ── Dunder ──────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return self.value


# Convenience short-name aliases — assigned after the class body so the
# Enum machinery doesn't try to register them as separate members.
TimeUnit.S = TimeUnit.SECOND
TimeUnit.MS = TimeUnit.MILLISECOND
TimeUnit.US = TimeUnit.MICROSECOND
TimeUnit.NS = TimeUnit.NANOSECOND
TimeUnit.D = TimeUnit.DAY


_UNIT_SECONDS: dict[str, float] = {
    "s":              1.0,
    "ms":             1.0e-3,
    "us":             1.0e-6,
    "ns":             1.0e-9,
    "d":              86400.0,
    "year_month":     float("nan"),
    "day_time":       float("nan"),
    "month_day_nano": float("nan"),
}


_UNIT_ORDER: dict[str, int] = {
    "s":              0,
    "ms":             1,
    "us":             2,
    "ns":             3,
    "d":              -2,
    "year_month":     -1,
    "day_time":       -1,
    "month_day_nano": -1,
}


def _build_timeunit_lookup() -> dict[str, TimeUnit]:
    """Pre-compute every accepted spelling → :class:`TimeUnit` member.

    Folds the alias table (lower-case keys) with the canonical token
    values (already TimeUnit values, but stored as plain strings),
    upper-case variants, and member names so :meth:`TimeUnit._from_str`
    resolves any common spelling with a single ``dict.get``.
    """
    out: dict[str, TimeUnit] = {}
    for alias, canonical in _TIMEUNIT_ALIASES.items():
        member = TimeUnit(canonical)
        out[alias] = member
        upper = alias.upper()
        if upper != alias:
            out[upper] = member
    for member in TimeUnit:
        out[member.value] = member
        out[member.value.upper()] = member
        out[member.name] = member
        out[member.name.lower()] = member
    return out


_TIMEUNIT_LOOKUP: dict[str, TimeUnit] = _build_timeunit_lookup()
