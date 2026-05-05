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

    # ── Parsing ─────────────────────────────────────────────────────────────

    @classmethod
    def parse(cls, value: Any, *, default: Any = ...) -> "TimeUnit":
        """Normalize *value* to a :class:`TimeUnit`.

        Accepts ``TimeUnit`` (returned as-is), strings (matched case-
        insensitively against the alias table or any canonical member
        value), and ``None`` — passing ``None`` returns *default* if
        provided, otherwise raises :class:`ValueError`.

        Raises:
            TypeError: when *value* is neither a string nor a
                ``TimeUnit``.
            ValueError: when the string can't be resolved to any
                canonical unit and no *default* is supplied.
        """
        if isinstance(value, cls):
            return value
        if value is None:
            if default is not ...:
                return default
            raise ValueError("TimeUnit cannot be parsed from None")
        if not isinstance(value, str):
            raise TypeError(
                f"Cannot parse {type(value).__name__} as TimeUnit; "
                f"expected str or TimeUnit, got {value!r}"
            )

        token = value.strip().lower().replace("-", "_").replace(" ", "_")
        if not token:
            if default is not ...:
                return default
            raise ValueError("TimeUnit string cannot be empty")

        canonical = _TIMEUNIT_ALIASES.get(token)
        if canonical is not None:
            return cls(canonical)

        # Allow exact value match for safety (e.g. ``TimeUnit("us")``
        # already routes here from member lookup but ``parse("US")``
        # benefits from the case-insensitive fallthrough).
        try:
            return cls(token)
        except ValueError:
            pass

        if default is not ...:
            return default
        raise ValueError(
            f"Unknown time unit: {value!r}. "
            f"Valid units are: {sorted({m.value for m in cls})!r}"
        )

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        """Return ``True`` when :meth:`parse` would succeed for *value*."""
        try:
            cls.parse(value)
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
