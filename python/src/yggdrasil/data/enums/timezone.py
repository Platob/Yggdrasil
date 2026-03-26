"""Timezone normalisation and conversion utilities.

The :class:`Timezone` dataclass wraps an IANA timezone identifier (e.g.
``"Europe/Paris"``, ``"UTC"``) and provides:

* **Parsing** from strings — IANA names, common abbreviations (``CET``,
  ``EST``, …), and fixed UTC offsets (``+01:00``, ``UTC-05``).
* **Conversion helpers** — ``localize`` naive datetimes, ``convert`` between
  zones, extract ``utc_offset`` at a given instant.
* **Polars integration** — ``from_polars_dtype`` to extract the tz from a
  ``pl.Datetime``, ``polars_normalize`` to map a Series/Expr of timezone
  strings to canonical IANA names.
* **Pre-built constants** — ``Timezone.UTC``, ``Timezone.CET``, etc.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Any, ClassVar, TYPE_CHECKING
from zoneinfo import ZoneInfo, available_timezones

if TYPE_CHECKING:
    import polars

__all__ = [
    "Timezone",
]

_TIMEZONE_ALIASES: dict[str, str] = {
    "UTC": "UTC",
    "ETC/UTC": "UTC",
    "+00:00": "UTC",
    "-00:00": "UTC",
    "GMT": "UTC",
    "Z": "UTC",
    "CET": "Europe/Paris",
    "CEST": "Europe/Paris",
    "WET": "Europe/Lisbon",
    "WEST": "Europe/Lisbon",
    "EET": "Europe/Helsinki",
    "EEST": "Europe/Helsinki",
    "ET": "America/New_York",
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "CT": "America/Chicago",
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
    "MT": "America/Denver",
    "MST": "America/Denver",
    "MDT": "America/Denver",
    "PT": "America/Los_Angeles",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    "JST": "Asia/Tokyo",
    "KST": "Asia/Seoul",
    "HKT": "Asia/Hong_Kong",
    "SGT": "Asia/Singapore",
}

_OFFSET_RE = re.compile(r"^([+-])(\d{2}):?(\d{2})$")


@lru_cache(maxsize=1)
def _available_timezones_cached() -> frozenset[str]:
    return frozenset(available_timezones())


@dataclass(slots=True, frozen=True)
class Timezone:
    """An immutable wrapper around an IANA timezone identifier.

    Instances are created via :meth:`parse` (accepts strings, ``ZoneInfo``,
    other ``Timezone`` objects, or ``None`` → UTC) or directly::

        tz = Timezone("Europe/Paris")
        tz = Timezone.parse("CET")      # → Timezone("Europe/Paris")
        tz = Timezone.parse("+01:00")   # → Timezone("Etc/GMT-1")
    """

    iana: str

    # ── Pre-built constants (set after class body) ───────────────────────────
    UTC: ClassVar["Timezone"]
    CET: ClassVar["Timezone"]          # Europe/Paris (CET/CEST)
    WET: ClassVar["Timezone"]          # Europe/Lisbon
    EET: ClassVar["Timezone"]          # Europe/Helsinki
    EASTERN: ClassVar["Timezone"]      # America/New_York
    CENTRAL: ClassVar["Timezone"]      # America/Chicago
    MOUNTAIN: ClassVar["Timezone"]     # America/Denver
    PACIFIC: ClassVar["Timezone"]      # America/Los_Angeles
    JST: ClassVar["Timezone"]          # Asia/Tokyo
    SGT: ClassVar["Timezone"]          # Asia/Singapore

    # ── dunder ───────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return self.iana

    def __repr__(self) -> str:
        return f"Timezone({self.iana!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Timezone):
            return self.iana == other.iana
        if isinstance(other, str):
            return self.iana == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.iana)

    # ── Parsing ──────────────────────────────────────────────────────────────

    @classmethod
    def parse(cls, obj: Any) -> "Timezone":
        """Parse *obj* into a :class:`Timezone`.

        Accepted types:

        * ``Timezone`` — returned as-is.
        * ``str`` — delegates to :meth:`parse_str`.
        * ``ZoneInfo`` — extracts the ``key`` attribute.
        * ``None`` — returns :attr:`UTC`.

        Raises:
            TypeError: For unsupported types.
        """
        if isinstance(obj, cls):
            return obj
        if obj is None:
            return cls.UTC
        if isinstance(obj, ZoneInfo):
            return cls(obj.key)
        if isinstance(obj, str):
            return cls.parse_str(obj)
        raise TypeError(f"Cannot parse {type(obj).__name__} as {cls.__name__}")

    @classmethod
    def parse_str(cls, s: str) -> "Timezone":
        """Parse a timezone string.

        Resolution order:

        1. Exact IANA name (``"Europe/Paris"``).
        2. Known alias (``"CET"`` → ``Europe/Paris``).
        3. UTC offset (``"+01:00"``, ``"UTC-05"``, ``"-0530"``).

        Raises:
            TypeError: If *s* is not a ``str``.
            ValueError: If the string cannot be resolved.
        """
        if not isinstance(s, str):
            raise TypeError(f"Expected str, got {type(s).__name__}")

        raw = s.strip()
        if not raw:
            raise ValueError("Timezone string cannot be empty")

        # Exact IANA timezone
        if raw in _available_timezones_cached():
            return cls(raw)

        upper = raw.upper()

        # Known aliases
        if upper in _TIMEZONE_ALIASES:
            return cls(_TIMEZONE_ALIASES[upper])

        # Normalize "UTC+01:00" / "UTC-0500"
        offset_part = raw
        if upper.startswith("UTC") and len(raw) > 3:
            offset_part = raw[3:].strip()

        # Normalize "+01:00" / "-0500" to IANA Etc/GMT±X where possible
        m = _OFFSET_RE.match(offset_part)
        if m:
            sign, hh, mm = m.groups()
            hours = int(hh)
            minutes = int(mm)

            if minutes != 0:
                raise ValueError(
                    f"Cannot normalize non-hour-aligned UTC offset to IANA timezone: {s!r}"
                )

            if hours == 0:
                return cls("UTC")

            # IANA Etc/GMT sign convention is reversed
            etc_sign = "-" if sign == "+" else "+"
            return cls(f"Etc/GMT{etc_sign}{hours}")

        raise ValueError(f"Unknown timezone: {s!r}")

    # ── ZoneInfo interop ─────────────────────────────────────────────────────

    @lru_cache(maxsize=64)
    def to_zoneinfo(self) -> ZoneInfo:
        """Return the ``zoneinfo.ZoneInfo`` for this timezone."""
        return ZoneInfo(self.iana)

    @property
    def key(self) -> str:
        """Alias for :attr:`iana` — matches ``ZoneInfo.key``."""
        return self.iana

    # ── Offset / introspection ───────────────────────────────────────────────

    def utc_offset(self, at: _dt.datetime | None = None) -> _dt.timedelta:
        """Return the UTC offset at instant *at* (default: now).

        The result accounts for DST transitions.

        >>> Timezone("Europe/Paris").utc_offset(datetime(2024, 7, 1))
        datetime.timedelta(seconds=7200)  # +02:00 (CEST)
        """
        if at is None:
            at = _dt.datetime.now(_dt.timezone.utc)
        elif at.tzinfo is None:
            at = at.replace(tzinfo=_dt.timezone.utc)
        return at.astimezone(self.to_zoneinfo()).utcoffset() or _dt.timedelta(0)

    def utc_offset_hours(self, at: _dt.datetime | None = None) -> float:
        """Return the UTC offset in fractional hours.

        >>> Timezone("Asia/Kolkata").utc_offset_hours()
        5.5
        """
        return self.utc_offset(at).total_seconds() / 3600

    def is_utc(self) -> bool:
        """Return ``True`` if this timezone is UTC (or an equivalent)."""
        return self.iana in ("UTC", "Etc/UTC", "Etc/GMT", "Etc/GMT+0", "Etc/GMT-0", "GMT")

    def is_fixed_offset(self) -> bool:
        """Return ``True`` if this timezone has a fixed UTC offset (no DST)."""
        return self.iana.startswith("Etc/") or self.is_utc()

    def is_dst(self, at: _dt.datetime | None = None) -> bool:
        """Return ``True`` if DST is active at instant *at* (default: now).

        >>> Timezone("Europe/Paris").is_dst(datetime(2024, 7, 1))
        True
        >>> Timezone("Europe/Paris").is_dst(datetime(2024, 1, 1))
        False
        """
        if at is None:
            at = _dt.datetime.now(_dt.timezone.utc)
        elif at.tzinfo is None:
            at = at.replace(tzinfo=_dt.timezone.utc)
        localized = at.astimezone(self.to_zoneinfo())
        dst = localized.dst()
        return dst is not None and dst != _dt.timedelta(0)

    def dst_offset(self, at: _dt.datetime | None = None) -> _dt.timedelta:
        """Return the DST adjustment at instant *at*.

        Returns ``timedelta(0)`` when DST is not active or for fixed-offset
        timezones.
        """
        if at is None:
            at = _dt.datetime.now(_dt.timezone.utc)
        elif at.tzinfo is None:
            at = at.replace(tzinfo=_dt.timezone.utc)
        localized = at.astimezone(self.to_zoneinfo())
        return localized.dst() or _dt.timedelta(0)

    def abbreviation(self, at: _dt.datetime | None = None) -> str:
        """Return the timezone abbreviation (e.g. ``"CET"``, ``"CEST"``) at *at*.

        >>> Timezone("Europe/Paris").abbreviation(datetime(2024, 7, 1))
        'CEST'
        >>> Timezone("Europe/Paris").abbreviation(datetime(2024, 1, 1))
        'CET'
        """
        if at is None:
            at = _dt.datetime.now(_dt.timezone.utc)
        elif at.tzinfo is None:
            at = at.replace(tzinfo=_dt.timezone.utc)
        return at.astimezone(self.to_zoneinfo()).strftime("%Z")

    def distance_to(self, other: "Timezone", at: _dt.datetime | None = None) -> _dt.timedelta:
        """Return the offset difference between ``self`` and *other* at *at*.

        A positive result means *other* is ahead of ``self``.

        >>> Timezone.UTC.distance_to(Timezone.CET, datetime(2024, 7, 1))
        datetime.timedelta(seconds=7200)
        """
        return other.utc_offset(at) - self.utc_offset(at)

    @property
    def tzinfo(self) -> _dt.tzinfo:
        """Return as a stdlib ``tzinfo`` (same as ``to_zoneinfo()``)."""
        return self.to_zoneinfo()

    # ── Clock ────────────────────────────────────────────────────────────────

    def now(self) -> _dt.datetime:
        """Return the current wall-clock time in this timezone.

        The returned datetime is timezone-aware.
        """
        return _dt.datetime.now(self.to_zoneinfo())

    def today(self) -> _dt.date:
        """Return today's date in this timezone."""
        return self.now().date()

    # ── Conversion helpers ───────────────────────────────────────────────────

    def localize(self, naive: _dt.datetime) -> _dt.datetime:
        """Stamp a naive datetime with this timezone (no conversion).

        Raises:
            ValueError: If *naive* is already timezone-aware.
        """
        if naive.tzinfo is not None:
            raise ValueError(
                f"Cannot localize a timezone-aware datetime "
                f"(has tzinfo={naive.tzinfo!r}); use convert() instead"
            )
        return naive.replace(tzinfo=self.to_zoneinfo())

    def convert(self, aware: _dt.datetime) -> _dt.datetime:
        """Convert a timezone-aware datetime to this timezone.

        Raises:
            ValueError: If *aware* is naive.
        """
        if aware.tzinfo is None:
            raise ValueError(
                "Cannot convert a naive datetime; use localize() first"
            )
        return aware.astimezone(self.to_zoneinfo())

    def midnight(self, date: _dt.date | None = None) -> _dt.datetime:
        """Return midnight (00:00) in this timezone for *date* (default: today)."""
        if date is None:
            date = self.today()
        return _dt.datetime(date.year, date.month, date.day, tzinfo=self.to_zoneinfo())

    # ── Polars integration ───────────────────────────────────────────────────

    @classmethod
    def from_polars_dtype(cls, dtype: "polars.DataType") -> "Timezone | None":
        """Extract the timezone from a Polars ``Datetime`` dtype.

        Returns ``None`` if the dtype is not ``Datetime`` or has no timezone.

        >>> Timezone.from_polars_dtype(pl.Datetime("us", "Europe/Paris"))
        Timezone('Europe/Paris')
        """
        import polars as pl

        if isinstance(dtype, pl.Datetime) and dtype.time_zone:
            return cls.parse(dtype.time_zone)
        return None

    @classmethod
    def polars_normalize(
        cls,
        col: "polars.Series | polars.Expr",
        *,
        lazy: bool = True,
        return_value: Literal["iana"] = "iana",
    ) -> "polars.Series | polars.Expr":
        """Normalize timezone strings in a Polars column using alias replacement only."""
        import polars as pl

        if return_value != "iana":
            raise ValueError(f"Unsupported return_value: {return_value!r}")
        if not isinstance(col, (pl.Series, pl.Expr)):
            raise TypeError(f"Expected polars.Series | polars.Expr, got {type(col).__name__}")

        normalized = (
            col.cast(pl.Utf8)
            .str.strip_chars()
            .str.to_uppercase()
        )
        return normalized.replace_strict(_TIMEZONE_ALIASES, default=None)

    # ── Arrow integration ────────────────────────────────────────────────────

    @classmethod
    def from_arrow_type(cls, dtype: Any) -> "Timezone | None":
        """Extract the timezone from a PyArrow ``TimestampType``.

        Returns ``None`` if the type has no ``tz`` attribute or it is empty.

        >>> import pyarrow as pa
        >>> Timezone.from_arrow_type(pa.timestamp("us", tz="Europe/Paris"))
        Timezone('Europe/Paris')
        """
        tz = getattr(dtype, "tz", None)
        if tz:
            return cls.parse(tz)
        return None

    def arrow_timestamp_type(self, unit: str = "us") -> Any:
        """Return a ``pa.timestamp(unit, tz=self.iana)`` type.

        >>> Timezone.CET.arrow_timestamp_type("ns")
        TimestampType(timestamp[ns, tz=Europe/Paris])
        """
        import pyarrow as pa
        return pa.timestamp(unit, tz=self.iana)

    # ── Enumeration ──────────────────────────────────────────────────────────

    @classmethod
    def all_iana(cls) -> frozenset[str]:
        """Return all IANA timezone identifiers available on this system."""
        return _available_timezones_cached()


# ── Class-level constants (can't be set inside a frozen dataclass body) ──────
Timezone.UTC = Timezone("UTC")
Timezone.CET = Timezone("Europe/Paris")
Timezone.WET = Timezone("Europe/Lisbon")
Timezone.EET = Timezone("Europe/Helsinki")
Timezone.EASTERN = Timezone("America/New_York")
Timezone.CENTRAL = Timezone("America/Chicago")
Timezone.MOUNTAIN = Timezone("America/Denver")
Timezone.PACIFIC = Timezone("America/Los_Angeles")
Timezone.JST = Timezone("Asia/Tokyo")
Timezone.SGT = Timezone("Asia/Singapore")
