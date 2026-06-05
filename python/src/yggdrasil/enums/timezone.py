"""Timezone normalisation and conversion utilities.

The :class:`Timezone` dataclass wraps an IANA timezone identifier (e.g.
``"Europe/Paris"``, ``"UTC"``) and provides:

* **Coercion** from any common Python timezone shape — strings (IANA
  names, abbreviations like ``CET`` / ``EST``, fixed offsets like
  ``+01:00`` / ``UTC-05``), ``ZoneInfo``, ``datetime.tzinfo``,
  ``datetime`` instances with a tzinfo attached, or other Timezone
  objects.
* **Conversion helpers** — ``localize`` naive datetimes, ``convert``
  between zones, extract ``utc_offset`` at a given instant.
* **Polars integration** — ``from_polars_type`` extracts the tz from a
  ``pl.Datetime``, ``polars_normalize`` maps a Series/Expr of timezone
  strings to canonical IANA names.
* **Pre-built constants** — :attr:`Timezone.UTC`, :attr:`Timezone.CET`,
  :attr:`Timezone.NAIVE`, etc.

The single coercion entry point is :meth:`Timezone.from_`. There is no
``parse`` / ``parse_str`` API anymore — every call site routes through
``from_`` so the same input rules apply everywhere.
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
    "UTC": "Etc/UTC",
    "ETC/UTC": "Etc/UTC",
    "+00:00": "Etc/UTC",
    "-00:00": "Etc/UTC",
    "GMT": "Etc/UTC",
    "Z": "Etc/UTC",
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

# Hot per-process cache of resolved Timezone strings. Pre-seeded after
# the class body with every alias and the named constants; misses
# fall through to ``_parse_timezone_string`` and cache the result.
_TIMEZONE_STR_CACHE: dict[str, "Timezone"] = {}

# Sentinel iana value for the naive Timezone — empty string keeps
# ``str(Timezone.NAIVE)`` invisible in pretty-printed output and makes
# ``bool(Timezone.NAIVE)`` falsy. Internal-only — callers should use
# :attr:`Timezone.NAIVE` rather than ``Timezone("")``.
_NAIVE_IANA = ""


@lru_cache(maxsize=1)
def _available_timezones_cached() -> frozenset[str]:
    return frozenset(available_timezones())


def _parse_timezone_string(s: str) -> str:
    """Resolve a timezone string to its canonical IANA name.

    Resolution order:

    1. Exact IANA name (``"Europe/Paris"``).
    2. Known alias (``"CET"`` → ``Europe/Paris``).
    3. UTC offset (``"+01:00"``, ``"UTC-05"``, ``"-0530"``).

    Pure helper — kept private; the public surface is
    :meth:`Timezone.from_`.

    Raises:
        ValueError: when the string can't be resolved.
    """
    raw = s.strip()
    if not raw:
        raise ValueError("Timezone string cannot be empty")

    # Plain UTC always folds to the canonical ``Etc/UTC`` — checked before the
    # available-zones lookup, since ``"UTC"`` is itself a valid IANA zone.
    if raw.upper() in ("UTC", "ETC/UTC"):
        return "Etc/UTC"

    if raw in _available_timezones_cached():
        return raw

    upper = raw.upper()
    if upper in _TIMEZONE_ALIASES:
        return _TIMEZONE_ALIASES[upper]

    offset_part = raw
    if upper.startswith("UTC") and len(raw) > 3:
        offset_part = raw[3:].strip()

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
            return "Etc/UTC"

        # IANA Etc/GMT sign convention is reversed.
        etc_sign = "-" if sign == "+" else "+"
        return f"Etc/GMT{etc_sign}{hours}"

    raise ValueError(f"Unknown timezone: {s!r}")


@dataclass(slots=True, frozen=True)
class Timezone:
    """An immutable wrapper around an IANA timezone identifier.

    Instances are created via :meth:`from_` (accepts strings,
    ``ZoneInfo``, ``datetime.tzinfo``, other ``Timezone`` objects, or
    ``None`` → UTC) or directly::

        tz = Timezone("Europe/Paris")
        tz = Timezone.from_("CET")        # → Timezone("Europe/Paris")
        tz = Timezone.from_("+01:00")     # → Timezone("Etc/GMT-1")
        tz = Timezone.from_(ZoneInfo("Europe/Paris"))

    The naive case is represented by :attr:`Timezone.NAIVE` — a
    sentinel instance whose ``iana`` is the empty string and whose
    ``__bool__`` is ``False``. Use it instead of ``None`` so a
    ``tz: Timezone`` field type can stay non-optional.
    """

    iana: str

    # ── Pre-built constants (set after class body) ───────────────────────────
    NAIVE: ClassVar["Timezone"]        # tz-less sentinel
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
        if self.iana == _NAIVE_IANA:
            return "Timezone.NAIVE"
        return f"Timezone({self.iana!r})"

    def __bool__(self) -> bool:
        # ``Timezone.NAIVE`` is falsy so callers can ``if self.tz:``
        # to mean "is timezone-aware".
        return bool(self.iana)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Timezone):
            return self.iana == other.iana
        if isinstance(other, str):
            return self.iana == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.iana)

    # ── Coercion ────────────────────────────────────────────────────────────

    @classmethod
    def from_(cls, obj: Any, *, default: Any = ...) -> "Timezone":
        """Coerce any Python value into a :class:`Timezone`.

        Accepts:

        * :class:`Timezone` (returned as-is — including ``Timezone.NAIVE``);
        * :class:`zoneinfo.ZoneInfo` (extracts ``key``);
        * a string — IANA name, alias (``CET`` / ``EST`` / ``Z`` / ``GMT``
          / ``Etc/UTC`` / ``+00:00`` / …), or fixed offset
          (``"+01:00"``, ``"UTC-05"``, ``"-0530"``);
        * ``datetime.tzinfo`` instance (``ZoneInfo``,
          ``datetime.timezone``, third-party zones) — the ``key`` /
          ``zone`` attribute is preferred; otherwise falls back to the
          fixed UTC offset;
        * timezone-aware ``datetime`` / ``time`` (extracts ``tzinfo``);
        * objects exposing a ``tz`` / ``time_zone`` / ``timezone`` /
          ``iana`` attribute (PyArrow ``TimestampType``, Polars
          ``Datetime``, foreign Timezone classes); the attribute is
          re-funneled through ``from_``;
        * ``None`` — returns :attr:`UTC` for backward compatibility
          unless ``default`` is supplied.

        ``default`` swallows unknown / unparseable input. Without it,
        bad input raises :class:`ValueError` (string parse failure) or
        :class:`TypeError` (unsupported value type).
        """
        if isinstance(obj, cls):
            return obj

        if obj is None:
            if default is not ...:
                return default
            return cls.UTC

        if isinstance(obj, ZoneInfo):
            return cls(obj.key)

        if isinstance(obj, _dt.datetime):
            return cls.from_(obj.tzinfo, default=default)
        if isinstance(obj, _dt.time):
            return cls.from_(obj.tzinfo, default=default)

        if isinstance(obj, _dt.tzinfo):
            for attr in ("key", "zone"):
                inner = getattr(obj, attr, None)
                if isinstance(inner, str) and inner:
                    return cls._from_str(inner, default=default)
            offset = obj.utcoffset(_dt.datetime.now(_dt.timezone.utc))
            if offset is None:
                if default is not ...:
                    return default
                raise ValueError(
                    f"Cannot derive Timezone from tzinfo without offset: {obj!r}"
                )
            total = int(offset.total_seconds())
            sign = "+" if total >= 0 else "-"
            hours, remainder = divmod(abs(total), 3600)
            minutes = remainder // 60
            return cls._from_str(f"{sign}{hours:02d}:{minutes:02d}", default=default)

        if isinstance(obj, str):
            return cls._from_str(obj, default=default)

        # Engine dtypes — Arrow ``TimestampType.tz``, Polars
        # ``Datetime.time_zone``, foreign Timezone classes with
        # ``iana`` / ``zone`` / ``timezone`` attribute.
        for attr in ("tz", "time_zone", "timezone", "iana"):
            inner = getattr(obj, attr, None)
            if inner is not None and inner is not obj:
                return cls.from_(inner, default=default)

        if default is not ...:
            return default
        raise TypeError(f"Cannot derive Timezone from {type(obj).__name__}: {obj!r}")

    @classmethod
    def _from_str(cls, s: str, *, default: Any = ...) -> "Timezone":
        # Fast path: most callers pass an already-canonical IANA name
        # (``"UTC"`` / ``"Europe/Paris"``) or a popular alias
        # (``"CET"`` / ``"EST"`` / ``"Z"``). A pre-seeded cache resolves
        # those without re-parsing or rebuilding the available-zones
        # frozenset.
        hit = _TIMEZONE_STR_CACHE.get(s)
        if hit is not None:
            return hit
        try:
            instance = cls(_parse_timezone_string(s))
        except (TypeError, ValueError):
            if default is not ...:
                return default
            raise
        # Memoize the input (the parser strips/normalises, so cache
        # the raw input the caller actually handed us — the next
        # identical call hits the cache without re-stripping).
        _TIMEZONE_STR_CACHE[s] = instance
        return instance

    # ── ZoneInfo interop ─────────────────────────────────────────────────────

    @lru_cache(maxsize=64)
    def to_zoneinfo(self) -> ZoneInfo:
        """Return the ``zoneinfo.ZoneInfo`` for this timezone.

        Raises :class:`ValueError` for :attr:`Timezone.NAIVE` since
        there is no concrete zone to materialize.
        """
        if self.is_naive():
            raise ValueError("Timezone.NAIVE has no ZoneInfo")
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
        if self.is_naive():
            raise ValueError("Timezone.NAIVE has no UTC offset")
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

    @property
    def utc_seconds_offset(self) -> int | None:
        """Fixed UTC offset in seconds, or ``None`` for non-fixed zones.

        Returns ``0`` for UTC and its equivalents (``Etc/UTC``,
        ``GMT``, ``Etc/GMT+0``, …), the parsed offset for
        ``Etc/GMT±N`` (note IANA's sign-flip — ``Etc/GMT-3`` means
        UTC+3, so the property reports ``+10800`` not ``-10800``),
        and ``None`` for any zone whose offset depends on DST or for
        :attr:`Timezone.NAIVE`. Use :meth:`utc_offset` instead when
        you need a DST-aware offset for a specific instant.

        Naming follows the "type-suffix unit" convention so
        ``utc_seconds_offset`` makes the unit explicit at the call
        site — ``utc_offset`` returns a :class:`datetime.timedelta`,
        ``utc_offset_hours`` returns fractional hours, and this
        property returns whole seconds (or ``None``).
        """
        if not self.is_fixed_offset():
            return None
        return int(self.utc_offset().total_seconds())

    def is_naive(self) -> bool:
        """Return ``True`` for :attr:`Timezone.NAIVE` — the tz-less sentinel."""
        return self.iana == _NAIVE_IANA

    def is_utc(self) -> bool:
        """Return ``True`` if this timezone is UTC (or an equivalent)."""
        return self.iana in ("UTC", "Etc/UTC", "Etc/GMT", "Etc/GMT+0", "Etc/GMT-0", "GMT")

    def is_fixed_offset(self) -> bool:
        """Return ``True`` if this timezone has a fixed UTC offset (no DST)."""
        if self.is_naive():
            return False
        return self.iana.startswith("Etc/") or self.is_utc()

    def is_dst(self, at: _dt.datetime | None = None) -> bool:
        """Return ``True`` if DST is active at instant *at* (default: now).

        >>> Timezone("Europe/Paris").is_dst(datetime(2024, 7, 1))
        True
        >>> Timezone("Europe/Paris").is_dst(datetime(2024, 1, 1))
        False
        """
        if self.is_naive():
            return False
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
        timezones, and for :attr:`Timezone.NAIVE`.
        """
        if self.is_naive():
            return _dt.timedelta(0)
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
        if self.is_naive():
            return ""
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
            ValueError: If *naive* is already timezone-aware, or if
                this is :attr:`Timezone.NAIVE`.
        """
        if self.is_naive():
            raise ValueError("Cannot localize against Timezone.NAIVE")
        if naive.tzinfo is not None:
            raise ValueError(
                f"Cannot localize a timezone-aware datetime "
                f"(has tzinfo={naive.tzinfo!r}); use convert() instead"
            )
        return naive.replace(tzinfo=self.to_zoneinfo())

    def convert(self, aware: _dt.datetime) -> _dt.datetime:
        """Convert a timezone-aware datetime to this timezone.

        Raises:
            ValueError: If *aware* is naive, or if this is
                :attr:`Timezone.NAIVE`.
        """
        if self.is_naive():
            raise ValueError("Cannot convert into Timezone.NAIVE")
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
    def from_polars_type(cls, dtype: "polars.DataType") -> "Timezone | None":
        """Extract the timezone from a Polars ``Datetime`` dtype.

        Returns ``None`` if the dtype is not ``Datetime`` or has no timezone.

        >>> Timezone.from_polars_type(pl.Datetime("us", "Europe/Paris"))
        Timezone('Europe/Paris')
        """
        import polars as pl

        if isinstance(dtype, pl.Datetime) and dtype.time_zone:
            return cls.from_(dtype.time_zone)
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
            return cls.from_(tz)
        return None

    def arrow_timestamp_type(self, unit: str = "us") -> Any:
        """Return a ``pa.timestamp(unit, tz=self.iana)`` type.

        >>> Timezone.CET.arrow_timestamp_type("ns")
        TimestampType(timestamp[ns, tz=Europe/Paris])
        """
        import pyarrow as pa
        if self.is_naive():
            return pa.timestamp(unit)
        return pa.timestamp(unit, tz=self.iana)

    # ── Enumeration ──────────────────────────────────────────────────────────

    @classmethod
    def all_iana(cls) -> frozenset[str]:
        """Return all IANA timezone identifiers available on this system."""
        return _available_timezones_cached()


# ── Class-level constants (can't be set inside a frozen dataclass body) ──────
Timezone.NAIVE = Timezone(_NAIVE_IANA)
Timezone.UTC = Timezone("Etc/UTC")
Timezone.CET = Timezone("Europe/Paris")
Timezone.WET = Timezone("Europe/Lisbon")
Timezone.EET = Timezone("Europe/Helsinki")
Timezone.EASTERN = Timezone("America/New_York")
Timezone.CENTRAL = Timezone("America/Chicago")
Timezone.MOUNTAIN = Timezone("America/Denver")
Timezone.PACIFIC = Timezone("America/Los_Angeles")
Timezone.JST = Timezone("Asia/Tokyo")
Timezone.SGT = Timezone("Asia/Singapore")


# Pre-seed the string-resolution cache: every alias key (and a few
# canonical IANA names that show up in tests / DataFrame schemas) is
# mapped to a singleton :class:`Timezone`. ``Timezone._from_str``
# hits this map first and only falls through to the parser for
# inputs we haven't seen yet.
def _seed_timezone_cache() -> None:
    available = _available_timezones_cached()
    seeds = {
        "UTC": Timezone.UTC,
        "Etc/UTC": Timezone.UTC,
        "Europe/Paris": Timezone.CET,
        "Europe/Lisbon": Timezone.WET,
        "Europe/Helsinki": Timezone.EET,
        "America/New_York": Timezone.EASTERN,
        "America/Chicago": Timezone.CENTRAL,
        "America/Denver": Timezone.MOUNTAIN,
        "America/Los_Angeles": Timezone.PACIFIC,
        "Asia/Tokyo": Timezone.JST,
        "Asia/Singapore": Timezone.SGT,
    }
    _TIMEZONE_STR_CACHE.update(seeds)
    # Folded alias variants — every alias key (upper-cased in the
    # table) plus its lower-case form land on the canonical Timezone
    # instance. Aliases that are *themselves* a valid IANA name
    # (e.g. ``"GMT"`` is a real zone) are skipped so the parser's
    # available-zones-first contract is preserved.
    for alias, canonical in _TIMEZONE_ALIASES.items():
        if alias in available:
            instance = Timezone(alias)
        else:
            instance = _TIMEZONE_STR_CACHE.get(canonical) or Timezone(canonical)
            _TIMEZONE_STR_CACHE.setdefault(canonical, instance)
        _TIMEZONE_STR_CACHE[alias] = instance
        lower = alias.lower()
        if lower != alias and lower not in available:
            _TIMEZONE_STR_CACHE[lower] = instance

    # ``"UTC"`` is itself a valid IANA zone, so the loop above pins it to a
    # literal ``Timezone("UTC")``. Re-fold every plain-UTC spelling onto the
    # single canonical ``Etc/UTC`` instance — the project standard.
    for spelling in ("UTC", "utc"):
        _TIMEZONE_STR_CACHE[spelling] = Timezone.UTC


_seed_timezone_cache()
