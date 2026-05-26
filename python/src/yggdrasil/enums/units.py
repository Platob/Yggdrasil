"""Physical-unit enums for cross-source curated views.

Curated and ``dash_*`` tables routinely need to carry **both** the
source-emitted value (in whatever unit the upstream feed publishes
— ``MWh``, ``MW``, ``BTU``, ``mi`` …) **and** a standardised
equivalent in a uniform canonical unit so downstream BI / ML can
group, sum, and compare across vendors without re-parsing strings
at the call site. :class:`~yggdrasil.fxrate.FxRate` already handles
currency conversion; this module is the matching surface for
physical quantities.

Each enum carries the ``(symbol, factor, offset)`` triple where the
**canonical** member has ``factor=1`` and ``offset=0``. Conversion
between any two members of the same family is::

    canonical = value * src.factor + src.offset
    target_value = (canonical - tgt.offset) / tgt.factor

Linear families (energy, power, mass, length, volume, pressure) ride
through with ``offset=0``; :class:`TemperatureUnit` is affine —
``°C → K`` is offset 273.15, ``°F → K`` is scaled+offset.

Public surface is uniform across every unit enum:

* :meth:`Unit.from_` — forgiving coercion (member / symbol / alias /
  case-insensitive variants);
* :meth:`Unit.is_valid` — boolean probe;
* :meth:`Unit.convert(value, source, target)` — scalar conversion;
* :meth:`Unit.to_canonical(value)` / :meth:`Unit.from_canonical(value)`
  — single-unit instance helpers;
* :attr:`Unit.symbol` — canonical short token (``"MWh"`` / ``"kg"`` /
  ``"°C"``);
* :attr:`Unit.factor`, :attr:`Unit.offset` — exposed so vectorised
  callers in :mod:`yggdrasil.databricks.standardize` can build
  per-row Polars / Spark expressions without re-deriving the math.

The canonical token per family is the SI base where one exists:
joule (J) for energy, watt (W) for power, kilogram (kg) for mass,
metre (m) for length, cubic metre (m³) for volume, kelvin (K) for
temperature, pascal (Pa) for pressure.

Aliases live in a module-level ``<Family>._ALIASES`` dict assigned
after the class body — Python's Enum metaclass would otherwise treat
a dict-valued class-body attribute as an enum member.
"""
from __future__ import annotations

from enum import Enum
from typing import Any


__all__ = [
    "Unit",
    "EnergyUnit",
    "PowerUnit",
    "MassUnit",
    "LengthUnit",
    "VolumeUnit",
    "TemperatureUnit",
    "PressureUnit",
    "UNIT_FAMILIES",
    "unit_family_for",
]


# ---------------------------------------------------------------------------
# Shared mixin — coercion + conversion logic, no Enum members itself.
# Concrete families inherit from ``Unit`` + ``Enum`` and declare their
# members as ``(symbol, factor, offset)`` tuples.
# ---------------------------------------------------------------------------


class Unit:
    """Mixin providing :meth:`from_` / :meth:`convert` / scalar helpers.

    Subclasses combine this with :class:`Enum` and declare members
    as ``(symbol, factor_to_canonical, offset_to_canonical)`` tuples
    (or ``(symbol, factor)`` — offset defaults to ``0.0``). The
    canonical member of each family has ``factor=1.0`` and
    ``offset=0.0``; conversion goes through that canonical pivot.

    Extra spellings (``"megawatthours"`` → ``MWH``) live in a
    ``<Family>._ALIASES`` dict assigned outside the enum body — the
    enum metaclass would treat a class-body dict as a member.
    """

    symbol: str
    factor: float
    offset: float

    def __init__(self, symbol: str, factor: float, offset: float = 0.0) -> None:
        self.symbol = symbol
        self.factor = float(factor)
        self.offset = float(offset)

    def __str__(self) -> str:
        return self.symbol

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}.{self.name}"

    # ── Coercion ────────────────────────────────────────────────────────────

    @classmethod
    def from_(cls, value: Any, *, default: Any = ...) -> "Unit":
        """Coerce *value* to a member of this unit family.

        Accepts: a member (passed through), a known symbol
        (``"MWh"`` / ``"°C"`` / ``"psi"``), the enum member name
        (``"KWH"``), or any alias the per-family ``_ALIASES`` dict
        knows (``"megawatthours"`` / ``"celsius"`` / ``"pound"``).
        Lookup is case-insensitive.

        Pass *default* to swallow unknown / unparseable input; without
        it, unknown strings raise :class:`ValueError` with a hint
        listing the valid symbols, and non-string non-member input
        raises :class:`TypeError`.
        """
        if isinstance(value, cls):
            return value
        if value is None:
            if default is not ...:
                return default
            raise ValueError(f"{cls.__name__} cannot be derived from None")
        if not isinstance(value, str):
            if default is not ...:
                return default
            raise TypeError(
                f"Cannot derive {cls.__name__} from {type(value).__name__}: {value!r}"
            )
        lookup = cls._lookup()
        # Fast path: exact match on the spelling the caller used.
        hit = lookup.get(value)
        if hit is not None:
            return hit
        token = value.strip()
        if not token:
            if default is not ...:
                return default
            raise ValueError(f"{cls.__name__} symbol cannot be empty")
        for candidate in (token, token.lower(), token.upper()):
            hit = lookup.get(candidate)
            if hit is not None:
                return hit
        if default is not ...:
            return default
        valid = sorted({m.symbol for m in cls})  # type: ignore[attr-defined]
        raise ValueError(
            f"Unknown {cls.__name__} token: {value!r}. "
            f"Valid symbols: {valid!r}."
        )

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        """``True`` when :meth:`from_` would resolve *value*."""
        try:
            cls.from_(value)
        except (TypeError, ValueError):
            return False
        return True

    # ── Conversion ──────────────────────────────────────────────────────────

    @classmethod
    def convert(cls, value: float, source: Any, target: Any) -> float:
        """Convert *value* from *source* unit to *target* unit (scalar)."""
        src = cls.from_(source)
        tgt = cls.from_(target)
        if src is tgt:
            return float(value)
        canonical = value * src.factor + src.offset
        return (canonical - tgt.offset) / tgt.factor

    def to_canonical(self, value: float) -> float:
        """Convert *value* in *self*'s unit to the family's canonical unit."""
        return value * self.factor + self.offset

    def from_canonical(self, value: float) -> float:
        """Convert *value* in the canonical unit to *self*'s unit."""
        return (value - self.offset) / self.factor

    @classmethod
    def canonical(cls) -> "Unit":
        """The family's canonical member (``factor=1.0``, ``offset=0.0``)."""
        cached = cls.__dict__.get("_CANONICAL")
        if cached is not None:
            return cached
        for member in cls:  # type: ignore[attr-defined]
            if member.factor == 1.0 and member.offset == 0.0:
                cls._CANONICAL = member  # type: ignore[attr-defined]
                return member
        raise RuntimeError(
            f"{cls.__name__} has no canonical member (factor=1, offset=0). "
            f"Every unit family must declare exactly one canonical token."
        )

    # ── Vectorised-helper hooks — used by yggdrasil.databricks.standardize ──

    @classmethod
    def factor_map(cls) -> dict[str, float]:
        """``{symbol: factor}`` for every member + alias, for per-row maps."""
        return {token: member.factor for token, member in cls._lookup().items()}

    @classmethod
    def offset_map(cls) -> dict[str, float]:
        """``{symbol: offset}`` for every member + alias, for per-row maps."""
        return {token: member.offset for token, member in cls._lookup().items()}

    # ── Internal lookup table ───────────────────────────────────────────────

    @classmethod
    def _lookup(cls) -> dict[str, "Unit"]:
        cached = cls.__dict__.get("_LOOKUP")
        if cached is not None:
            return cached
        out: dict[str, Unit] = {}
        for member in cls:  # type: ignore[attr-defined]
            for token in (member.symbol, member.symbol.lower(), member.symbol.upper(),
                          member.name, member.name.lower()):
                out[token] = member
        aliases: dict[str, str] = getattr(cls, "_ALIASES", {}) or {}
        for alias, member_name in aliases.items():
            try:
                member = cls[member_name]  # type: ignore[index]
            except KeyError as exc:
                raise RuntimeError(
                    f"{cls.__name__}._ALIASES references unknown member "
                    f"{member_name!r} (alias {alias!r})."
                ) from exc
            for token in (alias, alias.lower(), alias.upper()):
                out[token] = member
        cls._LOOKUP = out  # type: ignore[attr-defined]
        return out


# ---------------------------------------------------------------------------
# Energy — canonical = joule
# ---------------------------------------------------------------------------


# Useful constants — define once so member declarations read cleanly.
_WH_J = 3600.0           # 1 Wh = 3600 J
_BTU_J = 1055.05585262   # IT BTU
_CAL_J = 4.184           # thermochemical calorie
_THERM_J = 1.05505585262e8  # US therm


class EnergyUnit(Unit, Enum):
    """Energy unit (canonical = joule).

    Members cover SI scale (J / kJ / MJ / GJ / TJ), the watt-hour
    family that dominates electricity-market ingestion
    (Wh / kWh / MWh / GWh / TWh), and the imperial / domestic units
    (BTU / cal / kcal / therm).
    """

    J     = ("J",     1.0)
    KJ    = ("kJ",    1e3)
    MJ    = ("MJ",    1e6)
    GJ    = ("GJ",    1e9)
    TJ    = ("TJ",    1e12)
    WH    = ("Wh",    _WH_J)
    KWH   = ("kWh",   _WH_J * 1e3)
    MWH   = ("MWh",   _WH_J * 1e6)
    GWH   = ("GWh",   _WH_J * 1e9)
    TWH   = ("TWh",   _WH_J * 1e12)
    BTU   = ("BTU",   _BTU_J)
    CAL   = ("cal",   _CAL_J)
    KCAL  = ("kcal",  _CAL_J * 1e3)
    THERM = ("therm", _THERM_J)


EnergyUnit._ALIASES = {  # type: ignore[attr-defined]
    "joule":         "J",
    "joules":        "J",
    "kilojoule":     "KJ",
    "kilojoules":    "KJ",
    "megajoule":     "MJ",
    "megajoules":    "MJ",
    "gigajoule":     "GJ",
    "gigajoules":    "GJ",
    "terajoule":     "TJ",
    "terajoules":    "TJ",
    "watthour":      "WH",
    "watt-hour":     "WH",
    "watthours":     "WH",
    "watt_hours":    "WH",
    "kilowatthour":  "KWH",
    "kilowatt-hour": "KWH",
    "kilowatthours": "KWH",
    "megawatthour":  "MWH",
    "megawatt-hour": "MWH",
    "megawatthours": "MWH",
    "gigawatthour":  "GWH",
    "gigawatt-hour": "GWH",
    "terawatthour":  "TWH",
    "terawatt-hour": "TWH",
    "btu":           "BTU",
    "calorie":       "CAL",
    "calories":      "CAL",
    "kilocalorie":   "KCAL",
    "kilocalories":  "KCAL",
    "therms":        "THERM",
}


# ---------------------------------------------------------------------------
# Power — canonical = watt
# ---------------------------------------------------------------------------


_HP_W = 745.6998715822702  # mechanical horsepower (550 ft·lbf/s)


class PowerUnit(Unit, Enum):
    """Power unit (canonical = watt).

    SI scale (W / kW / MW / GW / TW) plus mechanical horsepower.
    Energy-market feeds publish capacity in MW, transmission limits
    in GW, household-scale appliances in W — having one enum cover
    all of them is what keeps schema-per-source curated views from
    re-stringing the unit token at every read.
    """

    W   = ("W",   1.0)
    KW  = ("kW",  1e3)
    MW  = ("MW",  1e6)
    GW  = ("GW",  1e9)
    TW  = ("TW",  1e12)
    HP  = ("hp",  _HP_W)


PowerUnit._ALIASES = {  # type: ignore[attr-defined]
    "watt":         "W",
    "watts":        "W",
    "kilowatt":     "KW",
    "kilowatts":    "KW",
    "megawatt":     "MW",
    "megawatts":    "MW",
    "gigawatt":     "GW",
    "gigawatts":    "GW",
    "terawatt":     "TW",
    "terawatts":    "TW",
    "horsepower":   "HP",
}


# ---------------------------------------------------------------------------
# Mass — canonical = kilogram
# ---------------------------------------------------------------------------


_LB_KG = 0.45359237   # international avoirdupois pound
_OZ_KG = _LB_KG / 16  # international avoirdupois ounce


class MassUnit(Unit, Enum):
    """Mass unit (canonical = kilogram).

    SI scale (mg / g / kg / t / Mt) plus the imperial pair (lb / oz)
    commonly used in commodity feeds.
    """

    KG  = ("kg",  1.0)
    G   = ("g",   1e-3)
    MG  = ("mg",  1e-6)
    T   = ("t",   1e3)
    MT  = ("Mt",  1e9)
    LB  = ("lb",  _LB_KG)
    OZ  = ("oz",  _OZ_KG)


MassUnit._ALIASES = {  # type: ignore[attr-defined]
    "kilogram":   "KG",
    "kilograms":  "KG",
    "gram":       "G",
    "grams":      "G",
    "milligram":  "MG",
    "milligrams": "MG",
    "tonne":      "T",
    "tonnes":     "T",
    "metric_ton": "T",
    "ton":        "T",   # short-token "ton" lands on the metric tonne
    "megatonne":  "MT",
    "megatonnes": "MT",
    "pound":      "LB",
    "pounds":     "LB",
    "lbs":        "LB",
    "ounce":      "OZ",
    "ounces":     "OZ",
}


# ---------------------------------------------------------------------------
# Length — canonical = metre
# ---------------------------------------------------------------------------


_IN_M = 0.0254
_FT_M = _IN_M * 12
_YD_M = _FT_M * 3
_MI_M = _FT_M * 5280
_NMI_M = 1852.0  # international nautical mile (exact)


class LengthUnit(Unit, Enum):
    """Length unit (canonical = metre).

    SI scale (mm / cm / m / km) plus the common imperial / aviation /
    maritime tokens (in / ft / yd / mi / nmi).
    """

    M    = ("m",    1.0)
    KM   = ("km",   1e3)
    CM   = ("cm",   1e-2)
    MM   = ("mm",   1e-3)
    IN   = ("in",   _IN_M)
    FT   = ("ft",   _FT_M)
    YD   = ("yd",   _YD_M)
    MI   = ("mi",   _MI_M)
    NMI  = ("nmi",  _NMI_M)


LengthUnit._ALIASES = {  # type: ignore[attr-defined]
    "metre":         "M",
    "metres":        "M",
    "meter":         "M",
    "meters":        "M",
    "kilometre":     "KM",
    "kilometres":    "KM",
    "kilometer":     "KM",
    "kilometers":    "KM",
    "centimetre":    "CM",
    "centimetres":   "CM",
    "centimeter":    "CM",
    "centimeters":   "CM",
    "millimetre":    "MM",
    "millimetres":   "MM",
    "millimeter":    "MM",
    "millimeters":   "MM",
    "inch":          "IN",
    "inches":        "IN",
    "foot":          "FT",
    "feet":          "FT",
    "yard":          "YD",
    "yards":         "YD",
    "mile":          "MI",
    "miles":         "MI",
    "nautical_mile":  "NMI",
    "nautical_miles": "NMI",
}


# ---------------------------------------------------------------------------
# Volume — canonical = cubic metre
# ---------------------------------------------------------------------------


_GAL_US_M3 = 0.003785411784  # US liquid gallon
_GAL_UK_M3 = 0.00454609      # imperial gallon
_BBL_M3    = 0.158987294928  # US oil barrel (42 US gal)


class VolumeUnit(Unit, Enum):
    """Volume unit (canonical = cubic metre).

    SI scale (mL / L / m³) plus the gallon variants and the oil
    barrel — commodity feeds split between US gallons, UK gallons,
    and the 42-gallon oil barrel, and a curated view that doesn't
    keep them straight ships wrong totals.
    """

    M3      = ("m3",      1.0)
    L       = ("L",       1e-3)
    ML      = ("mL",      1e-6)
    GAL_US  = ("gal_US",  _GAL_US_M3)
    GAL_UK  = ("gal_UK",  _GAL_UK_M3)
    BBL     = ("bbl",     _BBL_M3)


VolumeUnit._ALIASES = {  # type: ignore[attr-defined]
    "cubic_meter":   "M3",
    "cubic_metre":   "M3",
    "cubic_meters":  "M3",
    "cubic_metres":  "M3",
    "m^3":           "M3",
    "m³":            "M3",
    "litre":         "L",
    "litres":        "L",
    "liter":         "L",
    "liters":        "L",
    "millilitre":    "ML",
    "millilitres":   "ML",
    "milliliter":    "ML",
    "milliliters":   "ML",
    "gallon":         "GAL_US",  # default "gallon" → US (energy convention)
    "gallons":        "GAL_US",
    "gal":            "GAL_US",
    "us_gallon":      "GAL_US",
    "us_gallons":     "GAL_US",
    "uk_gallon":      "GAL_UK",
    "uk_gallons":     "GAL_UK",
    "imperial_gallon":  "GAL_UK",
    "imperial_gallons": "GAL_UK",
    "barrel":      "BBL",
    "barrels":     "BBL",
    "bbls":        "BBL",
    "oil_barrel":  "BBL",
}


# ---------------------------------------------------------------------------
# Temperature — canonical = kelvin (AFFINE)
# ---------------------------------------------------------------------------


# °F → K: K = (F - 32) * 5/9 + 273.15 = F * (5/9) + (273.15 - 32 * 5/9)
_F_FACTOR = 5.0 / 9.0
_F_OFFSET = 273.15 - 32.0 * _F_FACTOR  # ≈ 255.3722222...


class TemperatureUnit(Unit, Enum):
    """Temperature unit (canonical = kelvin).

    The one affine family — Celsius and Fahrenheit have non-zero
    offsets relative to kelvin, so the conversion is
    ``canonical = value * factor + offset`` (not pure scaling).
    Symbols use the standard degree glyphs (``°C`` / ``°F``) so the
    curated tables render legibly in BI tools; aliases cover the
    plain-ascii forms (``"C"`` / ``"F"`` / ``"K"``).
    """

    K  = ("K",  1.0,        0.0)
    C  = ("°C", 1.0,        273.15)
    F  = ("°F", _F_FACTOR,  _F_OFFSET)


TemperatureUnit._ALIASES = {  # type: ignore[attr-defined]
    "kelvin":     "K",
    "kelvins":    "K",
    "c":          "C",
    "celsius":    "C",
    "centigrade": "C",
    "f":          "F",
    "fahrenheit": "F",
}


# ---------------------------------------------------------------------------
# Pressure — canonical = pascal
# ---------------------------------------------------------------------------


_BAR_PA  = 1e5
_MBAR_PA = 1e2
_HPA_PA  = 1e2
_ATM_PA  = 101325.0
_PSI_PA  = 6894.757293168361   # exact: lbf/in² in SI
_TORR_PA = 101325.0 / 760.0    # 1 atm / 760


class PressureUnit(Unit, Enum):
    """Pressure unit (canonical = pascal).

    SI scale (Pa / kPa / MPa / hPa) plus the bar family
    (bar / mbar), atmosphere, psi, and torr. Weather feeds publish
    in hPa or mbar, oil & gas in psi or bar, process control in
    kPa or MPa — one enum keeps them aligned.
    """

    PA   = ("Pa",   1.0)
    KPA  = ("kPa",  1e3)
    MPA  = ("MPa",  1e6)
    HPA  = ("hPa",  _HPA_PA)
    BAR  = ("bar",  _BAR_PA)
    MBAR = ("mbar", _MBAR_PA)
    ATM  = ("atm",  _ATM_PA)
    PSI  = ("psi",  _PSI_PA)
    TORR = ("Torr", _TORR_PA)


PressureUnit._ALIASES = {  # type: ignore[attr-defined]
    "pascal":      "PA",
    "pascals":     "PA",
    "kilopascal":  "KPA",
    "kilopascals": "KPA",
    "megapascal":  "MPA",
    "megapascals": "MPA",
    "hectopascal":  "HPA",
    "hectopascals": "HPA",
    "millibar":    "MBAR",
    "millibars":   "MBAR",
    "atmosphere":  "ATM",
    "atmospheres": "ATM",
    "torrs":       "TORR",
    "mmhg":        "TORR",   # 1 mmHg ≈ 1 Torr to 1 ppm — good enough
}


# ---------------------------------------------------------------------------
# Family registry — keeps the curated/standardize helpers generic
# ---------------------------------------------------------------------------


#: Tuple of every concrete unit family in the public namespace. New
#: families register here and inherit the standardize-layer support
#: (Polars / Spark per-row expressions, dash-view field builders).
UNIT_FAMILIES: tuple[type[Unit], ...] = (
    EnergyUnit,
    PowerUnit,
    MassUnit,
    LengthUnit,
    VolumeUnit,
    TemperatureUnit,
    PressureUnit,
)


def unit_family_for(value: Any) -> type[Unit]:
    """Find the unit family that recognises *value* (member or symbol).

    Useful when a curated row carries a free-form unit token and the
    caller needs to dispatch to the right family without hard-coding
    a per-source switch. Returns the first family for which
    :meth:`Unit.is_valid` returns ``True``; raises :class:`ValueError`
    when no family claims the token (the message lists every family
    that was consulted).
    """
    if isinstance(value, Unit):
        return type(value)
    for family in UNIT_FAMILIES:
        if family.is_valid(value):
            return family
    raise ValueError(
        f"No unit family recognises {value!r}. Tried: "
        f"{[f.__name__ for f in UNIT_FAMILIES]!r}."
    )
