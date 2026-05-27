"""Behaviors of :mod:`yggdrasil.enums.units`.

Each unit family ships with the same surface: :meth:`Unit.from_`
forgiving coercion, :meth:`Unit.convert` scalar conversion, the
``factor`` / ``offset`` / ``symbol`` properties used by the
vectorised :mod:`yggdrasil.databricks.standardize` helpers, plus
:func:`unit_family_for` cross-family dispatch.

Tests parametrise across families so a regression in the shared
:class:`Unit` mixin shows up everywhere instead of once per family.
"""
from __future__ import annotations

import math

import pytest

from yggdrasil.enums.units import (
    EnergyUnit,
    LengthUnit,
    MassUnit,
    PowerUnit,
    PressureUnit,
    TemperatureUnit,
    Unit,
    UNIT_FAMILIES,
    VolumeUnit,
    unit_family_for,
)


# ---------------------------------------------------------------------------
# Cross-family invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", UNIT_FAMILIES)
class TestFamilyInvariants:
    """Every concrete family upholds these contract guarantees."""

    def test_canonical_is_unique(self, family: type[Unit]) -> None:
        canonicals = [m for m in family if m.factor == 1.0 and m.offset == 0.0]
        assert len(canonicals) == 1, (
            f"{family.__name__} must have exactly one canonical member "
            f"(factor=1.0, offset=0.0); got {[m.name for m in canonicals]!r}"
        )

    def test_canonical_classmethod_returns_unique(self, family: type[Unit]) -> None:
        m = family.canonical()
        assert m.factor == 1.0
        assert m.offset == 0.0
        # Cached on second call.
        assert family.canonical() is m

    def test_round_trip_via_canonical(self, family: type[Unit]) -> None:
        # For every (a, b) pair, converting a→b→a returns the original.
        for a in family:
            for b in family:
                v = 42.0
                back = family.convert(family.convert(v, a, b), b, a)
                assert math.isclose(back, v, rel_tol=1e-9, abs_tol=1e-9), (
                    f"Round-trip {a.symbol}→{b.symbol}→{a.symbol} failed: "
                    f"42.0 != {back}"
                )

    def test_from_passthrough_member(self, family: type[Unit]) -> None:
        m = next(iter(family))
        assert family.from_(m) is m

    def test_from_member_name(self, family: type[Unit]) -> None:
        m = next(iter(family))
        assert family.from_(m.name) is m
        assert family.from_(m.name.lower()) is m

    def test_from_symbol_case_insensitive(self, family: type[Unit]) -> None:
        m = next(iter(family))
        assert family.from_(m.symbol) is m
        assert family.from_(m.symbol.lower()) is m
        assert family.from_(m.symbol.upper()) is m

    def test_from_unknown_token_raises(self, family: type[Unit]) -> None:
        with pytest.raises(ValueError, match=family.__name__):
            family.from_("not-a-real-unit-token-xyz")

    def test_from_unknown_default(self, family: type[Unit]) -> None:
        sentinel = object()
        assert family.from_("not-a-real-unit-token-xyz", default=sentinel) is sentinel

    def test_from_none_raises(self, family: type[Unit]) -> None:
        with pytest.raises(ValueError):
            family.from_(None)

    def test_from_none_default(self, family: type[Unit]) -> None:
        assert family.from_(None, default=None) is None

    def test_from_int_raises(self, family: type[Unit]) -> None:
        with pytest.raises(TypeError):
            family.from_(123)

    def test_is_valid_member(self, family: type[Unit]) -> None:
        m = next(iter(family))
        assert family.is_valid(m) is True
        assert family.is_valid(m.symbol) is True

    def test_is_valid_unknown(self, family: type[Unit]) -> None:
        assert family.is_valid("def-not-a-unit") is False
        assert family.is_valid(None) is False
        assert family.is_valid(123) is False

    def test_factor_map_covers_all_members(self, family: type[Unit]) -> None:
        fm = family.factor_map()
        for m in family:
            assert m.symbol in fm
            assert fm[m.symbol] == m.factor

    def test_offset_map_covers_all_members(self, family: type[Unit]) -> None:
        om = family.offset_map()
        for m in family:
            assert m.symbol in om
            assert om[m.symbol] == m.offset

    def test_convert_same_unit_short_circuits(self, family: type[Unit]) -> None:
        m = next(iter(family))
        assert family.convert(7.5, m, m) == 7.5

    def test_to_from_canonical_round_trip(self, family: type[Unit]) -> None:
        for m in family:
            assert math.isclose(m.from_canonical(m.to_canonical(3.0)), 3.0,
                                rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------


class TestEnergyUnit:

    def test_canonical_is_joule(self) -> None:
        assert EnergyUnit.canonical() is EnergyUnit.J

    @pytest.mark.parametrize("value,source,target,expected", [
        (1.0,    "MWh", "kWh", 1_000.0),
        (1.0,    "MWh", "J",   3.6e9),
        (1.0,    "BTU", "J",   1055.05585262),
        (1.0,    "kcal","J",   4184.0),
        (1.0,    "TWh", "GWh", 1_000.0),
        (1000.0, "kWh", "MWh", 1.0),
    ])
    def test_known_conversions(self, value: float, source: str, target: str, expected: float) -> None:
        assert math.isclose(
            EnergyUnit.convert(value, source, target), expected, rel_tol=1e-9,
        )

    @pytest.mark.parametrize("alias,expected", [
        ("joule",           EnergyUnit.J),
        ("joules",          EnergyUnit.J),
        ("kilowatthour",    EnergyUnit.KWH),
        ("megawatthours",   EnergyUnit.MWH),
        ("MEGAWATTHOURS",   EnergyUnit.MWH),
        ("therms",          EnergyUnit.THERM),
        ("btu",             EnergyUnit.BTU),
    ])
    def test_aliases(self, alias: str, expected: EnergyUnit) -> None:
        assert EnergyUnit.from_(alias) is expected


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


class TestPowerUnit:

    def test_canonical_is_watt(self) -> None:
        assert PowerUnit.canonical() is PowerUnit.W

    @pytest.mark.parametrize("value,source,target,expected", [
        (1.0,    "MW", "kW", 1_000.0),
        (1.0,    "GW", "MW", 1_000.0),
        (1.0,    "hp", "W",  745.6998715822702),
        (1.0,    "TW", "GW", 1_000.0),
    ])
    def test_known_conversions(self, value: float, source: str, target: str, expected: float) -> None:
        assert math.isclose(
            PowerUnit.convert(value, source, target), expected, rel_tol=1e-9,
        )


# ---------------------------------------------------------------------------
# Mass
# ---------------------------------------------------------------------------


class TestMassUnit:

    def test_canonical_is_kilogram(self) -> None:
        assert MassUnit.canonical() is MassUnit.KG

    @pytest.mark.parametrize("value,source,target,expected", [
        (1.0,    "t",   "kg",  1_000.0),
        (1.0,    "kg",  "g",   1_000.0),
        (1.0,    "lb",  "kg",  0.45359237),
        (16.0,   "oz",  "lb",  1.0),
        (1.0,    "Mt",  "t",   1_000_000.0),
    ])
    def test_known_conversions(self, value: float, source: str, target: str, expected: float) -> None:
        assert math.isclose(
            MassUnit.convert(value, source, target), expected, rel_tol=1e-9,
        )

    def test_ton_alias_lands_on_metric_tonne(self) -> None:
        # The codebase uses "ton" == metric tonne (1000 kg) consistently;
        # if a feed needs the US short ton or UK long ton, the alias
        # table must be extended explicitly.
        assert MassUnit.from_("ton") is MassUnit.T


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------


class TestLengthUnit:

    def test_canonical_is_metre(self) -> None:
        assert LengthUnit.canonical() is LengthUnit.M

    @pytest.mark.parametrize("value,source,target,expected", [
        (1.0,    "km",  "m",   1_000.0),
        (1.0,    "mi",  "km",  1.609344),
        (1.0,    "ft",  "in",  12.0),
        (1.0,    "yd",  "ft",  3.0),
        (1.0,    "nmi", "m",   1852.0),
    ])
    def test_known_conversions(self, value: float, source: str, target: str, expected: float) -> None:
        assert math.isclose(
            LengthUnit.convert(value, source, target), expected, rel_tol=1e-9,
        )


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


class TestVolumeUnit:

    def test_canonical_is_m3(self) -> None:
        assert VolumeUnit.canonical() is VolumeUnit.M3

    @pytest.mark.parametrize("value,source,target,expected", [
        (1.0,    "m3",     "L",       1_000.0),
        (1.0,    "L",      "mL",      1_000.0),
        (1.0,    "gal_US", "L",       3.785411784),
        (1.0,    "gal_UK", "L",       4.54609),
        (1.0,    "bbl",    "gal_US",  42.0),
    ])
    def test_known_conversions(self, value: float, source: str, target: str, expected: float) -> None:
        assert math.isclose(
            VolumeUnit.convert(value, source, target), expected, rel_tol=1e-9,
        )

    def test_cubed_glyph_alias(self) -> None:
        assert VolumeUnit.from_("m³") is VolumeUnit.M3
        assert VolumeUnit.from_("m^3") is VolumeUnit.M3

    def test_default_gallon_is_us(self) -> None:
        # Energy / commodity feeds default "gallon" to US — the alias
        # table reflects that. UK gallons need explicit ``gal_UK``.
        assert VolumeUnit.from_("gallon") is VolumeUnit.GAL_US


# ---------------------------------------------------------------------------
# Temperature (affine — exercise the offset path)
# ---------------------------------------------------------------------------


class TestTemperatureUnit:

    def test_canonical_is_kelvin(self) -> None:
        assert TemperatureUnit.canonical() is TemperatureUnit.K

    @pytest.mark.parametrize("value,source,target,expected", [
        (0.0,    "C",  "K",   273.15),
        (100.0,  "C",  "K",   373.15),
        (100.0,  "C",  "F",   212.0),
        (0.0,    "C",  "F",   32.0),
        (32.0,   "F",  "C",   0.0),
        (212.0,  "F",  "C",   100.0),
        (212.0,  "F",  "K",   373.15),
        (273.15, "K",  "C",   0.0),
        (-40.0,  "C",  "F",   -40.0),     # the famous -40 equivalence
    ])
    def test_affine_conversions(self, value: float, source: str, target: str, expected: float) -> None:
        assert math.isclose(
            TemperatureUnit.convert(value, source, target), expected,
            rel_tol=1e-9, abs_tol=1e-9,
        )

    def test_celsius_symbol_carries_degree_glyph(self) -> None:
        assert TemperatureUnit.C.symbol == "°C"
        assert TemperatureUnit.F.symbol == "°F"

    @pytest.mark.parametrize("alias,expected", [
        ("celsius",    TemperatureUnit.C),
        ("centigrade", TemperatureUnit.C),
        ("fahrenheit", TemperatureUnit.F),
        ("kelvin",     TemperatureUnit.K),
        ("c",          TemperatureUnit.C),
        ("F",          TemperatureUnit.F),
    ])
    def test_aliases(self, alias: str, expected: TemperatureUnit) -> None:
        assert TemperatureUnit.from_(alias) is expected


# ---------------------------------------------------------------------------
# Pressure
# ---------------------------------------------------------------------------


class TestPressureUnit:

    def test_canonical_is_pascal(self) -> None:
        assert PressureUnit.canonical() is PressureUnit.PA

    @pytest.mark.parametrize("value,source,target,expected", [
        (1.0,    "bar",  "Pa",   1e5),
        (1.0,    "kPa",  "Pa",   1e3),
        (1.0,    "atm",  "Pa",   101325.0),
        (1.0,    "psi",  "Pa",   6894.757293168361),
        (760.0,  "Torr", "atm",  1.0),
        (1.0,    "hPa",  "mbar", 1.0),       # equal by SI definition
    ])
    def test_known_conversions(self, value: float, source: str, target: str, expected: float) -> None:
        assert math.isclose(
            PressureUnit.convert(value, source, target), expected, rel_tol=1e-9,
        )


# ---------------------------------------------------------------------------
# unit_family_for
# ---------------------------------------------------------------------------


class TestUnitFamilyFor:

    @pytest.mark.parametrize("token,expected_family", [
        ("MWh",   EnergyUnit),
        ("kWh",   EnergyUnit),
        ("MW",    PowerUnit),
        ("hp",    PowerUnit),
        ("kg",    MassUnit),
        ("lb",    MassUnit),
        ("m",     LengthUnit),
        ("mi",    LengthUnit),
        ("L",     VolumeUnit),
        ("m³",    VolumeUnit),
        ("°C",    TemperatureUnit),
        ("F",     TemperatureUnit),
        ("psi",   PressureUnit),
        ("bar",   PressureUnit),
    ])
    def test_resolves(self, token: str, expected_family: type[Unit]) -> None:
        assert unit_family_for(token) is expected_family

    def test_member_passthrough(self) -> None:
        assert unit_family_for(EnergyUnit.MWH) is EnergyUnit

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="No unit family"):
            unit_family_for("not-a-real-unit-xyz")
