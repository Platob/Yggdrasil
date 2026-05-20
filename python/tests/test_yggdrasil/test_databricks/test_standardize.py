"""Behaviors of :mod:`yggdrasil.databricks.standardize`.

The standardize helpers are the curated/dash layer of the codebase
— a thin layer that turns the unit / currency enums into Polars and
Spark expressions for "carry the source value AND the standardised
equivalent" patterns. Tests cover:

* :func:`polars_convert_unit` — static and per-row source units,
  linear and affine families;
* :func:`with_unit_equivalent` — DataFrame helper that appends the
  standardised column;
* :func:`with_currency_equivalents` — currency conversion via a
  stub :class:`FxRate` backend (no network);
* :func:`dash_dual_value_fields` — schema-field builder for the
  ``(source_value, source_unit, equivalent…)`` triplet.

Spark-side helpers live behind :class:`yggdrasil.spark.tests.SparkTestCase`
so the suite skips on base installs that don't ship pyspark.
"""
from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.data.enums.currency import Currency
from yggdrasil.data.enums.units import (
    EnergyUnit,
    PowerUnit,
    TemperatureUnit,
)
from yggdrasil.databricks.standardize import (
    DEFAULT_CURRENCY_TARGETS,
    dash_dual_value_fields,
    polars_convert_unit,
    spark_convert_unit,
    spark_with_currency_equivalents,
    spark_with_unit_equivalent,
    standardized_column_name,
    with_currency_equivalents,
    with_unit_equivalent,
)
from yggdrasil.polars.tests import PolarsTestCase
from yggdrasil.spark.tests import SparkTestCase

from ..test_fxrate._helpers import StubFxBackend, make_quote


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


class TestStandardizedColumnName:

    @pytest.mark.parametrize("base,target,expected", [
        ("value",  EnergyUnit.KWH,          "value_kwh"),
        ("value",  EnergyUnit.MWH,          "value_mwh"),
        ("temp",   TemperatureUnit.C,       "temp_c"),
        ("temp",   TemperatureUnit.F,       "temp_f"),
        ("vol",    "m³",                    "vol_m3"),
        ("price",  Currency.EUR,            "price_eur"),
        ("price",  Currency.JPY,            "price_jpy"),
    ])
    def test_naming(self, base: str, target, expected: str) -> None:
        assert standardized_column_name(base, target) == expected


# ---------------------------------------------------------------------------
# Polars: static unit conversion
# ---------------------------------------------------------------------------


class TestPolarsStaticUnit(PolarsTestCase):

    def test_static_same_unit_passthrough(self) -> None:
        df = self.df({"v": [1.0, 2.0, 3.0]})
        out = df.with_columns(
            polars_convert_unit("v", EnergyUnit.MWH, EnergyUnit.MWH).alias("k")
        )
        self.assertSeriesEqual(out["k"], self.series("k", [1.0, 2.0, 3.0]))

    def test_static_mwh_to_kwh(self) -> None:
        df = self.df({"v": [1.0, 2.0, 3.0]})
        out = df.with_columns(
            polars_convert_unit("v", EnergyUnit.MWH, EnergyUnit.KWH).alias("k")
        )
        self.assertSeriesEqual(out["k"], self.series("k", [1000.0, 2000.0, 3000.0]))

    def test_static_affine_celsius_to_kelvin(self) -> None:
        df = self.df({"v": [0.0, 100.0]})
        out = df.with_columns(
            polars_convert_unit("v", TemperatureUnit.C, TemperatureUnit.K).alias("k")
        )
        self.assertSeriesEqual(
            out["k"], self.series("k", [273.15, 373.15]),
        )

    def test_static_family_inferred_from_target(self) -> None:
        # With family unspecified, target='kWh' resolves via unit_family_for
        # to EnergyUnit; source='MWh' is then resolved within that family.
        df = self.df({"v": [1.0]})
        out = df.with_columns(
            polars_convert_unit("v", "MWh", "kWh").alias("k")
        )
        assert out["k"].to_list() == [1000.0]


# ---------------------------------------------------------------------------
# Polars: per-row source unit
# ---------------------------------------------------------------------------


class TestPolarsPerRowUnit(PolarsTestCase):

    def test_per_row_source_via_column(self) -> None:
        df = self.df({"v": [1.0, 2.0, 1000.0], "u": ["MWh", "GWh", "kWh"]})
        out = df.with_columns(
            polars_convert_unit("v", "u", EnergyUnit.KWH, family=EnergyUnit).alias("k")
        )
        # 1 MWh = 1000 kWh; 2 GWh = 2_000_000 kWh; 1000 kWh = 1000 kWh
        self.assertSeriesEqual(out["k"], self.series("k", [1000.0, 2_000_000.0, 1000.0]))

    def test_per_row_unknown_unit_emits_null(self) -> None:
        df = self.df({"v": [1.0, 2.0], "u": ["MWh", "not-a-unit"]})
        out = df.with_columns(
            polars_convert_unit("v", "u", EnergyUnit.KWH, family=EnergyUnit).alias("k")
        )
        vals = out["k"].to_list()
        assert vals[1] is None
        assert vals[0] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# with_unit_equivalent
# ---------------------------------------------------------------------------


class TestWithUnitEquivalent(PolarsTestCase):

    def test_static_source_unit(self) -> None:
        df = self.df({"v": [1.0, 2.0]})
        out = with_unit_equivalent(
            df, value_col="v", unit_col=EnergyUnit.MWH, target=EnergyUnit.KWH,
        )
        assert "v_kwh" in out.columns
        assert out["v_kwh"].to_list() == pytest.approx([1000.0, 2000.0])

    def test_static_source_unit_by_symbol(self) -> None:
        df = self.df({"v": [1.0]})
        out = with_unit_equivalent(
            df, value_col="v", unit_col="MWh", target="kWh", family=EnergyUnit,
        )
        assert out["v_kwh"].to_list() == pytest.approx([1000.0])

    def test_per_row_source(self) -> None:
        df = self.df({"v": [1.0, 1.0, 1.0], "u": ["MW", "GW", "kW"]})
        out = with_unit_equivalent(
            df, value_col="v", unit_col="u", target=PowerUnit.W,
        )
        assert "v_w" in out.columns
        assert out["v_w"].to_list() == pytest.approx([1e6, 1e9, 1e3])

    def test_lazyframe_returns_lazyframe(self) -> None:
        ldf = self.lazy({"v": [1.0], "u": ["MWh"]})
        out = with_unit_equivalent(
            ldf, value_col="v", unit_col="u", target=EnergyUnit.KWH,
        )
        assert isinstance(out, self.pl.LazyFrame)
        assert out.collect()["v_kwh"].to_list() == pytest.approx([1000.0])

    def test_custom_suffix(self) -> None:
        df = self.df({"v": [1.0]})
        out = with_unit_equivalent(
            df, value_col="v", unit_col=EnergyUnit.MWH, target=EnergyUnit.KWH,
            suffix="v_standardised",
        )
        assert "v_standardised" in out.columns
        assert "v_kwh" not in out.columns


# ---------------------------------------------------------------------------
# with_currency_equivalents (stub backend, no network)
# ---------------------------------------------------------------------------


def _stub_fx() -> "object":
    """Build an :class:`FxRate` with a deterministic stub backend."""
    from yggdrasil.fxrate import FxRate

    backend = StubFxBackend(quotes=(
        make_quote("EUR", "USD", value=1.10),
        make_quote("EUR", "CHF", value=0.97),
        make_quote("EUR", "JPY", value=160.0),
        make_quote("USD", "EUR", value=0.91),
        make_quote("USD", "CHF", value=0.88),
        make_quote("USD", "JPY", value=145.0),
        make_quote("GBP", "EUR", value=1.17),
        make_quote("GBP", "USD", value=1.29),
        make_quote("GBP", "CHF", value=1.13),
    ))
    return FxRate(backends=(backend,))


class TestWithCurrencyEquivalents(PolarsTestCase):

    def test_default_targets_three_equivalent_columns(self) -> None:
        df = self.df({"amount": [100.0]})
        out = with_currency_equivalents(
            df, value_col="amount", currency_col="EUR", fx=_stub_fx(),
        )
        for tgt in DEFAULT_CURRENCY_TARGETS:
            assert standardized_column_name("amount", tgt) in out.columns
        assert out["amount_eur"].to_list() == pytest.approx([100.0])   # same → identity
        assert out["amount_usd"].to_list() == pytest.approx([110.0])
        assert out["amount_chf"].to_list() == pytest.approx([97.0])

    def test_static_currency_symbol(self) -> None:
        df = self.df({"amount": [100.0, 200.0]})
        out = with_currency_equivalents(
            df, value_col="amount", currency_col="USD", fx=_stub_fx(),
            targets=("EUR",),
        )
        # USD 100 -> EUR 91 ; USD 200 -> EUR 182
        assert out["amount_eur"].to_list() == [91.0, 182.0]

    def test_static_currency_instance(self) -> None:
        df = self.df({"amount": [100.0]})
        out = with_currency_equivalents(
            df, value_col="amount", currency_col=Currency.GBP, fx=_stub_fx(),
            targets=(Currency.EUR, Currency.USD),
        )
        assert out["amount_eur"].to_list() == [117.0]
        assert out["amount_usd"].to_list() == [129.0]

    def test_per_row_currency(self) -> None:
        df = self.df({
            "amount": [100.0, 100.0, 100.0],
            "ccy":    ["EUR", "USD", "GBP"],
        })
        out = with_currency_equivalents(
            df, value_col="amount", currency_col="ccy", fx=_stub_fx(),
        )
        assert out["amount_eur"].to_list() == pytest.approx([100.0, 91.0, 117.0])
        assert out["amount_usd"].to_list() == pytest.approx([110.0, 100.0, 129.0])
        assert out["amount_chf"].to_list() == pytest.approx([97.0, 88.0, 113.0])

    def test_per_row_unknown_currency_emits_null(self) -> None:
        df = self.df({
            "amount": [100.0, 100.0],
            "ccy":    ["EUR", "XYZ"],
        })
        # Currency.parse('XYZ') succeeds (any 3-letter alpha is treated
        # as a valid ISO code), but the stub backend doesn't have rates
        # for XYZ — the equivalent column is null on that row.
        out = with_currency_equivalents(
            df, value_col="amount", currency_col="ccy", fx=_stub_fx(),
        )
        assert out["amount_eur"].to_list() == [100.0, None]

    def test_lazyframe_returns_lazyframe(self) -> None:
        ldf = self.lazy({"amount": [100.0], "ccy": ["EUR"]})
        out = with_currency_equivalents(
            ldf, value_col="amount", currency_col="ccy", fx=_stub_fx(),
        )
        assert isinstance(out, self.pl.LazyFrame)
        assert out.collect()["amount_usd"].to_list() == pytest.approx([110.0])

    def test_at_and_at_col_mutually_exclusive(self) -> None:
        df = self.df({"amount": [1.0], "ccy": ["EUR"], "ts": [dt.datetime.now(dt.timezone.utc)]})
        with pytest.raises(ValueError, match="pass at OR at_col"):
            with_currency_equivalents(
                df, value_col="amount", currency_col="ccy",
                at=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
                at_col="ts",
                fx=_stub_fx(),
            )

    def test_empty_targets_raises(self) -> None:
        df = self.df({"amount": [1.0]})
        with pytest.raises(ValueError, match="at least one target"):
            with_currency_equivalents(
                df, value_col="amount", currency_col="EUR",
                targets=(), fx=_stub_fx(),
            )


# ---------------------------------------------------------------------------
# dash_dual_value_fields — schema-field builder
# ---------------------------------------------------------------------------


class TestDashDualValueFields:

    def test_currency_default_targets(self) -> None:
        fields = dash_dual_value_fields("price", currency=True)
        names = [f.name for f in fields]
        assert names == ["price", "price_currency", "price_eur", "price_usd", "price_chf"]

    def test_currency_single_target(self) -> None:
        fields = dash_dual_value_fields("price", currency=True, target=Currency.EUR)
        assert [f.name for f in fields] == ["price", "price_currency", "price_eur"]

    def test_currency_explicit_list(self) -> None:
        fields = dash_dual_value_fields("price", currency=True, target=["EUR", "JPY"])
        assert [f.name for f in fields] == ["price", "price_currency", "price_eur", "price_jpy"]

    def test_currency_target_as_string(self) -> None:
        fields = dash_dual_value_fields("price", currency=True, target="GBP")
        assert [f.name for f in fields] == ["price", "price_currency", "price_gbp"]

    def test_unit_target(self) -> None:
        fields = dash_dual_value_fields("volume", target=EnergyUnit.KWH)
        assert [f.name for f in fields] == ["volume", "volume_unit", "volume_kwh"]

    def test_unit_target_temperature(self) -> None:
        fields = dash_dual_value_fields("temp", target=TemperatureUnit.K)
        assert [f.name for f in fields] == ["temp", "temp_unit", "temp_k"]

    def test_no_target_no_currency_raises(self) -> None:
        with pytest.raises(ValueError, match="target=None"):
            dash_dual_value_fields("foo")

    def test_unit_target_list_raises(self) -> None:
        with pytest.raises(TypeError, match="single target unit"):
            dash_dual_value_fields("foo", target=["kWh", "MWh"])

    def test_metadata_carries_standardized_unit(self) -> None:
        fields = dash_dual_value_fields("volume", target=EnergyUnit.KWH)
        equivalent = fields[2]  # volume_kwh
        meta = equivalent.metadata or {}
        # Metadata keys are bytes in Field; look up both spellings.
        assert any(
            meta.get(k) in (b"kWh", "kWh") for k in (b"standardized_unit", "standardized_unit")
        ) or "kWh" in str(meta)


# ---------------------------------------------------------------------------
# Spark — same shape as the Polars surface, exercised against a local
# SparkSession via :class:`SparkTestCase`. Skipped when pyspark isn't
# installed (matching the rest of the Spark suite).
# ---------------------------------------------------------------------------


class TestSparkConvertUnit(SparkTestCase):

    def _to_dict_list(self, df) -> list[dict]:
        return [r.asDict() for r in df.collect()]

    def test_static_unit_conversion(self) -> None:
        df = self.spark.createDataFrame(
            [(1.0,), (2.0,)], ["v"],
        )
        out = df.withColumn(
            "k",
            spark_convert_unit("v", EnergyUnit.MWH, EnergyUnit.KWH),
        )
        rows = self._to_dict_list(out)
        assert rows[0]["k"] == pytest.approx(1000.0)
        assert rows[1]["k"] == pytest.approx(2000.0)

    def test_static_affine_celsius_to_kelvin(self) -> None:
        df = self.spark.createDataFrame([(0.0,), (100.0,)], ["v"])
        out = df.withColumn(
            "k",
            spark_convert_unit("v", TemperatureUnit.C, TemperatureUnit.K),
        )
        rows = self._to_dict_list(out)
        assert rows[0]["k"] == pytest.approx(273.15)
        assert rows[1]["k"] == pytest.approx(373.15)

    def test_per_row_unit_via_column(self) -> None:
        df = self.spark.createDataFrame(
            [(1.0, "MWh"), (1.0, "GWh"), (1.0, "kWh")],
            ["v", "u"],
        )
        out = df.withColumn(
            "k",
            spark_convert_unit("v", "u", EnergyUnit.KWH, family=EnergyUnit),
        )
        rows = self._to_dict_list(out)
        # Sort by k for deterministic comparison.
        ks = sorted(r["k"] for r in rows)
        assert ks[0] == pytest.approx(1000.0)
        assert ks[1] == pytest.approx(1_000_000.0)
        assert ks[2] == pytest.approx(1_000_000_000.0)


class TestSparkWithUnitEquivalent(SparkTestCase):

    def test_static_source_unit(self) -> None:
        df = self.spark.createDataFrame([(1.0,), (2.0,)], ["v"])
        out = spark_with_unit_equivalent(
            df, value_col="v", unit_col=EnergyUnit.MWH, target=EnergyUnit.KWH,
        )
        assert "v_kwh" in out.columns
        rows = [r.asDict() for r in out.collect()]
        assert rows[0]["v_kwh"] == pytest.approx(1000.0)
        assert rows[1]["v_kwh"] == pytest.approx(2000.0)

    def test_per_row_source_unit(self) -> None:
        df = self.spark.createDataFrame(
            [(1.0, "MW"), (1.0, "GW")], ["v", "u"],
        )
        out = spark_with_unit_equivalent(
            df, value_col="v", unit_col="u", target=PowerUnit.W,
        )
        assert "v_w" in out.columns
        rows = sorted([r["v_w"] for r in out.collect()])
        assert rows[0] == pytest.approx(1e6)
        assert rows[1] == pytest.approx(1e9)


class TestSparkWithCurrencyEquivalents(SparkTestCase):

    def test_default_targets(self) -> None:
        df = self.spark.createDataFrame([(100.0,)], ["amount"])
        out = spark_with_currency_equivalents(
            df, value_col="amount", currency_col="EUR", fx=_stub_fx(),
        )
        for tgt in DEFAULT_CURRENCY_TARGETS:
            assert standardized_column_name("amount", tgt) in out.columns
        row = out.collect()[0]
        assert row["amount_eur"] == pytest.approx(100.0)
        assert row["amount_usd"] == pytest.approx(110.0)
        assert row["amount_chf"] == pytest.approx(97.0)

    def test_per_row_currency(self) -> None:
        df = self.spark.createDataFrame(
            [(100.0, "EUR"), (100.0, "USD"), (100.0, "GBP")],
            ["amount", "ccy"],
        )
        out = spark_with_currency_equivalents(
            df, value_col="amount", currency_col="ccy", fx=_stub_fx(),
        )
        rows = {r["ccy"]: r for r in out.collect()}
        assert rows["EUR"]["amount_usd"] == pytest.approx(110.0)
        assert rows["USD"]["amount_eur"] == pytest.approx(91.0)
        assert rows["GBP"]["amount_chf"] == pytest.approx(113.0)

    def test_at_and_at_col_mutually_exclusive(self) -> None:
        df = self.spark.createDataFrame([(1.0, "EUR")], ["amount", "ccy"])
        with pytest.raises(ValueError, match="pass at OR at_col"):
            spark_with_currency_equivalents(
                df, value_col="amount", currency_col="ccy",
                at=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
                at_col="ts",
                fx=_stub_fx(),
            )
