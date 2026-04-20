"""Tests for :class:`yggdrasil.data.statistics.DataStatistic` / :class:`KPI`
and their wiring into :class:`yggdrasil.io.buffer.media_options.MediaOptions`.
"""
from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.statistics import KPI, DataStatistic
from yggdrasil.io.buffer.media_options import MediaOptions


class TestKPI:
    def test_values(self):
        assert KPI.MIN.value == "min"
        assert KPI.DISTINCT.value == "distinct"

    def test_parse_roundtrip(self):
        for k in KPI:
            assert KPI.parse(k.value) is k
            assert KPI.parse(k) is k

    def test_parse_case_insensitive(self):
        assert KPI.parse("MIN") is KPI.MIN
        assert KPI.parse("Distinct") is KPI.DISTINCT

    def test_parse_aliases(self):
        assert KPI.parse("distinct_values") is KPI.DISTINCT
        assert KPI.parse("avg") is KPI.MEAN
        assert KPI.parse("average") is KPI.MEAN
        assert KPI.parse("nulls") is KPI.NULL_COUNT
        assert KPI.parse("nunique") is KPI.DISTINCT_COUNT

    def test_parse_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown KPI 'foo'"):
            KPI.parse("foo")

    def test_parse_non_string(self):
        with pytest.raises(TypeError, match="KPI must be str or KPI"):
            KPI.parse(42)  # type: ignore[arg-type]


class TestDataStatisticConstruction:
    def test_scalar_from_str(self):
        s = DataStatistic(fields="amount", kpis="min")
        assert s.field_names == ("amount",)
        assert s.kpis == frozenset({KPI.MIN})
        assert s.is_compound is False
        assert s.label == "amount.min"
        assert isinstance(s.fields[0], Field)

    def test_scalar_multi_kpi(self):
        s = DataStatistic(fields="amount", kpis=("min", "max", "count"))
        assert s.kpis == frozenset({KPI.MIN, KPI.MAX, KPI.COUNT})
        assert s.label == "amount.count,max,min"  # sorted for determinism

    def test_compound_fields(self):
        s = DataStatistic(fields=("year", "month"), kpis="distinct")
        assert s.field_names == ("year", "month")
        assert s.is_compound is True
        assert s.label == "(year,month).distinct"

    def test_accepts_Field_instances(self):
        f = Field("amount", pa.int64())
        s = DataStatistic(fields=f, kpis="min")
        assert s.fields[0] is f

    def test_accepts_pa_Field(self):
        pf = pa.field("amount", pa.int64())
        s = DataStatistic(fields=pf, kpis="min")
        assert s.fields[0].name == "amount"

    def test_mixed_field_types(self):
        s = DataStatistic(
            fields=[Field("a", pa.int64()), "b", pa.field("c", pa.string())],
            kpis="distinct",
        )
        assert s.field_names == ("a", "b", "c")

    def test_empty_fields_rejected(self):
        with pytest.raises(ValueError, match="at least one field"):
            DataStatistic(fields=[], kpis="min")

    def test_empty_kpis_rejected(self):
        with pytest.raises(ValueError, match="at least one KPI"):
            DataStatistic(fields="x", kpis=[])

    def test_duplicate_field_rejected(self):
        with pytest.raises(ValueError, match="duplicate column 'x'"):
            DataStatistic(fields=("x", "y", "x"), kpis="distinct")

    def test_bad_field_type(self):
        with pytest.raises(TypeError, match="must be a Field / str / pa.Field"):
            DataStatistic(fields=42, kpis="min")  # type: ignore[arg-type]

    def test_bad_kpi_type(self):
        with pytest.raises(TypeError, match="must be a KPI, str"):
            DataStatistic(fields="x", kpis=42)  # type: ignore[arg-type]

    def test_is_frozen(self):
        s = DataStatistic(fields="x", kpis="min")
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.label = "nope"  # type: ignore[misc]

    def test_explicit_label(self):
        s = DataStatistic(fields="x", kpis="min", label="custom")
        assert s.label == "custom"

    def test_label_type_checked(self):
        with pytest.raises(TypeError, match="label must be str"):
            DataStatistic(fields="x", kpis="min", label=123)  # type: ignore[arg-type]

    def test_compound_rejects_scalar_only_kpi(self):
        with pytest.raises(ValueError, match=r"\{min\}.*compound key"):
            DataStatistic(fields=("year", "month"), kpis=("min", "distinct"))

    def test_compound_rejects_multiple_scalar_kpis(self):
        with pytest.raises(ValueError, match=r"\{max, min, sum\}.*compound"):
            DataStatistic(
                fields=("year", "month"),
                kpis=("min", "max", "sum", "distinct"),
            )


class TestDataStatisticParse:
    def test_scalar_single_kpi(self):
        s = DataStatistic.parse("amount.min")
        assert s.field_names == ("amount",)
        assert s.kpis == frozenset({KPI.MIN})

    def test_scalar_multi_kpi(self):
        s = DataStatistic.parse("amount.min,max,count")
        assert s.kpis == frozenset({KPI.MIN, KPI.MAX, KPI.COUNT})

    def test_compound_distinct(self):
        s = DataStatistic.parse("(year,month).distinct")
        assert s.field_names == ("year", "month")
        assert s.kpis == frozenset({KPI.DISTINCT})

    def test_compound_with_whitespace(self):
        s = DataStatistic.parse("( year , month ) . distinct , distinct_count")
        assert s.field_names == ("year", "month")
        assert s.kpis == frozenset({KPI.DISTINCT, KPI.DISTINCT_COUNT})

    def test_dotted_column_via_parens(self):
        s = DataStatistic.parse("(ns.col).min")
        assert s.field_names == ("ns.col",)
        assert s.kpis == frozenset({KPI.MIN})

    def test_passthrough_instance(self):
        s = DataStatistic(fields="amount", kpis="min")
        assert DataStatistic.parse(s) is s

    def test_missing_kpi_suffix(self):
        with pytest.raises(ValueError, match="missing '.kpi' suffix"):
            DataStatistic.parse("amount")

    def test_trailing_dot(self):
        with pytest.raises(ValueError, match="no KPI after"):
            DataStatistic.parse("amount.")

    def test_empty_spec(self):
        with pytest.raises(ValueError, match="empty spec"):
            DataStatistic.parse("")

    def test_unbalanced_parens(self):
        with pytest.raises(ValueError, match=r"unbalanced '\('"):
            DataStatistic.parse("(a.min")
        with pytest.raises(ValueError, match=r"unbalanced '\)'"):
            DataStatistic.parse("a).min")

    def test_mixed_comma_outside_parens(self):
        with pytest.raises(ValueError, match="mixed parens / commas"):
            DataStatistic.parse("a,b.min")

    def test_unknown_kpi_in_dsl(self):
        with pytest.raises(ValueError, match="Unknown KPI 'bogus'"):
            DataStatistic.parse("amount.bogus")

    def test_non_string(self):
        with pytest.raises(TypeError, match="expects a str"):
            DataStatistic.parse(42)  # type: ignore[arg-type]


class TestParseMany:
    def test_none_passthrough(self):
        assert DataStatistic.parse_many(None) is None

    def test_mixed(self):
        stats = DataStatistic.parse_many(
            [
                "amount.min,max",
                DataStatistic(fields="id", kpis="count"),
                "(year,month).distinct",
            ]
        )
        assert stats is not None
        labels = [s.label for s in stats]
        assert labels == ["amount.max,min", "id.count", "(year,month).distinct"]

    def test_single_scalar_rejected(self):
        with pytest.raises(TypeError, match="Wrap it in a list"):
            DataStatistic.parse_many("amount.min")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Wrap it in a list"):
            DataStatistic.parse_many(DataStatistic(fields="x", kpis="min"))  # type: ignore[arg-type]

    def test_duplicate_label(self):
        with pytest.raises(ValueError, match=r"Duplicate statistics entry with label 'amount.min'"):
            DataStatistic.parse_many(["amount.min", "amount.min"])

    def test_duplicate_label_via_instance(self):
        with pytest.raises(ValueError, match="Duplicate statistics entry"):
            DataStatistic.parse_many(
                [
                    DataStatistic(fields="a", kpis="min", label="dup"),
                    DataStatistic(fields="b", kpis="max", label="dup"),
                ]
            )

    def test_entry_type_check(self):
        with pytest.raises(TypeError, match="DataStatistic or DSL strings"):
            DataStatistic.parse_many([42])  # type: ignore[list-item]


class TestMediaOptionsStatistics:
    def test_default_is_none(self):
        assert MediaOptions().statistics is None

    def test_dsl_strings(self):
        mo = MediaOptions(statistics=["amount.min,max", "(year,month).distinct"])
        assert mo.statistics is not None
        assert [s.label for s in mo.statistics] == [
            "amount.max,min",
            "(year,month).distinct",
        ]

    def test_partition_tuple_distinct(self):
        mo = MediaOptions(statistics=["(year,month).distinct"])
        assert mo.statistics is not None
        s = mo.statistics[0]
        assert s.is_compound
        assert KPI.DISTINCT in s.kpis

    def test_mixed_instance_and_dsl(self):
        mo = MediaOptions(
            statistics=[
                DataStatistic(fields="id", kpis="count"),
                "amount.min",
            ]
        )
        assert mo.statistics is not None
        assert len(mo.statistics) == 2

    def test_single_string_rejected(self):
        with pytest.raises(TypeError, match="Wrap it in a list"):
            MediaOptions(statistics="amount.min")  # type: ignore[arg-type]

    def test_check_parameters_passthrough(self):
        mo = MediaOptions.check_parameters(statistics=["amount.min"])
        assert mo.statistics is not None
        assert mo.statistics[0].label == "amount.min"

    def test_check_parameters_keeps_existing(self):
        base = MediaOptions(statistics=["amount.min"])
        merged = MediaOptions.check_parameters(options=base)
        assert merged.statistics is not None
        assert [s.label for s in merged.statistics] == ["amount.min"]

    def test_exported_from_data_package(self):
        import yggdrasil.data as data_pkg

        assert data_pkg.DataStatistic is DataStatistic
        assert data_pkg.KPI is KPI
