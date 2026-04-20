"""Tests for :class:`yggdrasil.data.statistics.DataStatisticsConfig` and
its wiring into :class:`yggdrasil.io.buffer.media_options.MediaOptions`.
"""
from __future__ import annotations

import dataclasses

import pytest

from yggdrasil.data.statistics import DataStatisticsConfig
from yggdrasil.io.buffer.media_options import MediaOptions


class TestDataStatisticsConfig:
    def test_defaults(self):
        cfg = DataStatisticsConfig(field="amount")
        assert cfg.field == "amount"
        # cheap / useful flags on by default
        assert cfg.count is True
        assert cfg.null_count is True
        assert cfg.min is True
        assert cfg.max is True
        # expensive flags off by default
        assert cfg.distinct_count is False
        assert cfg.distinct_values is False
        assert cfg.sum is False
        assert cfg.mean is False
        assert cfg.byte_size is False
        assert cfg.is_compound is False
        assert cfg.field_names == ("amount",)

    def test_is_frozen(self):
        cfg = DataStatisticsConfig(field="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.field = "y"  # type: ignore[misc]

    def test_field_must_be_nonempty_str(self):
        with pytest.raises(TypeError, match="must be a str or a tuple/list of str"):
            DataStatisticsConfig(field=123)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-empty"):
            DataStatisticsConfig(field="")

    def test_flags_must_be_bool(self):
        with pytest.raises(TypeError, match=r"\.min must be bool"):
            DataStatisticsConfig(field="x", min="true")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match=r"\.count must be bool"):
            DataStatisticsConfig(field="x", count=1)  # type: ignore[arg-type]

    def test_coerce_instance(self):
        cfg = DataStatisticsConfig(field="a", sum=True)
        assert DataStatisticsConfig.coerce(cfg) is cfg

    def test_coerce_string(self):
        cfg = DataStatisticsConfig.coerce("price")
        assert cfg == DataStatisticsConfig(field="price")

    def test_coerce_dict(self):
        cfg = DataStatisticsConfig.coerce({"field": "price", "sum": True, "max": False})
        assert cfg.field == "price"
        assert cfg.sum is True
        assert cfg.max is False

    def test_coerce_dict_missing_field(self):
        with pytest.raises(ValueError, match="'field' key"):
            DataStatisticsConfig.coerce({"sum": True})

    def test_coerce_unknown_type(self):
        with pytest.raises(TypeError, match="must be a DataStatisticsConfig"):
            DataStatisticsConfig.coerce(42)  # type: ignore[arg-type]

    def test_coerce_many_none(self):
        assert DataStatisticsConfig.coerce_many(None) is None

    def test_coerce_many_mixed(self):
        out = DataStatisticsConfig.coerce_many(
            ["a", {"field": "b", "sum": True}, DataStatisticsConfig(field="c")]
        )
        assert out is not None
        assert [c.field for c in out] == ["a", "b", "c"]
        assert out[1].sum is True

    def test_coerce_many_rejects_single_scalar(self):
        with pytest.raises(TypeError, match="Wrap it in a list"):
            DataStatisticsConfig.coerce_many("only_one")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Wrap it in a list"):
            DataStatisticsConfig.coerce_many({"field": "x"})  # type: ignore[arg-type]

    def test_coerce_many_duplicate_field(self):
        with pytest.raises(ValueError, match="Duplicate statistics entry for field 'x'"):
            DataStatisticsConfig.coerce_many(["x", {"field": "x", "sum": True}])


class TestCompoundFieldKey:
    def test_tuple_field(self):
        cfg = DataStatisticsConfig(field=("year", "month"), distinct_values=True)
        assert cfg.field == ("year", "month")
        assert cfg.is_compound is True
        assert cfg.field_names == ("year", "month")
        assert cfg.distinct_values is True

    def test_list_field_normalizes_to_tuple(self):
        cfg = DataStatisticsConfig(field=["year", "month"])
        assert cfg.field == ("year", "month")
        assert cfg.is_compound is True

    def test_single_element_tuple_collapses_to_str(self):
        cfg = DataStatisticsConfig(field=("only",))
        assert cfg.field == "only"
        assert cfg.is_compound is False
        assert cfg.field_names == ("only",)

    def test_empty_tuple_rejected(self):
        with pytest.raises(ValueError, match="at least one column name"):
            DataStatisticsConfig(field=())

    def test_tuple_with_non_str_rejected(self):
        with pytest.raises(TypeError, match="must contain only str"):
            DataStatisticsConfig(field=("a", 1))  # type: ignore[arg-type]

    def test_tuple_with_empty_string_rejected(self):
        with pytest.raises(ValueError, match="empty string at index 1"):
            DataStatisticsConfig(field=("a", ""))

    def test_tuple_with_duplicate_column_rejected(self):
        with pytest.raises(ValueError, match="duplicate column 'a'"):
            DataStatisticsConfig(field=("a", "b", "a"))

    def test_compound_key_zeroes_scalar_flags(self):
        # min/max/sum/mean/byte_size are nonsense for a tuple key.
        # Defaults have min=True, max=True — normalization silently
        # zeroes them rather than raising.
        cfg = DataStatisticsConfig(
            field=("year", "month"),
            min=True,
            max=True,
            sum=True,
            mean=True,
            byte_size=True,
        )
        assert cfg.min is False
        assert cfg.max is False
        assert cfg.sum is False
        assert cfg.mean is False
        assert cfg.byte_size is False
        # These stay meaningful
        assert cfg.count is True
        assert cfg.null_count is True

    def test_compound_key_keeps_distinct_flags(self):
        cfg = DataStatisticsConfig(
            field=("year", "month"),
            distinct_count=True,
            distinct_values=True,
        )
        assert cfg.distinct_count is True
        assert cfg.distinct_values is True

    def test_invalid_field_type(self):
        with pytest.raises(TypeError, match="must be a str or a tuple/list of str"):
            DataStatisticsConfig(field=123)  # type: ignore[arg-type]

    def test_coerce_tuple(self):
        cfg = DataStatisticsConfig.coerce(("year", "month"))
        assert cfg.field == ("year", "month")

    def test_coerce_list(self):
        cfg = DataStatisticsConfig.coerce(["year", "month"])
        assert cfg.field == ("year", "month")

    def test_coerce_dict_tuple_field(self):
        cfg = DataStatisticsConfig.coerce(
            {"field": ("year", "month"), "distinct_values": True}
        )
        assert cfg.field == ("year", "month")
        assert cfg.distinct_values is True

    def test_coerce_many_mixes_scalar_and_compound(self):
        out = DataStatisticsConfig.coerce_many(
            [
                "id",
                ("year", "month"),
                {"field": "amount", "sum": True},
            ]
        )
        assert out is not None
        assert [c.field for c in out] == ["id", ("year", "month"), "amount"]

    def test_coerce_many_scalar_and_compound_are_distinct_keys(self):
        # "year" (scalar stats) and ("year", "month") (partition tuples) are
        # different targets — both should be accepted side-by-side.
        out = DataStatisticsConfig.coerce_many(["year", ("year", "month")])
        assert out is not None
        assert len(out) == 2

    def test_coerce_many_duplicate_tuple_key(self):
        with pytest.raises(ValueError, match=r"Duplicate statistics entry"):
            DataStatisticsConfig.coerce_many(
                [("year", "month"), ("year", "month")]
            )


class TestMediaOptionsStatistics:
    def test_default_is_none(self):
        assert MediaOptions().statistics is None

    def test_accepts_list_of_configs(self):
        mo = MediaOptions(statistics=[DataStatisticsConfig(field="id")])
        assert mo.statistics == [DataStatisticsConfig(field="id")]

    def test_accepts_field_names_and_dicts(self):
        mo = MediaOptions(statistics=["id", {"field": "ts", "max": False}])
        assert mo.statistics is not None
        assert [c.field for c in mo.statistics] == ["id", "ts"]
        assert mo.statistics[1].max is False

    def test_accepts_partition_tuple_for_distinct_values(self):
        # The partition-pruning use case: capture the distinct
        # (year, month) tuples seen in the batch.
        mo = MediaOptions(
            statistics=[
                {"field": ("year", "month"), "distinct_values": True},
            ]
        )
        assert mo.statistics is not None
        cfg = mo.statistics[0]
        assert cfg.field == ("year", "month")
        assert cfg.distinct_values is True
        # Scalar-only flags zeroed automatically.
        assert cfg.min is False
        assert cfg.max is False

    def test_rejects_duplicate_fields(self):
        with pytest.raises(ValueError, match="Duplicate statistics entry"):
            MediaOptions(statistics=["a", "a"])

    def test_rejects_single_scalar(self):
        with pytest.raises(TypeError, match="Wrap it in a list"):
            MediaOptions(statistics="a")  # type: ignore[arg-type]

    def test_check_parameters_passthrough(self):
        mo = MediaOptions.check_parameters(
            statistics=[DataStatisticsConfig(field="x", count=False)],
        )
        assert mo.statistics is not None
        assert mo.statistics[0].field == "x"
        assert mo.statistics[0].count is False

    def test_check_parameters_keeps_existing_when_sentinel(self):
        base = MediaOptions(statistics=["a"])
        merged = MediaOptions.check_parameters(options=base)
        assert merged.statistics is not None
        assert [c.field for c in merged.statistics] == ["a"]

    def test_exported_from_data_package(self):
        import yggdrasil.data as data_pkg

        assert data_pkg.DataStatisticsConfig is DataStatisticsConfig
