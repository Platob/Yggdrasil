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
        assert cfg.sum is False
        assert cfg.mean is False
        assert cfg.byte_size is False

    def test_is_frozen(self):
        cfg = DataStatisticsConfig(field="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.field = "y"  # type: ignore[misc]

    def test_field_must_be_nonempty_str(self):
        with pytest.raises(TypeError, match="must be a non-empty str"):
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
