from __future__ import annotations

import pathlib
import sys

import polars as pl
import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from yggdrasil.data import (
    DATA_CAST_REGISTRY,
    DataCastRegistry,
    DataCaster,
    DataUtility,
)


def test_ensure_polars_type() -> None:
    """Test DataUtility.ensure_polars_type method."""
    # Direct type
    dtype = pl.Int32
    result = DataUtility.ensure_polars_type(dtype)
    assert result == dtype

    # String lookup is not implemented yet
    with pytest.raises(ValueError):
        DataUtility.ensure_polars_type("Int32")


def test_create_series() -> None:
    """Test DataUtility.create_series method."""
    series = DataUtility.create_series("test", pl.Int32)
    assert series.name == "test"
    assert series.dtype == pl.Int32
    assert len(series) == 0


def test_can_convert_types_same_type() -> None:
    """Test DataUtility.can_convert_types with the same type."""
    assert DataUtility.can_convert_types(pl.Int32, pl.Int32)
    assert DataUtility.can_convert_types(pl.Float64, pl.Float64)
    assert DataUtility.can_convert_types(pl.Utf8, pl.Utf8)


def test_can_convert_types_numeric_widening() -> None:
    """Test DataUtility.can_convert_types with numeric widening."""
    # Int32 to Int64 should be safe
    assert DataUtility.can_convert_types(pl.Int32, pl.Int64, safe=True)
    # Int64 to Int32 is unsafe
    assert not DataUtility.can_convert_types(pl.Int64, pl.Int32, safe=True)
    # Float32 to Float64 is safe
    assert DataUtility.can_convert_types(pl.Float32, pl.Float64, safe=True)


def test_can_convert_types_string_types() -> None:
    """Test DataUtility.can_convert_types with string types."""
    assert DataUtility.can_convert_types(pl.Utf8, pl.String, safe=True)
    assert DataUtility.can_convert_types(pl.String, pl.Utf8, safe=True)


def test_cast_series_integer_widening() -> None:
    """Test casting a series from Int32 to Int64."""
    source_dtype = pl.Int32
    target_dtype = pl.Int64
    caster = DataCaster(source_dtype=source_dtype, target_dtype=target_dtype)

    series = pl.Series("numbers", [1, 2, 3], dtype=source_dtype)
    cast_series = caster.cast_series(series)

    assert cast_series.dtype == target_dtype
    assert cast_series.to_list() == [1, 2, 3]


def test_cast_scalar_integer_to_float() -> None:
    """Test casting a scalar from Int32 to Float64."""
    source_dtype = pl.Int32
    target_dtype = pl.Float64
    caster = DataCaster(source_dtype=source_dtype, target_dtype=target_dtype)

    scalar = 42
    cast_scalar = caster.cast_scalar(scalar)

    assert isinstance(cast_scalar, float)
    assert cast_scalar == 42.0


def test_registry_builds_and_caches_caster() -> None:
    """Test the registry builds and caches casters."""
    registry = DataCastRegistry()
    source_dtype = pl.Int32
    target_dtype = pl.Int64

    first = registry.get_or_build(source_dtype, target_dtype)
    second = registry.get_or_build(source_dtype, target_dtype)

    assert isinstance(first, DataCaster)
    assert first is second


def test_singleton_registry_is_shared_instance() -> None:
    """Test the singleton registry pattern."""
    instance_one = DataCastRegistry.instance()
    instance_two = DataCastRegistry.instance()

    assert instance_one is instance_two
    assert DATA_CAST_REGISTRY is instance_one