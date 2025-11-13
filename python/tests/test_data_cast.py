from __future__ import annotations

import pathlib
import sys
from typing import List

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


# Tests for nested type handling

def test_is_nested_type() -> None:
    """Test DataUtility.is_nested_type method."""
    # Test with simple types
    assert not DataUtility.is_nested_type(pl.Int32)
    assert not DataUtility.is_nested_type(pl.Float64)
    assert not DataUtility.is_nested_type(pl.Utf8)

    # Test with List type
    assert DataUtility.is_nested_type(pl.List)

    # Test with Struct type
    assert DataUtility.is_nested_type(pl.Struct)

    # Test with parametrized list type
    if hasattr(pl, "list_"):  # Check if this feature exists in the installed polars version
        list_int = pl.list_(pl.Int32)
        assert DataUtility.is_nested_type(list_int)

    # Test with parametrized struct type
    if hasattr(pl, "struct"):
        struct_type = pl.struct({"name": pl.Utf8, "age": pl.Int32})
        assert DataUtility.is_nested_type(struct_type)


def test_get_inner_type() -> None:
    """Test DataUtility.get_inner_type method."""
    # Test with non-nested types
    assert DataUtility.get_inner_type(pl.Int32) is None

    # Test with list types if available
    if hasattr(pl, "list_"):
        list_int = pl.list_(pl.Int32)
        inner_type = DataUtility.get_inner_type(list_int)
        assert inner_type == pl.Int32


def test_get_nested_fields() -> None:
    """Test DataUtility.get_nested_fields method."""
    # Simple types have no nested fields
    assert len(DataUtility.get_nested_fields(pl.Int32)) == 0

    # Test with list types if available
    if hasattr(pl, "list_"):
        list_int = pl.list_(pl.Int32)
        fields = DataUtility.get_nested_fields(list_int)
        assert len(fields) == 1
        assert fields[0][0] == "_inner"
        assert fields[0][1] == pl.Int32

    # Test with struct types if available
    if hasattr(pl, "struct"):
        # Create a struct type
        struct_type = pl.struct({"name": pl.Utf8, "age": pl.Int32})

        # Get fields from the struct
        fields = DataUtility.get_nested_fields(struct_type)

        # Should have two fields: name and age
        assert len(fields) == 2

        # Check field names and types
        name_field = next((f for f in fields if f[0] == "name"), None)
        age_field = next((f for f in fields if f[0] == "age"), None)

        assert name_field is not None
        assert age_field is not None
        assert name_field[1] == pl.Utf8
        assert age_field[1] == pl.Int32


def test_can_convert_types_with_nested_types() -> None:
    """Test DataUtility.can_convert_types with nested types."""
    # Test with list types if available
    if hasattr(pl, "list_"):
        # Same list types should be convertible
        list_int32 = pl.list_(pl.Int32)
        assert DataUtility.can_convert_types(list_int32, list_int32)

        # List of Int32 to List of Int64 should be safely convertible
        list_int64 = pl.list_(pl.Int64)
        assert DataUtility.can_convert_types(list_int32, list_int64, safe=True)

        # List of Int64 to List of Int32 should not be safely convertible
        assert not DataUtility.can_convert_types(list_int64, list_int32, safe=True)

        # List to non-list should not be convertible
        assert not DataUtility.can_convert_types(list_int32, pl.Int32)
        assert not DataUtility.can_convert_types(pl.Int32, list_int32)

    # Test with struct types if available
    if hasattr(pl, "struct"):
        # Create struct types
        struct1 = pl.struct({"name": pl.Utf8, "age": pl.Int32})
        struct2 = pl.struct({"name": pl.Utf8, "age": pl.Int64})
        struct3 = pl.struct({"name": pl.Utf8, "salary": pl.Float64})

        # Same struct should be convertible
        assert DataUtility.can_convert_types(struct1, struct1)

        # Struct with wider numeric types should be safely convertible
        assert DataUtility.can_convert_types(struct1, struct2, safe=True)

        # Struct with different field names should not be convertible when check_names=True
        assert not DataUtility.can_convert_types(struct1, struct3, safe=True, check_names=True)

        # Struct to non-struct should not be convertible
        assert not DataUtility.can_convert_types(struct1, pl.Int32)
        assert not DataUtility.can_convert_types(pl.Int32, struct1)

    # Test with map/dict types if available (depends on Polars version)
    if hasattr(pl, "map"):
        # Create map types
        map_str_int = pl.map(pl.Utf8, pl.Int32)
        map_str_int64 = pl.map(pl.Utf8, pl.Int64)
        map_int_str = pl.map(pl.Int32, pl.Utf8)

        # Same map type should be convertible
        assert DataUtility.can_convert_types(map_str_int, map_str_int)

        # Map with wider value type should be safely convertible
        assert DataUtility.can_convert_types(map_str_int, map_str_int64, safe=True)

        # Different key types should not be safely convertible
        assert not DataUtility.can_convert_types(map_str_int, map_int_str, safe=True)


def test_cast_series_with_list_type() -> None:
    """Test casting series with list types."""
    if hasattr(pl, "list_"):
        # Create list types
        source_dtype = pl.list_(pl.Int32)
        target_dtype = pl.list_(pl.Int64)

        # Create a caster
        caster = DataCaster(source_dtype=source_dtype, target_dtype=target_dtype)

        # Create a series with nested data
        series = pl.Series("nested", [[1, 2], [3, 4]], dtype=source_dtype)

        # Cast the series
        cast_series = caster.cast_series(series)

        # Check the result
        assert cast_series.dtype == target_dtype
        assert cast_series.to_list() == [[1, 2], [3, 4]]


def test_cast_series_with_struct_type() -> None:
    """Test casting series with struct types."""
    if hasattr(pl, "struct"):
        # Create struct types
        source_dtype = pl.struct({"name": pl.Utf8, "age": pl.Int32})
        target_dtype = pl.struct({"name": pl.Utf8, "age": pl.Int64})

        # Create a caster
        caster = DataCaster(source_dtype=source_dtype, target_dtype=target_dtype)

        # Create a series with struct data
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        series = pl.Series("people", data, dtype=source_dtype)

        # Cast the series
        cast_series = caster.cast_series(series)

        # Check the result
        assert cast_series.dtype == target_dtype
        # Extract the struct values
        result_data = cast_series.to_list()
        assert len(result_data) == 2
        assert result_data[0]["name"] == "Alice"
        assert result_data[0]["age"] == 30
        assert result_data[1]["name"] == "Bob"
        assert result_data[1]["age"] == 25


def test_cast_series_with_map_type() -> None:
    """Test casting series with map types."""
    if hasattr(pl, "map"):
        # Create map types
        source_dtype = pl.map(pl.Utf8, pl.Int32)
        target_dtype = pl.map(pl.Utf8, pl.Int64)

        # Create a caster
        caster = DataCaster(source_dtype=source_dtype, target_dtype=target_dtype)

        # Create a series with map data
        data = [{"name": 30, "age": 25}, {"Alice": 35, "Bob": 40}]
        series = pl.Series("mappings", data, dtype=source_dtype)

        # Cast the series
        cast_series = caster.cast_series(series)

        # Check the result
        assert cast_series.dtype == target_dtype
        # Extract the map values
        result_data = cast_series.to_list()
        assert len(result_data) == 2
        assert result_data[0]["name"] == 30
        assert result_data[0]["age"] == 25
        assert result_data[1]["Alice"] == 35
        assert result_data[1]["Bob"] == 40