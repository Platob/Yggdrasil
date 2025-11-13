from __future__ import annotations

import pathlib
import sys
import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from yggdrasil.data.arrow.arrow_cast import (
    ARROW_CAST_REGISTRY,
    ArrowCaster,
    HAS_POLARS,
)

# Skip all tests if Polars is not available
pytestmark = pytest.mark.skipif(not HAS_POLARS, reason="Polars is not installed")

if HAS_POLARS:
    import polars as pl
    import pyarrow as pa


@pytest.mark.skipif(not HAS_POLARS, reason="Polars is not installed")
def test_cast_polars_series_to_wider_integer():
    """Test casting a Polars Series with int32 to int64."""
    if not HAS_POLARS:
        return

    # Create a Polars Series with int32 data
    series = pl.Series("numbers", [1, 2, 3], dtype=pl.Int32)

    # Set up the caster
    source_field = pa.field("numbers", pa.int32())
    target_field = pa.field("numbers", pa.int64())
    caster = ArrowCaster(source_field=source_field, target_field=target_field)

    # Cast the Polars Series
    cast_array = caster.cast_array(series)

    # Check the result
    assert cast_array.type == pa.int64()
    assert cast_array.to_pylist() == [1, 2, 3]


@pytest.mark.skipif(not HAS_POLARS, reason="Polars is not installed")
def test_cast_registry_with_polars_series():
    """Test using the registry with a Polars Series."""
    if not HAS_POLARS:
        return

    # Create a Polars Series with string data
    series = pl.Series("names", ["Alice", "Bob", "Charlie"])

    # Set up source and target fields
    source_field = pa.field("names", pa.string())
    target_field = pa.field("names", pa.large_string())

    # Get a caster from the registry
    caster = ARROW_CAST_REGISTRY.get_or_build(source_field, target_field)

    # Cast the Polars Series
    cast_array = caster.cast_array(series)

    # Check the result
    assert cast_array.type == pa.large_string()
    assert cast_array.to_pylist() == ["Alice", "Bob", "Charlie"]