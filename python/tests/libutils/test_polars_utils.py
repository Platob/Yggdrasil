"""Unit tests for the polars_utils module."""

import pytest
import pyarrow as pa

# Check if polars is available
has_polars = False
try:
    import polars as pl
    has_polars = True
except ImportError:
    pass

# Skip all tests if polars is not installed
pytestmark = pytest.mark.skipif(not has_polars, reason="Polars not installed")

if has_polars:
    from yggdrasil.libutils.polars_utils import (
        polars_to_arrow_type,
        POLARS_TO_ARROW_TYPE_MAP
    )


@pytest.mark.skipif(not has_polars, reason="Polars not installed")
class TestPolarsToArrowTypeMap:
    """Test the POLARS_TO_ARROW_TYPE_MAP dictionary."""

    def test_primitive_type_mappings(self):
        """Test that primitive Polars types map to the expected Arrow types."""
        # Check Boolean mapping
        assert POLARS_TO_ARROW_TYPE_MAP[pl.Boolean()] == pa.bool_()

        # Check integer mappings
        assert POLARS_TO_ARROW_TYPE_MAP[pl.Int8()] == pa.int8()
        assert POLARS_TO_ARROW_TYPE_MAP[pl.Int64()] == pa.int64()
        assert POLARS_TO_ARROW_TYPE_MAP[pl.UInt32()] == pa.uint32()

        # Check float mappings
        assert POLARS_TO_ARROW_TYPE_MAP[pl.Float32()] == pa.float32()
        assert POLARS_TO_ARROW_TYPE_MAP[pl.Float64()] == pa.float64()

        # Check string/binary mappings
        assert POLARS_TO_ARROW_TYPE_MAP[pl.Utf8()] == pa.string()
        assert POLARS_TO_ARROW_TYPE_MAP[pl.Binary()] == pa.binary()

        # Check date/time mappings
        assert POLARS_TO_ARROW_TYPE_MAP[pl.Date()] == pa.date32()

        # Check null mapping
        assert POLARS_TO_ARROW_TYPE_MAP[pl.Null()] == pa.null()


@pytest.mark.skipif(not has_polars, reason="Polars not installed")
class TestPolarsToArrowType:
    """Test the polars_to_arrow_type function."""

    def test_primitive_types(self):
        """Test conversion of primitive types using dictionary lookup."""
        # Test a few primitive types
        assert polars_to_arrow_type(pl.Boolean()) == pa.bool_()
        assert polars_to_arrow_type(pl.Int32()) == pa.int32()
        assert polars_to_arrow_type(pl.Utf8()) == pa.string()

    def test_datetime_type(self):
        """Test conversion of Datetime types."""
        # Test with different time units and timezones
        assert polars_to_arrow_type(pl.Datetime("ns")) == pa.timestamp("ns")
        assert polars_to_arrow_type(pl.Datetime("ms", "UTC")) == pa.timestamp("ms", "UTC")

    def test_time_type(self):
        """Test conversion of Time type."""
        assert polars_to_arrow_type(pl.Time()) == pa.time64("ns")

    def test_decimal_type(self):
        """Test conversion of Decimal types."""
        # Test decimal128
        assert polars_to_arrow_type(pl.Decimal(10, 2)) == pa.decimal128(10, 2)

        # Test decimal256 (for large precision)
        assert polars_to_arrow_type(pl.Decimal(40, 2)) == pa.decimal256(40, 2)

    def test_list_type(self):
        """Test conversion of List types."""
        # Simple list of integers
        list_type = pl.List(pl.Int32())
        arrow_type = polars_to_arrow_type(list_type)
        assert pa.types.is_list(arrow_type)
        assert arrow_type.value_type == pa.int32()

        # Nested list
        nested_list_type = pl.List(pl.List(pl.Utf8()))
        arrow_nested_type = polars_to_arrow_type(nested_list_type)
        assert pa.types.is_list(arrow_nested_type)
        assert pa.types.is_list(arrow_nested_type.value_type)
        assert arrow_nested_type.value_type.value_type == pa.string()

    def test_categorical_type(self):
        """Test conversion of Categorical type."""
        cat_type = pl.Categorical()
        arrow_type = polars_to_arrow_type(cat_type)
        assert pa.types.is_dictionary(arrow_type)
        assert arrow_type.index_type == pa.int32()
        assert arrow_type.value_type == pa.string()

    def test_struct_type(self):
        """Test conversion of Struct type."""
        # Create a struct with two fields
        fields = [
            pl.Field("a", pl.Int32()),
            pl.Field("b", pl.Utf8())
        ]
        struct_type = pl.Struct(fields)

        arrow_type = polars_to_arrow_type(struct_type)
        assert pa.types.is_struct(arrow_type)
        assert len(arrow_type) == 2
        assert arrow_type.field("a").type == pa.int32()
        assert arrow_type.field("b").type == pa.string()

    def test_unsupported_type(self):
        """Test that TypeError is raised for unsupported types."""
        class UnsupportedType:
            pass

        with pytest.raises(TypeError):
            # This should fail because we're passing an instance, not a Polars type
            polars_to_arrow_type(UnsupportedType())


if __name__ == "__main__":
    pytest.main()