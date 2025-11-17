"""Unit tests for the arrow_utils module."""

import pyarrow as pa
import pytest

# Check if optional dependencies are available
has_polars = False
has_pandas = False
has_numpy = False

try:
    import polars as pl
    has_polars = True
except ImportError:
    pass

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    pass

try:
    import numpy as np
    has_numpy = True
except ImportError:
    pass

from yggdrasil.libutils.arrow_utils import (
    PYTHON_TO_ARROW_TYPE_MAP,
    default_uuid,
    uuid_to_bytes,
    safe_arrow_tabular,
    arrow_default_scalar,
    array_nulls,
    array_length,
    get_child_array,
    dump_arrow_field_metadata,
    refine_arrow_field,
)


def test_default_uuid():
    """Test that default_uuid returns a nil UUID (all zeros)."""
    import uuid

    nil_uuid = default_uuid()
    assert isinstance(nil_uuid, uuid.UUID)
    assert str(nil_uuid) == '00000000-0000-0000-0000-000000000000'
    assert int(nil_uuid) == 0


def test_uuid_to_bytes():
    """Test that uuid_to_bytes correctly converts a UUID to bytes."""
    import uuid

    # Test with nil UUID
    nil_uuid = default_uuid()
    bytes_val = uuid_to_bytes(nil_uuid)
    assert isinstance(bytes_val, bytes)
    assert len(bytes_val) == 16  # UUID is 16 bytes
    assert bytes_val == b'\x00' * 16  # Nil UUID is all zeros

    # Test with a random UUID
    random_uuid = uuid.uuid4()
    bytes_val = uuid_to_bytes(random_uuid)
    assert isinstance(bytes_val, bytes)
    assert len(bytes_val) == 16
    # Verify roundtrip conversion
    assert uuid.UUID(bytes=bytes_val) == random_uuid


def test_python_to_arrow_type_map_mappings():
    """Test that PYTHON_TO_ARROW_TYPE_MAP contains expected mappings."""
    import datetime as dt
    import decimal as dec
    import enum
    import uuid
    import zoneinfo
    from datetime import timezone
    from typing import Dict, List, Set, Tuple

    # Check basic Python types
    assert PYTHON_TO_ARROW_TYPE_MAP[bool] == pa.bool_()
    assert PYTHON_TO_ARROW_TYPE_MAP[int] == pa.int64()
    assert PYTHON_TO_ARROW_TYPE_MAP[str] == pa.utf8()
    assert PYTHON_TO_ARROW_TYPE_MAP[bytes] == pa.binary()

    # Check decimal and date/time types
    assert PYTHON_TO_ARROW_TYPE_MAP[dec.Decimal] == pa.decimal128(38, 18)
    assert PYTHON_TO_ARROW_TYPE_MAP[dt.datetime] == pa.timestamp("us")
    assert PYTHON_TO_ARROW_TYPE_MAP[timezone].equals(pa.timestamp("us", tz="UTC"))
    assert PYTHON_TO_ARROW_TYPE_MAP[zoneinfo.ZoneInfo].equals(pa.timestamp("us", tz="UTC"))
    assert PYTHON_TO_ARROW_TYPE_MAP[dt.date] == pa.date32()
    assert PYTHON_TO_ARROW_TYPE_MAP[dt.time] == pa.time64("us")
    assert PYTHON_TO_ARROW_TYPE_MAP[dt.timedelta] == pa.duration("us")

    # Check additional types
    assert PYTHON_TO_ARROW_TYPE_MAP[uuid.UUID] == pa.uuid()
    assert PYTHON_TO_ARROW_TYPE_MAP[type(None)] == pa.null()
    assert PYTHON_TO_ARROW_TYPE_MAP[enum.Enum] == pa.string()

    # Check container types
    assert pa.types.is_list(PYTHON_TO_ARROW_TYPE_MAP[list])
    assert pa.types.is_list(PYTHON_TO_ARROW_TYPE_MAP[tuple])
    assert pa.types.is_list(PYTHON_TO_ARROW_TYPE_MAP[set])
    assert pa.types.is_list(PYTHON_TO_ARROW_TYPE_MAP[frozenset])
    assert pa.types.is_map(PYTHON_TO_ARROW_TYPE_MAP[dict])

    # Check typing types
    assert pa.types.is_map(PYTHON_TO_ARROW_TYPE_MAP[Dict])
    assert pa.types.is_list(PYTHON_TO_ARROW_TYPE_MAP[List])
    assert pa.types.is_list(PYTHON_TO_ARROW_TYPE_MAP[Set])
    assert pa.types.is_list(PYTHON_TO_ARROW_TYPE_MAP[Tuple])


class TestSafeArrowTabular:
    """Test safe_arrow_tabular function with various inputs."""

    def test_with_arrow_table(self):
        """Test that pa.Table objects are returned as-is."""
        table = pa.table({'a': [1, 2, 3]})
        result = safe_arrow_tabular(table)
        assert result is table

    def test_with_arrow_record_batch(self):
        """Test that pa.RecordBatch objects are returned as-is."""
        batch = pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], ['a'])
        result = safe_arrow_tabular(batch)
        assert result is batch

    @pytest.mark.skipif(not has_polars, reason="Polars not installed")
    def test_with_polars_dataframe(self):
        """Test conversion from polars DataFrame."""
        df = pl.DataFrame({'a': [1, 2, 3]})
        result = safe_arrow_tabular(df)
        assert isinstance(result, pa.Table)
        assert result.column_names == ['a']

    @pytest.mark.skipif(not has_pandas, reason="Pandas not installed")
    def test_with_pandas_dataframe(self):
        """Test conversion from pandas DataFrame."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = safe_arrow_tabular(df)
        assert isinstance(result, pa.Table)
        assert result.column_names == ['a']

    @pytest.mark.skipif(not has_pandas, reason="Pandas not installed")
    def test_with_pandas_series(self):
        """Test conversion from pandas Series."""
        series = pd.Series([1, 2, 3], name='a')
        result = safe_arrow_tabular(series)
        assert isinstance(result, pa.Table)
        assert result.column_names == ['a']

    def test_with_invalid_type(self):
        """Test that TypeError is raised for invalid types."""
        with pytest.raises(TypeError):
            safe_arrow_tabular([1, 2, 3])


class TestArrowDefaultScalar:
    """Test arrow_default_scalar function."""

    def test_with_nullable_types(self):
        """Test that null scalars are returned for nullable types."""
        for arrow_type in [pa.int32(), pa.utf8(), pa.bool_()]:
            scalar = arrow_default_scalar(arrow_type, nullable=True)
            assert scalar.type == arrow_type

    def test_with_primitive_types(self):
        """Test default values for primitive types."""
        # Integer defaults to 0
        scalar = arrow_default_scalar(pa.int32(), nullable=False)
        assert scalar.as_py() == 0

        # String defaults to empty string
        scalar = arrow_default_scalar(pa.string(), nullable=False)
        assert scalar.as_py() == ""

        # Bool defaults to False
        scalar = arrow_default_scalar(pa.bool_(), nullable=False)
        assert scalar.as_py() is False

    def test_with_list_type(self):
        """Test default value for list types."""
        list_type = pa.list_(pa.int32())
        scalar = arrow_default_scalar(list_type, nullable=False)
        assert isinstance(scalar, pa.Scalar)
        assert scalar.type == list_type

    def test_with_struct_type(self):
        """Test default value for struct types."""
        struct_type = pa.struct([
            pa.field('a', pa.int32()),
            pa.field('b', pa.string())
        ])
        scalar = arrow_default_scalar(struct_type, nullable=False)
        assert isinstance(scalar, pa.Scalar)
        assert scalar.type == struct_type

    def test_with_map_type(self):
        """Test default value for map types."""
        map_type = pa.map_(pa.string(), pa.int32())
        scalar = arrow_default_scalar(map_type, nullable=False)
        assert isinstance(scalar, pa.Scalar)
        assert scalar.type == map_type

    def test_with_dictionary_type(self):
        """Test default value for dictionary types."""
        dict_type = pa.dictionary(pa.int8(), pa.string())
        scalar = arrow_default_scalar(dict_type, nullable=False)
        assert isinstance(scalar, pa.Scalar)
        # We return a string scalar for dictionary types as a simplification
        assert scalar.type == pa.string()

    def test_with_timestamp_tz_types(self):
        """Test default values for timezone-aware timestamp types."""
        # Test pre-defined timezone types
        timestamp_s_utc = pa.timestamp('s', tz='UTC')
        scalar = arrow_default_scalar(timestamp_s_utc, nullable=False)
        assert isinstance(scalar, pa.Scalar)
        assert scalar.type == timestamp_s_utc

        # Test custom timezone
        timestamp_us_ny = pa.timestamp('us', tz='America/New_York')
        scalar = arrow_default_scalar(timestamp_us_ny, nullable=False)
        assert isinstance(scalar, pa.Scalar)
        assert scalar.type == timestamp_us_ny

    def test_with_uuid_type(self):
        """Test default value for UUID type."""
        import uuid

        # Test the UUID type gets the nil UUID (all zeros) as default
        uuid_type = pa.uuid()
        scalar = arrow_default_scalar(uuid_type, nullable=False)
        assert isinstance(scalar, pa.Scalar)
        assert scalar.type == uuid_type

        # PyArrow might return bytes or UUID depending on version
        # so we need to handle both cases
        value = scalar.as_py()
        if isinstance(value, bytes):
            # Convert bytes to UUID if needed
            value = uuid.UUID(bytes=value)

        # Verify the UUID value is the nil UUID
        assert value == default_uuid()
        assert str(value) == '00000000-0000-0000-0000-000000000000'

    def test_with_time_types(self):
        """Test default values for time types."""
        # Time32
        time32_s_type = pa.time32("s")
        scalar = arrow_default_scalar(time32_s_type, nullable=False)
        assert isinstance(scalar, pa.Scalar)
        assert scalar.type == time32_s_type

        time32_ms_type = pa.time32("ms")
        scalar = arrow_default_scalar(time32_ms_type, nullable=False)
        assert isinstance(scalar, pa.Scalar)
        assert scalar.type == time32_ms_type

        # Time64
        time64_us_type = pa.time64("us")
        scalar = arrow_default_scalar(time64_us_type, nullable=False)
        assert isinstance(scalar, pa.Scalar)
        assert scalar.type == time64_us_type

        time64_ns_type = pa.time64("ns")
        scalar = arrow_default_scalar(time64_ns_type, nullable=False)
        assert isinstance(scalar, pa.Scalar)
        assert scalar.type == time64_ns_type

    def test_with_duration_types(self):
        """Test default values for duration types."""
        for unit in ["s", "ms", "us", "ns"]:
            duration_type = pa.duration(unit)
            scalar = arrow_default_scalar(duration_type, nullable=False)
            assert isinstance(scalar, pa.Scalar)
            assert scalar.type == duration_type

    def test_with_union_type(self):
        """Test default value for union types."""
        # Create a sparse union type
        union_fields = [
            pa.field("int_field", pa.int32()),
            pa.field("str_field", pa.string())
        ]
        sparse_union_type = pa.sparse_union(union_fields, type_codes=[0, 1])
        scalar = arrow_default_scalar(sparse_union_type, nullable=False)
        assert isinstance(scalar, pa.Scalar)
        # For unions, we return a scalar of the first field type
        assert scalar.type == pa.int32()

        # Create a dense union type
        dense_union_type = pa.dense_union(union_fields, type_codes=[0, 1])
        scalar = arrow_default_scalar(dense_union_type, nullable=False)
        assert isinstance(scalar, pa.Scalar)
        # For unions, we return a scalar of the first field type
        assert scalar.type == pa.int32()

    def test_with_extension_type(self):
        """Test default value for extension types."""
        try:
            # Create a simple extension type - UuidType
            class UuidType(pa.ExtensionType):
                def __init__(self):
                    pa.ExtensionType.__init__(self, pa.string(), 'uuid-type')

                def __arrow_ext_serialize__(self):
                    return b''

                @classmethod
                def __arrow_ext_deserialize__(cls, storage_type, serialized):
                    return UuidType()

                def __arrow_ext_scalar_from_value__(self, value):
                    return str(value)

            # Register the extension type
            uuid_type = UuidType()
            pa.register_extension_type(uuid_type)

            # Test arrow_default_scalar with our extension type
            scalar = arrow_default_scalar(uuid_type, nullable=False)
            assert isinstance(scalar, pa.Scalar)
            assert scalar.type == uuid_type
        except (TypeError, NotImplementedError):
            # Skip if extension types not fully supported in this PyArrow version
            pytest.skip("Extension type testing not fully supported in this PyArrow version")


class TestArrayNulls:
    """Test array_nulls function."""

    def test_with_nullable_field(self):
        """Test creating array of nulls for nullable field."""
        field = pa.field('test', pa.int32(), nullable=True)
        arr = array_nulls(field, size=5)
        assert len(arr) == 5
        assert arr.null_count == 5
        assert arr.type == pa.int32()

    def test_with_non_nullable_field(self):
        """Test creating array of defaults for non-nullable field."""
        field = pa.field('test', pa.int32(), nullable=False)
        arr = array_nulls(field, size=5)
        assert len(arr) == 5
        assert arr.null_count == 0
        assert arr.type == pa.int32()
        assert arr.to_pylist() == [0, 0, 0, 0, 0]

    def test_with_custom_default(self):
        """Test creating array with custom default value."""
        field = pa.field('test', pa.int32(), nullable=False)
        default = pa.scalar(42, pa.int32())
        arr = array_nulls(field, size=3, default=default)
        assert arr.to_pylist() == [42, 42, 42]


def test_array_length():
    """Test array_length function."""
    # Test with pa.Array
    arr = pa.array([1, 2, 3])
    assert array_length(arr) == 3

    # Test with pa.ChunkedArray
    chunked = pa.chunked_array([[1, 2], [3, 4, 5]])
    assert array_length(chunked) == 5


class TestGetChildArray:
    """Test get_child_array function."""

    def test_with_found_field(self):
        """Test extracting field that exists in the struct array."""
        # Create a struct array with two fields
        struct_data = [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}, {'a': 3, 'b': 'z'}]
        struct_arr = pa.array(struct_data)

        # Extract the 'a' field
        a_field = pa.field('a', pa.int64())
        result = get_child_array(struct_arr, a_field)

        assert result.type == pa.int64()
        assert result.to_pylist() == [1, 2, 3]

    def test_with_missing_field_strict(self):
        """Test behavior when field is not found with strict_names=True."""
        struct_data = [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}, {'a': 3, 'b': 'z'}]
        struct_arr = pa.array(struct_data)

        missing_field = pa.field('c', pa.int64())

        with pytest.raises(pa.ArrowInvalid):
            get_child_array(struct_arr, missing_field, strict_names=True)

    def test_with_missing_field_not_strict(self):
        """Test behavior when field is not found with strict_names=False."""
        struct_data = [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}, {'a': 3, 'b': 'z'}]
        struct_arr = pa.array(struct_data)

        missing_field = pa.field('c', pa.int64(), nullable=True)

        result = get_child_array(struct_arr, missing_field, strict_names=False)

        assert result.type == pa.int64()
        assert result.null_count == 3


class TestDumpArrowFieldMetadata:
    """Test dump_arrow_field_metadata function."""

    def test_basic_metadata_extraction(self):
        """Test extracting metadata from a simple field."""
        field = pa.field('test', pa.int32(), metadata={b'key1': b'value1', b'key2': b'value2'})
        metadata = dump_arrow_field_metadata(field)

        # Should include field metadata and type info
        assert 'test' in metadata
        assert metadata['test']['key1'] == 'value1'
        assert metadata['test']['key2'] == 'value2'
        assert metadata['test']['type'] == 'int32'
        assert metadata['test']['base_type'] == 'int32'
        assert metadata['test']['nullable'] == 'true'

    def test_exclude_keys(self):
        """Test excluding specific keys from metadata."""
        field = pa.field('test', pa.int32(), metadata={b'key1': b'value1', b'key2': b'value2'})

        # Exclude 'key1' and 'type'
        metadata = dump_arrow_field_metadata(field, exclude_keys=['key1', 'type'])

        # key1 should be excluded, but key2 should remain
        assert 'test' in metadata
        assert 'key1' not in metadata['test']
        assert metadata['test']['key2'] == 'value2'
        assert 'type' not in metadata['test']
        assert metadata['test']['base_type'] == 'int32'  # Not excluded

    def test_recursive_with_exclude_keys(self):
        """Test recursive metadata extraction with key exclusion."""
        # Create a nested struct field with metadata
        struct_field = pa.field(
            'parent',
            pa.struct([
                pa.field('child1', pa.int32(), metadata={b'c1key1': b'c1val1', b'c1key2': b'c1val2'}),
                pa.field('child2', pa.string(), metadata={b'c2key1': b'c2val1', b'c2key2': b'c2val2'})
            ]),
            metadata={b'pkey1': b'pval1', b'pkey2': b'pval2'}
        )

        # Exclude 'c1key1' and any 'type' keys
        metadata = dump_arrow_field_metadata(struct_field, exclude_keys=['c1key1', 'type'])

        # Check parent metadata
        assert 'parent' in metadata
        assert metadata['parent']['pkey1'] == 'pval1'
        assert metadata['parent']['pkey2'] == 'pval2'
        assert 'type' not in metadata['parent']

        # Check child1 metadata
        assert 'parent.child1' in metadata
        assert 'c1key1' not in metadata['parent.child1']
        assert metadata['parent.child1']['c1key2'] == 'c1val2'
        assert 'type' not in metadata['parent.child1']

        # Check child2 metadata
        assert 'parent.child2' in metadata
        assert metadata['parent.child2']['c2key1'] == 'c2val1'
        assert metadata['parent.child2']['c2key2'] == 'c2val2'
        assert 'type' not in metadata['parent.child2']

    def test_exclude_all_type_info(self):
        """Test excluding all type-related keys."""
        field = pa.field('test', pa.float64(), metadata={b'description': b'A test field'})

        # Exclude all type-related keys
        metadata = dump_arrow_field_metadata(field, exclude_keys=['type', 'base_type', 'nullable'])

        # Should only have the description left
        assert 'test' in metadata
        assert metadata['test']['description'] == 'A test field'
        assert 'type' not in metadata['test']
        assert 'base_type' not in metadata['test']
        assert 'nullable' not in metadata['test']


class TestRefineArrowField:
    """Test refine_arrow_field function."""

    def test_refine_metadata_only(self):
        """Test refining field's metadata without changing type."""
        original = pa.field('test', pa.int32(), metadata={b'key1': b'value1'})

        # Add a new key and update an existing one
        refined = refine_arrow_field(
            original,
            metadata={'key1': 'new_value', 'key2': 'value2'}
        )

        # Type should remain the same
        assert refined.type == original.type
        assert refined.type == pa.int32()

        # Name and nullable should be preserved
        assert refined.name == 'test'
        assert refined.nullable == original.nullable

        # Metadata should be updated
        assert b'key1' in refined.metadata
        assert b'key2' in refined.metadata
        assert refined.metadata[b'key1'] == b'new_value'
        assert refined.metadata[b'key2'] == b'value2'

    def test_refine_type_via_type_metadata(self):
        """Test refining field's type via 'type' metadata."""
        original = pa.field('test', pa.int32(), metadata={b'key1': b'value1'})

        # Add type metadata to change the type
        refined = refine_arrow_field(
            original,
            metadata={'type': 'int64'}
        )

        # Type should be updated based on the 'type' metadata
        assert refined.type != original.type
        assert refined.type == pa.int64()

        # Name and nullable should be preserved
        assert refined.name == 'test'
        assert refined.nullable == original.nullable

        # Metadata should be updated
        assert b'key1' in refined.metadata
        assert b'type' in refined.metadata

    def test_refine_decimal_type_via_metadata(self):
        """Test refining decimal type via precision/scale metadata."""
        original = pa.field('price', pa.decimal128(10, 2))

        # Change precision and scale via metadata
        refined = refine_arrow_field(
            original,
            metadata={'precision': '15', 'scale': '5'}
        )

        # Type should be updated with new precision and scale
        assert refined.type.precision == 15
        assert refined.type.scale == 5
        assert pa.types.is_decimal(refined.type)

    def test_refine_time_type_via_metadata(self):
        """Test refining time type via unit metadata."""
        original = pa.field('time', pa.time32('s'))

        # Change time unit via metadata
        refined = refine_arrow_field(
            original,
            metadata={'unit': 'ms'}
        )

        # Type should be updated with new unit
        assert refined.type.unit == 'ms'
        assert pa.types.is_time32(refined.type)  # Still a time32 type

        # Now change to a microsecond unit, which requires a time64 type
        refined2 = refine_arrow_field(
            original,
            metadata={'unit': 'us'}
        )

        # Type should change from time32 to time64
        assert refined2.type.unit == 'us'
        assert pa.types.is_time64(refined2.type)  # Changed to time64

    def test_refine_timestamp_type_via_metadata(self):
        """Test refining timestamp type via unit and timezone metadata."""
        original = pa.field('ts', pa.timestamp('ms'))

        # Change timestamp unit via metadata
        refined = refine_arrow_field(
            original,
            metadata={'unit': 'us'}
        )

        # Unit should be updated
        assert refined.type.unit == 'us'
        assert refined.type.tz is None  # Still no timezone

        # Change both unit and timezone
        refined2 = refine_arrow_field(
            original,
            metadata={'unit': 'ns', 'tz': 'UTC'}
        )

        # Both unit and timezone should be updated
        assert refined2.type.unit == 'ns'
        assert refined2.type.tz == 'UTC'

    def test_with_none_values(self):
        """Test refining with None values for metadata."""
        original = pa.field('test', pa.int32(), metadata={b'key1': b'value1'})

        # Passing None for metadata
        refined = refine_arrow_field(original)

        # Should essentially be a copy with the same properties
        assert refined.type == original.type
        assert refined.name == original.name
        assert refined.nullable == original.nullable
        assert refined.metadata == original.metadata

    def test_with_invalid_type_metadata(self):
        """Test handling of invalid type metadata."""
        original = pa.field('test', pa.int32())

        # Add invalid type metadata
        refined = refine_arrow_field(
            original,
            metadata={'type': 'nonexistent_type'}
        )

        # Type should remain unchanged when type metadata is invalid
        assert refined.type == original.type


if __name__ == "__main__":
    pytest.main()