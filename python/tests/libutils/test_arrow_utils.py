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
    """Test the dump_arrow_field_metadata function with the new nested JSON structure."""

    def test_simple_field_metadata(self):
        """Test extracting metadata from a simple field."""
        field = pa.field('test', pa.int32(), metadata={b'description': b'A test field'})
        result = dump_arrow_field_metadata(field)

        assert 'test' in result
        assert 'metadata' in result['test']
        assert result['test']['metadata']['description'] == 'A test field'
        assert 'type' in result['test']
        assert result['test']['type'] == 'integer'

    def test_nested_struct_recursive(self):
        """Test extracting metadata from a struct field with nested fields recursively."""
        # Create a nested struct type with metadata at multiple levels
        address_fields = [
            pa.field('street', pa.string(), metadata={b'description': b'Street name'}),
            pa.field('city', pa.string(), metadata={b'description': b'City name'}),
            pa.field('zip', pa.string(), metadata={b'description': b'ZIP code'})
        ]
        address_field = pa.field('address', pa.struct(address_fields),
                                metadata={b'description': b'Address information'})

        person_fields = [
            pa.field('name', pa.string(), metadata={b'description': b'Person name'}),
            pa.field('age', pa.int32(), metadata={b'description': b'Age in years'}),
            address_field
        ]
        person_field = pa.field('person', pa.struct(person_fields),
                               metadata={b'description': b'Person information'})

        # Test with recursive=False (default)
        non_recursive_result = dump_arrow_field_metadata(person_field)
        assert 'person' in non_recursive_result
        assert 'metadata' in non_recursive_result['person']
        assert non_recursive_result['person']['metadata']['description'] == 'Person information'
        assert 'children' not in non_recursive_result['person']

        # Test with recursive=True
        recursive_result = dump_arrow_field_metadata(person_field, recursive=True)
        assert 'person' in recursive_result
        assert recursive_result['person']['metadata']['description'] == 'Person information'

        # Check for children dictionary
        assert 'children' in recursive_result['person']

        # Check first level of nesting
        assert 'name' in recursive_result['person']['children']
        assert recursive_result['person']['children']['name']['metadata']['description'] == 'Person name'
        assert 'age' in recursive_result['person']['children']
        assert recursive_result['person']['children']['age']['metadata']['description'] == 'Age in years'
        assert 'address' in recursive_result['person']['children']
        assert recursive_result['person']['children']['address']['metadata']['description'] == 'Address information'

        # Check second level of nesting
        assert 'children' in recursive_result['person']['children']['address']
        assert 'street' in recursive_result['person']['children']['address']['children']
        assert recursive_result['person']['children']['address']['children']['street']['metadata']['description'] == 'Street name'
        assert 'city' in recursive_result['person']['children']['address']['children']
        assert recursive_result['person']['children']['address']['children']['city']['metadata']['description'] == 'City name'
        assert 'zip' in recursive_result['person']['children']['address']['children']
        assert recursive_result['person']['children']['address']['children']['zip']['metadata']['description'] == 'ZIP code'

    def test_list_field_recursive(self):
        """Test extracting metadata from a list field recursively."""
        # Create a list field with item metadata
        item_field = pa.field('item', pa.string(), metadata={b'description': b'List item'})
        list_type = pa.list_(item_field)
        list_field = pa.field('items', list_type, metadata={b'description': b'List of items'})

        # Test with recursive=True
        result = dump_arrow_field_metadata(list_field, recursive=True)

        assert 'items' in result
        assert result['items']['metadata']['description'] == 'List of items'
        assert 'items' in result['items']
        assert 'metadata' in result['items']['items']
        assert result['items']['items']['metadata']['description'] == 'List item'

    def test_map_field_recursive(self):
        """Test extracting metadata from a map field recursively."""
        # Create a map field with key and item metadata
        key_field = pa.field('key', pa.string(), metadata={b'description': b'Map key'})
        value_field = pa.field('value', pa.int32(), metadata={b'description': b'Map value'})
        map_type = pa.map_(key_type=key_field.type, item_type=value_field.type)
        # In PyArrow, map fields don't retain the metadata from key_field and value_field
        # We need to recreate these in the map type
        map_field = pa.field('mappings', map_type, metadata={b'description': b'Map of key-values'})

        # Test with recursive=True
        result = dump_arrow_field_metadata(map_field, recursive=True)

        assert 'mappings' in result
        assert result['mappings']['metadata']['description'] == 'Map of key-values'
        assert 'keys' in result['mappings']
        assert 'values' in result['mappings']

    def test_nested_list_of_structs_recursive(self):
        """Test extracting metadata from a list of structs field recursively."""
        # Create a struct for the list items
        struct_fields = [
            pa.field('name', pa.string(), metadata={b'description': b'Item name'}),
            pa.field('value', pa.float64(), metadata={b'description': b'Item value'})
        ]
        struct_type = pa.struct(struct_fields)
        struct_field = pa.field('struct_item', struct_type, metadata={b'description': b'Struct item'})

        # Create a list of structs
        list_type = pa.list_(struct_field)
        list_field = pa.field('items', list_type, metadata={b'description': b'List of structs'})

        # Test with recursive=True
        result = dump_arrow_field_metadata(list_field, recursive=True)

        assert 'items' in result
        assert result['items']['metadata']['description'] == 'List of structs'
        assert 'items' in result['items']
        assert result['items']['items']['metadata']['description'] == 'Struct item'
        assert 'children' in result['items']['items']
        assert 'name' in result['items']['items']['children']
        assert result['items']['items']['children']['name']['metadata']['description'] == 'Item name'
        assert 'value' in result['items']['items']['children']
        assert result['items']['items']['children']['value']['metadata']['description'] == 'Item value'

    def test_union_field_recursive(self):
        """Test extracting metadata from a union field recursively."""
        # Create fields for the union
        str_field = pa.field('str_val', pa.string(), metadata={b'description': b'String value'})
        int_field = pa.field('int_val', pa.int32(), metadata={b'description': b'Integer value'})

        # Create a sparse union field
        union_type = pa.sparse_union([str_field, int_field], type_codes=[0, 1])
        union_field = pa.field('variant', union_type, metadata={b'description': b'Variant data'})

        # Test with recursive=True
        result = dump_arrow_field_metadata(union_field, recursive=True)

        assert 'variant' in result
        assert result['variant']['metadata']['description'] == 'Variant data'

        # Check union variants
        assert 'variants' in result['variant']
        assert 'str_val' in result['variant']['variants']
        assert result['variant']['variants']['str_val']['metadata']['description'] == 'String value'
        assert 'int_val' in result['variant']['variants']
        assert result['variant']['variants']['int_val']['metadata']['description'] == 'Integer value'

    def test_type_specific_metadata(self):
        """Test that type-specific metadata is correctly extracted."""
        # Test decimal type
        decimal_field = pa.field('amount', pa.decimal128(precision=10, scale=2))
        result = dump_arrow_field_metadata(decimal_field)

        assert 'amount' in result
        assert 'type' in result['amount']
        assert result['amount']['type'] == 'decimal'
        # Type-specific attributes are top-level keys
        assert result['amount']['precision'] == '10'
        assert result['amount']['scale'] == '2'

        # Test timestamp type
        ts_field = pa.field('timestamp', pa.timestamp('ms', tz='UTC'))
        result = dump_arrow_field_metadata(ts_field)

        assert 'timestamp' in result
        assert 'type' in result['timestamp']
        assert result['timestamp']['type'] == 'timestamp'
        # Type-specific attributes are top-level keys
        assert result['timestamp']['timeunit'] == 'ms'
        assert result['timestamp']['timezone'] == 'UTC'


class TestRefineArrowField:
    """Test the refine_arrow_field function."""

    def test_no_refinement_needed(self):
        """Test that fields without relevant metadata are returned unchanged."""
        # Field with no metadata
        field = pa.field('test', pa.int32())
        result = refine_arrow_field(field)
        assert result.equals(field)
        assert result.metadata is None

        # Field with unrelated metadata
        field = pa.field('test', pa.int32(), metadata={b'description': b'A test field'})
        result = refine_arrow_field(field)
        assert result.equals(field)
        assert result.metadata == {b'description': b'A test field'}

    def test_decimal_refinement(self):
        """Test refinement of float fields to decimal based on metadata."""
        # Using data_type directive with precision and scale
        field = pa.field('amount', pa.float64(), metadata={
            b'data_type': b'decimal',
            b'precision': b'10',
            b'scale': b'2',
            b'description': b'Transaction amount'
        })
        result = refine_arrow_field(field)
        assert result.name == 'amount'
        assert pa.types.is_decimal(result.type)
        assert result.type.precision == 10
        assert result.type.scale == 2
        assert result.metadata == {b'description': b'Transaction amount'}

        # Using just precision and scale without data_type
        field = pa.field('price', pa.float64(), metadata={
            b'precision': b'8',
            b'scale': b'4',
            b'description': b'Item price'
        })
        result = refine_arrow_field(field)
        assert result.name == 'price'
        assert pa.types.is_decimal(result.type)
        assert result.type.precision == 8
        assert result.type.scale == 4
        assert result.metadata == {b'description': b'Item price'}

        # Invalid precision or scale should keep the original type
        field = pa.field('invalid', pa.float64(), metadata={
            b'precision': b'not_a_number',
            b'scale': b'2',
            b'description': b'Invalid precision'
        })
        result = refine_arrow_field(field)
        assert result.name == 'invalid'
        assert pa.types.is_floating(result.type)  # Original type preserved
        assert result.metadata == {
            b'precision': b'not_a_number',
            b'scale': b'2',
            b'description': b'Invalid precision'
        }

    def test_date_refinement(self):
        """Test refinement to date type based on metadata."""
        field = pa.field('date_field', pa.string(), metadata={
            b'data_type': b'date',
            b'format': b'YYYY-MM-DD',
            b'description': b'Date of transaction'
        })
        result = refine_arrow_field(field)
        assert result.name == 'date_field'
        assert pa.types.is_date(result.type)
        assert result.metadata == {b'format': b'YYYY-MM-DD', b'description': b'Date of transaction'}

    def test_time_refinement(self):
        """Test refinement to time type based on metadata."""
        # Time32 with seconds unit
        field = pa.field('time_s', pa.string(), metadata={
            b'data_type': b'time',
            b'unit': b's',
            b'description': b'Time in seconds'
        })
        result = refine_arrow_field(field)
        assert result.name == 'time_s'
        assert pa.types.is_time32(result.type)
        assert result.type.unit == 's'
        assert result.metadata == {b'description': b'Time in seconds'}

        # Time64 with microseconds unit
        field = pa.field('time_us', pa.string(), metadata={
            b'data_type': b'time',
            b'unit': b'us',
            b'description': b'Time with microseconds'
        })
        result = refine_arrow_field(field)
        assert result.name == 'time_us'
        assert pa.types.is_time64(result.type)
        assert result.type.unit == 'us'
        assert result.metadata == {b'description': b'Time with microseconds'}

        # Invalid unit should keep original type
        field = pa.field('time_invalid', pa.string(), metadata={
            b'data_type': b'time',
            b'unit': b'invalid',
            b'description': b'Invalid time unit'
        })
        result = refine_arrow_field(field)
        assert result.name == 'time_invalid'
        assert pa.types.is_string(result.type)  # Original type preserved
        # data_type is marked as used even with invalid unit
        assert b'data_type' not in result.metadata
        # but unit is preserved since it's invalid
        assert b'unit' in result.metadata
        assert result.metadata[b'unit'] == b'invalid'
        # description is preserved
        assert b'description' in result.metadata
        assert result.metadata[b'description'] == b'Invalid time unit'

    def test_timestamp_refinement(self):
        """Test refinement to timestamp type based on metadata."""
        # Timestamp with unit only
        field = pa.field('ts_ms', pa.int64(), metadata={
            b'data_type': b'timestamp',
            b'unit': b'ms',
            b'description': b'Timestamp in milliseconds'
        })
        result = refine_arrow_field(field)
        assert result.name == 'ts_ms'
        assert pa.types.is_timestamp(result.type)
        assert result.type.unit == 'ms'
        assert result.type.tz is None
        assert result.metadata == {b'description': b'Timestamp in milliseconds'}

        # Timestamp with unit and timezone
        field = pa.field('ts_utc', pa.int64(), metadata={
            b'data_type': b'timestamp',
            b'unit': b'us',
            b'timezone': b'UTC',
            b'description': b'Timestamp in UTC'
        })
        result = refine_arrow_field(field)
        assert result.name == 'ts_utc'
        assert pa.types.is_timestamp(result.type)
        assert result.type.unit == 'us'
        assert result.type.tz == 'UTC'
        assert result.metadata == {b'description': b'Timestamp in UTC'}

        # Adding timezone to existing timestamp type
        field = pa.field('ts_existing', pa.timestamp('ns'), metadata={
            b'timezone': b'America/New_York',
            b'description': b'Timestamp with timezone'
        })
        result = refine_arrow_field(field)
        assert result.name == 'ts_existing'
        assert pa.types.is_timestamp(result.type)
        assert result.type.unit == 'ns'  # Preserves original unit
        assert result.type.tz == 'America/New_York'
        assert result.metadata == {b'description': b'Timestamp with timezone'}

    def test_duration_refinement(self):
        """Test refinement to duration type based on metadata."""
        field = pa.field('duration', pa.int64(), metadata={
            b'data_type': b'duration',
            b'unit': b'us',
            b'description': b'Duration in microseconds'
        })
        result = refine_arrow_field(field)
        assert result.name == 'duration'
        assert pa.types.is_duration(result.type)
        assert result.type.unit == 'us'
        assert result.metadata == {b'description': b'Duration in microseconds'}

    def test_metadata_cleanup(self):
        """Test that used metadata is properly cleaned up."""
        # Metadata with multiple keys, some used for refinement
        field = pa.field('mixed', pa.float64(), metadata={
            b'data_type': b'decimal',
            b'precision': b'12',
            b'scale': b'3',
            b'description': b'Mixed metadata',
            b'source': b'external',
            b'nullable': b'false'
        })
        result = refine_arrow_field(field)
        assert result.name == 'mixed'
        assert pa.types.is_decimal(result.type)
        assert result.type.precision == 12
        assert result.type.scale == 3
        # Check that used keys are removed
        assert b'data_type' not in result.metadata
        assert b'precision' not in result.metadata
        assert b'scale' not in result.metadata
        # Check that unused keys are preserved
        assert result.metadata == {
            b'description': b'Mixed metadata',
            b'source': b'external',
            b'nullable': b'false'
        }

    def test_all_metadata_used(self):
        """Test case where all metadata is used for refinement."""
        field = pa.field('all_used', pa.int64(), metadata={
            b'data_type': b'timestamp',
            b'unit': b'ms',
            b'timezone': b'UTC'
        })
        result = refine_arrow_field(field)
        assert result.name == 'all_used'
        assert pa.types.is_timestamp(result.type)
        assert result.type.unit == 'ms'
        assert result.type.tz == 'UTC'
        # All metadata was used, so metadata should be None
        assert result.metadata is None or len(result.metadata) == 0

    def test_complex_refinement(self):
        """Test handling of complex data type refinements."""
        field = pa.field('complex', pa.string(), metadata={
            b'data_type': b'decimal',  # Doesn't match original type well
            b'precision': b'9',
            b'scale': b'3',
            b'description': b'Complex type conversion'
        })
        result = refine_arrow_field(field)
        assert result.name == 'complex'
        assert pa.types.is_decimal(result.type)
        assert result.type.precision == 9
        assert result.type.scale == 3
        assert result.metadata == {b'description': b'Complex type conversion'}


if __name__ == "__main__":
    pytest.main()