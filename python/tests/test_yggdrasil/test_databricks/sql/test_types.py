"""Unit tests for yggdrasil.databricks.sql.types module."""

import json
import unittest

import pyarrow as pa
import pytest
from databricks.sdk.service.catalog import ColumnInfo as CatalogColumnInfo, ColumnTypeName
from databricks.sdk.service.sql import ColumnInfo as SQLColumnInfo

from yggdrasil.databricks.sql.types import (
    STRING_TYPE_MAP,
    arrow_field_to_column_info,
    arrow_field_to_ddl,
    arrow_field_to_type_json,
    arrow_type_to_column_type_name,
    arrow_type_to_sql_type,
    arrow_type_to_type_text,
    column_info_to_arrow_field,
    parse_sql_type_to_pa,
)


class TestParseTypeToArrow(unittest.TestCase):
    """Test parsing SQL type strings to Arrow DataTypes."""

    def test_primitive_types(self):
        """Test parsing primitive scalar types."""
        # Boolean
        assert parse_sql_type_to_pa("BOOLEAN") == pa.bool_()
        assert parse_sql_type_to_pa("BOOL") == pa.bool_()

        # Integers
        assert parse_sql_type_to_pa("TINYINT") == pa.int8()
        assert parse_sql_type_to_pa("BYTE") == pa.int8()
        assert parse_sql_type_to_pa("SMALLINT") == pa.int16()
        assert parse_sql_type_to_pa("SHORT") == pa.int16()
        assert parse_sql_type_to_pa("INT") == pa.int32()
        assert parse_sql_type_to_pa("INTEGER") == pa.int32()
        assert parse_sql_type_to_pa("BIGINT") == pa.int64()
        assert parse_sql_type_to_pa("LONG") == pa.int64()

        # Floats
        assert parse_sql_type_to_pa("FLOAT") == pa.float32()
        assert parse_sql_type_to_pa("REAL") == pa.float32()
        assert parse_sql_type_to_pa("DOUBLE") == pa.float64()
        assert parse_sql_type_to_pa("DOUBLE PRECISION") == pa.float64()

        # String
        assert parse_sql_type_to_pa("STRING") == pa.string()
        assert parse_sql_type_to_pa("VARCHAR") == pa.string()
        assert parse_sql_type_to_pa("TEXT") == pa.large_string()

        # Binary
        assert parse_sql_type_to_pa("BINARY") == pa.binary()
        assert parse_sql_type_to_pa("VARBINARY") == pa.binary()

        # Date/Time
        assert parse_sql_type_to_pa("DATE") == pa.date32()
        assert parse_sql_type_to_pa("TIMESTAMP") == pa.timestamp("us", "UTC")
        assert parse_sql_type_to_pa("TIMESTAMP_NTZ") == pa.timestamp("us")

    def test_decimal_parsing(self):
        """Test DECIMAL(precision, scale) parsing."""
        assert parse_sql_type_to_pa("DECIMAL(10, 2)") == pa.decimal128(10, 2)
        assert parse_sql_type_to_pa("DECIMAL(18,6)") == pa.decimal128(18, 6)
        assert parse_sql_type_to_pa("DECIMAL(38, 18)") == pa.decimal128(38, 18)

    def test_array_parsing(self):
        """Test ARRAY<element_type> parsing."""
        assert parse_sql_type_to_pa("ARRAY<INT>") == pa.list_(pa.int32())
        assert parse_sql_type_to_pa("ARRAY<STRING>") == pa.list_(pa.string())
        assert parse_sql_type_to_pa("ARRAY<DOUBLE>") == pa.list_(pa.float64())

        # Nested arrays
        nested = parse_sql_type_to_pa("ARRAY<ARRAY<STRING>>")
        assert pa.types.is_list(nested)
        assert pa.types.is_list(nested.value_type)

    def test_map_parsing(self):
        """Test MAP<key_type, value_type> parsing."""
        map_type = parse_sql_type_to_pa("MAP<STRING, INT>")
        assert pa.types.is_map(map_type)
        assert map_type.key_type == pa.string()
        assert map_type.item_type == pa.int32()

        # Complex value types
        map_type = parse_sql_type_to_pa("MAP<STRING, ARRAY<DOUBLE>>")
        assert pa.types.is_map(map_type)
        assert pa.types.is_list(map_type.item_type)

    def test_struct_parsing(self):
        """Test STRUCT<field1:type1, field2:type2> parsing."""
        struct_type = parse_sql_type_to_pa("STRUCT<name:STRING, age:INT>")
        assert pa.types.is_struct(struct_type)
        assert len(struct_type) == 2
        assert struct_type.field(0).name == "name"
        assert struct_type.field(0).type == pa.string()
        assert struct_type.field(1).name == "age"
        assert struct_type.field(1).type == pa.int32()

        # Nested struct
        nested = parse_sql_type_to_pa("STRUCT<id:INT, data:STRUCT<x:DOUBLE, y:DOUBLE>>")
        assert pa.types.is_struct(nested)
        assert pa.types.is_struct(nested.field(1).type)

    def test_void_null_parsing(self):
        """Test VOID and NULL type parsing."""
        assert parse_sql_type_to_pa("VOID") == pa.null()
        assert parse_sql_type_to_pa("NULL") == pa.null()

    def test_unknown_type_defaults_to_string(self):
        """Test that unknown types degrade gracefully to string."""
        assert parse_sql_type_to_pa("UNKNOWN_TYPE") == pa.string()
        assert parse_sql_type_to_pa("CUSTOM_TYPE") == pa.string()

    def test_empty_string_raises_error(self):
        """Test that empty type string raises ValueError."""
        with pytest.raises(ValueError, match="Empty type string"):
            parse_sql_type_to_pa("")


class TestArrowToTypeText(unittest.TestCase):
    """Test converting Arrow types to SQL type strings."""

    def test_boolean(self):
        """Test boolean type conversion."""
        assert arrow_type_to_type_text(pa.bool_()) == "BOOLEAN"

    def test_integer_types(self):
        """Test integer type conversions use standard SQL names."""
        assert arrow_type_to_type_text(pa.int8()) == "TINYINT"
        assert arrow_type_to_type_text(pa.int16()) == "SMALLINT"
        assert arrow_type_to_type_text(pa.int32()) == "INT"
        assert arrow_type_to_type_text(pa.int64()) == "BIGINT"

    def test_unsigned_integer_widening(self):
        """Test unsigned integers are widened to signed types."""
        assert arrow_type_to_type_text(pa.uint8()) == "SMALLINT"
        assert arrow_type_to_type_text(pa.uint16()) == "INT"
        assert arrow_type_to_type_text(pa.uint32()) == "BIGINT"
        assert arrow_type_to_type_text(pa.uint64()) == "DECIMAL(20,0)"

    def test_float_types(self):
        """Test float type conversions."""
        assert arrow_type_to_type_text(pa.float16()) == "FLOAT"
        assert arrow_type_to_type_text(pa.float32()) == "FLOAT"
        assert arrow_type_to_type_text(pa.float64()) == "DOUBLE"

    def test_decimal_types(self):
        """Test decimal type conversions preserve precision and scale."""
        assert arrow_type_to_type_text(pa.decimal128(10, 2)) == "DECIMAL(10,2)"
        assert arrow_type_to_type_text(pa.decimal128(18, 6)) == "DECIMAL(18,6)"
        assert arrow_type_to_type_text(pa.decimal128(38, 18)) == "DECIMAL(38,18)"

    def test_string_types(self):
        """Test string type conversions."""
        assert arrow_type_to_type_text(pa.string()) == "STRING"
        assert arrow_type_to_type_text(pa.large_string()) == "STRING"

    def test_binary_types(self):
        """Test binary type conversions."""
        assert arrow_type_to_type_text(pa.binary()) == "BINARY"
        assert arrow_type_to_type_text(pa.large_binary()) == "BINARY"

    def test_date_type(self):
        """Test date type conversion."""
        assert arrow_type_to_type_text(pa.date32()) == "DATE"
        assert arrow_type_to_type_text(pa.date64()) == "DATE"

    def test_time_type(self):
        """Test time types convert to STRING (no native support)."""
        assert arrow_type_to_type_text(pa.time32("s")) == "STRING"
        assert arrow_type_to_type_text(pa.time64("us")) == "STRING"

    def test_timestamp_timezone_handling(self):
        """Test timestamp tz-aware vs naive conversion."""
        # tz-aware → TIMESTAMP
        assert arrow_type_to_type_text(pa.timestamp("us", "UTC")) == "TIMESTAMP"
        assert arrow_type_to_type_text(pa.timestamp("ns", "America/New_York")) == "TIMESTAMP"

        # naive → TIMESTAMP_NTZ
        assert arrow_type_to_type_text(pa.timestamp("us")) == "TIMESTAMP_NTZ"
        assert arrow_type_to_type_text(pa.timestamp("ns")) == "TIMESTAMP_NTZ"

    def test_duration_types(self):
        """Test duration types convert to BIGINT."""
        assert arrow_type_to_type_text(pa.duration("s")) == "BIGINT"
        assert arrow_type_to_type_text(pa.duration("ms")) == "BIGINT"
        assert arrow_type_to_type_text(pa.duration("us")) == "BIGINT"
        assert arrow_type_to_type_text(pa.duration("ns")) == "BIGINT"

    def test_array_types(self):
        """Test array type conversion."""
        assert arrow_type_to_type_text(pa.list_(pa.int32())) == "ARRAY<INT>"
        assert arrow_type_to_type_text(pa.list_(pa.string())) == "ARRAY<STRING>"
        assert arrow_type_to_type_text(pa.large_list(pa.float64())) == "ARRAY<DOUBLE>"

        # Nested arrays
        nested = pa.list_(pa.list_(pa.string()))
        assert arrow_type_to_type_text(nested) == "ARRAY<ARRAY<STRING>>"

    def test_map_types(self):
        """Test map type conversion."""
        map_type = pa.map_(pa.string(), pa.int32())
        assert arrow_type_to_type_text(map_type) == "MAP<STRING,INT>"

        # Complex value types
        map_type = pa.map_(pa.string(), pa.list_(pa.float64()))
        assert arrow_type_to_type_text(map_type) == "MAP<STRING,ARRAY<DOUBLE>>"

    def test_struct_types(self):
        """Test struct type conversion."""
        struct_type = pa.struct([
            pa.field("name", pa.string()),
            pa.field("age", pa.int32()),
        ])
        result = arrow_type_to_type_text(struct_type)
        assert result == "STRUCT<name:STRING,age:INT>"

    def test_null_type(self):
        """Test null type conversion."""
        assert arrow_type_to_type_text(pa.null()) == "VOID"


class TestArrowToColumnTypeName(unittest.TestCase):
    """Test converting Arrow types to ColumnTypeName enum."""

    def test_primitive_mappings(self):
        """Test primitive type mappings to ColumnTypeName."""
        assert arrow_type_to_column_type_name(pa.bool_()) == ColumnTypeName.BOOLEAN
        assert arrow_type_to_column_type_name(pa.int8()) == ColumnTypeName.BYTE
        assert arrow_type_to_column_type_name(pa.int16()) == ColumnTypeName.SHORT
        assert arrow_type_to_column_type_name(pa.int32()) == ColumnTypeName.INT
        assert arrow_type_to_column_type_name(pa.int64()) == ColumnTypeName.LONG
        assert arrow_type_to_column_type_name(pa.float32()) == ColumnTypeName.FLOAT
        assert arrow_type_to_column_type_name(pa.float64()) == ColumnTypeName.DOUBLE
        assert arrow_type_to_column_type_name(pa.string()) == ColumnTypeName.STRING
        assert arrow_type_to_column_type_name(pa.binary()) == ColumnTypeName.BINARY
        assert arrow_type_to_column_type_name(pa.date32()) == ColumnTypeName.DATE

    def test_unsigned_integer_mappings(self):
        """Test unsigned integers map to widened types."""
        assert arrow_type_to_column_type_name(pa.uint8()) == ColumnTypeName.SHORT
        assert arrow_type_to_column_type_name(pa.uint16()) == ColumnTypeName.INT
        assert arrow_type_to_column_type_name(pa.uint32()) == ColumnTypeName.LONG
        assert arrow_type_to_column_type_name(pa.uint64()) == ColumnTypeName.DECIMAL

    def test_decimal_mapping(self):
        """Test decimal types map to DECIMAL regardless of parameters."""
        assert arrow_type_to_column_type_name(pa.decimal128(10, 2)) == ColumnTypeName.DECIMAL
        assert arrow_type_to_column_type_name(pa.decimal128(38, 18)) == ColumnTypeName.DECIMAL

    def test_timestamp_timezone_mapping(self):
        """Test timestamp tz handling in ColumnTypeName mapping."""
        # tz-aware
        assert arrow_type_to_column_type_name(pa.timestamp("us", "UTC")) == ColumnTypeName.TIMESTAMP

        # Test with explicit timezone string
        with_tz = pa.timestamp("us", tz="America/New_York")
        assert arrow_type_to_column_type_name(with_tz) == ColumnTypeName.TIMESTAMP

    def test_duration_mapping(self):
        """Test duration types map to LONG."""
        assert arrow_type_to_column_type_name(pa.duration("s")) == ColumnTypeName.LONG
        assert arrow_type_to_column_type_name(pa.duration("us")) == ColumnTypeName.LONG

    def test_nested_type_mappings(self):
        """Test nested types map correctly."""
        assert arrow_type_to_column_type_name(pa.list_(pa.int32())) == ColumnTypeName.ARRAY
        assert arrow_type_to_column_type_name(pa.map_(pa.string(), pa.int32())) == ColumnTypeName.MAP
        assert arrow_type_to_column_type_name(pa.struct([pa.field("x", pa.int32())])) == ColumnTypeName.STRUCT


class TestArrowToSqlType(unittest.TestCase):
    """Test arrow_type_to_sql_type function for DDL generation."""

    def test_primitive_types(self):
        """Test primitive type conversions for DDL."""
        assert arrow_type_to_sql_type(pa.bool_()) == "BOOLEAN"
        assert arrow_type_to_sql_type(pa.int8()) == "TINYINT"
        assert arrow_type_to_sql_type(pa.int16()) == "SMALLINT"
        assert arrow_type_to_sql_type(pa.int32()) == "INT"
        assert arrow_type_to_sql_type(pa.int64()) == "BIGINT"
        assert arrow_type_to_sql_type(pa.float32()) == "FLOAT"
        assert arrow_type_to_sql_type(pa.float64()) == "DOUBLE"
        assert arrow_type_to_sql_type(pa.string()) == "STRING"
        assert arrow_type_to_sql_type(pa.binary()) == "BINARY"
        assert arrow_type_to_sql_type(pa.date32()) == "DATE"

    def test_decimal_with_spacing(self):
        """Test decimal DDL has spacing in parameters."""
        assert arrow_type_to_sql_type(pa.decimal128(18, 6)) == "DECIMAL(18, 6)"

    def test_timestamp_handling(self):
        """Test timestamp DDL generation."""
        assert arrow_type_to_sql_type(pa.timestamp("us", "UTC")) == "TIMESTAMP"
        assert arrow_type_to_sql_type(pa.timestamp("us")) == "TIMESTAMP_NTZ"

    def test_duration_to_bigint(self):
        """Test duration types convert to BIGINT for DDL."""
        assert arrow_type_to_sql_type(pa.duration("us")) == "BIGINT"

    def test_void_type(self):
        """Test void/null type."""
        assert arrow_type_to_sql_type(pa.null()) == "VOID"


class TestArrowFieldToColumnInfo(unittest.TestCase):
    """Test converting Arrow fields to Databricks ColumnInfo."""

    def test_simple_field(self):
        """Test conversion of simple field."""
        field = pa.field("name", pa.string(), nullable=True)
        col_info = arrow_field_to_column_info(field, position=0)

        assert col_info.name == "name"
        assert col_info.position == 0
        assert col_info.nullable is True
        assert col_info.type_name == ColumnTypeName.STRING
        assert col_info.type_text == "STRING"

    def test_non_nullable_field(self):
        """Test non-nullable field conversion."""
        field = pa.field("id", pa.int64(), nullable=False)
        col_info = arrow_field_to_column_info(field, position=1)

        assert col_info.nullable is False

    def test_decimal_precision_scale(self):
        """Test decimal type includes precision and scale."""
        field = pa.field("price", pa.decimal128(18, 6))
        col_info = arrow_field_to_column_info(field, position=0)

        assert col_info.type_name == ColumnTypeName.DECIMAL
        assert col_info.type_precision == 18
        assert col_info.type_scale == 6

    def test_uint64_decimal_precision(self):
        """Test uint64 widened to DECIMAL(20,0)."""
        field = pa.field("big_num", pa.uint64())
        col_info = arrow_field_to_column_info(field, position=0)

        assert col_info.type_name == ColumnTypeName.DECIMAL
        assert col_info.type_precision == 20
        assert col_info.type_scale == 0

    def test_field_with_comment(self):
        """Test field with comment metadata."""
        field = pa.field("price", pa.float64(), metadata={b"comment": b"Trade price in USD"})
        col_info = arrow_field_to_column_info(field, position=0)

        assert col_info.comment == "Trade price in USD"

    def test_type_json_format(self):
        """Test type_json is properly formatted."""
        field = pa.field("amount", pa.decimal128(10, 2), nullable=False)
        col_info = arrow_field_to_column_info(field, position=0)

        type_json = json.loads(col_info.type_json)
        assert type_json["name"] == "amount"
        assert type_json["type"] == "decimal(10,2)"
        assert type_json["nullable"] is False


class TestColumnInfoToArrowField(unittest.TestCase):
    """Test converting Databricks ColumnInfo to Arrow fields."""

    def test_catalog_column_info(self):
        """Test conversion from CatalogColumnInfo."""
        col = CatalogColumnInfo(
            name="price",
            type_text="DECIMAL(18,6)",
            type_name=ColumnTypeName.DECIMAL,
            type_json='{"name":"price","type":"decimal(18,6)","nullable":false,"metadata":{"comment":"USD"}}',
            nullable=False,
            position=0,
        )
        field = column_info_to_arrow_field(col)

        assert field.name == "price"
        assert field.type == pa.decimal128(18, 6)
        assert field.nullable is False
        assert field.metadata[b"comment"] == b"USD"

    def test_sql_column_info(self):
        """Test conversion from SQLColumnInfo (no metadata)."""
        col = SQLColumnInfo(
            name="amount",
            type_text="DOUBLE",
        )
        field = column_info_to_arrow_field(col)

        assert field.name == "amount"
        assert field.type == pa.float64()
        assert field.nullable is True  # SQL results assume nullable
        assert field.metadata == {}

    def test_comment_extraction(self):
        """Test comment is extracted from col.comment if not in metadata."""
        col = CatalogColumnInfo(
            name="id",
            type_text="BIGINT",
            type_name=ColumnTypeName.LONG,
            type_json='{"name":"id","type":"bigint","nullable":false,"metadata":{}}',
            nullable=False,
            position=0,
            comment="Primary key",
        )
        field = column_info_to_arrow_field(col)

        assert field.metadata[b"comment"] == b"Primary key"


class TestArrowFieldToDDL(unittest.TestCase):
    """Test DDL generation from Arrow fields."""

    def test_simple_field_ddl(self):
        """Test DDL for simple field."""
        field = pa.field("name", pa.string())
        ddl = arrow_field_to_ddl(field)
        assert ddl == "`name` STRING"

    def test_not_null_constraint(self):
        """Test NOT NULL constraint in DDL."""
        field = pa.field("id", pa.int64(), nullable=False)
        ddl = arrow_field_to_ddl(field)
        assert "NOT NULL" in ddl

    def test_comment_in_ddl(self):
        """Test COMMENT clause in DDL."""
        field = pa.field("price", pa.decimal128(18, 6), metadata={b"comment": b"Trade price"})
        ddl = arrow_field_to_ddl(field)
        assert "COMMENT 'Trade price'" in ddl

    def test_decimal_ddl(self):
        """Test decimal type DDL."""
        field = pa.field("amount", pa.decimal128(10, 2))
        ddl = arrow_field_to_ddl(field)
        assert "DECIMAL(10, 2)" in ddl

    def test_struct_ddl(self):
        """Test nested STRUCT type DDL."""
        field = pa.field("data", pa.struct([
            pa.field("x", pa.float64()),
            pa.field("y", pa.float64()),
        ]))
        ddl = arrow_field_to_ddl(field)
        assert "STRUCT<" in ddl
        assert "x" in ddl and "y" in ddl

    def test_array_ddl(self):
        """Test ARRAY type DDL."""
        field = pa.field("tags", pa.list_(pa.string()))
        ddl = arrow_field_to_ddl(field)
        assert "ARRAY<STRING>" in ddl

    def test_map_ddl(self):
        """Test MAP type DDL."""
        field = pa.field("attrs", pa.map_(pa.string(), pa.int32()))
        ddl = arrow_field_to_ddl(field)
        assert "MAP<STRING, INT>" in ddl

    def test_ddl_without_name(self):
        """Test DDL generation without column name."""
        field = pa.field("col", pa.int32())
        ddl = arrow_field_to_ddl(field, put_name=False)
        assert "col" not in ddl
        assert ddl == "INT"

    def test_sql_injection_prevention(self):
        """Test SQL string escaping for comments."""
        field = pa.field("col", pa.string(), metadata={b"comment": b"O'Reilly's book"})
        ddl = arrow_field_to_ddl(field)
        assert "O''Reilly''s book" in ddl

    def test_backtick_escaping(self):
        """Test backtick escaping in identifiers."""
        field = pa.field("col`with`backticks", pa.int32())
        ddl = arrow_field_to_ddl(field)
        assert "`col``with``backticks`" in ddl


class TestArrowFieldToTypeJson(unittest.TestCase):
    """Test type_json generation."""

    def test_simple_type_json(self):
        """Test type_json for simple field."""
        field = pa.field("name", pa.string())
        type_json = arrow_field_to_type_json(field)

        parsed = json.loads(type_json)
        assert parsed["name"] == "name"
        assert parsed["type"] == "string"
        assert parsed["nullable"] is True
        assert parsed["metadata"] == {}

    def test_type_json_with_metadata(self):
        """Test type_json includes metadata."""
        field = pa.field("price", pa.float64(), metadata={
            b"comment": b"Trade price",
            b"unit": b"USD"
        })
        type_json = arrow_field_to_type_json(field)

        parsed = json.loads(type_json)
        assert parsed["metadata"]["comment"] == "Trade price"
        assert parsed["metadata"]["unit"] == "USD"

    def test_type_json_lowercased(self):
        """Test type field is lowercased."""
        field = pa.field("col", pa.int32())
        type_json = arrow_field_to_type_json(field)

        parsed = json.loads(type_json)
        assert parsed["type"] == "int"  # lowercased


class TestRoundTripConversions(unittest.TestCase):
    """Test round-trip conversions: Arrow → SQL → Arrow."""

    def test_arrow_to_sql_to_arrow(self):
        """Test Arrow → SQL → Arrow preserves type."""
        test_cases = [
            (pa.bool_(), "BOOLEAN"),
            (pa.int8(), "TINYINT"),
            (pa.int16(), "SMALLINT"),
            (pa.int32(), "INT"),
            (pa.int64(), "BIGINT"),
            (pa.float32(), "FLOAT"),
            (pa.float64(), "DOUBLE"),
            (pa.string(), "STRING"),
            (pa.binary(), "BINARY"),
            (pa.date32(), "DATE"),
            (pa.decimal128(18, 6), "DECIMAL(18,6)"),
            (pa.timestamp("us", "UTC"), "TIMESTAMP"),
            (pa.timestamp("us"), "TIMESTAMP_NTZ"),
        ]

        for arrow_type, expected_sql in test_cases:
            with self.subTest(arrow_type=arrow_type):
                sql_type = arrow_type_to_type_text(arrow_type)
                assert sql_type == expected_sql

                recovered = parse_sql_type_to_pa(sql_type)

                # For comparison, check type equality accounting for differences
                if pa.types.is_decimal(arrow_type):
                    assert pa.types.is_decimal(recovered)
                    assert recovered.precision == arrow_type.precision
                    assert recovered.scale == arrow_type.scale
                elif pa.types.is_timestamp(arrow_type):
                    assert pa.types.is_timestamp(recovered)
                    # Note: timezone may differ slightly but semantics preserved
                else:
                    assert recovered == arrow_type

    def test_nested_types_roundtrip(self):
        """Test nested types preserve structure."""
        # Array
        array_type = pa.list_(pa.int32())
        sql = arrow_type_to_type_text(array_type)
        recovered = parse_sql_type_to_pa(sql)
        assert pa.types.is_list(recovered)
        assert recovered.value_type == pa.int32()

        # Map
        map_type = pa.map_(pa.string(), pa.float64())
        sql = arrow_type_to_type_text(map_type)
        recovered = parse_sql_type_to_pa(sql)
        assert pa.types.is_map(recovered)
        assert recovered.key_type == pa.string()
        assert recovered.item_type == pa.float64()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_duration_types_all_units(self):
        """Test all duration units map correctly."""
        for unit in ["s", "ms", "us", "ns"]:
            duration = pa.duration(unit)
            assert arrow_type_to_type_text(duration) == "BIGINT"
            assert arrow_type_to_column_type_name(duration) == ColumnTypeName.LONG
            assert arrow_type_to_sql_type(duration) == "BIGINT"

    def test_large_types(self):
        """Test large string and binary types."""
        assert arrow_type_to_type_text(pa.large_string()) == "STRING"
        assert arrow_type_to_type_text(pa.large_binary()) == "BINARY"

    def test_null_type_handling(self):
        """Test null type conversions."""
        assert arrow_type_to_type_text(pa.null()) == "VOID"
        assert arrow_type_to_sql_type(pa.null()) == "VOID"
        assert parse_sql_type_to_pa("VOID") == pa.null()
        assert parse_sql_type_to_pa("NULL") == pa.null()

    def test_time_types_fallback(self):
        """Test time types fall back to STRING."""
        for time_type in [pa.time32("s"), pa.time32("ms"), pa.time64("us"), pa.time64("ns")]:
            assert arrow_type_to_type_text(time_type) == "STRING"
            assert arrow_type_to_column_type_name(time_type) == ColumnTypeName.STRING

    def test_deeply_nested_types(self):
        """Test deeply nested type structures."""
        nested = pa.list_(pa.list_(pa.list_(pa.int32())))
        sql = arrow_type_to_type_text(nested)
        assert sql == "ARRAY<ARRAY<ARRAY<INT>>>"

        recovered = parse_sql_type_to_pa(sql)
        assert pa.types.is_list(recovered)
        assert pa.types.is_list(recovered.value_type)
        assert pa.types.is_list(recovered.value_type.value_type)


class TestStringTypeMap(unittest.TestCase):
    """Test STRING_TYPE_MAP dictionary."""

    def test_all_basic_types_present(self):
        """Test basic types are in the map."""
        assert "BOOLEAN" in STRING_TYPE_MAP
        assert "INT" in STRING_TYPE_MAP
        assert "BIGINT" in STRING_TYPE_MAP
        assert "DOUBLE" in STRING_TYPE_MAP
        assert "STRING" in STRING_TYPE_MAP
        assert "BINARY" in STRING_TYPE_MAP
        assert "DATE" in STRING_TYPE_MAP
        assert "TIMESTAMP" in STRING_TYPE_MAP

    def test_aliases_present(self):
        """Test common aliases are in the map."""
        assert "LONG" in STRING_TYPE_MAP
        assert "BYTE" in STRING_TYPE_MAP
        assert "SHORT" in STRING_TYPE_MAP
        assert STRING_TYPE_MAP["LONG"] == pa.int64()
        assert STRING_TYPE_MAP["BYTE"] == pa.int8()


if __name__ == "__main__":
    unittest.main()

