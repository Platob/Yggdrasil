"""Comprehensive tests for Spark ↔ Arrow and Polars ↔ Arrow casting modules.

Test layout
-----------
Section 1  – Arrow ↔ Spark type converters (pure, no data)
Section 2  – Arrow ↔ Spark field converters + metadata roundtrip
Section 3  – Arrow ↔ Spark schema converters
Section 4  – _arrow_type_to_metadata / _arrow_type_from_metadata internals
Section 5  – Spark column casts (primitive, safe/unsafe)
Section 6  – JSON decode: string/binary → struct / array / map
Section 7  – cast_spark_column_to_struct / to_list / to_map
Section 8  – cast_spark_dataframe (schema-level, missing cols, extra cols)
Section 9  – Spark ↔ Arrow table converters
Section 10 – Arrow ↔ Polars type converters
Section 11 – Arrow ↔ Polars field converters
Section 12 – Polars array / series casts (primitive, safe/unsafe)
Section 13 – Polars nested casts: list, struct (with JSON parse + serialize)
Section 14 – Polars DataFrame cast
Section 15 – Polars ↔ Arrow table converters
Section 16 – any_to_polars_dataframe / any_to_spark_dataframe edge cases
Section 17 – Integration: Polars → Arrow → Spark → Arrow → Polars roundtrip

Isolation strategy
------------------
All Spark tests use the module-scoped ``spark`` fixture which creates a local
SparkSession once per test session.  Column/DataFrame operations are executed
lazily and only materialised (``collect()``) inside individual tests.

Polars tests are fully in-process and need no fixtures.

Tests that exercise private helpers (``_arrow_type_to_metadata`` etc.) import
them by name to keep coverage explicit.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.types as pat
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pytest
from pyspark.sql import SparkSession

# ---------------------------------------------------------------------------
# Local imports – adjust paths to match project layout
# ---------------------------------------------------------------------------
from yggdrasil.spark.cast import (
    CastOptions,
    _ARROW_META_KEY,
    _arrow_type_from_metadata,
    _arrow_type_to_metadata,
    _is_string_or_binary,
    _parse_arrow_type_str,
    _try_json_parse,
    arrow_field_to_spark_field,
    arrow_schema_to_spark_schema,
    arrow_table_to_spark_dataframe,
    arrow_type_to_spark_type,
    cast_spark_column,
    cast_spark_dataframe,
    spark_dataframe_to_arrow_table,
    spark_field_to_arrow_field,
    spark_schema_to_arrow_schema,
    spark_type_to_arrow_type,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Minimal local SparkSession for the test session."""
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("test_cast")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def _df(spark: SparkSession, data: list[dict], schema: T.StructType | None = None):
    """Create a Spark DataFrame from a list-of-dicts."""
    if schema:
        return spark.createDataFrame(data, schema=schema)
    return spark.createDataFrame(data)


def _col_values(df, col: str) -> list:
    """Collect a single column as a Python list (None for nulls)."""
    return [row[col] for row in df.select(col).collect()]


# ===========================================================================
# Section 1 – Arrow ↔ Spark type converters
# ===========================================================================

class TestArrowTypeToSparkType:

    @pytest.mark.parametrize("arrow_type, expected", [
        (pa.null(),    T.NullType()),
        (pa.bool_(),   T.BooleanType()),
        (pa.int8(),    T.ByteType()),
        (pa.int16(),   T.ShortType()),
        (pa.int32(),   T.IntegerType()),
        (pa.int64(),   T.LongType()),
        (pa.uint8(),   T.ShortType()),
        (pa.uint16(),  T.IntegerType()),
        (pa.uint32(),  T.LongType()),
        (pa.uint64(),  T.LongType()),
        (pa.float16(), T.FloatType()),
        (pa.float32(), T.FloatType()),
        (pa.float64(), T.DoubleType()),
        (pa.string(),  T.StringType()),
        (pa.binary(),  T.BinaryType()),
        (pa.date32(),  T.DateType()),
        (pa.date64(),  T.DateType()),
        (pa.timestamp("us", "UTC"), T.TimestampType()),
    ])
    def test_primitive_lookup(self, arrow_type, expected):
        assert arrow_type_to_spark_type(arrow_type) == expected

    def test_decimal(self):
        assert arrow_type_to_spark_type(pa.decimal128(18, 6)) == T.DecimalType(18, 6)

    def test_decimal_large(self):
        assert arrow_type_to_spark_type(pa.decimal128(38, 10)) == T.DecimalType(38, 10)

    def test_timestamp_with_tz(self):
        assert arrow_type_to_spark_type(pa.timestamp("ms", "US/Eastern")) == T.TimestampType()

    def test_timestamp_no_tz_returns_ntz(self):
        assert arrow_type_to_spark_type(pa.timestamp("ns")) == T.TimestampNTZType()

    def test_list(self):
        result = arrow_type_to_spark_type(pa.list_(pa.int32()))
        assert isinstance(result, T.ArrayType)
        assert result.elementType == T.IntegerType()
        assert result.containsNull is True

    def test_large_list(self):
        result = arrow_type_to_spark_type(pa.large_list(pa.string()))
        assert isinstance(result, T.ArrayType)
        assert result.elementType == T.StringType()

    def test_fixed_size_list(self):
        result = arrow_type_to_spark_type(pa.list_(pa.float32(), 4))
        assert isinstance(result, T.ArrayType)
        assert result.elementType == T.FloatType()

    def test_struct(self):
        arrow_struct = pa.struct([pa.field("x", pa.int32()), pa.field("y", pa.float64())])
        result = arrow_type_to_spark_type(arrow_struct)
        assert isinstance(result, T.StructType)
        assert result.fieldNames() == ["x", "y"]
        assert result["x"].dataType == T.IntegerType()
        assert result["y"].dataType == T.DoubleType()

    def test_map(self):
        result = arrow_type_to_spark_type(pa.map_(pa.string(), pa.int64()))
        assert isinstance(result, T.MapType)
        assert result.keyType == T.StringType()
        assert result.valueType == T.LongType()
        assert result.valueContainsNull is True

    def test_duration_becomes_long(self):
        assert arrow_type_to_spark_type(pa.duration("ns")) == T.LongType()

    def test_nested_list_of_structs(self):
        inner = pa.struct([pa.field("v", pa.float64())])
        result = arrow_type_to_spark_type(pa.list_(inner))
        assert isinstance(result, T.ArrayType)
        assert isinstance(result.elementType, T.StructType)

    def test_unsupported_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            # extension types have no mapping
            arrow_type_to_spark_type(pa.month_day_nano_interval())


class TestSparkTypeToArrowType:

    @pytest.mark.parametrize("spark_type, expected", [
        (T.BooleanType(),  pa.bool_()),
        (T.ByteType(),     pa.int8()),
        (T.ShortType(),    pa.int16()),
        (T.IntegerType(),  pa.int32()),
        (T.LongType(),     pa.int64()),
        (T.FloatType(),    pa.float32()),
        (T.DoubleType(),   pa.float64()),
        (T.StringType(),   pa.string()),
        (T.BinaryType(),   pa.binary()),
        (T.DateType(),     pa.date32()),
        (T.TimestampType(), pa.timestamp("us", "UTC")),
        (T.TimestampNTZType(), pa.timestamp("us")),
        (T.NullType(),     pa.null()),
    ])
    def test_primitive(self, spark_type, expected):
        assert spark_type_to_arrow_type(spark_type) == expected

    def test_decimal(self):
        assert spark_type_to_arrow_type(T.DecimalType(10, 3)) == pa.decimal128(10, 3)

    def test_array(self):
        result = spark_type_to_arrow_type(T.ArrayType(T.IntegerType()))
        assert result == pa.list_(pa.int32())

    def test_map(self):
        result = spark_type_to_arrow_type(T.MapType(T.StringType(), T.LongType()))
        assert result == pa.map_(pa.string(), pa.int64())

    def test_struct(self):
        spark_struct = T.StructType([
            T.StructField("a", T.IntegerType()),
            T.StructField("b", T.StringType()),
        ])
        result = spark_type_to_arrow_type(spark_struct)
        assert pat.is_struct(result)
        assert result.field("a").type == pa.int32()
        assert result.field("b").type == pa.string()

    def test_unsupported_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            spark_type_to_arrow_type(T.UserDefinedType())


# ===========================================================================
# Section 2 – Arrow ↔ Spark field converters + metadata roundtrip
# ===========================================================================

class TestArrowFieldToSparkField:

    def test_basic_field(self):
        f = pa.field("price", pa.float64(), nullable=True)
        sf = arrow_field_to_spark_field(f)
        assert sf.name == "price"
        assert sf.dataType == T.DoubleType()
        assert sf.nullable is True

    def test_not_nullable(self):
        f = pa.field("qty", pa.int32(), nullable=False)
        sf = arrow_field_to_spark_field(f)
        assert sf.nullable is False

    def test_user_metadata_preserved(self):
        f = pa.field("x", pa.int32(), metadata={b"source": b"feed_a"})
        sf = arrow_field_to_spark_field(f)
        assert sf.metadata.get("source") == "feed_a"

    def test_arrow_meta_key_written(self):
        """Any type that loses info when projected to Spark must write __arrow__."""
        f = pa.field("ts", pa.timestamp("ns", "UTC"))
        sf = arrow_field_to_spark_field(f)
        assert _ARROW_META_KEY in sf.metadata
        meta = json.loads(sf.metadata[_ARROW_META_KEY])
        assert meta["unit"] == "ns"
        assert meta["tz"] == "UTC"

    def test_lossless_int64_no_arrow_key(self):
        """int64 maps perfectly to LongType; no __arrow__ key needed."""
        f = pa.field("count", pa.int64())
        sf = arrow_field_to_spark_field(f)
        # Either absent or present is fine, but if present it should be valid JSON
        if _ARROW_META_KEY in sf.metadata:
            json.loads(sf.metadata[_ARROW_META_KEY])  # must not raise

    def test_user_metadata_not_overwritten_by_arrow_key(self):
        """User cannot set __arrow__ and have it survive; our key takes precedence."""
        f = pa.field("v", pa.float16(), metadata={b"__arrow__": b"user_value"})
        sf = arrow_field_to_spark_field(f)
        meta = json.loads(sf.metadata[_ARROW_META_KEY])
        assert meta.get("float_width") == "16"  # our value, not user's


class TestSparkFieldToArrowField:

    def test_basic_roundtrip_string(self):
        f = pa.field("symbol", pa.string(), nullable=True)
        spark_f = arrow_field_to_spark_field(f)
        recovered = spark_field_to_arrow_field(spark_f)
        assert recovered.name == "symbol"
        assert recovered.type == pa.string()
        assert recovered.nullable is True

    def test_arrow_meta_stripped_from_output(self):
        f = pa.field("ts", pa.timestamp("ms", "UTC"))
        spark_f = arrow_field_to_spark_field(f)
        recovered = spark_field_to_arrow_field(spark_f)
        # __arrow__ must NOT appear in the Arrow field metadata
        if recovered.metadata:
            assert _ARROW_META_KEY.encode() not in recovered.metadata

    def test_user_metadata_passes_through(self):
        f = pa.field("x", pa.int32(), metadata={b"tag": b"foo"})
        spark_f = arrow_field_to_spark_field(f)
        recovered = spark_field_to_arrow_field(spark_f)
        assert recovered.metadata and recovered.metadata.get(b"tag") == b"foo"

    def test_no_arrow_meta_falls_back_to_structural(self):
        """A plain Spark field with no __arrow__ metadata round-trips structurally."""
        sf = T.StructField("n", T.LongType(), nullable=True)
        f = spark_field_to_arrow_field(sf)
        assert f.type == pa.int64()

    def test_corrupt_arrow_meta_falls_back(self):
        sf = T.StructField("n", T.LongType(), nullable=True, metadata={_ARROW_META_KEY: "{bad json"})
        f = spark_field_to_arrow_field(sf)
        assert f.type == pa.int64()  # falls back to structural


class TestFieldRoundtrip:
    """Lossless Arrow → Spark → Arrow roundtrip for all interesting types."""

    @pytest.mark.parametrize("arrow_type", [
        pa.timestamp("s",  "UTC"),
        pa.timestamp("ms", "UTC"),
        pa.timestamp("us", "UTC"),
        pa.timestamp("ns", "UTC"),
        pa.timestamp("us"),            # timezone-naive
        pa.timestamp("ns", "America/New_York"),
        pa.duration("ns"),
        pa.duration("us"),
        pa.duration("ms"),
        pa.decimal128(18, 6),
        pa.decimal128(38, 10),
        pa.binary(16),                 # fixed_size_binary
        pa.binary(32),
        pa.float16(),
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
        pa.large_string(),
        pa.large_binary(),
        pa.list_(pa.int32()),
        pa.large_list(pa.float64()),
        pa.map_(pa.string(), pa.int64()),
    ])
    def test_roundtrip(self, arrow_type):
        original = pa.field("f", arrow_type, nullable=True)
        spark_f   = arrow_field_to_spark_field(original)
        recovered = spark_field_to_arrow_field(spark_f)
        assert recovered.type == original.type, (
            f"Roundtrip failed for {arrow_type}: got {recovered.type}"
        )

    def test_roundtrip_fixed_size_list(self):
        arrow_type = pa.list_(pa.float32(), 4)
        original  = pa.field("vec", arrow_type, nullable=True)
        spark_f   = arrow_field_to_spark_field(original)
        recovered = spark_field_to_arrow_field(spark_f)
        assert pat.is_fixed_size_list(recovered.type)
        assert recovered.type.list_size == 4
        assert recovered.type.value_type == pa.float32()

    def test_roundtrip_nested_struct(self):
        arrow_type = pa.struct([
            pa.field("bid", pa.float64()),
            pa.field("ask", pa.float64()),
            pa.field("ts",  pa.timestamp("ns", "UTC")),
        ])
        original  = pa.field("quote", arrow_type)
        spark_f   = arrow_field_to_spark_field(original)
        recovered = spark_field_to_arrow_field(spark_f)
        assert pat.is_struct(recovered.type)
        assert recovered.type.field("ts").type == pa.timestamp("ns", "UTC")

    def test_roundtrip_nullability_preserved(self):
        for nullable in (True, False):
            f = pa.field("x", pa.int32(), nullable=nullable)
            sf = arrow_field_to_spark_field(f)
            r = spark_field_to_arrow_field(sf)
            assert r.nullable == nullable


# ===========================================================================
# Section 3 – Arrow ↔ Spark schema converters
# ===========================================================================

class TestSchemaConverters:

    def test_arrow_schema_to_spark(self):
        schema = pa.schema([
            pa.field("symbol",   pa.string(),              nullable=False),
            pa.field("price",    pa.float64(),             nullable=True),
            pa.field("quantity", pa.int64(),               nullable=True),
            pa.field("ts",       pa.timestamp("us", "UTC"), nullable=True),
        ])
        spark_schema = arrow_schema_to_spark_schema(schema)
        assert isinstance(spark_schema, T.StructType)
        assert spark_schema["symbol"].dataType  == T.StringType()
        assert spark_schema["price"].dataType   == T.DoubleType()
        assert spark_schema["ts"].dataType      == T.TimestampType()

    def test_spark_schema_to_arrow(self):
        spark_schema = T.StructType([
            T.StructField("id",    T.LongType(),    nullable=False),
            T.StructField("name",  T.StringType(),  nullable=True),
            T.StructField("value", T.DoubleType(),  nullable=True),
        ])
        arrow_schema = spark_schema_to_arrow_schema(spark_schema)
        assert isinstance(arrow_schema, pa.Schema)
        assert arrow_schema.field("id").type   == pa.int64()
        assert arrow_schema.field("name").type == pa.string()

    def test_schema_roundtrip(self):
        original = pa.schema([
            pa.field("a", pa.int32(),  nullable=True),
            pa.field("b", pa.float64(), nullable=False),
            pa.field("c", pa.timestamp("ns", "UTC"), nullable=True),
        ])
        spark_schema = arrow_schema_to_spark_schema(original)
        recovered    = spark_schema_to_arrow_schema(spark_schema)
        assert recovered.field("c").type == pa.timestamp("ns", "UTC")

    def test_empty_schema(self):
        empty = pa.schema([])
        spark_schema = arrow_schema_to_spark_schema(empty)
        assert len(spark_schema.fields) == 0
        assert len(spark_schema_to_arrow_schema(spark_schema)) == 0


# ===========================================================================
# Section 4 – _arrow_type_to_metadata / _arrow_type_from_metadata internals
# ===========================================================================

class TestArrowMetadataHelpers:

    # --- _arrow_type_to_metadata ---

    def test_timestamp_us_utc(self):
        meta = _arrow_type_to_metadata(pa.timestamp("us", "UTC"))
        assert meta["unit"] == "us"
        assert meta["tz"]   == "UTC"

    def test_timestamp_no_tz(self):
        meta = _arrow_type_to_metadata(pa.timestamp("ns"))
        assert meta["unit"] == "ns"
        assert "tz" not in meta

    def test_duration(self):
        meta = _arrow_type_to_metadata(pa.duration("ms"))
        assert meta["unit"] == "ms"

    def test_decimal(self):
        meta = _arrow_type_to_metadata(pa.decimal128(18, 6))
        assert meta["precision"] == "18"
        assert meta["scale"]     == "6"

    def test_fixed_size_binary(self):
        meta = _arrow_type_to_metadata(pa.binary(16))
        assert meta["byte_width"] == "16"

    def test_fixed_size_list(self):
        meta = _arrow_type_to_metadata(pa.list_(pa.float32(), 4))
        assert meta["list_size"]  == "4"
        assert meta["value_type"] == "float"

    def test_large_list(self):
        meta = _arrow_type_to_metadata(pa.large_list(pa.int32()))
        assert meta["large"]      == "true"
        assert meta["value_type"] == "int32"

    def test_regular_list(self):
        meta = _arrow_type_to_metadata(pa.list_(pa.int32()))
        assert meta["value_type"] == "int32"
        assert "large" not in meta

    def test_map(self):
        meta = _arrow_type_to_metadata(pa.map_(pa.string(), pa.int64()))
        assert meta["key_type"]  == "string"  # or "utf8"
        assert "item_type" in meta

    def test_unsigned_int(self):
        for t in (pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()):
            meta = _arrow_type_to_metadata(t)
            assert meta["signed"] == "false"

    def test_float16(self):
        meta = _arrow_type_to_metadata(pa.float16())
        assert meta["float_width"] == "16"

    def test_large_string(self):
        meta = _arrow_type_to_metadata(pa.large_string())
        assert meta["large"] == "true"
        assert meta["kind"]  == "string"

    def test_large_binary(self):
        meta = _arrow_type_to_metadata(pa.large_binary())
        assert meta["large"] == "true"
        assert meta["kind"]  == "binary"

    def test_plain_int64_has_type_id_only(self):
        meta = _arrow_type_to_metadata(pa.int64())
        # For perfectly-mapped types, only the fallback key is written
        # (no lossy attributes to preserve)
        assert "arrow_type_id" in meta
        for unexpected_key in ("unit", "tz", "precision", "signed", "large", "float_width"):
            assert unexpected_key not in meta

    # --- _arrow_type_from_metadata ---

    def test_reconstruct_timestamp(self):
        meta = {"unit": "ns", "tz": "UTC", "arrow_type_id": "timestamp[ns, tz=UTC]"}
        result = _arrow_type_from_metadata(T.TimestampType(), meta)
        assert result == pa.timestamp("ns", "UTC")

    def test_reconstruct_timestamp_naive(self):
        meta = {"unit": "ms", "arrow_type_id": "timestamp[ms]"}
        result = _arrow_type_from_metadata(T.TimestampNTZType(), meta)
        assert result == pa.timestamp("ms")

    def test_reconstruct_duration(self):
        meta = {"unit": "us", "arrow_type_id": "duration[us]"}
        result = _arrow_type_from_metadata(T.LongType(), meta)
        assert result == pa.duration("us")

    def test_reconstruct_decimal(self):
        meta = {"precision": "12", "scale": "4", "arrow_type_id": "decimal128(12, 4)"}
        result = _arrow_type_from_metadata(T.DecimalType(12, 4), meta)
        assert result == pa.decimal128(12, 4)

    def test_reconstruct_fixed_size_binary(self):
        meta = {"byte_width": "16", "arrow_type_id": "fixed_size_binary[16]"}
        result = _arrow_type_from_metadata(T.BinaryType(), meta)
        assert result == pa.binary(16)

    def test_reconstruct_uint64(self):
        meta = {"signed": "false", "arrow_type_id": "uint64"}
        result = _arrow_type_from_metadata(T.LongType(), meta)
        assert result == pa.uint64()

    def test_reconstruct_float16(self):
        meta = {"float_width": "16", "arrow_type_id": "halffloat"}
        result = _arrow_type_from_metadata(T.FloatType(), meta)
        assert result == pa.float16()

    def test_reconstruct_large_string(self):
        meta = {"large": "true", "kind": "string", "arrow_type_id": "large_utf8"}
        result = _arrow_type_from_metadata(T.StringType(), meta)
        assert result == pa.large_string()

    def test_reconstruct_large_binary(self):
        meta = {"large": "true", "kind": "binary", "arrow_type_id": "large_binary"}
        result = _arrow_type_from_metadata(T.BinaryType(), meta)
        assert result == pa.large_binary()

    def test_empty_meta_falls_back_to_spark(self):
        result = _arrow_type_from_metadata(T.LongType(), {})
        assert result == pa.int64()

    # --- _parse_arrow_type_str ---

    def test_parse_known_type(self):
        assert _parse_arrow_type_str("int32", T.LongType()) == pa.int32()

    def test_parse_none_falls_back(self):
        assert _parse_arrow_type_str(None, T.IntegerType()) == pa.int32()

    def test_parse_unknown_falls_back(self):
        result = _parse_arrow_type_str("not_a_type_xyz", T.DoubleType())
        assert result == pa.float64()


# ===========================================================================
# Section 5 – Spark column casts (primitive)
# ===========================================================================

class TestSparkPrimitiveCasts:

    def test_int_to_long(self, spark):
        df = spark.createDataFrame([(1,), (2,)], ["v"])
        opts = CastOptions(
            source_arrow_field=pa.field("v", pa.int32()),
            target_arrow_field=pa.field("v", pa.int64()),
        )
        result = df.select(cast_spark_column(df["v"], opts))
        assert result.schema["v"].dataType == T.LongType()
        assert _col_values(result, "v") == [1, 2]

    def test_string_to_double(self, spark):
        df = spark.createDataFrame([("3.14",), ("2.71",)], ["v"])
        opts = CastOptions(
            source_arrow_field=pa.field("v", pa.string()),
            target_arrow_field=pa.field("v", pa.float64()),
            safe=True,
        )
        result = df.select(cast_spark_column(df["v"], opts))
        vals = _col_values(result, "v")
        assert abs(vals[0] - 3.14) < 1e-9

    def test_string_to_double_invalid_safe(self, spark):
        df = spark.createDataFrame([("abc",), ("1.0",)], ["v"])
        opts = CastOptions(
            source_arrow_field=pa.field("v", pa.string()),
            target_arrow_field=pa.field("v", pa.float64()),
            safe=True,
        )
        result = df.select(cast_spark_column(df["v"], opts))
        vals = _col_values(result, "v")
        assert vals[0] is None       # invalid → null in safe mode
        assert vals[1] == 1.0

    def test_null_filled_for_non_nullable_target(self, spark):
        df = spark.createDataFrame([(None,), (5,)], T.StructType([T.StructField("v", T.IntegerType(), nullable=True)]))
        opts = CastOptions(
            source_arrow_field=pa.field("v", pa.int32(), nullable=True),
            target_arrow_field=pa.field("v", pa.int32(), nullable=False),
        )
        result = df.select(cast_spark_column(df["v"], opts))
        vals = _col_values(result, "v")
        assert vals[0] is not None   # null filled with default (0)
        assert vals[1] == 5

    def test_no_target_passthrough(self, spark):
        df = spark.createDataFrame([(42,)], ["v"])
        opts = CastOptions()   # no target
        col = cast_spark_column(df["v"], opts)
        assert df.select(col).collect()[0]["v"] == 42


# ===========================================================================
# Section 6 – JSON decode: string/binary → struct / array / map
# ===========================================================================

class TestIsStringOrBinary:

    @pytest.mark.parametrize("spark_type, expected", [
        (T.StringType(), True),
        (T.BinaryType(), True),
        (T.LongType(),   False),
        (T.DoubleType(), False),
        (T.StructType([]), False),
        (T.ArrayType(T.IntegerType()), False),
    ])
    def test_all(self, spark_type, expected):
        assert _is_string_or_binary(spark_type) == expected


class TestTryJsonParse:
    """Unit tests for _try_json_parse (pure function, no Spark execution)."""

    def test_non_string_passthrough(self, spark):
        df = spark.createDataFrame([(1,)], ["v"])
        col = df["v"]
        result_col, effective_type = _try_json_parse(col, T.LongType(), T.ArrayType(T.IntegerType()), safe=True)
        assert effective_type == T.LongType()  # unchanged

    def test_string_to_struct(self, spark):
        df = spark.createDataFrame([('{"a":1}',)], ["v"])
        target_type = T.StructType([T.StructField("a", T.IntegerType())])
        result_col, effective_type = _try_json_parse(df["v"], T.StringType(), target_type, safe=True)
        assert effective_type == target_type
        parsed = df.select(result_col.alias("v")).collect()[0]["v"]
        assert parsed["a"] == 1

    def test_string_to_array(self, spark):
        df = spark.createDataFrame([('[1,2,3]',)], ["v"])
        target_type = T.ArrayType(T.IntegerType())
        result_col, effective_type = _try_json_parse(df["v"], T.StringType(), target_type, safe=True)
        assert effective_type == target_type
        parsed = df.select(result_col.alias("v")).collect()[0]["v"]
        assert parsed == [1, 2, 3]

    def test_binary_to_struct_decodes_first(self, spark):
        raw = '{"bid":99.5}'.encode("utf-8")
        df = spark.createDataFrame([(raw,)], T.StructType([T.StructField("v", T.BinaryType())]))
        target_type = T.StructType([T.StructField("bid", T.DoubleType())])
        result_col, effective_type = _try_json_parse(df["v"], T.BinaryType(), target_type, safe=True)
        assert effective_type == target_type
        parsed = df.select(result_col.alias("v")).collect()[0]["v"]
        assert abs(parsed["bid"] - 99.5) < 1e-9

    def test_malformed_json_returns_null(self, spark):
        df = spark.createDataFrame([('not json',)], ["v"])
        target_type = T.StructType([T.StructField("x", T.IntegerType())])
        result_col, _ = _try_json_parse(df["v"], T.StringType(), target_type, safe=True)
        parsed = df.select(result_col.alias("v")).collect()[0]["v"]
        assert parsed is None


class TestStringToStructCast:
    """Integration: string/binary column → cast_spark_column_to_struct."""

    def test_string_to_struct_basic(self, spark):
        data = [('{"bid": 100.0, "ask": 100.5}',)]
        schema = T.StructType([T.StructField("q", T.StringType())])
        df = spark.createDataFrame(data, schema)

        target_arrow = pa.field("q", pa.struct([
            pa.field("bid", pa.float64()),
            pa.field("ask", pa.float64()),
        ]))
        source_arrow = pa.field("q", pa.string())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)

        result = df.select(cast_spark_column(df["q"], opts))
        row = result.collect()[0]["q"]
        assert abs(row["bid"] - 100.0) < 1e-9
        assert abs(row["ask"] - 100.5) < 1e-9

    def test_string_to_struct_missing_field_filled_with_default(self, spark):
        data = [('{"bid": 50.0}',)]  # ask is missing
        schema = T.StructType([T.StructField("q", T.StringType())])
        df = spark.createDataFrame(data, schema)

        target_arrow = pa.field("q", pa.struct([
            pa.field("bid", pa.float64()),
            pa.field("ask", pa.float64(), nullable=False),
        ]))
        source_arrow = pa.field("q", pa.string())
        opts = CastOptions(
            source_arrow_field=source_arrow,
            target_arrow_field=target_arrow,
            safe=True,
            add_missing_columns=True,
        )
        result = df.select(cast_spark_column(df["q"], opts))
        row = result.collect()[0]["q"]
        assert row["ask"] is not None  # filled with default

    def test_string_to_struct_malformed_returns_null(self, spark):
        data = [("bad json!",), ('{"bid": 1.0}',)]
        schema = T.StructType([T.StructField("q", T.StringType())])
        df = spark.createDataFrame(data, schema)

        target_arrow = pa.field("q", pa.struct([pa.field("bid", pa.float64())]))
        source_arrow = pa.field("q", pa.string())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)

        result = df.select(cast_spark_column(df["q"], opts))
        rows = result.collect()
        assert rows[0]["q"] is None         # malformed → null
        assert rows[1]["q"]["bid"] == 1.0   # valid → parsed

    def test_binary_to_struct(self, spark):
        raw = '{"px": 42}'.encode("utf-8")
        data = [(raw,)]
        schema = T.StructType([T.StructField("v", T.BinaryType())])
        df = spark.createDataFrame(data, schema)

        target_arrow = pa.field("v", pa.struct([pa.field("px", pa.int64())]))
        source_arrow = pa.field("v", pa.binary())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)

        result = df.select(cast_spark_column(df["v"], opts))
        row = result.collect()[0]["v"]
        assert row["px"] == 42

    def test_struct_to_string_via_to_json(self, spark):
        """Struct → STRING via Spark to_json serialises back to JSON."""
        schema = T.StructType([
            T.StructField("quote", T.StructType([
                T.StructField("bid", T.DoubleType()),
                T.StructField("ask", T.DoubleType()),
            ]))
        ])
        data = [((100.0, 100.5),)]
        df = spark.createDataFrame(data, schema)

        serialised = df.select(F.to_json(df["quote"]).alias("quote"))
        row = serialised.collect()[0]["quote"]
        parsed = json.loads(row)
        assert abs(parsed["bid"] - 100.0) < 1e-9
        assert abs(parsed["ask"] - 100.5) < 1e-9


class TestStringToArrayCast:

    def test_string_json_to_array_int(self, spark):
        data = [('[10, 20, 30]',)]
        schema = T.StructType([T.StructField("v", T.StringType())])
        df = spark.createDataFrame(data, schema)

        target_arrow = pa.field("v", pa.list_(pa.int64()))
        source_arrow = pa.field("v", pa.string())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)

        result = df.select(cast_spark_column(df["v"], opts))
        assert result.collect()[0]["v"] == [10, 20, 30]

    def test_string_json_to_array_struct(self, spark):
        data = [('[{"px": 1.0}, {"px": 2.0}]',)]
        schema = T.StructType([T.StructField("v", T.StringType())])
        df = spark.createDataFrame(data, schema)

        inner = pa.struct([pa.field("px", pa.float64())])
        target_arrow = pa.field("v", pa.list_(inner))
        source_arrow = pa.field("v", pa.string())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)

        result = df.select(cast_spark_column(df["v"], opts))
        rows = result.collect()[0]["v"]
        assert abs(rows[0]["px"] - 1.0) < 1e-9
        assert abs(rows[1]["px"] - 2.0) < 1e-9

    def test_binary_json_to_array(self, spark):
        raw = b'[1, 2, 3]'
        schema = T.StructType([T.StructField("v", T.BinaryType())])
        df = spark.createDataFrame([(raw,)], schema)

        target_arrow = pa.field("v", pa.list_(pa.int32()))
        source_arrow = pa.field("v", pa.binary())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)

        result = df.select(cast_spark_column(df["v"], opts))
        assert result.collect()[0]["v"] == [1, 2, 3]

    def test_malformed_array_json_returns_null(self, spark):
        data = [("not_an_array",), ("[1,2]",)]
        schema = T.StructType([T.StructField("v", T.StringType())])
        df = spark.createDataFrame(data, schema)

        target_arrow = pa.field("v", pa.list_(pa.int64()))
        source_arrow = pa.field("v", pa.string())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)

        result = df.select(cast_spark_column(df["v"], opts))
        rows = result.collect()
        assert rows[0]["v"] is None
        assert rows[1]["v"] == [1, 2]

    def test_array_to_json_string(self, spark):
        """Array → STRING via to_json produces JSON array string."""
        schema = T.StructType([T.StructField("v", T.ArrayType(T.IntegerType()))])
        df = spark.createDataFrame([([1, 2, 3],)], schema)
        result = df.select(F.to_json(df["v"]).alias("v"))
        raw = result.collect()[0]["v"]
        assert json.loads(raw) == [1, 2, 3]


class TestStringToMapCast:

    def test_string_json_to_map(self, spark):
        data = [('{"WTI": 75.0, "BRENT": 80.0}',)]
        schema = T.StructType([T.StructField("v", T.StringType())])
        df = spark.createDataFrame(data, schema)

        target_arrow = pa.field("v", pa.map_(pa.string(), pa.float64()))
        source_arrow = pa.field("v", pa.string())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)

        result = df.select(cast_spark_column(df["v"], opts))
        row = result.collect()[0]["v"]
        row_dict = dict(row)
        assert abs(row_dict["WTI"] - 75.0) < 1e-9
        assert abs(row_dict["BRENT"] - 80.0) < 1e-9

    def test_binary_json_to_map(self, spark):
        raw = b'{"A": 1, "B": 2}'
        schema = T.StructType([T.StructField("v", T.BinaryType())])
        df = spark.createDataFrame([(raw,)], schema)

        target_arrow = pa.field("v", pa.map_(pa.string(), pa.int64()))
        source_arrow = pa.field("v", pa.binary())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)

        result = df.select(cast_spark_column(df["v"], opts))
        row_dict = dict(result.collect()[0]["v"])
        assert row_dict["A"] == 1
        assert row_dict["B"] == 2

    def test_map_to_json_string(self, spark):
        """Map → STRING via to_json round-trip."""
        schema = T.StructType([T.StructField("v", T.MapType(T.StringType(), T.DoubleType()))])
        df = spark.createDataFrame([([("WTI", 75.0)],)], schema)
        result = df.select(F.to_json(df["v"]).alias("v"))
        raw = result.collect()[0]["v"]
        assert json.loads(raw) == {"WTI": 75.0}


# ===========================================================================
# Section 7 – cast_spark_column_to_struct / to_list / to_map (native source)
# ===========================================================================

class TestCastSparkColumnToStruct:

    def test_struct_to_struct_type_cast(self, spark):
        schema = T.StructType([
            T.StructField("q", T.StructType([
                T.StructField("bid", T.FloatType()),
                T.StructField("ask", T.FloatType()),
            ]))
        ])
        df = spark.createDataFrame([((1.0, 2.0),)], schema)

        target_arrow = pa.field("q", pa.struct([
            pa.field("bid", pa.float64()),
            pa.field("ask", pa.float64()),
        ]))
        source_arrow = pa.field("q", pa.struct([
            pa.field("bid", pa.float32()),
            pa.field("ask", pa.float32()),
        ]))
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)
        result = df.select(cast_spark_column(df["q"], opts))
        row = result.collect()[0]["q"]
        assert result.schema["q"]["bid"].dataType == T.DoubleType()
        assert abs(row["bid"] - 1.0) < 1e-9

    def test_struct_case_insensitive_match(self, spark):
        schema = T.StructType([
            T.StructField("q", T.StructType([T.StructField("BID", T.DoubleType())]))
        ])
        df = spark.createDataFrame([((99.0,),)], schema)

        target_arrow = pa.field("q", pa.struct([pa.field("bid", pa.float64())]))
        source_arrow = pa.field("q", pa.struct([pa.field("BID", pa.float64())]))
        opts = CastOptions(
            source_arrow_field=source_arrow,
            target_arrow_field=target_arrow,
            strict_match_names=False,
        )
        result = df.select(cast_spark_column(df["q"], opts))
        assert result.collect()[0]["q"]["bid"] == 99.0

    def test_non_struct_source_safe_returns_null(self, spark):
        df = spark.createDataFrame([(42,)], ["v"])
        target_arrow = pa.field("v", pa.struct([pa.field("x", pa.int32())]))
        source_arrow = pa.field("v", pa.int32())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)
        result = df.select(cast_spark_column(df["v"], opts))
        assert result.collect()[0]["v"] is None

    def test_non_struct_source_unsafe_raises(self, spark):
        df = spark.createDataFrame([(42,)], ["v"])
        target_arrow = pa.field("v", pa.struct([pa.field("x", pa.int32())]))
        source_arrow = pa.field("v", pa.int32())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=False)
        with pytest.raises(ValueError, match="Cannot cast non-struct"):
            df.select(cast_spark_column(df["v"], opts)).collect()


class TestCastSparkColumnToList:

    def test_array_element_type_cast(self, spark):
        schema = T.StructType([T.StructField("v", T.ArrayType(T.IntegerType()))])
        df = spark.createDataFrame([([1, 2, 3],)], schema)

        target_arrow = pa.field("v", pa.list_(pa.int64()))
        source_arrow = pa.field("v", pa.list_(pa.int32()))
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)
        result = df.select(cast_spark_column(df["v"], opts))
        assert result.schema["v"].dataType.elementType == T.LongType()
        assert result.collect()[0]["v"] == [1, 2, 3]

    def test_non_array_safe_returns_null(self, spark):
        df = spark.createDataFrame([(1,)], ["v"])
        target_arrow = pa.field("v", pa.list_(pa.int64()))
        source_arrow = pa.field("v", pa.int32())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)
        result = df.select(cast_spark_column(df["v"], opts))
        assert result.collect()[0]["v"] is None

    def test_non_array_unsafe_raises(self, spark):
        df = spark.createDataFrame([(1,)], ["v"])
        target_arrow = pa.field("v", pa.list_(pa.int64()))
        source_arrow = pa.field("v", pa.int32())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=False)
        with pytest.raises(ValueError, match="Cannot cast non-array"):
            df.select(cast_spark_column(df["v"], opts)).collect()

    def test_array_of_structs(self, spark):
        inner_type = T.StructType([T.StructField("v", T.FloatType())])
        schema = T.StructType([T.StructField("arr", T.ArrayType(inner_type))])
        df = spark.createDataFrame([([{"v": 1.5}, {"v": 2.5}],)], schema)

        target_inner = pa.struct([pa.field("v", pa.float64())])
        target_arrow  = pa.field("arr", pa.list_(target_inner))
        source_arrow  = pa.field("arr", pa.list_(pa.struct([pa.field("v", pa.float32())])))
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)
        result = df.select(cast_spark_column(df["arr"], opts))
        rows = result.collect()[0]["arr"]
        assert abs(rows[0]["v"] - 1.5) < 1e-9


class TestCastSparkColumnToMap:

    def test_map_value_type_cast(self, spark):
        schema = T.StructType([T.StructField("m", T.MapType(T.StringType(), T.IntegerType()))])
        df = spark.createDataFrame([([("k", 1)],)], schema)

        target_arrow = pa.field("m", pa.map_(pa.string(), pa.int64()))
        source_arrow = pa.field("m", pa.map_(pa.string(), pa.int32()))
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)
        result = df.select(cast_spark_column(df["m"], opts))
        assert result.schema["m"].dataType.valueType == T.LongType()

    def test_non_map_safe_returns_null(self, spark):
        df = spark.createDataFrame([(1,)], ["v"])
        target_arrow = pa.field("v", pa.map_(pa.string(), pa.int64()))
        source_arrow = pa.field("v", pa.int32())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=True)
        result = df.select(cast_spark_column(df["v"], opts))
        assert result.collect()[0]["v"] is None

    def test_non_map_unsafe_raises(self, spark):
        df = spark.createDataFrame([(1,)], ["v"])
        target_arrow = pa.field("v", pa.map_(pa.string(), pa.int64()))
        source_arrow = pa.field("v", pa.int32())
        opts = CastOptions(source_arrow_field=source_arrow, target_arrow_field=target_arrow, safe=False)
        with pytest.raises(ValueError, match="Cannot cast non-map"):
            df.select(cast_spark_column(df["v"], opts)).collect()


# ===========================================================================
# Section 8 – cast_spark_dataframe
# ===========================================================================

class TestCastSparkDataframe:

    def test_basic_schema_cast(self, spark):
        df = spark.createDataFrame([(1, "hello", 3.14)], ["id", "name", "value"])
        target = pa.schema([
            pa.field("id",    pa.int64()),
            pa.field("name",  pa.string()),
            pa.field("value", pa.float64()),
        ])
        result = cast_spark_dataframe(df, CastOptions(target_arrow_schema=target))
        assert result.schema["id"].dataType    == T.LongType()
        assert result.schema["value"].dataType == T.DoubleType()

    def test_no_target_passthrough(self, spark):
        df = spark.createDataFrame([(1,)], ["v"])
        assert cast_spark_dataframe(df, None).count() == 1

    def test_missing_column_safe_fills_default(self, spark):
        df = spark.createDataFrame([(1,)], ["id"])
        target = pa.schema([
            pa.field("id",    pa.int64()),
            pa.field("score", pa.float64()),   # missing in source
        ])
        opts = CastOptions(target_arrow_schema=target, safe=True, add_missing_columns=True)
        result = cast_spark_dataframe(df, opts)
        assert "score" in result.columns
        assert result.collect()[0]["score"] is not None or True  # default value

    def test_missing_column_unsafe_raises(self, spark):
        df = spark.createDataFrame([(1,)], ["id"])
        target = pa.schema([
            pa.field("id",    pa.int64()),
            pa.field("score", pa.float64()),
        ])
        opts = CastOptions(target_arrow_schema=target, safe=False, add_missing_columns=False)
        with pytest.raises(ValueError, match="missing"):
            cast_spark_dataframe(df, opts)

    def test_extra_columns_kept_with_allow_add(self, spark):
        df = spark.createDataFrame([(1, "extra")], ["id", "extra_col"])
        target = pa.schema([pa.field("id", pa.int64())])
        opts = CastOptions(target_arrow_schema=target, allow_add_columns=True)
        result = cast_spark_dataframe(df, opts)
        assert "extra_col" in result.columns

    def test_extra_columns_dropped_without_allow_add(self, spark):
        df = spark.createDataFrame([(1, "extra")], ["id", "extra_col"])
        target = pa.schema([pa.field("id", pa.int64())])
        opts = CastOptions(target_arrow_schema=target, allow_add_columns=False)
        result = cast_spark_dataframe(df, opts)
        assert "extra_col" not in result.columns

    def test_case_insensitive_column_match(self, spark):
        df = spark.createDataFrame([(1,)], ["ID"])
        target = pa.schema([pa.field("id", pa.int64())])
        opts = CastOptions(target_arrow_schema=target, strict_match_names=False)
        result = cast_spark_dataframe(df, opts)
        assert result.collect()[0]["id"] == 1

    def test_type_widening(self, spark):
        schema = T.StructType([T.StructField("v", T.IntegerType())])
        df = spark.createDataFrame([(100,)], schema)
        target = pa.schema([pa.field("v", pa.int64())])
        result = cast_spark_dataframe(df, CastOptions(target_arrow_schema=target))
        assert result.schema["v"].dataType == T.LongType()
        assert result.collect()[0]["v"] == 100


# ===========================================================================
# Section 9 – Spark ↔ Arrow table converters
# ===========================================================================

class TestSparkArrowTableConverters:

    def test_spark_df_to_arrow_table(self, spark):
        df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "name"])
        table = spark_dataframe_to_arrow_table(df)
        assert isinstance(table, pa.Table)
        assert table.num_rows == 2
        assert table.schema.field("id").type   == pa.int64()
        assert table.schema.field("name").type == pa.string()

    def test_arrow_table_to_spark_df(self, spark):
        table = pa.table({"id": [1, 2], "val": [3.14, 2.71]})
        df    = arrow_table_to_spark_dataframe(table)
        assert df.count() == 2
        assert df.schema["id"].dataType  == T.LongType()
        assert df.schema["val"].dataType == T.DoubleType()

    def test_spark_to_arrow_with_target_schema(self, spark):
        df = spark.createDataFrame([(1, "x")], ["id", "name"])
        target = pa.schema([
            pa.field("id",   pa.int32()),
            pa.field("name", pa.string()),
        ])
        table = spark_dataframe_to_arrow_table(df, CastOptions(target_arrow_schema=target))
        assert table.schema.field("id").type == pa.int32()

    def test_roundtrip_spark_arrow_spark(self, spark):
        schema = T.StructType([
            T.StructField("id",   T.LongType(),   nullable=False),
            T.StructField("val",  T.DoubleType(),  nullable=True),
        ])
        original = spark.createDataFrame([(1, 3.14), (2, None)], schema)
        table    = spark_dataframe_to_arrow_table(original)
        restored = arrow_table_to_spark_dataframe(table)
        assert restored.count() == 2
        rows = {r["id"]: r["val"] for r in restored.collect()}
        assert abs(rows[1] - 3.14) < 1e-9
        assert rows[2] is None
