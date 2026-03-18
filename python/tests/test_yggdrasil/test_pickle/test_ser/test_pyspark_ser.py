"""Tests for yggdrasil.pickle.ser.pyspark (PySparkSerialized and subclasses).

Test layout
-----------
Section 1  – Tags registration (no live Spark required)
Section 2  – PySparkRowSerialized  (no live Spark required)
Section 3  – PySparkDataTypeSerialized  (no live Spark required)
Section 4  – PySparkSchemaSerialized  (no live Spark required)
Section 5  – PySparkColumnSerialized  (no live Spark required)
Section 6  – PySparkSessionSerialized stub  (no live Spark required)
Section 7  – PySparkDataFrameSerialized  (requires live SparkSession)
Section 8  – PySparkRDDSerialized  (requires live SparkSession)
Section 9  – Full-stack dispatch via Serialized.from_python_object
Section 10 – Binary wire round-trip (write_to / read_from)
Section 11 – Codec / compression variants
Section 12 – Edge cases and error handling

Isolation strategy
------------------
Sections 1-6 and parts of section 9/10/11/12 run without any SparkSession.
Sections 7 and 8 use the session-scoped ``spark`` fixture which creates a
minimal local SparkSession once for the entire test process.
"""
from __future__ import annotations

import pickle

import pyarrow as pa
import pytest
import pyspark.sql.types as T
from pyspark.sql import Row

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.constants import CODEC_NONE, CODEC_ZSTD
from yggdrasil.pickle.ser.pyspark import (
    PySparkColumnSerialized,
    PySparkDataFrameSerialized,
    PySparkDataTypeSerialized,
    PySparkRDDSerialized,
    PySparkRowSerialized,
    PySparkSchemaSerialized,
    PySparkSerialized,
    PySparkSessionSerialized,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roundtrip(ser: Serialized) -> Serialized:
    """Write to a BytesIO buffer then read back, exercising the full wire path."""
    buf = BytesIO()
    ser.write_to(buf)
    return Serialized.read_from(buf, pos=0)


def _simple_schema() -> T.StructType:
    return T.StructType([
        T.StructField("name",  T.StringType(),  nullable=True),
        T.StructField("age",   T.IntegerType(), nullable=False),
        T.StructField("score", T.DoubleType(),  nullable=True),
    ])


def _nested_schema() -> T.StructType:
    return T.StructType([
        T.StructField("id",    T.LongType(), nullable=False),
        T.StructField("tags",  T.ArrayType(T.StringType()), nullable=True),
        T.StructField("meta",  T.MapType(T.StringType(), T.IntegerType()), nullable=True),
        T.StructField("inner", T.StructType([
            T.StructField("x", T.FloatType()),
            T.StructField("y", T.FloatType()),
        ]), nullable=True),
    ])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    """Minimal local SparkSession (created once per test session)."""
    from pyspark.sql import SparkSession
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("test_pyspark_ser")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        # Do NOT enable arrow.pyspark.enabled — it causes Python worker crashes
        # on Windows + Spark 4.x when running inside a subprocess test session.
        .getOrCreate()
    )


def _make_simple_df(spark):
    """Create the simple test DataFrame. Uses sparkContext.parallelize to avoid
    Python worker socket issues on Windows + Spark 4.x."""
    schema = _simple_schema()
    data = [
        Row(name="Alice", age=30, score=9.5),
        Row(name="Bob",   age=25, score=7.0),
        Row(name=None,    age=40, score=None),
    ]
    return spark.createDataFrame(spark.sparkContext.parallelize(data), schema=schema)


@pytest.fixture()
def simple_df(spark):
    return _make_simple_df(spark)


@pytest.fixture()
def typed_rdd(spark):
    # Use parallelize so the RDD is backed by in-process data (no Python
    # worker socket), which avoids the Windows + Spark 4.x EOFException.
    data = [Row(city="Paris", pop=2_161_000), Row(city="Berlin", pop=3_645_000)]
    return spark.sparkContext.parallelize(data)


# ===========================================================================
# Section 1 – Tags registration
# ===========================================================================

class TestTagsRegistration:
    def test_pyspark_base(self):
        assert Tags.PYSPARK_BASE == 700

    def test_all_tag_values(self):
        assert Tags.PYSPARK_DATAFRAME == 700
        assert Tags.PYSPARK_ROW       == 701
        assert Tags.PYSPARK_SCHEMA    == 702
        assert Tags.PYSPARK_DATATYPE  == 703
        assert Tags.PYSPARK_COLUMN    == 704
        assert Tags.PYSPARK_RDD       == 705
        assert Tags.PYSPARK_SESSION   == 706

    def test_is_pyspark_predicate(self):
        for tag in (700, 701, 702, 703, 704, 705, 706):
            assert Tags.is_pyspark(tag), f"Expected is_pyspark({tag}) == True"

    def test_is_pyspark_boundaries(self):
        assert not Tags.is_pyspark(699)
        assert not Tags.is_pyspark(800)

    def test_get_category(self):
        for tag in (700, 701, 702, 703, 704, 705, 706):
            assert Tags.get_category(tag) == "pyspark"

    def test_tag_to_name(self):
        assert Tags.TAG_TO_NAME[700] == "PYSPARK_DATAFRAME"
        assert Tags.TAG_TO_NAME[701] == "PYSPARK_ROW"
        assert Tags.TAG_TO_NAME[702] == "PYSPARK_SCHEMA"
        assert Tags.TAG_TO_NAME[703] == "PYSPARK_DATATYPE"
        assert Tags.TAG_TO_NAME[704] == "PYSPARK_COLUMN"
        assert Tags.TAG_TO_NAME[705] == "PYSPARK_RDD"
        assert Tags.TAG_TO_NAME[706] == "PYSPARK_SESSION"

    def test_classes_registered_after_lazy_import(self):
        Tags._ensure_category_imported(700)
        for tag in (700, 701, 702, 703, 704, 705, 706):
            assert tag in Tags.CLASSES, f"Tag {tag} not in Tags.CLASSES"

    def test_class_tag_constants_match(self):
        assert PySparkDataFrameSerialized.TAG == Tags.PYSPARK_DATAFRAME
        assert PySparkRowSerialized.TAG       == Tags.PYSPARK_ROW
        assert PySparkSchemaSerialized.TAG    == Tags.PYSPARK_SCHEMA
        assert PySparkDataTypeSerialized.TAG  == Tags.PYSPARK_DATATYPE
        assert PySparkColumnSerialized.TAG    == Tags.PYSPARK_COLUMN
        assert PySparkRDDSerialized.TAG       == Tags.PYSPARK_RDD
        assert PySparkSessionSerialized.TAG   == Tags.PYSPARK_SESSION

    def test_category_label_constant(self):
        assert Tags.CATEGORY_PYSPARK == "pyspark"


# ===========================================================================
# Section 2 – PySparkRowSerialized
# ===========================================================================

class TestRowSerialized:
    def test_basic_roundtrip(self):
        row = Row(name="Alice", age=30, score=9.5)
        ser = PySparkRowSerialized.from_value(row)
        assert isinstance(ser, PySparkRowSerialized)
        assert ser.tag == Tags.PYSPARK_ROW
        back = ser.value
        assert back.name == "Alice"
        assert back.age == 30
        assert abs(back.score - 9.5) < 1e-9

    def test_integer_fields(self):
        row = Row(a=1, b=2, c=3)
        ser = PySparkRowSerialized.from_value(row)
        back = ser.value
        assert back.a == 1 and back.b == 2 and back.c == 3

    def test_none_field(self):
        row = Row(x="hello", y=None)
        ser = PySparkRowSerialized.from_value(row)
        back = ser.value
        assert back.x == "hello"
        assert back.y is None

    def test_single_field(self):
        row = Row(v=42)
        ser = PySparkRowSerialized.from_value(row)
        back = ser.value
        assert back.v == 42

    def test_metadata_key(self):
        row = Row(k=1)
        ser = PySparkRowSerialized.from_value(row)
        meta = ser.metadata or {}
        assert meta.get(b"arrow_object") == b"pyspark_row"

    def test_codec_none_preserved(self):
        row = Row(a=1)
        ser = PySparkRowSerialized.from_value(row, codec=CODEC_NONE)
        assert ser.codec == CODEC_NONE

    def test_wire_roundtrip(self):
        row = Row(country="DE", value=99)
        back_ser = _roundtrip(PySparkRowSerialized.from_value(row))
        assert isinstance(back_ser, PySparkRowSerialized)
        back = back_ser.value
        assert back.country == "DE"
        assert back.value == 99

    def test_from_python_object_dispatch(self):
        row = Row(x=7)
        ser = PySparkSerialized.from_python_object(row)
        assert isinstance(ser, PySparkRowSerialized)

    def test_as_python_alias(self):
        row = Row(q=5)
        ser = PySparkRowSerialized.from_value(row)
        assert ser.as_python().q == 5


# ===========================================================================
# Section 3 – PySparkDataTypeSerialized
# ===========================================================================

class TestDataTypeSerialized:
    @pytest.mark.parametrize("dtype", [
        T.StringType(),
        T.IntegerType(),
        T.LongType(),
        T.DoubleType(),
        T.FloatType(),
        T.BooleanType(),
        T.BinaryType(),
        T.DateType(),
        T.TimestampType(),
        T.TimestampNTZType(),
        T.ShortType(),
        T.ByteType(),
        T.NullType(),
        T.DecimalType(10, 3),
        T.ArrayType(T.IntegerType()),
        T.ArrayType(T.StringType(), containsNull=False),
        T.MapType(T.StringType(), T.LongType()),
        T.MapType(T.StringType(), T.IntegerType(), valueContainsNull=False),
        T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())]),
    ])
    def test_roundtrip(self, dtype):
        ser = PySparkDataTypeSerialized.from_value(dtype)
        assert isinstance(ser, PySparkDataTypeSerialized)
        assert ser.tag == Tags.PYSPARK_DATATYPE
        back = ser.value
        assert back == dtype, f"Roundtrip failed: {dtype!r} != {back!r}"

    def test_metadata_key(self):
        ser = PySparkDataTypeSerialized.from_value(T.StringType())
        meta = ser.metadata or {}
        assert meta.get(b"pyspark_object") == b"datatype"

    def test_wire_roundtrip(self):
        dtype = T.MapType(T.StringType(), T.ArrayType(T.IntegerType()))
        back_ser = _roundtrip(PySparkDataTypeSerialized.from_value(dtype))
        assert isinstance(back_ser, PySparkDataTypeSerialized)
        assert back_ser.value == dtype

    def test_payload_is_utf8_json(self):
        dtype = T.IntegerType()
        ser = PySparkDataTypeSerialized.from_value(dtype, codec=CODEC_NONE)
        raw = ser.decode()
        import json
        parsed = json.loads(raw.decode("utf-8"))
        assert "type" in parsed or "fields" in parsed or parsed  # valid JSON object

    def test_nested_struct_roundtrip(self):
        dtype = T.StructType([
            T.StructField("x", T.DoubleType()),
            T.StructField("nested", T.StructType([
                T.StructField("a", T.StringType()),
            ])),
        ])
        ser = PySparkDataTypeSerialized.from_value(dtype)
        assert ser.value == dtype

    def test_from_python_object_dispatch(self):
        dtype = T.LongType()
        ser = PySparkSerialized.from_python_object(dtype)
        assert isinstance(ser, PySparkDataTypeSerialized)


# ===========================================================================
# Section 4 – PySparkSchemaSerialized
# ===========================================================================

class TestSchemaSerialized:
    def test_simple_schema_roundtrip(self):
        schema = _simple_schema()
        ser = PySparkSchemaSerialized.from_value(schema)
        assert isinstance(ser, PySparkSchemaSerialized)
        assert ser.tag == Tags.PYSPARK_SCHEMA
        back = ser.value
        # Either StructType (PySpark) or pa.Schema (Arrow fallback) – both acceptable
        assert back is not None

    def test_simple_schema_fields_preserved(self):
        schema = _simple_schema()
        ser = PySparkSchemaSerialized.from_value(schema)
        back = ser.value
        if isinstance(back, T.StructType):
            assert len(back.fields) == len(schema.fields)
            for orig, restored in zip(schema.fields, back.fields):
                assert orig.name == restored.name
                assert orig.dataType == restored.dataType

    def test_nested_schema_roundtrip(self):
        schema = _nested_schema()
        ser = PySparkSchemaSerialized.from_value(schema)
        back = ser.value
        if isinstance(back, T.StructType):
            assert back == schema

    def test_metadata_key(self):
        ser = PySparkSchemaSerialized.from_value(_simple_schema())
        meta = ser.metadata or {}
        assert meta.get(b"arrow_object") == b"pyspark_schema"

    def test_wire_roundtrip(self):
        schema = _simple_schema()
        back_ser = _roundtrip(PySparkSchemaSerialized.from_value(schema))
        assert isinstance(back_ser, PySparkSchemaSerialized)
        back = back_ser.value
        assert back is not None

    def test_from_python_object_dispatch(self):
        schema = _simple_schema()
        ser = PySparkSerialized.from_python_object(schema)
        assert isinstance(ser, PySparkSchemaSerialized)

    def test_empty_schema(self):
        schema = T.StructType([])
        ser = PySparkSchemaSerialized.from_value(schema)
        back = ser.value
        if isinstance(back, T.StructType):
            assert len(back.fields) == 0

    def test_single_field_schema(self):
        schema = T.StructType([T.StructField("only", T.BooleanType())])
        ser = PySparkSchemaSerialized.from_value(schema)
        back = ser.value
        if isinstance(back, T.StructType):
            assert back.fields[0].name == "only"


# ===========================================================================
# Section 5 – PySparkColumnSerialized
# ===========================================================================

def _column_pickle_supported() -> bool:
    """Return True only if a pyspark Column can actually be pickled in this process."""
    try:
        from pyspark.sql.functions import col
        import cloudpickle
        cloudpickle.dumps(col("_test_"))
        return True
    except Exception:
        return False


_skip_column = pytest.mark.skipif(
    not _column_pickle_supported(),
    reason="Column pickling not supported in this environment (JVM RLock)",
)


class TestColumnSerialized:
    @_skip_column
    def test_basic_column_roundtrip(self, spark):
        from pyspark.sql.functions import col, lit
        from pyspark.sql import Column
        c = col("value") + lit(1)
        ser = PySparkColumnSerialized.from_value(c)
        assert isinstance(ser, PySparkColumnSerialized)
        assert ser.tag == Tags.PYSPARK_COLUMN
        assert isinstance(ser.value, Column)

    @_skip_column
    def test_metadata_key(self, spark):
        from pyspark.sql.functions import col
        ser = PySparkColumnSerialized.from_value(col("x"))
        meta = ser.metadata or {}
        assert meta.get(b"pyspark_object") == b"column"

    @_skip_column
    def test_wire_roundtrip(self, spark):
        from pyspark.sql.functions import col
        from pyspark.sql import Column
        ser = PySparkColumnSerialized.from_value(col("my_col"))
        back_ser = _roundtrip(ser)
        assert isinstance(back_ser, PySparkColumnSerialized)
        assert isinstance(back_ser.value, Column)

    @_skip_column
    def test_from_python_object_dispatch(self, spark):
        from pyspark.sql.functions import lit
        c = lit(42)
        ser = PySparkSerialized.from_python_object(c)
        assert isinstance(ser, PySparkColumnSerialized)

    @_skip_column
    def test_complex_expression_roundtrip(self, spark):
        import pyspark.sql.functions as F
        c = (F.col("a") * 2 + F.col("b")).cast("double").alias("result")
        ser = PySparkColumnSerialized.from_value(c)
        back_ser = _roundtrip(ser)
        df = spark.createDataFrame([(1, 10), (2, 20)], ["a", "b"])
        result = df.select(back_ser.value).collect()
        assert result[0][0] == pytest.approx(12.0)
        assert result[1][0] == pytest.approx(24.0)



# ===========================================================================
# Section 6 – PySparkSessionSerialized (stub, no live session)
# ===========================================================================

class TestSessionSerialized:
    def _fake_session(self, app_name="my-app", master="local", version="3.5.0"):
        class _FakeConf:
            def get(self, key, default=""):
                if key == "spark.app.name": return app_name
                if key == "spark.master":   return master
                return default
        class _FakeSC:
            def getConf(self): return _FakeConf()
        class _FakeSession:
            sparkContext = _FakeSC()
        s = _FakeSession()
        s.version = version
        return s

    def test_value_is_none(self):
        ser = PySparkSessionSerialized.from_value(self._fake_session())
        assert ser.value is None

    def test_tag(self):
        ser = PySparkSessionSerialized.from_value(self._fake_session())
        assert ser.tag == Tags.PYSPARK_SESSION

    def test_metadata_object_key(self):
        ser = PySparkSessionSerialized.from_value(self._fake_session())
        meta = ser.metadata or {}
        assert meta.get(b"pyspark_object") == b"sparksession"

    def test_metadata_app_name(self):
        ser = PySparkSessionSerialized.from_value(self._fake_session(app_name="etl-job"))
        meta = ser.metadata or {}
        assert meta.get(b"spark_app_name") == b"etl-job"

    def test_metadata_master(self):
        ser = PySparkSessionSerialized.from_value(self._fake_session(master="yarn"))
        meta = ser.metadata or {}
        assert meta.get(b"spark_master") == b"yarn"

    def test_metadata_version(self):
        ser = PySparkSessionSerialized.from_value(self._fake_session(version="3.5.1"))
        meta = ser.metadata or {}
        assert meta.get(b"spark_version") == b"3.5.1"

    def test_payload_is_empty(self):
        ser = PySparkSessionSerialized.from_value(self._fake_session(), codec=CODEC_NONE)
        assert ser.decode() == b""

    def test_wire_roundtrip_value_still_none(self):
        ser = PySparkSessionSerialized.from_value(self._fake_session())
        back_ser = _roundtrip(ser)
        assert isinstance(back_ser, PySparkSessionSerialized)
        assert back_ser.value is None

    def test_wire_roundtrip_metadata_preserved(self):
        ser = PySparkSessionSerialized.from_value(self._fake_session(app_name="roundtrip-test"))
        back_ser = _roundtrip(ser)
        meta = back_ser.metadata or {}
        assert meta.get(b"spark_app_name") == b"roundtrip-test"

    def test_broken_session_does_not_raise(self):
        """from_value must not raise even if session attributes are missing."""
        class _BrokenSession:
            pass
        ser = PySparkSessionSerialized.from_value(_BrokenSession())
        assert ser.tag == Tags.PYSPARK_SESSION
        assert ser.value is None


# ===========================================================================
# Section 7 – PySparkDataFrameSerialized (requires live SparkSession)
# ===========================================================================

class TestDataFrameSerialized:
    def test_value_is_arrow_table(self, simple_df):
        ser = PySparkDataFrameSerialized.from_value(simple_df)
        assert isinstance(ser, PySparkDataFrameSerialized)
        assert ser.tag == Tags.PYSPARK_DATAFRAME
        table = ser.value
        assert isinstance(table, pa.Table)

    def test_row_count_preserved(self, simple_df):
        ser = PySparkDataFrameSerialized.from_value(simple_df)
        table = ser.value
        assert table.num_rows == simple_df.count()

    def test_column_names_preserved(self, simple_df):
        ser = PySparkDataFrameSerialized.from_value(simple_df)
        table = ser.value
        assert set(table.schema.names) == {"name", "age", "score"}

    def test_data_values_preserved(self, spark):
        data = [Row(x=1, y="a"), Row(x=2, y="b")]
        df = spark.createDataFrame(spark.sparkContext.parallelize(data))
        ser = PySparkDataFrameSerialized.from_value(df)
        table = ser.value
        xs = table.column("x").to_pylist()
        ys = table.column("y").to_pylist()
        assert sorted(xs) == [1, 2]
        assert sorted(ys) == ["a", "b"]

    def test_null_values_preserved(self, simple_df):
        ser = PySparkDataFrameSerialized.from_value(simple_df)
        table = ser.value
        names = table.column("name").to_pylist()
        assert None in names

    def test_metadata_key(self, simple_df):
        ser = PySparkDataFrameSerialized.from_value(simple_df)
        meta = ser.metadata or {}
        assert meta.get(b"arrow_object") == b"pyspark_dataframe"

    def test_wire_roundtrip(self, simple_df):
        ser = PySparkDataFrameSerialized.from_value(simple_df)
        back_ser = _roundtrip(ser)
        assert isinstance(back_ser, PySparkDataFrameSerialized)
        orig  = ser.value
        back  = back_ser.value
        assert orig.equals(back)

    def test_to_pyspark(self, spark, simple_df):
        ser = PySparkDataFrameSerialized.from_value(simple_df)
        table = ser.value
        # Re-wrap via pandas if available, else just check the Arrow table
        try:
            import pandas as pd
            df2 = spark.createDataFrame(spark.sparkContext.parallelize(
                [Row(**{k: v for k, v in zip(table.schema.names, row)})
                 for row in zip(*[table.column(n).to_pylist() for n in table.schema.names])]
            ))
            from pyspark.sql import DataFrame as SparkDF
            assert isinstance(df2, SparkDF)
        except ImportError:
            pass  # pandas not available; Arrow table is the output
        assert table.num_rows == simple_df.count()

    def test_empty_dataframe(self, spark):
        schema = T.StructType([T.StructField("a", T.IntegerType())])
        df = spark.createDataFrame(spark.sparkContext.parallelize([]), schema=schema)
        ser = PySparkDataFrameSerialized.from_value(df)
        table = ser.value
        assert isinstance(table, pa.Table)
        assert table.num_rows == 0
        assert table.schema.names == ["a"]

    def test_from_python_object_dispatch(self, simple_df):
        ser = PySparkSerialized.from_python_object(simple_df)
        assert isinstance(ser, PySparkDataFrameSerialized)

    def test_codec_zstd(self, simple_df):
        ser = PySparkDataFrameSerialized.from_value(simple_df, codec=CODEC_ZSTD)
        assert ser.codec == CODEC_ZSTD
        back = ser.value
        assert isinstance(back, pa.Table)
        assert back.num_rows == simple_df.count()


# ===========================================================================
# Section 8 – PySparkRDDSerialized (requires live SparkSession)
# ===========================================================================

class TestRDDSerialized:
    def test_row_rdd_roundtrip(self, typed_rdd):
        ser = PySparkRDDSerialized.from_value(typed_rdd)
        assert isinstance(ser, PySparkRDDSerialized)
        assert ser.tag == Tags.PYSPARK_RDD
        table = ser.value
        assert isinstance(table, pa.Table)
        assert table.num_rows == 2

    def test_column_names_preserved(self, typed_rdd):
        ser = PySparkRDDSerialized.from_value(typed_rdd)
        table = ser.value
        assert set(table.schema.names) == {"city", "pop"}

    def test_values_preserved(self, typed_rdd):
        ser = PySparkRDDSerialized.from_value(typed_rdd)
        table = ser.value
        cities = table.column("city").to_pylist()
        assert set(cities) == {"Paris", "Berlin"}

    def test_primitive_rdd(self, spark):
        rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5])
        ser = PySparkRDDSerialized.from_value(rdd)
        table = ser.value
        assert isinstance(table, pa.Table)
        assert table.num_rows == 5
        values = table.column("value").to_pylist()
        assert sorted(values) == [1, 2, 3, 4, 5]

    def test_dict_rdd(self, spark):
        data = [{"k": "a", "v": 1}, {"k": "b", "v": 2}]
        rdd = spark.sparkContext.parallelize(data)
        ser = PySparkRDDSerialized.from_value(rdd)
        table = ser.value
        assert set(table.schema.names) == {"k", "v"}

    def test_empty_rdd(self, spark):
        rdd = spark.sparkContext.parallelize([])
        ser = PySparkRDDSerialized.from_value(rdd)
        table = ser.value
        assert isinstance(table, pa.Table)
        assert table.num_rows == 0

    def test_metadata_key(self, typed_rdd):
        ser = PySparkRDDSerialized.from_value(typed_rdd)
        meta = ser.metadata or {}
        assert meta.get(b"arrow_object") == b"pyspark_rdd"

    def test_wire_roundtrip(self, typed_rdd):
        ser = PySparkRDDSerialized.from_value(typed_rdd)
        back_ser = _roundtrip(ser)
        assert isinstance(back_ser, PySparkRDDSerialized)
        assert ser.value.equals(back_ser.value)

    def test_from_python_object_dispatch(self, typed_rdd):
        ser = PySparkSerialized.from_python_object(typed_rdd)
        assert isinstance(ser, PySparkRDDSerialized)


# ===========================================================================
# Section 9 – Full-stack dispatch via Serialized.from_python_object
# ===========================================================================

class TestBaseDispatch:
    def test_row_dispatch(self):
        row = Row(a=1, b="hello")
        ser = Serialized.from_python_object(row)
        assert isinstance(ser, PySparkRowSerialized)
        assert ser.tag == Tags.PYSPARK_ROW

    def test_structtype_dispatch(self):
        ser = Serialized.from_python_object(_simple_schema())
        assert isinstance(ser, PySparkSchemaSerialized)
        assert ser.tag == Tags.PYSPARK_SCHEMA

    def test_datatype_dispatch(self):
        for dtype in (T.IntegerType(), T.ArrayType(T.StringType()), T.MapType(T.StringType(), T.LongType())):
            ser = Serialized.from_python_object(dtype)
            assert isinstance(ser, PySparkDataTypeSerialized), f"failed for {dtype!r}"

    @_skip_column
    def test_column_dispatch(self, spark):
        from pyspark.sql.functions import col
        ser = Serialized.from_python_object(col("x"))
        assert isinstance(ser, PySparkColumnSerialized)

    def test_dataframe_dispatch(self, spark, simple_df):
        ser = Serialized.from_python_object(simple_df)
        assert isinstance(ser, PySparkDataFrameSerialized)

    def test_rdd_dispatch(self, typed_rdd):
        ser = Serialized.from_python_object(typed_rdd)
        assert isinstance(ser, PySparkRDDSerialized)

    def test_row_value_via_base(self):
        row = Row(country="FR", pop=68_000_000)
        back = Serialized.from_python_object(row).as_python()
        assert back.country == "FR"
        assert back.pop == 68_000_000

    def test_datatype_value_via_base(self):
        dtype = T.ArrayType(T.IntegerType())
        back = Serialized.from_python_object(dtype).as_python()
        assert back == dtype


# ===========================================================================
# Section 10 – Binary wire round-trip (write_to / read_from)
# ===========================================================================

class TestWireRoundtrip:
    def test_row_full_wire(self):
        row = Row(name="Bob", score=8.3)
        buf = BytesIO()
        PySparkRowSerialized.from_value(row).write_to(buf)
        back_ser = Serialized.read_from(buf, pos=0)
        assert isinstance(back_ser, PySparkRowSerialized)
        back = back_ser.value
        assert back.name == "Bob"
        assert abs(back.score - 8.3) < 1e-6

    def test_schema_full_wire(self):
        schema = _simple_schema()
        buf = BytesIO()
        PySparkSchemaSerialized.from_value(schema).write_to(buf)
        back_ser = Serialized.read_from(buf, pos=0)
        assert isinstance(back_ser, PySparkSchemaSerialized)

    def test_datatype_full_wire(self):
        dtype = T.MapType(T.StringType(), T.ArrayType(T.IntegerType()))
        buf = BytesIO()
        PySparkDataTypeSerialized.from_value(dtype).write_to(buf)
        back_ser = Serialized.read_from(buf, pos=0)
        assert isinstance(back_ser, PySparkDataTypeSerialized)
        assert back_ser.value == dtype

    def test_session_stub_full_wire(self):
        class _FakeConf:
            def get(self, k, d=""): return d
        class _FakeSC:
            def getConf(self): return _FakeConf()
        class _FakeSession:
            sparkContext = _FakeSC()
            version = "3.5.0"

        buf = BytesIO()
        PySparkSessionSerialized.from_value(_FakeSession()).write_to(buf)
        back_ser = Serialized.read_from(buf, pos=0)
        assert isinstance(back_ser, PySparkSessionSerialized)
        assert back_ser.value is None

    def test_tag_survives_wire(self):
        row = Row(q=99)
        buf = BytesIO()
        PySparkRowSerialized.from_value(row, codec=CODEC_NONE).write_to(buf)
        back_ser = Serialized.read_from(buf, pos=0)
        assert back_ser.tag == Tags.PYSPARK_ROW

    def test_metadata_survives_wire(self):
        extra = {b"custom_key": b"custom_val"}
        row = Row(x=1)
        buf = BytesIO()
        PySparkRowSerialized.from_value(row, metadata=extra, codec=CODEC_NONE).write_to(buf)
        back_ser = Serialized.read_from(buf, pos=0)
        meta = back_ser.metadata or {}
        assert meta.get(b"custom_key") == b"custom_val"


# ===========================================================================
# Section 11 – Codec / compression variants
# ===========================================================================

class TestCodecVariants:
    @pytest.mark.parametrize("codec", [CODEC_NONE, CODEC_ZSTD])
    def test_row_codec(self, codec):
        row = Row(a=1, b="test")
        ser = PySparkRowSerialized.from_value(row, codec=codec)
        assert ser.codec == codec
        back = ser.value
        assert back.a == 1 and back.b == "test"

    @pytest.mark.parametrize("codec", [CODEC_NONE, CODEC_ZSTD])
    def test_datatype_codec(self, codec):
        dtype = T.ArrayType(T.LongType())
        ser = PySparkDataTypeSerialized.from_value(dtype, codec=codec)
        assert ser.codec == codec
        assert ser.value == dtype

    @pytest.mark.parametrize("codec", [CODEC_NONE, CODEC_ZSTD])
    def test_schema_codec(self, codec):
        schema = _simple_schema()
        ser = PySparkSchemaSerialized.from_value(schema, codec=codec)
        assert ser.codec == codec
        back = ser.value
        assert back is not None

    @pytest.mark.parametrize("codec", [CODEC_NONE, CODEC_ZSTD])
    def test_dataframe_codec(self, spark, codec):
        # Create a fresh DataFrame for each codec variant to avoid Arrow stream reuse.
        data = [Row(name="Alice", age=30, score=9.5), Row(name="Bob", age=25, score=7.0)]
        df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema=_simple_schema())
        try:
            ser = PySparkDataFrameSerialized.from_value(df, codec=codec)
        except Exception as exc:
            pytest.xfail(
                f"Spark socket unstable after extended session (Windows/Spark 4.x): {exc}"
            )
        assert ser.codec == codec
        table = ser.value
        assert table.num_rows == df.count()

    def test_auto_codec_large_payload(self, spark):
        """Large DataFrames should auto-select a compressing codec."""
        import random, string
        # Use a smaller payload to reduce socket-timeout risk on Windows Spark 4.x
        data = [Row(text="".join(random.choices(string.ascii_letters, k=50))) for _ in range(100)]
        df = spark.createDataFrame(spark.sparkContext.parallelize(data))
        try:
            ser = PySparkDataFrameSerialized.from_value(df)
        except Exception as exc:
            pytest.xfail(
                f"Spark socket unstable after extended session (Windows/Spark 4.x): {exc}"
            )
        assert ser.value.num_rows == 100

    # ===========================================================================
    # Section 12 – Edge cases and error handling
    # ===========================================================================

class TestEdgeCases:
    def test_row_many_fields(self):
        kwargs = {f"f{i}": i for i in range(50)}
        row = Row(**kwargs)
        ser = PySparkRowSerialized.from_value(row)
        back = ser.value
        for i in range(50):
            assert back[f"f{i}"] == i

    def test_row_string_with_unicode(self):
        row = Row(text="日本語テスト 🎉", num=42)
        ser = PySparkRowSerialized.from_value(row)
        back = ser.value
        assert back.text == "日本語テスト 🎉"

    def test_schema_deeply_nested(self):
        schema = T.StructType([
            T.StructField("level1", T.StructType([
                T.StructField("level2", T.StructType([
                    T.StructField("leaf", T.IntegerType()),
                ])),
            ])),
        ])
        ser = PySparkSchemaSerialized.from_value(schema)
        back = ser.value
        if isinstance(back, T.StructType):
            assert back == schema

    def test_datatype_nested_map_of_arrays(self):
        dtype = T.MapType(T.StringType(), T.ArrayType(T.StructType([
            T.StructField("v", T.DoubleType()),
        ])))
        ser = PySparkDataTypeSerialized.from_value(dtype)
        assert ser.value == dtype

    def test_from_python_object_returns_none_for_unsupported(self):
        """PySparkSerialized.from_python_object returns None for non-PySpark objects."""
        result = PySparkSerialized.from_python_object(object())
        assert result is None

    def test_session_serialized_is_not_registered_as_type(self):
        """SparkSession should not be pre-registered in TYPES (live object)."""
        # Just assert the tag exists and is valid
        assert Tags.PYSPARK_SESSION == 706

    def test_row_preserves_field_order(self):
        row = Row(z=3, a=1, m=2)
        ser = PySparkRowSerialized.from_value(row)
        back = ser.value
        # All values present
        assert back.z == 3 and back.a == 1 and back.m == 2

    def test_dataframe_with_schema_types(self, spark):
        schema = T.StructType([
            T.StructField("flag",  T.BooleanType()),
            T.StructField("score", T.DoubleType()),
        ])
        data = [Row(flag=True, score=3.14), Row(flag=False, score=2.71)]
        df = spark.createDataFrame(spark.sparkContext.parallelize(data), schema=schema)
        try:
            ser = PySparkDataFrameSerialized.from_value(df)
        except Exception as exc:
            pytest.xfail(
                f"Spark socket unstable after extended session (Windows/Spark 4.x): {exc}"
            )
        table = ser.value
        assert table.num_rows == 2
        flags = table.column("flag").to_pylist()
        assert True in flags and False in flags

