# test_spark_cast.py

import pyarrow as pa
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession, functions as F, types as T

from yggdrasil.types.cast.spark_cast import (
    cast_spark_dataframe,
    cast_spark_column,
)
from yggdrasil.types.cast.arrow_cast import ArrowCastOptions
from yggdrasil.types import convert


# ---------------------------------------------------------------------------
# SparkSession fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("yggdrasil-spark-cast-tests")
        .getOrCreate()
    )
    yield spark
    spark.stop()


# ---------------------------------------------------------------------------
# DataFrame casting tests (pure Spark, Arrow-driven types)
# ---------------------------------------------------------------------------

def test_cast_spark_dataframe_no_target_schema_is_noop(spark):
    df = spark.createDataFrame([(1, "x"), (2, "y")], ["a", "b"])

    # options with no target_field -> target_schema is None
    opts = ArrowCastOptions.__safe_init__(target_field=None)

    result = cast_spark_dataframe(df, opts)

    assert result.schema == df.schema
    assert result.collect() == df.collect()


def test_cast_spark_dataframe_numeric_cast_and_fill_non_nullable(spark):
    df = spark.createDataFrame(
        [
            (1,),
            (None,),
            (3,),
        ],
        ["a"],
    )

    target_schema = pa.schema(
        [pa.field("a", pa.float64(), nullable=False)],
    )
    opts = ArrowCastOptions.__safe_init__(target_field=target_schema)

    result = cast_spark_dataframe(df, opts)

    # Schema: a -> DoubleType, nullable False
    assert result.schema["a"].dataType == T.DoubleType()
    assert result.schema["a"].nullable is False

    values = [row.a for row in result.orderBy("a").collect()]
    # Null should be replaced by 0.0
    assert sorted(values) == [0.0, 1.0, 3.0]


def test_cast_spark_dataframe_case_insensitive_match(spark):
    df = spark.createDataFrame([(1,), (2,)], ["A"])

    target_schema = pa.schema(
        [pa.field("a", pa.int64(), nullable=False)],
    )
    opts = ArrowCastOptions.__safe_init__(
        target_field=target_schema,
        strict_match_names=False,
    )

    result = cast_spark_dataframe(df, opts)

    assert result.columns == ["a"]
    assert result.schema["a"].dataType == T.LongType()
    assert [r.a for r in result.collect()] == [1, 2]


def test_cast_spark_dataframe_missing_column_add_missing_false_raises(spark):
    df = spark.createDataFrame([(1,), (2,)], ["a"])

    target_schema = pa.schema(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("b", pa.int32(), nullable=True),
        ]
    )

    opts = ArrowCastOptions.__safe_init__(
        target_field=target_schema,
        add_missing_columns=False,
        strict_match_names=True,
    )

    with pytest.raises(TypeError, match="Missing column b"):
        cast_spark_dataframe(df, opts)


def test_cast_spark_dataframe_add_missing_column_with_defaults(spark):
    df = spark.createDataFrame([(1,), (2,)], ["a"])

    target_schema = pa.schema(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("b", pa.string(), nullable=False),
        ]
    )

    opts = ArrowCastOptions.__safe_init__(
        target_field=target_schema,
        add_missing_columns=True,
        strict_match_names=True,
    )

    result = cast_spark_dataframe(df, opts)

    assert result.columns == ["a", "b"]
    assert result.schema["b"].dataType == T.StringType()
    # default_arrow_python_value(string) -> ""
    assert [r.b for r in result.collect()] == ["", ""]


def test_cast_spark_dataframe_allow_add_columns_false_drops_extras(spark):
    df = spark.createDataFrame([(1, "x")], ["a", "extra"])

    target_schema = pa.schema(
        [pa.field("a", pa.int32(), nullable=True)],
    )

    opts = ArrowCastOptions.__safe_init__(
        target_field=target_schema,
        allow_add_columns=False,
        strict_match_names=True,
    )

    result = cast_spark_dataframe(df, opts)

    assert result.columns == ["a"]
    assert [tuple(r) for r in result.collect()] == [(1,)]


def test_cast_spark_dataframe_allow_add_columns_true_keeps_extras(spark):
    df = spark.createDataFrame([(1, "x")], ["a", "extra"])

    target_schema = pa.schema(
        [pa.field("a", pa.int32(), nullable=True)],
    )

    opts = ArrowCastOptions.__safe_init__(
        target_field=target_schema,
        allow_add_columns=True,
        strict_match_names=True,
    )

    result = cast_spark_dataframe(df, opts)

    # Target columns first, then extras
    assert result.columns == ["a", "extra"]
    assert [tuple(r) for r in result.collect()] == [(1, "x")]


# ---------------------------------------------------------------------------
# Column casting tests (pure Spark, Arrow-driven types)
# ---------------------------------------------------------------------------

def test_cast_spark_column_simple_type_cast(spark):
    df = spark.createDataFrame([(1,), (2,)], ["a"])

    # Cast column to float64 using Arrow type
    casted_col = cast_spark_column(F.col("a"), pa.float64())
    result = df.select(casted_col.alias("a"))

    assert result.schema["a"].dataType == T.DoubleType()
    assert [r.a for r in result.collect()] == [1.0, 2.0]


def test_cast_spark_column_fill_non_nullable_with_default(spark):
    df = spark.createDataFrame(
        [
            (1,),
            (None,),
            (3,),
        ],
        ["a"],
    )

    target_field = pa.field("a", pa.int32(), nullable=False)
    opts = ArrowCastOptions.__safe_init__(target_field=target_field)

    casted_col = cast_spark_column(F.col("a"), opts)
    result = df.select(casted_col.alias("a"))

    assert result.schema["a"].dataType == T.IntegerType()
    values = [r.a for r in result.orderBy("a").collect()]
    assert sorted(values) == [0, 1, 3]


def test_cast_spark_column_schema_target_uses_first_field(spark):
    df = spark.createDataFrame([(1,)], ["a"])

    schema = pa.schema(
        [pa.field("a", pa.string(), nullable=True)]
    )
    opts = ArrowCastOptions.__safe_init__(target_field=schema)

    casted_col = cast_spark_column(F.col("a"), opts)
    result = df.select(casted_col.alias("a"))

    assert result.schema["a"].dataType == T.StringType()
    assert [r.a for r in result.collect()] == ["1"]


# ---------------------------------------------------------------------------
# Integration with convert(...) registry
# ---------------------------------------------------------------------------

def test_convert_dataframe_to_dataframe_uses_cast_spark_dataframe(spark):
    df = spark.createDataFrame([(1,), (2,)], ["a"])

    target_schema = pa.schema(
        [pa.field("a", pa.int64(), nullable=False)],
    )

    # `convert(df, DataFrame)` should hit the registered converter:
    casted = convert(df, pyspark.sql.DataFrame, options=ArrowCastOptions.__safe_init__(target_field=target_schema))

    assert casted.schema["a"].dataType == T.LongType()
    assert casted.schema["a"].nullable is False
    assert [r.a for r in casted.collect()] == [1, 2]


def test_convert_column_to_column_uses_cast_spark_column(spark):
    df = spark.createDataFrame([(1,), (2,)], ["a"])

    target_type = pa.float64()
    casted_col = convert(F.col("a"), pyspark.sql.Column, options=target_type)

    result = df.select(casted_col.alias("a"))
    assert result.schema["a"].dataType == T.DoubleType()
    assert [r.a for r in result.collect()] == [1.0, 2.0]
