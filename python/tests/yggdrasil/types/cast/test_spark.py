# test_spark_cast.py

import pyarrow as pa
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession, Row, functions as F, types as T

from yggdrasil.types.cast.spark_cast import (
    cast_spark_dataframe,
    cast_spark_column,
)
from yggdrasil.types.cast.cast_options import CastOptions
from yggdrasil.types import convert


# ---------------------------------------------------------------------------
# SparkSession fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("yggdrasil-spark-cast-tests")
        .getOrCreate()
    )
    yield spark
    spark.stop()


# ---------------------------------------------------------------------------
# DataFrame casting tests (pure Spark, Arrow-driven types)
# ---------------------------------------------------------------------------

def test_cast_spark_dataframe_struct_defaults_and_casefold(spark):
    df = spark.createDataFrame(
        [({"score": 1},), ({"score": None},)],
        schema="payload struct<score:int>",
    )

    target_schema = pa.schema(
        [
            pa.field(
                "payload",
                pa.struct(
                    [
                        pa.field("score", pa.int64(), nullable=False),
                        pa.field("label", pa.string(), nullable=False),
                    ]
                ),
                nullable=False,
            )
        ]
    )

    opts = CastOptions.safe_init(
        target_field=target_schema,
        add_missing_columns=True,
        strict_match_names=False,
    )

    result = cast_spark_dataframe(df, opts)

    assert result.schema["payload"].dataType == T.StructType(
        [
            T.StructField("score", T.LongType(), nullable=False),
            T.StructField("label", T.StringType(), nullable=False),
        ]
    )
    assert [r.payload.asDict(recursive=True) for r in result.collect()] == [
        {"score": 1, "label": ""},
        {"score": 0, "label": ""},
    ]


def test_cast_spark_dataframe_adds_missing_columns_and_drops_extra(spark):
    df = spark.createDataFrame([(1, True)], ["id", "extra"])

    target_schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("flag", pa.bool_(), nullable=False),
        ]
    )

    opts = CastOptions.safe_init(
        target_field=target_schema,
        add_missing_columns=True,
        allow_add_columns=False,
    )

    result = cast_spark_dataframe(df, opts)

    assert result.columns == ["id", "flag"]
    assert result.schema["flag"].nullable is False
    assert [tuple(r) for r in result.collect()] == [(1, False)]


def test_cast_spark_dataframe_allows_extras_when_opted_in(spark):
    df = spark.createDataFrame([(1, "x")], ["a", "extra"])

    target_schema = pa.schema([pa.field("a", pa.int32(), nullable=True)])
    opts = CastOptions.safe_init(
        target_field=target_schema,
        allow_add_columns=True,
        strict_match_names=True,
    )

    result = cast_spark_dataframe(df, opts)

    assert result.columns == ["a", "extra"]
    assert [tuple(r) for r in result.collect()] == [(1, "x")]


# ---------------------------------------------------------------------------
# Column casting tests (pure Spark, Arrow-driven types)
# ---------------------------------------------------------------------------

def test_cast_spark_column_struct_child_defaults(spark):
    df = spark.createDataFrame(
        [({"count": None, "label": "a"},)],
        schema="info struct<count:int,label:string>",
    )

    target_field = pa.field(
        "info",
        pa.struct(
            [
                pa.field("count", pa.int64(), nullable=False),
                pa.field("label", pa.string(), nullable=False),
            ]
        ),
        nullable=False,
    )

    casted_col = cast_spark_column(F.col("info"), target_field)
    result = df.select(casted_col.alias("info"))

    assert result.schema["info"].dataType == T.StructType(
        [
            T.StructField("count", T.LongType(), nullable=False),
            T.StructField("label", T.StringType(), nullable=False),
        ]
    )
    assert [r.info.asDict(recursive=True) for r in result.collect()] == [
        {"count": 0, "label": "a"}
    ]


def test_cast_spark_column_schema_target_prefers_first_field(spark):
    df = spark.createDataFrame([(1,)], ["a"])

    schema = pa.schema([pa.field("a", pa.string(), nullable=True)])
    opts = CastOptions.safe_init(target_field=schema)

    casted_col = cast_spark_column(F.col("a"), opts)
    result = df.select(casted_col.alias("a"))

    assert result.schema["a"].dataType == T.StringType()
    assert [r.a for r in result.collect()] == ["1"]


# ---------------------------------------------------------------------------
# Integration with convert(...) registry
# ---------------------------------------------------------------------------

def test_convert_dataframe_to_dataframe_applies_nested_cast(spark):
    df = spark.createDataFrame(
        [(Row(score="4"),)],
        schema=T.StructType(
            [
                T.StructField(
                    "payload",
                    T.StructType([T.StructField("score", T.StringType(), True)]),
                    True,
                )
            ]
        ),
    )

    target_schema = pa.schema(
        [
            pa.field(
                "payload",
                pa.struct([pa.field("score", pa.int64(), nullable=False)]),
                nullable=False,
            )
        ]
    )

    casted = convert(
        df,
        pyspark.sql.DataFrame,
        options=CastOptions.safe_init(target_field=target_schema),
    )

    assert casted.schema["payload"].dataType == T.StructType(
        [T.StructField("score", T.LongType(), nullable=False)]
    )
    assert [r.payload.score for r in casted.collect()] == [4]


def test_convert_column_to_column_uses_target_dtype(spark):
    df = spark.createDataFrame([(1,), (2,)], ["a"])
    target_type = pa.float64()

    casted_col = convert(F.col("a"), pyspark.sql.Column, options=target_type)
    result = df.select(casted_col.alias("a"))

    assert result.schema["a"].dataType == T.DoubleType()
    assert [r.a for r in result.collect()] == [1.0, 2.0]
