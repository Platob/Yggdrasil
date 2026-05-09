"""Spark-side casts for :class:`MapType`.

Skipped without pyspark. Coverage:

* Dtype probes — ``to_spark`` / ``handles_spark_type`` /
  ``from_spark_type`` round-trips.
* Compute — ``cast_spark_map_column`` / ``cast_spark_list_column_to_map``
  / ``cast_spark_struct_column_to_map`` against a real DataFrame.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.nested import MapType
from yggdrasil.data.types.nested.map import (
    cast_spark_list_column_to_map,
    cast_spark_map_column,
    cast_spark_struct_column_to_map,
)

from ._helpers import normalize_map_like

pytest.importorskip("pyspark")

from yggdrasil.spark.tests import spark  # noqa: E402,F401
from yggdrasil.spark.cast import spark_dataframe_to_arrow  # noqa: E402


# ---------------------------------------------------------------------------
# Dtype probes
# ---------------------------------------------------------------------------


class TestSparkDtype:

    def test_to_spark_emits_map_type(
        self, spark, int64_type, string_type  # noqa: F811
    ) -> None:
        from pyspark.sql.types import MapType as SparkMapType

        dtype = MapType.from_key_value(string_type, int64_type)

        assert isinstance(dtype.to_spark(), SparkMapType)

    def test_handles_spark_type_only_for_map(self, spark) -> None:  # noqa: F811
        from pyspark.sql.types import (
            LongType,
            MapType as SparkMapType,
            StringType,
        )

        assert (
            MapType.handles_spark_type(SparkMapType(StringType(), LongType())) is True
        )
        assert MapType.handles_spark_type(LongType()) is False

    def test_from_spark_type_preserves_kv(self, spark) -> None:  # noqa: F811
        from pyspark.sql.types import (
            LongType,
            MapType as SparkMapType,
            StringType,
        )

        rebuilt = MapType.from_spark_type(SparkMapType(StringType(), LongType()))

        assert isinstance(rebuilt, MapType)
        assert rebuilt.key_field.name == "key"
        assert rebuilt.value_field.name == "value"


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


class TestSparkMapColumn:

    def test_recasts_values(
        self,
        spark,  # noqa: F811
        source_map_field: Field,
        target_map_field: Field,
    ) -> None:
        import pyspark.sql.functions as F

        frame = spark.createDataFrame(
            [({"a": 1, "b": 2},), ({"x": 3},), (None,)],
            schema="source_map map<string,bigint>",
        )

        result = frame.select(
            cast_spark_map_column(
                F.col("source_map"),
                CastOptions(
                    source_field=source_map_field,
                    target_field=target_map_field,
                ),
            ).alias("target_map")
        )

        rows = [
            normalize_map_like(v)
            for v in result.toPandas()["target_map"].tolist()
        ]
        assert rows == [{"a": "1", "b": "2"}, {"x": "3"}, None]


class TestSparkListColumnToMap:

    def test_list_of_struct_collapses_into_map(
        self,
        spark,  # noqa: F811
        source_list_of_struct_field: Field,
        target_map_field: Field,
    ) -> None:
        import pyspark.sql.functions as F

        frame = spark.createDataFrame(
            [
                ([{"key": "a", "value": 1}, {"key": "b", "value": 2}],),
                ([{"key": "x", "value": 3}],),
                (None,),
            ],
            schema="source_entries array<struct<key:string,value:bigint>>",
        )

        result = frame.select(
            cast_spark_list_column_to_map(
                F.col("source_entries"),
                CastOptions(
                    source_field=source_list_of_struct_field,
                    target_field=target_map_field,
                ),
            ).alias("target_map")
        )

        rows = [
            normalize_map_like(v)
            for v in spark_dataframe_to_arrow(result)["target_map"].to_pylist()
        ]
        assert rows == [{"a": "1", "b": "2"}, {"x": "3"}, None]


class TestSparkStructColumnToMap:

    def test_field_names_become_keys(
        self,
        spark,  # noqa: F811
        source_struct_to_map_field: Field,
        target_map_field: Field,
    ) -> None:
        import pyspark.sql.functions as F

        frame = spark.createDataFrame(
            [
                ({"a": 1, "b": 2, "c": 3},),
                ({"a": 4, "b": None, "c": 6},),
                (None,),
            ],
            schema="source_struct struct<a:bigint,b:bigint,c:bigint>",
        )

        result = frame.select(
            cast_spark_struct_column_to_map(
                F.col("source_struct"),
                CastOptions(
                    source_field=source_struct_to_map_field,
                    target_field=target_map_field,
                ),
            ).alias("target_map")
        )

        rows = [
            normalize_map_like(v)
            for v in result.toPandas()["target_map"].tolist()
        ]
        assert rows == [
            {"a": "1", "b": "2", "c": "3"},
            {"a": "4", "b": None, "c": "6"},
            None,
        ]
