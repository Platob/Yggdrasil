"""Spark-side casts for :class:`ArrayType`.

Live SparkSession is required — the whole module skips if pyspark
isn't installed. Coverage:

* Dtype probes — ``to_spark`` / ``from_spark_type`` / ``handles_spark_type``.
* Compute — ``cast_spark_list_column`` against a real DataFrame.
* Short-circuit — ``target_field=None`` is a passthrough.
* Rejections — non-array source raises.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested.array import (
    ArrayType,
    cast_spark_list_column,
)
from yggdrasil.data.types.primitive import IntegerType

pytest.importorskip("pyspark")

from yggdrasil.spark.tests import spark  # noqa: E402,F401
from yggdrasil.spark.cast import spark_dataframe_to_arrow  # noqa: E402


# ---------------------------------------------------------------------------
# Dtype probes
# ---------------------------------------------------------------------------


class TestSparkDtype:

    def test_to_spark_emits_array_type(self, spark) -> None:  # noqa: F811
        from pyspark.sql.types import ArrayType as SparkArrayType

        dtype = ArrayType(
            item_field=Field(
                name="item",
                dtype=IntegerType(byte_size=8, signed=True),
                nullable=True,
            ),
        )

        assert isinstance(dtype.to_spark(), SparkArrayType)

    def test_from_spark_type_preserves_item(self, spark) -> None:  # noqa: F811
        from pyspark.sql.types import ArrayType as SparkArrayType, LongType

        rebuilt = ArrayType.from_spark_type(
            SparkArrayType(LongType(), containsNull=False)
        )

        assert isinstance(rebuilt, ArrayType)
        assert rebuilt.item_field.dtype.type_id == DataTypeId.INTEGER
        assert rebuilt.item_field.nullable is False

    def test_handles_spark_type_only_for_array(self, spark) -> None:  # noqa: F811
        from pyspark.sql.types import ArrayType as SparkArrayType, LongType

        assert ArrayType.handles_spark_type(SparkArrayType(LongType())) is True
        assert ArrayType.handles_spark_type(LongType()) is False

    def test_from_spark_type_rejects_non_array(self, spark) -> None:  # noqa: F811
        from pyspark.sql.types import LongType

        with pytest.raises(TypeError, match="Unsupported Spark data type"):
            ArrayType.from_spark_type(LongType())


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


class TestSparkCast:

    def test_cast_changes_item_dtype(
        self,
        spark,  # noqa: F811
        source_array_field: Field,
        target_array_field: Field,
    ) -> None:
        import pyspark.sql.functions as F

        frame = spark.createDataFrame(
            [([1, 2],), ([3, None],), (None,)],
            schema="source_array array<bigint>",
        )

        result = frame.select(
            cast_spark_list_column(
                F.col("source_array"),
                CastOptions(
                    source_field=source_array_field,
                    target_field=target_array_field,
                ),
            ).alias("target_array")
        )

        rows = spark_dataframe_to_arrow(result)["target_array"].to_pylist()
        assert rows == [["1", "2"], ["3", None], None]

    def test_target_none_returns_input_column(
        self,
        spark,  # noqa: F811
        source_array_field: Field,
    ) -> None:
        import pyspark.sql.functions as F

        column = F.col("source_array")

        out = cast_spark_list_column(
            column,
            CastOptions(source_field=source_array_field, target_field=None),
        )

        assert out is column

    def test_non_array_source_raises(
        self,
        spark,  # noqa: F811
        source_map_field: Field,
        target_array_field: Field,
    ) -> None:
        import pyspark.sql.functions as F

        with pytest.raises(TypeError, match="Cannot cast"):
            cast_spark_list_column(
                F.col("anything"),
                CastOptions(
                    source_field=source_map_field,
                    target_field=target_array_field,
                ),
            )
