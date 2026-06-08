"""Runtime-import regression guards for ``DataType`` engine helpers.

Specific helpers (`fill_polars_array_nulls`, `fill_spark_column_nulls`)
historically blew up with ``NameError`` when an engine module wasn't
imported at the call site. Both should now work without any import
prelude — these tests fail loud if either regresses.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.types.primitive import IntegerType
from yggdrasil.polars.tests import PolarsTestCase
from yggdrasil.spark.tests import SparkTestCase


class TestPolarsRuntimeImport(PolarsTestCase):

    def test_fill_polars_array_nulls_does_not_NameError(self) -> None:
        dtype = IntegerType(byte_size=8, signed=True)
        series = self.pl.Series("x", [1, None, 3])

        # Must not raise NameError — the helper is responsible for its own imports.
        dtype.fill_polars_array_nulls(
            series,
            nullable=False,
            default_scalar=pa.scalar(0, type=pa.int64()),
        )


class TestSparkRuntimeImport(SparkTestCase):

    def test_fill_spark_column_nulls_does_not_NameError(self) -> None:
        df = self.spark.createDataFrame([(1,), (None,), (3,)], ["x"])
        dtype = IntegerType(byte_size=8, signed=True)

        dtype.fill_spark_column_nulls(
            df["x"],
            nullable=False,
            default_scalar=pa.scalar(0, type=pa.int64()),
        )
