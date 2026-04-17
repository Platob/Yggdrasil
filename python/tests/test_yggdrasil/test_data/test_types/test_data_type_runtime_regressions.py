from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.types.primitive import IntegerType
from yggdrasil.polars.tests import PolarsTestCase
from yggdrasil.spark.tests import SparkTestCase


class TestDataTypePolarsRegressions(PolarsTestCase):

    def test_fill_polars_array_nulls_raises_name_error_until_runtime_import_is_fixed(self):
        dtype = IntegerType(byte_size=8, signed=True)
        series = self.pl.Series("x", [1, None, 3])

        dtype.fill_polars_array_nulls(
            series,
            nullable=False,
            default_scalar=pa.scalar(0, type=pa.int64()),
        )


class TestDataTypeSparkRegressions(SparkTestCase):

    def test_fill_spark_column_nulls_raises_name_error_until_runtime_import_is_fixed(self):
        df = self.spark.createDataFrame([(1,), (None,), (3,)], ["x"])
        dtype = IntegerType(byte_size=8, signed=True)
        dtype.fill_spark_column_nulls(
            df["x"],
            nullable=False,
            default_scalar=pa.scalar(0, type=pa.int64()),
        )
