from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.data_field import Field
from yggdrasil.polars.tests import PolarsTestCase
from yggdrasil.spark.tests import SparkTestCase


class TestFieldPolarsRegressions(PolarsTestCase):

    def test_fill_polars_array_nulls_propagates_dtype_runtime_name_error(self):
        src = Field("value", pa.int64(), nullable=False, default=0)
        series = self.pl.Series("value", [1, None, 3])

        src.fill_polars_array_nulls(series)


class TestFieldSparkRegressions(SparkTestCase):

    def test_fill_spark_column_nulls_propagates_dtype_runtime_name_error(self):
        df = self.spark.createDataFrame([(1,), (None,), (3,)], ["value"])
        src = Field("value", pa.int64(), nullable=False, default=0)
        src.fill_spark_column_nulls(df["value"])
