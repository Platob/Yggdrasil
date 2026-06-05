"""Runtime-import regression guards on the engine-side fill helpers.

``Field.fill_polars_array_nulls`` and ``Field.fill_spark_column_nulls``
historically tripped a ``NameError`` when the engine module wasn't
already imported at the call site. Both should now work without any
import prelude — these tests fail loud if either regresses.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.data_field import Field
from yggdrasil.polars.tests import PolarsTestCase
from yggdrasil.spark.tests import SparkTestCase


class TestPolarsRuntimeImport(PolarsTestCase):

    def test_fill_polars_array_nulls_does_not_NameError(self) -> None:
        src = Field("value", pa.int64(), nullable=False, default=0)
        series = self.pl.Series("value", [1, None, 3])

        # Helper must not raise NameError on a cold import path.
        src.fill_polars_array_nulls(series)


class TestSparkRuntimeImport(SparkTestCase):

    def test_fill_spark_column_nulls_does_not_NameError(self) -> None:
        df = self.spark.createDataFrame([(1,), (None,), (3,)], ["value"])
        src = Field("value", pa.int64(), nullable=False, default=0)

        src.fill_spark_column_nulls(df["value"])
