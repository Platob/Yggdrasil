"""Tests for the engine-side curation wrappers.

Each engine method (polars, pandas, spark) goes through the Arrow
bridge — so we mostly assert the round-trip preserves the inferred
types and Field names, and that the dispatcher falls back to as-is
for columns no Curator subclass claims.
"""

from __future__ import annotations

import unittest

from yggdrasil.data.types import Int64Type, StringType, TimestampType
from yggdrasil.io.curation import Curator, StringCurator
from yggdrasil.pandas.tests import PandasTestCase
from yggdrasil.polars.tests import PolarsTestCase
from yggdrasil.spark.tests import SparkTestCase


class TestPolarsCuration(PolarsTestCase):
    """Round-trip through polars: Series → Arrow → curate → Series."""

    def test_series_round_trip_preserves_name(self):
        series = self.pl.Series("id", ["1", "2", "3"])
        field, curated = StringCurator().curate_polars_series(series)
        self.assertEqual(field.name, "id")
        self.assertEqual(field.dtype, Int64Type())
        self.assertEqual(curated.name, "id")
        self.assertEqual(curated.to_list(), [1, 2, 3])

    def test_series_name_override(self):
        series = self.pl.Series("raw", ["1", "2"])
        field, curated = StringCurator().curate_polars_series(series, name="id")
        self.assertEqual(field.name, "id")
        self.assertEqual(curated.name, "id")

    def test_dataframe_mixed_columns(self):
        df = self.pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "when": [
                    "2024-01-01T10:00:00+02:00",
                    "2024-01-01T11:00:00-05:00",
                    None,
                ],
                "label": ["a", "b", "c"],
            }
        )
        schema, curated = Curator.curate_polars_dataframe(df)
        self.assertEqual(curated.columns, ["id", "when", "label"])
        self.assertEqual(curated.schema["id"], self.pl.Int64)
        self.assertEqual(
            curated.schema["when"],
            self.pl.Datetime(time_unit="us", time_zone="UTC"),
        )
        self.assertEqual(curated.schema["label"], self.pl.String)
        self.assertIsInstance(schema[1].dtype, TimestampType)

    def test_pretyped_columns_pass_through(self):
        df = self.pl.DataFrame(
            {"id": self.pl.Series("id", [1, 2, 3]), "name": ["a", "b", "c"]}
        )
        schema, curated = Curator.curate_polars_dataframe(df)
        self.assertEqual(curated.schema["id"], self.pl.Int64)
        self.assertEqual(curated.schema["name"], self.pl.String)


class TestPandasCuration(PandasTestCase):
    """Round-trip through pandas: Series → Arrow → curate → Series."""

    def test_series_round_trip_preserves_name(self):
        series = self.pd.Series(["1", "2", "3"], name="id")
        field, curated = StringCurator().curate_pandas_series(series)
        self.assertEqual(field.name, "id")
        self.assertEqual(field.dtype, Int64Type())
        self.assertEqual(curated.name, "id")
        self.assertEqual(curated.tolist(), [1, 2, 3])

    def test_unnamed_series_yields_default_name(self):
        series = self.pd.Series(["1", "2"])  # name is None
        field, _ = StringCurator().curate_pandas_series(series)
        self.assertEqual(field.name, "")

    def test_dataframe_mixed_columns(self):
        df = self.pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "amount": ["1.5", "2.5", "3.5"],
                "label": ["x", "y", "z"],
            }
        )
        schema, curated = Curator.curate_pandas_dataframe(df)
        self.assertEqual(list(curated.columns), ["id", "amount", "label"])
        self.assertEqual(curated["id"].tolist(), [1, 2, 3])
        self.assertEqual(curated["amount"].tolist(), [1.5, 2.5, 3.5])
        self.assertEqual(schema[0].dtype, Int64Type())
        self.assertEqual(schema[2].dtype, StringType())


class TestSparkCuration(SparkTestCase):
    """Spark round-trip via ``toPandas`` + ``createDataFrame``."""

    def test_dataframe_round_trip(self):
        df = self.spark.createDataFrame(
            [("1", "a"), ("2", "b"), ("3", "c")], ["id", "label"]
        )
        schema, curated = Curator.curate_spark_dataframe(df)
        names_to_types = {f.name: f.dataType for f in curated.schema.fields}
        # Spark's createDataFrame round-trip from pandas turns ints into
        # LongType; the inferred yggdrasil dtype is the load-bearing
        # assertion here.
        self.assertIn("id", names_to_types)
        self.assertIn("label", names_to_types)
        self.assertEqual(schema[0].dtype, Int64Type())
        self.assertEqual(schema[1].dtype, StringType())
        rows = curated.collect()
        self.assertEqual([r["id"] for r in rows], [1, 2, 3])
        self.assertEqual([r["label"] for r in rows], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
