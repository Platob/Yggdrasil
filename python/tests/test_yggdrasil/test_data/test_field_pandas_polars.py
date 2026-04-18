from __future__ import annotations

import pyarrow as pa
from yggdrasil.data.constants import DEFAULT_FIELD_NAME

from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import IntegerType, StringType
from yggdrasil.pandas.tests import PandasTestCase
from yggdrasil.polars.tests import PolarsTestCase


class TestFieldPandas(PandasTestCase):

    def test_from_pandas_series(self):
        series = self.pd.Series([1, 2, None], name="qty", dtype="Int64")

        out = Field.from_pandas(series)

        self.assertEqual(out.name, "qty")
        self.assertIsInstance(out.dtype, IntegerType)
        self.assertTrue(out.nullable)

    def test_from_pandas_dataframe(self):
        df = self.pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        out = Field.from_pandas(df)

        self.assertEqual(out.name, DEFAULT_FIELD_NAME)
        self.assertIsInstance(out.dtype, StructType)
        self.assertFalse(out.nullable)
        self.assertEqual(out.arrow_type.field("a").type, pa.int64())
        # pandas 3.0+ defaults strings to StringDtype -> arrow large_string.
        b_type = out.arrow_type.field("b").type
        self.assertTrue(
            pa.types.is_string(b_type) or pa.types.is_large_string(b_type),
            f"Expected string/large_string, got {b_type!r}",
        )


class TestFieldPolars(PolarsTestCase):

    def test_from_polars_series(self):
        series = self.pl.Series("name", ["a", None, "b"])

        out = Field.from_polars(series)

        self.assertEqual(out.name, "name")
        self.assertIsInstance(out.dtype, StringType)
        self.assertTrue(out.nullable)

    def test_from_polars_dataframe(self):
        df = self.pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        out = Field.from_polars(df)

        self.assertEqual(out.name, DEFAULT_FIELD_NAME)
        self.assertIsInstance(out.dtype, StructType)
        self.assertFalse(out.nullable)
