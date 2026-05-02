"""``Field.from_pandas`` and ``Field.from_polars``.

Both engines expose Series, DataFrame, and dtype-class shapes.
:meth:`Field.from_pandas` and :meth:`Field.from_polars` resolve them
to a yggdrasil :class:`Field` — series-level inputs preserve the
column name; DataFrame-level inputs lift to a struct field with the
default name.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.constants import DEFAULT_FIELD_NAME
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import IntegerType, StringType
from yggdrasil.pandas.tests import PandasTestCase
from yggdrasil.polars.tests import PolarsTestCase


class TestFromPandas(PandasTestCase):

    def test_series_keeps_name_and_promotes_to_integer(self) -> None:
        series = self.pd.Series([1, 2, None], name="qty", dtype="Int64")

        out = Field.from_pandas(series)

        self.assertEqual(out.name, "qty")
        self.assertIsInstance(out.dtype, IntegerType)
        self.assertTrue(out.nullable)

    def test_dataframe_lifts_to_struct_with_default_name(self) -> None:
        df = self.pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        out = Field.from_pandas(df)

        self.assertEqual(out.name, DEFAULT_FIELD_NAME)
        self.assertIsInstance(out.dtype, StructType)
        self.assertFalse(out.nullable)
        self.assertEqual(out.arrow_type.field("a").type, pa.int64())
        # Pandas 3.0+ defaults strings to StringDtype → arrow large_string.
        b_type = out.arrow_type.field("b").type
        self.assertTrue(
            pa.types.is_string(b_type) or pa.types.is_large_string(b_type),
            f"Expected string/large_string, got {b_type!r}",
        )


class TestFromPolars(PolarsTestCase):

    def test_series_keeps_name_and_promotes_to_string(self) -> None:
        series = self.pl.Series("name", ["a", None, "b"])

        out = Field.from_polars(series)

        self.assertEqual(out.name, "name")
        self.assertIsInstance(out.dtype, StringType)
        self.assertTrue(out.nullable)

    def test_dataframe_lifts_to_struct_with_default_name(self) -> None:
        df = self.pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        out = Field.from_polars(df)

        self.assertEqual(out.name, DEFAULT_FIELD_NAME)
        self.assertIsInstance(out.dtype, StructType)
        self.assertFalse(out.nullable)
