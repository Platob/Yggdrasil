"""``DataType.from_pandas`` and ``DataType.from_polars`` dispatch.

Both engines expose dtypes, scalars, and frame objects; the
``from_*`` constructors take any of those shapes and resolve to a
yggdrasil :class:`DataType`. The frame variants flatten to
:class:`StructType`; everything else lands on the matching
primitive subclass.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import (
    BooleanType,
    DurationType,
    FloatingPointType,
    IntegerType,
    StringType,
    TimestampType,
)
from yggdrasil.pandas.tests import PandasTestCase
from yggdrasil.polars.tests import PolarsTestCase


class TestFromPandas(PandasTestCase):

    def test_nullable_int_dtype_resolves_to_integer(self) -> None:
        series = self.pd.Series([1, 2, None], dtype="Int32")

        dtype = DataType.from_pandas(series.dtype)

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.to_arrow(), pa.int32())

    def test_float64_series_resolves_to_floating_point(self) -> None:
        series = self.pd.Series([1.5, 2.5, 3.5])

        dtype = DataType.from_pandas(series.dtype)

        self.assertIsInstance(dtype, FloatingPointType)

    def test_object_series_of_strings_resolves_to_string(self) -> None:
        series = self.pd.Series(["a", "b", "c"])

        self.assertIsInstance(DataType.from_pandas(series.dtype), StringType)

    def test_bool_series_resolves_to_boolean(self) -> None:
        series = self.pd.Series([True, False, True])

        self.assertIsInstance(DataType.from_pandas(series.dtype), BooleanType)

    def test_timestamp_value_carries_unit_and_tz(self) -> None:
        value = self.pd.Timestamp("2025-01-01T12:00:00Z")

        dtype = DataType.from_pandas(value)

        self.assertIsInstance(dtype, TimestampType)
        self.assertEqual(dtype.unit, "ns")
        self.assertIsNotNone(dtype.tz)

    def test_timedelta_value_carries_unit(self) -> None:
        value = self.pd.Timedelta("1 day")

        dtype = DataType.from_pandas(value)

        self.assertIsInstance(dtype, DurationType)
        self.assertEqual(dtype.unit, "ns")

    def test_dataframe_resolves_to_struct(self) -> None:
        df = self.pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        dtype = DataType.from_pandas(df)

        self.assertIsInstance(dtype, StructType)
        arrow_struct = dtype.to_arrow()
        self.assertEqual(arrow_struct.field("a").type, pa.int64())
        # Modern pandas uses large_string for string columns; older keeps
        # plain string. Both are acceptable.
        b_type = arrow_struct.field("b").type
        self.assertTrue(
            pa.types.is_string(b_type) or pa.types.is_large_string(b_type),
            f"Expected string/large_string, got {b_type!r}",
        )


class TestFromPolars(PolarsTestCase):

    def test_int_series_resolves_to_integer(self) -> None:
        s = self.pl.Series("a", [1, 2, 3], dtype=self.pl.Int64)

        dtype = DataType.from_polars(s)

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.to_arrow(), pa.int64())

    def test_float_series_resolves_to_floating_point(self) -> None:
        s = self.pl.Series("f", [1.0, 2.0], dtype=self.pl.Float64)

        dtype = DataType.from_polars(s)

        self.assertIsInstance(dtype, FloatingPointType)
        self.assertEqual(dtype.to_arrow(), pa.float64())

    def test_string_series_resolves_to_string(self) -> None:
        s = self.pl.Series("s", ["a", "b"], dtype=self.pl.Utf8)

        self.assertIsInstance(DataType.from_polars(s), StringType)

    def test_dtype_class_resolves_directly(self) -> None:
        dtype = DataType.from_polars(self.pl.Int32)

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.to_arrow(), pa.int32())

    def test_dataframe_resolves_to_struct(self) -> None:
        df = self.pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        dtype = DataType.from_polars(df)

        self.assertIsInstance(dtype, StructType)
        self.assertEqual(
            dtype.to_arrow(),
            pa.struct(
                [
                    pa.field("a", pa.int64(), nullable=True),
                    pa.field("b", pa.string(), nullable=True),
                ]
            ),
        )
