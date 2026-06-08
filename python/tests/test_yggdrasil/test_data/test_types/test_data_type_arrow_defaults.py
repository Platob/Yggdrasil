"""``DataType.default_arrow_*`` and ``fill_arrow_array_nulls``.

Two related surfaces:

* ``default_arrow_scalar`` / ``default_arrow_array`` produce the
  fallback value/array used whenever a non-nullable target meets a
  null source.
* ``fill_arrow_array_nulls`` applies that fallback in-place over an
  existing array (Array or ChunkedArray).

The tests cover: empty arrays, fixed-size arrays, chunked arrays,
explicit ``default_scalar`` overrides, and the nullable-keep-null
short-circuit.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types.primitive import BooleanType, IntegerType, StringType


class TestDefaultArrowScalar(ArrowTestCase):

    def test_nullable_returns_typed_null(self) -> None:
        scalar = StringType().default_arrow_scalar(nullable=True)

        self.assertEqual(scalar.type, pa.string())
        self.assertIsNone(scalar.as_py())

    def test_non_nullable_returns_typed_default_value(self) -> None:
        cases = [
            (IntegerType(byte_size=8, signed=True), pa.int64()),
            (StringType(), pa.string()),
            (BooleanType(), pa.bool_()),
        ]
        for dtype, expected_type in cases:
            with self.subTest(dtype=dtype):
                scalar = dtype.default_arrow_scalar(nullable=False)
                self.assertEqual(scalar.type, expected_type)
                self.assertIsNotNone(scalar.as_py())


class TestDefaultArrowArray(ArrowTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.dtype = IntegerType(byte_size=8, signed=True)

    def test_no_size_returns_empty_array(self) -> None:
        out = self.dtype.default_arrow_array(nullable=True)

        self.assertIsInstance(out, pa.Array)
        self.assertEqual(len(out), 0)
        self.assertEqual(out.type, pa.int64())

    def test_nullable_size_fills_with_nulls(self) -> None:
        out = self.dtype.default_arrow_array(nullable=True, size=4)

        self.assertEqual(len(out), 4)
        self.assertEqual(out.type, pa.int64())
        self.assertEqual(out.to_pylist(), [None, None, None, None])

    def test_explicit_default_scalar_overrides(self) -> None:
        out = self.dtype.default_arrow_array(
            nullable=False,
            size=3,
            default_scalar=pa.scalar(7, type=pa.int64()),
        )

        self.assertIsInstance(out, pa.Array)
        self.assertEqual(out.type, pa.int64())
        self.assertEqual(out.to_pylist(), [7, 7, 7])

    def test_chunked_layout_with_explicit_default_scalar(self) -> None:
        out = self.dtype.default_arrow_array(
            nullable=False,
            chunks=[2, 0, 3],
            default_scalar=pa.scalar(11, type=pa.int64()),
        )

        self.assertIsInstance(out, pa.ChunkedArray)
        self.assertEqual(out.type, pa.int64())
        self.assertEqual([len(chunk) for chunk in out.chunks], [2, 0, 3])
        self.assertEqual(out.to_pylist(), [11, 11, 11, 11, 11])

    def test_empty_chunks_list_returns_empty_chunked_array(self) -> None:
        out = BooleanType().default_arrow_array(
            nullable=False,
            chunks=[],
            default_scalar=pa.scalar(False, type=pa.bool_()),
        )

        self.assertIsInstance(out, pa.ChunkedArray)
        self.assertEqual(len(out.chunks), 0)
        self.assertEqual(out.type, pa.bool_())


class TestFillArrowArrayNulls(ArrowTestCase):

    def test_nullable_target_keeps_nulls(self) -> None:
        dtype = IntegerType(byte_size=8, signed=True)
        arr = pa.array([1, None, 3], type=pa.int64())

        out = dtype.fill_arrow_array_nulls(arr, nullable=True)

        self.assertEqual(out.to_pylist(), [1, None, 3])

    def test_non_nullable_target_uses_explicit_default(self) -> None:
        dtype = IntegerType(byte_size=8, signed=True)
        arr = pa.array([1, None, 3], type=pa.int64())

        out = dtype.fill_arrow_array_nulls(
            arr,
            nullable=False,
            default_scalar=pa.scalar(0, type=pa.int64()),
        )

        self.assertEqual(out.to_pylist(), [1, 0, 3])

    def test_chunked_array_preserves_chunked_shape(self) -> None:
        dtype = StringType()
        arr = pa.chunked_array(
            [
                pa.array(["a", None], type=pa.string()),
                pa.array([None, "b"], type=pa.string()),
            ],
            type=pa.string(),
        )

        out = dtype.fill_arrow_array_nulls(
            arr,
            nullable=False,
            default_scalar=pa.scalar("", type=pa.string()),
        )

        self.assertIsInstance(out, pa.ChunkedArray)
        self.assertEqual(out.to_pylist(), ["a", "", "", "b"])
