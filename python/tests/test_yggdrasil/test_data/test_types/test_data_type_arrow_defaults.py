from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types.primitive import BooleanType, IntegerType, StringType


class TestDataTypeArrowDefaults(ArrowTestCase):

    def test_default_arrow_scalar_nullable_true_is_null_scalar(self):
        pa = self.pa
        dtype = StringType()

        scalar = dtype.default_arrow_scalar(nullable=True)

        self.assertEqual(scalar.type, pa.string())
        self.assertIsNone(scalar.as_py())

    def test_default_arrow_array_empty_without_chunks(self):
        pa = self.pa
        dtype = IntegerType(byte_size=8, signed=True)

        out = dtype.default_arrow_array(nullable=True)

        self.assertIsInstance(out, pa.Array)
        self.assertEqual(len(out), 0)
        self.assertEqual(out.type, pa.int64())

    def test_default_arrow_array_uses_explicit_default_scalar(self):
        pa = self.pa
        dtype = IntegerType(byte_size=8, signed=True)

        out = dtype.default_arrow_array(
            nullable=False,
            size=3,
            default_scalar=pa.scalar(7, type=pa.int64()),
        )

        self.assertIsInstance(out, pa.Array)
        self.assertEqual(out.type, pa.int64())
        self.assertEqual(out.to_pylist(), [7, 7, 7])

    def test_default_arrow_array_chunked_uses_explicit_default_scalar(self):
        pa = self.pa
        dtype = IntegerType(byte_size=8, signed=True)

        out = dtype.default_arrow_array(
            nullable=False,
            chunks=[2, 0, 3],
            default_scalar=pa.scalar(11, type=pa.int64()),
        )

        self.assertIsInstance(out, pa.ChunkedArray)
        self.assertEqual(out.type, pa.int64())
        self.assertEqual([len(chunk) for chunk in out.chunks], [2, 0, 3])
        self.assertEqual(out.to_pylist(), [11, 11, 11, 11, 11])

    def test_fill_arrow_array_nulls_nullable_true_keeps_nulls(self):
        pa = self.pa
        dtype = IntegerType(byte_size=8, signed=True)
        arr = pa.array([1, None, 3], type=pa.int64())

        out = dtype.fill_arrow_array_nulls(arr, nullable=True)

        self.assertEqual(out.to_pylist(), [1, None, 3])

    def test_fill_arrow_array_nulls_nullable_false_uses_explicit_default(self):
        pa = self.pa
        dtype = IntegerType(byte_size=8, signed=True)
        arr = pa.array([1, None, 3], type=pa.int64())

        out = dtype.fill_arrow_array_nulls(
            arr,
            nullable=False,
            default_scalar=pa.scalar(0, type=pa.int64()),
        )

        self.assertEqual(out.to_pylist(), [1, 0, 3])

    def test_fill_arrow_chunked_array_nulls_uses_explicit_default(self):
        pa = self.pa
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

    def test_default_arrow_array_zero_size_with_explicit_chunks_empty_list(self):
        pa = self.pa
        dtype = BooleanType()

        out = dtype.default_arrow_array(
            nullable=False,
            chunks=[],
            default_scalar=pa.scalar(False, type=pa.bool_()),
        )

        self.assertIsInstance(out, pa.ChunkedArray)
        self.assertEqual(len(out.chunks), 0)
        self.assertEqual(out.type, pa.bool_())

    def test_default_arrow_scalar_nullable_false_returns_typed_value(self):
        pa = self.pa
        for dtype, expected_type in [
            (IntegerType(byte_size=8, signed=True), pa.int64()),
            (StringType(), pa.string()),
            (BooleanType(), pa.bool_()),
        ]:
            with self.subTest(dtype=dtype):
                scalar = dtype.default_arrow_scalar(nullable=False)
                self.assertEqual(scalar.type, expected_type)
                self.assertIsNotNone(scalar.as_py())

    def test_default_arrow_array_nullable_true_with_nonzero_size(self):
        pa = self.pa
        dtype = IntegerType(byte_size=8, signed=True)

        out = dtype.default_arrow_array(nullable=True, size=4)

        self.assertEqual(len(out), 4)
        self.assertEqual(out.type, pa.int64())
        self.assertEqual(out.to_pylist(), [None, None, None, None])
