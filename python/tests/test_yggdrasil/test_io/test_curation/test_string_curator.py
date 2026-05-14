"""Tests for :class:`yggdrasil.io.curation.StringCurator`.

Covers the auto-typing rule order, null normalization, the
"timestamps uniformize to a single tz" guarantee, and the dispatcher
on :meth:`Curator.pick`.
"""

from __future__ import annotations

import datetime as dt
import unittest

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.enums.timezone import Timezone
from yggdrasil.data.types import (
    BooleanType,
    DateType,
    Float64Type,
    Int64Type,
    NullType,
    StringType,
    TimestampType,
    TimeType,
)
from yggdrasil.io.curation import Curator, CurationResult, StringCurator


class TestStringCuratorTypeInference(ArrowTestCase):
    """Each trial picks the most specific type that absorbs every cell."""

    def setUp(self) -> None:
        super().setUp()
        self.curator = StringCurator()

    def _curate(self, values, **overrides):
        curator = StringCurator(**overrides) if overrides else self.curator
        arr = self.pa.array(values, type=self.pa.string())
        return curator.curate(arr)

    def test_integer_column(self):
        result = self._curate(["1", "2", "3"])
        self.assertEqual(result.dtype, Int64Type())
        self.assertEqual(result.array.to_pylist(), [1, 2, 3])

    def test_float_column(self):
        result = self._curate(["1.5", "2.5", "3"])
        self.assertEqual(result.dtype, Float64Type())
        self.assertEqual(result.array.to_pylist(), [1.5, 2.5, 3.0])

    def test_boolean_mixed_case_and_aliases(self):
        result = self._curate(["true", "False", "YES", "no", None])
        self.assertEqual(result.dtype, BooleanType())
        self.assertEqual(result.array.to_pylist(), [True, False, True, False, None])

    def test_string_fallback_when_nothing_matches(self):
        result = self._curate(["hello", "world", "1"])
        self.assertEqual(result.dtype, StringType())
        self.assertEqual(result.array.to_pylist(), ["hello", "world", "1"])

    def test_all_null_collapses_to_null_type(self):
        result = self._curate(["", "NA", "n/a", None])
        self.assertEqual(result.dtype, NullType())
        self.assertEqual(result.array.null_count, 4)

    def test_iso_date_column(self):
        result = self._curate(["2024-01-15", "2024-02-20"])
        self.assertEqual(result.dtype, DateType(byte_size=4, unit="d"))
        self.assertEqual(
            result.array.to_pylist(),
            [dt.date(2024, 1, 15), dt.date(2024, 2, 20)],
        )

    def test_day_first_slash_date_matches_excel_eu(self):
        # "01/02/2024" should parse as 1 Feb (day-first), matching the
        # polars catalogue in ``primitive/temporal.py``.
        result = self._curate(["01/02/2024", "15/02/2024"])
        self.assertEqual(result.dtype, DateType(byte_size=4, unit="d"))
        self.assertEqual(
            result.array.to_pylist(),
            [dt.date(2024, 2, 1), dt.date(2024, 2, 15)],
        )

    def test_time_column(self):
        result = self._curate(["10:30:45", "11:00:00"])
        self.assertEqual(result.dtype, TimeType(byte_size=8, unit="us"))
        self.assertEqual(
            result.array.to_pylist(),
            [dt.time(10, 30, 45), dt.time(11, 0, 0)],
        )

    def test_naive_timestamp_column(self):
        result = self._curate(["2024-01-15T10:30:00", "2024-02-20T11:00:00"])
        self.assertEqual(
            result.dtype,
            TimestampType(unit="us", tz=Timezone.NAIVE),
        )
        self.assertEqual(
            result.array.to_pylist(),
            [dt.datetime(2024, 1, 15, 10, 30), dt.datetime(2024, 2, 20, 11, 0)],
        )

    def test_aware_timestamps_uniformize_to_utc(self):
        # Two different source offsets — output is a single
        # ``timestamp[us, tz=UTC]`` and both cells point at the same
        # instant the string originally encoded.
        result = self._curate(
            [
                "2024-01-15T10:30:00+02:00",  # 08:30 UTC
                "2024-01-15T11:30:00-05:00",  # 16:30 UTC
            ]
        )
        self.assertEqual(result.dtype.tz, Timezone.from_("UTC"))
        self.assertEqual(str(result.array.type), "timestamp[us, tz=UTC]")
        instants = result.array.to_pylist()
        self.assertEqual(
            instants[0],
            dt.datetime(2024, 1, 15, 8, 30, tzinfo=dt.timezone.utc),
        )
        self.assertEqual(
            instants[1],
            dt.datetime(2024, 1, 15, 16, 30, tzinfo=dt.timezone.utc),
        )

    def test_aware_timestamps_uniformize_to_custom_tz(self):
        result = self._curate(["2024-01-15T10:30:00+02:00"], target_tz="Europe/Paris")
        self.assertEqual(str(result.array.type), "timestamp[us, tz=Europe/Paris]")

    def test_assume_naive_tz_stamps_then_normalizes(self):
        # Naive Paris timestamps → uniformized to UTC.
        # 10:30 Paris (winter) = 09:30 UTC; 12:00 Paris (summer) = 10:00 UTC.
        result = self._curate(
            ["2024-01-15 10:30:00", "2024-06-15 12:00:00"],
            target_tz="UTC",
            assume_naive_tz="Europe/Paris",
        )
        self.assertEqual(str(result.array.type), "timestamp[us, tz=UTC]")
        self.assertEqual(
            result.array.to_pylist(),
            [
                dt.datetime(2024, 1, 15, 9, 30, tzinfo=dt.timezone.utc),
                dt.datetime(2024, 6, 15, 10, 0, tzinfo=dt.timezone.utc),
            ],
        )


class TestStringCuratorCleaning(ArrowTestCase):
    """Whitespace + null-token normalization runs before any trial."""

    def test_whitespace_is_trimmed_before_inference(self):
        curator = StringCurator()
        result = curator.curate(self.pa.array(["  42  ", "\t7", "9\n"]))
        self.assertEqual(result.dtype, Int64Type())
        self.assertEqual(result.array.to_pylist(), [42, 7, 9])

    def test_null_tokens_replace_with_real_nulls(self):
        curator = StringCurator()
        result = curator.curate(self.pa.array(["1", "NA", "n/a", "3"]))
        self.assertEqual(result.dtype, Int64Type())
        self.assertEqual(result.array.to_pylist(), [1, None, None, 3])

    def test_disable_null_tokens_keeps_strings(self):
        # An empty ``null_tokens`` set means "NA" stays literal, which
        # breaks integer parsing → falls through to StringType.
        curator = StringCurator(null_tokens=frozenset())
        result = curator.curate(self.pa.array(["1", "NA", "3"]))
        self.assertEqual(result.dtype, StringType())

    def test_disable_trim_preserves_whitespace(self):
        # "  42  " is not parseable as int unless we trim — disabling
        # trim should send the column to string fallback.
        curator = StringCurator(trim=False)
        result = curator.curate(self.pa.array(["  42  ", "  7"]))
        self.assertEqual(result.dtype, StringType())


class TestStringCuratorArrayShapes(ArrowTestCase):
    """The curator accepts plain Array, ChunkedArray, and string variants."""

    def test_chunked_array_round_trip_preserves_shape(self):
        curator = StringCurator()
        arr = self.pa.chunked_array(
            [self.pa.array(["1", "2"]), self.pa.array(["3", "4"])]
        )
        result = curator.curate(arr)
        self.assertIsInstance(result.array, self.pa.ChunkedArray)
        self.assertEqual(result.array.to_pylist(), [1, 2, 3, 4])
        self.assertEqual(result.dtype, Int64Type())

    def test_large_string_is_handled(self):
        curator = StringCurator()
        arr = self.pa.array(["1.5", "2.5"], type=self.pa.large_string())
        result = curator.curate(arr)
        self.assertEqual(result.dtype, Float64Type())

    def test_curate_unpacks_to_array_and_dtype(self):
        curator = StringCurator()
        arr, dtype = curator.curate(self.pa.array(["1", "2", "3"]))
        self.assertEqual(dtype, Int64Type())
        self.assertEqual(arr.to_pylist(), [1, 2, 3])


class TestCuratorDispatch(ArrowTestCase):
    """``Curator.pick`` routes to the subclass whose ``handles`` matches."""

    def test_pick_returns_string_curator_for_strings(self):
        curator = Curator.pick(self.pa.array(["1", "2"]))
        self.assertIsInstance(curator, StringCurator)

    def test_pick_raises_when_no_subclass_matches(self):
        # Binary arrays have no Curator subclass; integers / floats /
        # nested types do, so use ``pa.binary()`` to exercise the
        # "nothing handles it" path.
        with self.assertRaisesRegex(TypeError, "No Curator subclass handles"):
            Curator.pick(self.pa.array([b"x", b"y"]))

    def test_curator_call_rejects_wrong_dtype(self):
        with self.assertRaisesRegex(TypeError, "cannot curate Arrow type"):
            StringCurator()(self.pa.array([b"x", b"y"]))

    def test_infer_only_returns_dtype(self):
        curator = StringCurator()
        dtype = curator.infer(self.pa.array(["1.5", "2.5"]))
        self.assertEqual(dtype, Float64Type())

    def test_curation_result_is_a_namedtuple_like(self):
        result = StringCurator().curate(self.pa.array(["1"]))
        self.assertIsInstance(result, CurationResult)


class TestCurateArrowArray(ArrowTestCase):
    """``curate_arrow_array`` wraps the inferred dtype into a Field."""

    def test_returns_field_and_curated_array(self):
        from yggdrasil.data.data_field import Field

        field, curated = StringCurator().curate_arrow_array(
            self.pa.array(["1", "2", "3"]), name="id"
        )
        self.assertIsInstance(field, Field)
        self.assertEqual(field.name, "id")
        self.assertEqual(field.dtype, Int64Type())
        self.assertEqual(curated.to_pylist(), [1, 2, 3])

    def test_default_name_is_blank(self):
        field, _ = StringCurator().curate_arrow_array(self.pa.array(["a", "b"]))
        self.assertEqual(field.name, "")

    def test_nullable_flag_is_forwarded(self):
        field, _ = StringCurator().curate_arrow_array(
            self.pa.array(["1"]), name="id", nullable=False
        )
        self.assertFalse(field.nullable)

    def test_wrong_dtype_raises_via_call(self):
        with self.assertRaisesRegex(TypeError, "cannot curate Arrow type"):
            StringCurator().curate_arrow_array(self.pa.array([b"x", b"y"]))


class TestCurateArrowTabular(ArrowTestCase):
    """``Curator.curate_arrow_tabular`` routes per column."""

    def _mixed_table(self):
        return self.pa.table(
            {
                "id": ["1", "2", "3"],
                "amount": ["1.5", "2.5", "3.5"],
                "when": [
                    "2024-01-15T10:00:00+02:00",
                    "2024-01-15T11:00:00-05:00",
                    None,
                ],
                "flag": ["true", "false", "yes"],
                "label": ["a", "b", "c"],
            }
        )

    def test_table_returns_schema_and_typed_table(self):
        from yggdrasil.data.schema import StructField

        schema, curated = Curator.curate_arrow_tabular(self._mixed_table())
        self.assertIsInstance(schema, StructField)
        self.assertIsInstance(curated, self.pa.Table)

        types = {name: curated.schema.field(name).type for name in curated.schema.names}
        self.assertEqual(types["id"], self.pa.int64())
        self.assertEqual(types["amount"], self.pa.float64())
        self.assertEqual(str(types["when"]), "timestamp[us, tz=UTC]")
        self.assertEqual(types["flag"], self.pa.bool_())
        self.assertEqual(types["label"], self.pa.string())

    def test_record_batch_round_trips_as_record_batch(self):
        batch = self.pa.record_batch({"id": ["1", "2"], "label": ["a", "b"]})
        schema, curated = Curator.curate_arrow_tabular(batch)
        self.assertIsInstance(curated, self.pa.RecordBatch)
        self.assertEqual(curated.column(0).to_pylist(), [1, 2])
        self.assertEqual(schema[0].dtype, Int64Type())

    def test_pretyped_int_columns_get_shrunk(self):
        # IntegerCurator picks up pre-typed ints and downcasts to the
        # narrowest width that holds the range (int64([1..3]) → int8).
        table = self.pa.table(
            {"id": self.pa.array([1, 2, 3]), "label": ["a", "b", "c"]}
        )
        schema, curated = Curator.curate_arrow_tabular(table)
        self.assertEqual(curated.schema.field("id").type, self.pa.uint8())
        self.assertEqual(curated.schema.field("label").type, self.pa.string())

    def test_unhandled_columns_pass_through(self):
        # Binary has no Curator subclass — should land in the output
        # with its Field carrying the original dtype.
        table = self.pa.table(
            {"blob": self.pa.array([b"x", b"y"]), "label": ["a", "b"]}
        )
        schema, curated = Curator.curate_arrow_tabular(table)
        self.assertEqual(curated.schema.field("blob").type, self.pa.binary())

    def test_rejects_non_arrow_tabular(self):
        with self.assertRaisesRegex(TypeError, "expects a pyarrow Table"):
            Curator.curate_arrow_tabular({"id": [1, 2, 3]})  # plain dict

    def test_curator_kwargs_are_forwarded(self):
        # ``parse_temporal=False`` keeps timestamp strings as strings.
        schema, curated = Curator.curate_arrow_tabular(
            self.pa.table({"when": ["2024-01-15T10:00:00"]}),
            parse_temporal=False,
        )
        self.assertEqual(curated.schema.field("when").type, self.pa.string())


if __name__ == "__main__":
    unittest.main()
