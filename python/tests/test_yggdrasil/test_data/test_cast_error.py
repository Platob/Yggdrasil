"""Tests for :class:`yggdrasil.exceptions.CastError` and the
permissive-by-default behaviour of :class:`CastOptions`.

What changed
------------
* ``CastOptions.safe`` defaults to ``False`` — the wider data plane
  (CSV-from-web, JSON-from-API, partial joins) is messy enough that a
  strict cast is almost never what callers want. Strict semantics
  (parse / overflow errors → raise) are opt-in via ``safe=True``.
* Cast failures inside the per-column rebuild are wrapped in
  :class:`CastError`. The error message names both the source and the
  target field so a multi-column write surfaces *which* column blew up
  instead of a bare ``ArrowInvalid`` deep in the stack.
* ``CastError`` subclasses :class:`pyarrow.ArrowInvalid` so existing
  ``except pa.ArrowInvalid`` handlers keep catching the same failure
  class.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.exceptions import CastError


class TestSafeDefaultIsFalse(ArrowTestCase):
    """``CastOptions.safe`` defaults to ``False`` — permissive mode."""

    def test_default_is_false(self) -> None:
        self.assertFalse(CastOptions().safe)

    def test_explicit_true_still_works(self) -> None:
        self.assertTrue(CastOptions(safe=True).safe)


class TestBadJsonRowsNullOutByDefault(ArrowTestCase):
    """Reproduces the user-reported failure: a string column being cast
    to a JSON-shaped nested target used to fail the whole batch on the
    first malformed row. The default behaviour now nulls out bad rows
    so the rest of the batch lands."""

    def _schemas(self):
        pa = self.pa
        src = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("payload", pa.string()),
        ])
        tgt = pa.schema([
            pa.field("id", pa.int64()),
            pa.field(
                "payload",
                pa.list_(
                    pa.struct(
                        [pa.field("a", pa.int64()), pa.field("b", pa.string())]
                    )
                ),
            ),
        ])
        return src, tgt

    def test_default_permissive_nulls_bad_rows(self) -> None:
        pa = self.pa
        src, tgt = self._schemas()
        tbl = pa.table(
            {"id": [1, 2, 3], "payload": ["pypsa", '[{"a": 1, "b": "x"}]', "nope"]},
            schema=src,
        )
        opts = CastOptions(
            source_field=Field.from_arrow(src),
            target_field=Field.from_arrow(tgt),
        )
        out = opts.cast_arrow_tabular(tbl)

        self.assertEqual(out.num_rows, 3)
        self.assertEqual(out.column("id").to_pylist(), [1, 2, 3])
        # Bad rows null out, the good row keeps its structured value.
        rows = out.column("payload").to_pylist()
        self.assertIsNone(rows[0])
        self.assertEqual(rows[1], [{"a": 1, "b": "x"}])
        self.assertIsNone(rows[2])

    def test_strict_safe_true_raises_cast_error(self) -> None:
        pa = self.pa
        src, tgt = self._schemas()
        tbl = pa.table(
            {"id": [1], "payload": ["pypsa"]},
            schema=src,
        )
        opts = CastOptions(
            source_field=Field.from_arrow(src),
            target_field=Field.from_arrow(tgt),
            safe=True,
        )
        with self.assertRaises(CastError) as ctx:
            opts.cast_arrow_tabular(tbl)

        err = ctx.exception
        # Source / target fields are accessible programmatically.
        self.assertIsNotNone(err.source_field)
        self.assertIsNotNone(err.target_field)
        self.assertEqual(err.source_field.name, "payload")
        self.assertEqual(err.target_field.name, "payload")
        # Message mentions the failing column on both ends.
        msg = str(err)
        self.assertIn("payload: string", msg)
        self.assertIn("payload: list<", msg)
        self.assertIn("pypsa", msg)


class TestCastErrorBackCompat(ArrowTestCase):
    """``CastError`` still flows through ``except pa.ArrowInvalid``."""

    def test_caught_by_pyarrow_arrow_invalid(self) -> None:
        pa = self.pa
        src = pa.schema([pa.field("payload", pa.string())])
        tgt = pa.schema(
            [
                pa.field(
                    "payload",
                    pa.list_(pa.field("item", pa.int64())),
                )
            ]
        )
        tbl = pa.table({"payload": ["not-json"]}, schema=src)
        opts = CastOptions(
            source_field=Field.from_arrow(src),
            target_field=Field.from_arrow(tgt),
            safe=True,
        )

        # Existing callers catch pa.ArrowInvalid — the CastError
        # subclass should still land in that net.
        with self.assertRaises(pa.ArrowInvalid):
            opts.cast_arrow_tabular(tbl)

    def test_is_subclass_of_value_error(self) -> None:
        # ArrowInvalid is itself a ValueError, so CastError inherits
        # that too — useful for callers that catch ValueError at a
        # generic boundary.
        err = CastError("bad cast")
        self.assertIsInstance(err, ValueError)
        self.assertIsInstance(err, pa.ArrowInvalid)


class TestCastErrorMessageShape(ArrowTestCase):
    """``CastError`` renders source / target on a single line for logs."""

    def test_constructed_message_includes_both_sides(self) -> None:
        src = Field.from_arrow(self.pa.field("price", self.pa.string()))
        tgt = Field.from_arrow(self.pa.field("price", self.pa.float64()))
        err = CastError("bad value 'oops'", source_field=src, target_field=tgt)
        msg = str(err)
        self.assertIn("price: string", msg)
        self.assertIn("price: double", msg)
        self.assertIn("bad value 'oops'", msg)
        # Single-line — nested types must stay readable in log lines.
        self.assertNotIn("\n", msg)

    def test_missing_fields_render_question_mark(self) -> None:
        err = CastError("dunno")
        msg = str(err)
        self.assertIn("? -> ?", msg)
        self.assertIn("dunno", msg)

    def test_keeps_original_exception(self) -> None:
        original = ValueError("inner")
        err = CastError("outer", original=original)
        self.assertIs(err.original, original)
