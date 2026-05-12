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
            source=Field.from_arrow(src),
            target=Field.from_arrow(tgt),
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
            source=Field.from_arrow(src),
            target=Field.from_arrow(tgt),
            safe=True,
        )
        with self.assertRaises(CastError) as ctx:
            opts.cast_arrow_tabular(tbl)

        err = ctx.exception
        # Source / target fields are accessible programmatically.
        self.assertIsNotNone(err.source)
        self.assertIsNotNone(err.target)
        self.assertEqual(err.source.name, "payload")
        self.assertEqual(err.target.name, "payload")
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
            source=Field.from_arrow(src),
            target=Field.from_arrow(tgt),
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
        err = CastError("bad value 'oops'", source=src, target=tgt)
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


class TestNestedCastErrorPropagation(ArrowTestCase):
    """A leaf failure deep inside a nested target binds the *leaf* field.

    The wrap on :meth:`Field.cast_arrow_array` is the single seam — every
    nested rebuild (struct → struct children, list → struct by index,
    map → struct via map_lookup, tabular per-column) funnels through it
    before hitting the dtype-level cast. So the leaf cast wraps first
    with its own field, and the outer call sees a ``CastError`` that it
    re-raises unchanged (instead of clobbering the context with the
    outer wrapper field).
    """

    def test_struct_child_leaf_failure(self) -> None:
        # struct → struct with an inner string→int that fails.
        pa = self.pa
        src = pa.schema([
            pa.field("row", pa.struct([
                pa.field("kept", pa.string()),
                pa.field("bad",  pa.string()),
            ])),
        ])
        tgt = pa.schema([
            pa.field("row", pa.struct([
                pa.field("kept", pa.string()),
                pa.field("bad",  pa.int64()),
            ])),
        ])
        tbl = pa.table(
            {"row": [{"kept": "ok", "bad": "not-a-number"}]}, schema=src,
        )
        opts = CastOptions(
            source=Field.from_arrow(src),
            target=Field.from_arrow(tgt),
            safe=True,
        )
        with self.assertRaises(CastError) as ctx:
            opts.cast_arrow_tabular(tbl)

        # The leaf is what got bound — not the outer ``row`` field.
        self.assertEqual(ctx.exception.target.name, "bad")
        self.assertIn("bad: string", str(ctx.exception))
        self.assertIn("bad: int64", str(ctx.exception))

    def test_list_of_struct_leaf_failure(self) -> None:
        pa = self.pa
        src = pa.schema([
            pa.field("payload", pa.list_(pa.struct([
                pa.field("count", pa.string()),
            ]))),
        ])
        tgt = pa.schema([
            pa.field("payload", pa.list_(pa.struct([
                pa.field("count", pa.int64()),
            ]))),
        ])
        tbl = pa.table({"payload": [[{"count": "bad"}]]}, schema=src)
        opts = CastOptions(
            source=Field.from_arrow(src),
            target=Field.from_arrow(tgt),
            safe=True,
        )
        with self.assertRaises(CastError) as ctx:
            opts.cast_arrow_tabular(tbl)
        self.assertEqual(ctx.exception.target.name, "count")

    def test_deeply_nested_leaf_failure(self) -> None:
        # struct → struct → struct: the innermost leaf is what gets bound.
        pa = self.pa
        src = pa.schema([
            pa.field("outer", pa.struct([
                pa.field("inner", pa.struct([
                    pa.field("val", pa.string()),
                ])),
            ])),
        ])
        tgt = pa.schema([
            pa.field("outer", pa.struct([
                pa.field("inner", pa.struct([
                    pa.field("val", pa.int64()),
                ])),
            ])),
        ])
        tbl = pa.table(
            {"outer": [{"inner": {"val": "abc"}}]}, schema=src,
        )
        opts = CastOptions(
            source=Field.from_arrow(src),
            target=Field.from_arrow(tgt),
            safe=True,
        )
        with self.assertRaises(CastError) as ctx:
            opts.cast_arrow_tabular(tbl)
        # Bound to the deepest leaf, not the outer ``outer`` / ``inner``
        # wrappers.
        self.assertEqual(ctx.exception.target.name, "val")
        self.assertIn("val: string", str(ctx.exception))
        self.assertIn("val: int64", str(ctx.exception))

    def test_cast_arrow_array_direct_wraps(self) -> None:
        # Direct :meth:`Field.cast_arrow_array` call (no tabular shell)
        # also wraps — same code path, same field context.
        pa = self.pa
        src_field = Field.from_arrow(pa.field("amount", pa.string()))
        tgt_field = Field.from_arrow(pa.field("amount", pa.int64()))
        arr = pa.array(["nope"], type=pa.string())
        with self.assertRaises(CastError) as ctx:
            tgt_field.cast_arrow_array(
                arr, source=src_field, safe=True,
            )
        self.assertEqual(ctx.exception.target.name, "amount")
        self.assertIsNotNone(ctx.exception.original)
