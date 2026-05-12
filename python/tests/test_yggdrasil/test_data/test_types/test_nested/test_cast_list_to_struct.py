"""Correctness tests for the vectorised ``cast_arrow_list_array`` path.

``struct_arrow.cast_arrow_list_array`` extracts each row's list values
position-by-position into struct children — used when the wider data
plane lands a list-shaped column that the target schema declares as a
struct with N children.

The vectorised implementation uses ``pc.take`` over the flat
``values`` buffer plus a ``pc.list_value_length`` mask. These tests
pin down the semantics it must preserve:

1. Position ``i`` of each row maps to the target's i-th child.
2. Rows shorter than the target's child count null out the missing
   positions (instead of raising — the wider plane often hands us
   short lists).
3. Null parent rows produce null in every child position.
4. Inner nulls inside a valid row carry through to the child.
5. Rows longer than the target's child count keep only the leading
   positions — the excess is dropped.
6. ``LargeListArray`` and the fallback ``to_pylist`` path produce the
   same output as the fast path.
"""
from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions


def _list_int_to_struct_3(self) -> tuple[Field, Field]:
    src = Field.from_arrow(self.pa.field(
        "x", self.pa.list_(self.pa.field("item", self.pa.int32())),
    ))
    tgt = Field.from_arrow(self.pa.field(
        "x", self.pa.struct([
            self.pa.field("c0", self.pa.int32()),
            self.pa.field("c1", self.pa.int32()),
            self.pa.field("c2", self.pa.int32()),
        ]),
    ))
    return src, tgt


def _cast(self, arr, src, tgt):
    """Run the cast under test — returns a ``pa.StructArray``."""
    return CastOptions(
        source=src, target=tgt,
    ).cast_arrow_array(arr)


class TestListToStructPositional(ArrowTestCase):
    """Positional extraction: list[i] → struct.child[i]."""

    def test_equal_length_rows(self) -> None:
        src, tgt = _list_int_to_struct_3(self)
        arr = self.pa.array(
            [[1, 2, 3], [4, 5, 6]],
            type=self.pa.list_(self.pa.field("item", self.pa.int32())),
        )
        out = _cast(self, arr, src, tgt)
        self.assertEqual(out.field("c0").to_pylist(), [1, 4])
        self.assertEqual(out.field("c1").to_pylist(), [2, 5])
        self.assertEqual(out.field("c2").to_pylist(), [3, 6])

    def test_short_rows_null_out(self) -> None:
        # Row has fewer items than target has children → missing
        # positions null out.
        src, tgt = _list_int_to_struct_3(self)
        arr = self.pa.array(
            [[1, 2], [10]],
            type=self.pa.list_(self.pa.field("item", self.pa.int32())),
        )
        out = _cast(self, arr, src, tgt)
        self.assertEqual(out.field("c0").to_pylist(), [1, 10])
        self.assertEqual(out.field("c1").to_pylist(), [2, None])
        self.assertEqual(out.field("c2").to_pylist(), [None, None])

    def test_long_rows_truncate(self) -> None:
        # Row has more items than target — extras get dropped silently
        # (matches the prior behaviour). The target's child count is
        # the contract.
        src, tgt = _list_int_to_struct_3(self)
        arr = self.pa.array(
            [[1, 2, 3, 99], [4, 5, 6, 77, 88]],
            type=self.pa.list_(self.pa.field("item", self.pa.int32())),
        )
        out = _cast(self, arr, src, tgt)
        self.assertEqual(out.field("c0").to_pylist(), [1, 4])
        self.assertEqual(out.field("c1").to_pylist(), [2, 5])
        self.assertEqual(out.field("c2").to_pylist(), [3, 6])

    def test_null_parent_rows(self) -> None:
        # Null parent rows propagate to all children. The struct's
        # own validity mask carries the parent-null bit.
        src, tgt = _list_int_to_struct_3(self)
        arr = self.pa.array(
            [[1, 2, 3], None, [4, 5, 6]],
            type=self.pa.list_(self.pa.field("item", self.pa.int32())),
        )
        out = _cast(self, arr, src, tgt)
        self.assertFalse(out.is_valid()[1].as_py())
        self.assertEqual(out.field("c0").to_pylist(), [1, None, 4])
        self.assertEqual(out.field("c1").to_pylist(), [2, None, 5])
        self.assertEqual(out.field("c2").to_pylist(), [3, None, 6])

    def test_inner_nulls_preserved(self) -> None:
        # An inner null inside an otherwise valid row stays a null
        # in the corresponding child position.
        src, tgt = _list_int_to_struct_3(self)
        arr = self.pa.array(
            [[1, None, 3], [None, 5, None]],
            type=self.pa.list_(self.pa.field("item", self.pa.int32())),
        )
        out = _cast(self, arr, src, tgt)
        self.assertEqual(out.field("c0").to_pylist(), [1, None])
        self.assertEqual(out.field("c1").to_pylist(), [None, 5])
        self.assertEqual(out.field("c2").to_pylist(), [3, None])

    def test_empty_array(self) -> None:
        src, tgt = _list_int_to_struct_3(self)
        arr = self.pa.array(
            [], type=self.pa.list_(self.pa.field("item", self.pa.int32())),
        )
        out = _cast(self, arr, src, tgt)
        self.assertEqual(len(out), 0)
        self.assertEqual(out.field("c0").to_pylist(), [])

    def test_large_list_array(self) -> None:
        # ``LargeListArray`` has int64 offsets — the vectorised path
        # must handle both list shapes via the same offset+take pattern.
        src = Field.from_arrow(self.pa.field(
            "x", self.pa.large_list(self.pa.field("item", self.pa.int32())),
        ))
        tgt = Field.from_arrow(self.pa.field(
            "x", self.pa.struct([
                self.pa.field("c0", self.pa.int32()),
                self.pa.field("c1", self.pa.int32()),
            ]),
        ))
        arr = self.pa.array(
            [[10, 20], None, [30]],
            type=self.pa.large_list(self.pa.field("item", self.pa.int32())),
        )
        out = _cast(self, arr, src, tgt)
        self.assertEqual(out.field("c0").to_pylist(), [10, None, 30])
        self.assertEqual(out.field("c1").to_pylist(), [20, None, None])

    def test_list_of_struct_inner(self) -> None:
        # The inner type itself is a struct — positional extraction
        # must hand each struct value through to the corresponding
        # child unchanged.
        pa = self.pa
        inner = pa.struct([
            pa.field("a", pa.int32()),
            pa.field("b", pa.string()),
        ])
        src = Field.from_arrow(pa.field(
            "x", pa.list_(pa.field("item", inner)),
        ))
        tgt = Field.from_arrow(pa.field(
            "x", pa.struct([
                pa.field("c0", inner),
                pa.field("c1", inner),
            ]),
        ))
        arr = pa.array(
            [
                [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
                None,
                [{"a": 3, "b": "z"}],
            ],
            type=pa.list_(pa.field("item", inner)),
        )
        out = _cast(self, arr, src, tgt)
        self.assertEqual(
            out.field("c0").to_pylist(),
            [{"a": 1, "b": "x"}, None, {"a": 3, "b": "z"}],
        )
        self.assertEqual(
            out.field("c1").to_pylist(),
            [{"a": 2, "b": "y"}, None, None],
        )

    def test_inner_type_widen_runs(self) -> None:
        # The child cast still runs — confirm an int32→int64 widen
        # on the inner type applies in the vectorised path.
        pa = self.pa
        src = Field.from_arrow(pa.field(
            "x", pa.list_(pa.field("item", pa.int32())),
        ))
        tgt = Field.from_arrow(pa.field(
            "x", pa.struct([
                pa.field("c0", pa.int64()),
                pa.field("c1", pa.int64()),
            ]),
        ))
        arr = pa.array(
            [[1, 2], [3]],
            type=pa.list_(pa.field("item", pa.int32())),
        )
        out = _cast(self, arr, src, tgt)
        self.assertEqual(out.field("c0").type, pa.int64())
        self.assertEqual(out.field("c1").type, pa.int64())
        self.assertEqual(out.field("c0").to_pylist(), [1, 3])
        self.assertEqual(out.field("c1").to_pylist(), [2, None])


class TestListToStructFallbackParity(ArrowTestCase):
    """The fast path and the ``to_pylist`` fallback must produce the
    same output bit-for-bit on a representative input.

    Run the same data through both and compare. The fast path is
    selected by type isinstance; we exercise the fallback by reaching
    into ``_extract_list_positions`` directly with a vanilla list
    array — the type check still routes it down the fast path, so we
    instead compare against a hand-built ``to_pylist`` walk.
    """

    def test_parity_with_legacy_walk(self) -> None:
        pa = self.pa
        arr = pa.array(
            [[1, 2, 3], None, [4], [5, 6], [], [7, 8, 9, 10]],
            type=pa.list_(pa.field("item", pa.int32())),
        )
        src = Field.from_arrow(pa.field(
            "x", pa.list_(pa.field("item", pa.int32())),
        ))
        tgt = Field.from_arrow(pa.field(
            "x", pa.struct([
                pa.field("c0", pa.int32()),
                pa.field("c1", pa.int32()),
                pa.field("c2", pa.int32()),
            ]),
        ))
        out = _cast(self, arr, src, tgt)

        # Hand-rolled legacy semantics — the contract we're matching.
        py = arr.to_pylist()
        expected = [
            [
                None if row is None or i >= len(row) else row[i]
                for row in py
            ]
            for i in range(3)
        ]
        for i, name in enumerate(("c0", "c1", "c2")):
            self.assertEqual(out.field(name).to_pylist(), expected[i])
