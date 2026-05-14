"""Tests for :class:`NestedCurator`.

Recursive curation of struct / list / map arrays. Each leaf must
land in the same inferred type the :class:`StringCurator` would
produce on its own, and the parent's validity + offsets must
round-trip exactly.
"""

from __future__ import annotations

import unittest

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types import (
    ArrayType,
    Float64Type,
    MapType,
    StructType,
)
from yggdrasil.io.curation import Curator, NestedCurator


class TestStructCuration(ArrowTestCase):
    """Each struct field is curated independently; parent validity survives."""

    def test_pick_returns_nested_curator_for_struct(self):
        arr = self.pa.array([{"id": "1"}])
        self.assertIsInstance(Curator.pick(arr), NestedCurator)

    def test_struct_of_strings_becomes_typed(self):
        arr = self.pa.array(
            [
                {"id": "1", "amount": "1.5"},
                {"id": "2", "amount": "2.5"},
                None,
            ]
        )
        result = Curator.pick(arr).curate(arr)
        self.assertEqual(str(result.array.type), "struct<id: int64, amount: double>")
        self.assertEqual(result.array.field("id").to_pylist(), [1, 2, None])
        self.assertEqual(result.array.field("amount").to_pylist(), [1.5, 2.5, None])
        # Parent struct's null mask survives the rebuild.
        self.assertEqual(result.array.is_null().to_pylist(), [False, False, True])

    def test_struct_dtype_is_a_structtype(self):
        arr = self.pa.array([{"id": "1", "name": "a"}])
        result = Curator.pick(arr).curate(arr)
        self.assertIsInstance(result.dtype, StructType)
        names = [f.name for f in result.dtype.fields]
        self.assertEqual(names, ["id", "name"])

    def test_struct_child_unhandled_dtype_passes_through(self):
        # The ``blob`` column is binary — no Curator subclass claims
        # it, so it lands on the rebuild as-is.
        arr = self.pa.StructArray.from_arrays(
            [
                self.pa.array(["1", "2"]),
                self.pa.array([b"x", b"y"]),
            ],
            names=["id", "blob"],
        )
        result = Curator.pick(arr).curate(arr)
        self.assertEqual(str(result.array.type), "struct<id: int64, blob: binary>")
        self.assertEqual(result.array.field("blob").to_pylist(), [b"x", b"y"])


class TestListCuration(ArrowTestCase):
    """List values are curated; offsets + null mask preserved."""

    def test_list_of_strings_becomes_list_of_int(self):
        arr = self.pa.array([["1", "2"], ["3"], None, []])
        result = Curator.pick(arr).curate(arr)
        self.assertEqual(str(result.array.type), "list<item: int64>")
        self.assertEqual(result.array.to_pylist(), [[1, 2], [3], None, []])

    def test_list_dtype_is_an_arraytype(self):
        arr = self.pa.array([["1.5"], ["2.5"]])
        result = Curator.pick(arr).curate(arr)
        self.assertIsInstance(result.dtype, ArrayType)
        self.assertEqual(result.dtype.item_field.dtype, Float64Type())

    def test_large_list_keeps_large_list_shape(self):
        arr = self.pa.array(
            [["1", "2"], ["3"]], type=self.pa.large_list(self.pa.string())
        )
        result = Curator.pick(arr).curate(arr)
        self.assertEqual(str(result.array.type), "large_list<item: int64>")


class TestMapCuration(ArrowTestCase):
    """Map keys and items curated separately; keys stay non-nullable."""

    def test_map_string_to_string_values_become_int(self):
        arr = self.pa.array(
            [[("a", "1"), ("b", "2")], None, [("c", "3")]],
            type=self.pa.map_(self.pa.string(), self.pa.string()),
        )
        result = Curator.pick(arr).curate(arr)
        self.assertEqual(str(result.array.type), "map<string, int64>")
        self.assertEqual(
            result.array.to_pylist(),
            [[("a", 1), ("b", 2)], None, [("c", 3)]],
        )

    def test_map_dtype_is_a_maptype(self):
        arr = self.pa.array(
            [[("a", "1")]], type=self.pa.map_(self.pa.string(), self.pa.string())
        )
        result = Curator.pick(arr).curate(arr)
        self.assertIsInstance(result.dtype, MapType)


class TestRecursiveCuration(ArrowTestCase):
    """Nested-of-nested: list<struct<...>> and struct<list<...>>."""

    def test_list_of_structs_descends_two_layers(self):
        arr = self.pa.array(
            [
                [{"a": "1", "b": "2024-01-15T10:00:00+02:00"}],
                None,
                [
                    {"a": "3", "b": "2024-06-15T11:00:00-05:00"},
                    {"a": "4", "b": "2024-07-15T12:00:00+00:00"},
                ],
            ]
        )
        result = Curator.pick(arr).curate(arr)
        # Outer list survives; inner struct's fields got auto-typed.
        self.assertTrue(str(result.array.type).startswith("list<item: struct<"))
        struct_type = result.array.type.value_type
        self.assertEqual(struct_type.field("a").type, self.pa.int64())
        self.assertEqual(str(struct_type.field("b").type), "timestamp[us, tz=UTC]")

    def test_inner_curator_kwargs_forward(self):
        # ``target_tz="Europe/Paris"`` set on the NestedCurator should
        # propagate to the inner StringCurator picked for the leaf
        # timestamp column.
        arr = self.pa.array([{"when": "2024-01-15T10:00:00+02:00"}])
        result = NestedCurator(target_tz="Europe/Paris").curate(arr)
        self.assertEqual(
            str(result.array.type),
            "struct<when: timestamp[us, tz=Europe/Paris]>",
        )

    def test_struct_of_list_of_strings(self):
        arr = self.pa.array([{"vals": ["1", "2", "3"]}, {"vals": ["4"]}, None])
        result = Curator.pick(arr).curate(arr)
        self.assertEqual(str(result.array.type), "struct<vals: list<item: int64>>")
        # The outer struct's null mask survives — Arrow stores
        # validity at the parent level, so the child list at the null
        # slot is materialised as an empty list (the standard Arrow
        # shape), but ``arr[2]`` reads back as ``None``.
        self.assertEqual(
            result.array.to_pylist(),
            [{"vals": [1, 2, 3]}, {"vals": [4]}, None],
        )


class TestTabularRoutesToNested(ArrowTestCase):
    """The tabular dispatcher should now route nested columns through NestedCurator."""

    def test_table_with_nested_column(self):
        table = self.pa.table(
            {
                "id": ["1", "2"],
                "tags": [["a", "b"], ["c"]],
                "meta": [
                    {"score": "1.5", "ok": "true"},
                    {"score": "2.5", "ok": "false"},
                ],
            }
        )
        schema, curated = Curator.curate_arrow_tabular(table)
        types = {n: curated.schema.field(n).type for n in curated.schema.names}
        self.assertEqual(types["id"], self.pa.int64())
        self.assertEqual(str(types["tags"]), "list<item: string>")
        self.assertEqual(str(types["meta"]), "struct<score: double, ok: bool>")


if __name__ == "__main__":
    unittest.main()
