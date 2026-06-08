""":class:`UnionType` — DataType-layer wrapper for Python ``Union`` / ``Optional``.

The user-visible contract is the ``to_field()`` unwrap: when materialising
a union into a :class:`Field`, ``NullType`` arms drop into the
``nullable`` flag and a single non-null arm un-nests so the field's
``dtype`` is the inner type, not the union. The full rule is exercised
below, plus the engine-projection delegation and the to/from-dict
round trip.
"""
from __future__ import annotations

import unittest
from typing import Optional, Union

import pyarrow as pa

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types import (
    IntegerType,
    NullType,
    StringType,
    UnionType,
)
from yggdrasil.data.types.base import DataType


class TestUnionTypeConstruction(ArrowTestCase):

    def test_construct_with_tuple_members(self):
        u = UnionType(members=(IntegerType(), NullType()))
        self.assertEqual(len(u.members), 2)
        self.assertTrue(u.nullable)

    def test_construct_with_list_members_coerced_to_tuple(self):
        u = UnionType(members=[IntegerType(), StringType()])
        self.assertIsInstance(u.members, tuple)
        self.assertFalse(u.nullable)

    def test_rejects_non_datatype_members(self):
        with self.assertRaises(TypeError) as cm:
            UnionType(members=(int, NullType()))
        # Helpful error message
        self.assertIn("DataType", str(cm.exception))

    def test_nullable_property_tracks_nulltype_membership(self):
        self.assertTrue(UnionType(members=(IntegerType(), NullType())).nullable)
        self.assertFalse(UnionType(members=(IntegerType(), StringType())).nullable)
        self.assertTrue(UnionType(members=(NullType(),)).nullable)

    def test_non_null_members(self):
        u = UnionType(members=(IntegerType(), StringType(), NullType()))
        non_null = u.non_null_members
        self.assertEqual(len(non_null), 2)
        self.assertIsInstance(non_null[0], IntegerType)
        self.assertIsInstance(non_null[1], StringType)


class TestUnionTypeToField(ArrowTestCase):
    """``to_field`` is the bridge from union-rich DataType to nullable-flat Field.

    Spec: drop ``NullType`` arms into the ``nullable`` flag; un-nest
    the union when only one non-null arm remains.
    """

    def test_int_plus_null_unwraps_to_int_with_nullable(self):
        u = UnionType(members=(IntegerType(), NullType()))
        f = u.to_field("id")
        self.assertEqual(f.name, "id")
        self.assertIsInstance(f.dtype, IntegerType)
        self.assertTrue(f.nullable)

    def test_multi_arm_with_null_keeps_union_drops_null(self):
        u = UnionType(members=(IntegerType(), StringType(), NullType()))
        f = u.to_field("v")
        self.assertIsInstance(f.dtype, UnionType)
        self.assertEqual(len(f.dtype.members), 2)
        self.assertFalse(any(isinstance(m, NullType) for m in f.dtype.members))
        self.assertTrue(f.nullable)

    def test_multi_arm_without_null_keeps_union(self):
        u = UnionType(members=(IntegerType(), StringType()))
        f = u.to_field("v", nullable=False)
        self.assertIsInstance(f.dtype, UnionType)
        self.assertEqual(len(f.dtype.members), 2)
        self.assertFalse(f.nullable)

    def test_single_non_null_member_unwraps(self):
        # Even without NullType in the union, a single-member UnionType
        # should un-nest to the inner type.
        u = UnionType(members=(IntegerType(),))
        f = u.to_field("only", nullable=False)
        self.assertIsInstance(f.dtype, IntegerType)
        self.assertFalse(f.nullable)

    def test_all_null_collapses_to_nulltype(self):
        u = UnionType(members=(NullType(), NullType()))
        f = u.to_field("nothing")
        self.assertIsInstance(f.dtype, NullType)
        self.assertTrue(f.nullable)

    def test_empty_union_collapses_to_nulltype(self):
        u = UnionType(members=())
        f = u.to_field("nothing")
        self.assertIsInstance(f.dtype, NullType)
        self.assertTrue(f.nullable)

    def test_null_membership_overrides_nullable_false(self):
        # Even when the caller passes ``nullable=False``, the presence
        # of NullType in the union should force nullable=True — the
        # union's content is the stronger signal of intent.
        u = UnionType(members=(IntegerType(), NullType()))
        f = u.to_field("id", nullable=False)
        self.assertTrue(f.nullable)


class TestUnionTypeToPyhint(ArrowTestCase):
    """``to_pyhint`` builds back the Python ``Union`` / ``Optional`` form."""

    def test_int_plus_null_round_trips(self):
        u = UnionType(members=(IntegerType(), NullType()))
        # ``int | None`` and ``Optional[int]`` are the same typing form.
        self.assertEqual(u.to_pyhint(), Optional[int])

    def test_multi_arm(self):
        u = UnionType(members=(IntegerType(), StringType()))
        self.assertEqual(u.to_pyhint(), Union[int, str])

    def test_multi_arm_with_null(self):
        u = UnionType(members=(IntegerType(), StringType(), NullType()))
        self.assertEqual(u.to_pyhint(), Union[int, str, None])

    def test_single_member_collapses(self):
        u = UnionType(members=(IntegerType(),))
        self.assertIs(u.to_pyhint(), int)

    def test_empty_union(self):
        u = UnionType(members=())
        self.assertIs(u.to_pyhint(), type(None))


class TestUnionTypeEngineProjection(ArrowTestCase):
    """``to_arrow`` / ``to_polars`` / ``to_spark`` delegate to a single member.

    Single non-null member → that member's projection. Multi-arm non-null
    → StringType. Zero non-null → NullType. Same answer the legacy
    ``from_pytype`` path would have returned, so engine round-trips
    stay unchanged.
    """

    def test_optional_int_arrow_is_int64(self):
        u = UnionType(members=(IntegerType(), NullType()))
        self.assertEqual(u.to_arrow(), pa.int64())

    def test_multi_arm_arrow_is_string(self):
        u = UnionType(members=(IntegerType(), StringType()))
        self.assertEqual(u.to_arrow(), pa.string())

    def test_all_null_arrow_is_null(self):
        u = UnionType(members=(NullType(),))
        self.assertEqual(u.to_arrow(), pa.null())


class TestUnionTypeSerialization(ArrowTestCase):
    """``to_dict`` / ``from_dict`` round-trip the members."""

    def test_to_dict_carries_id_name_members(self):
        u = UnionType(members=(IntegerType(), NullType()))
        d = u.to_dict()
        self.assertEqual(d["name"], "UNION")
        self.assertIn("members", d)
        self.assertEqual(len(d["members"]), 2)

    def test_from_dict_rebuilds_shape(self):
        u = UnionType(members=(IntegerType(), NullType()))
        d = u.to_dict()
        rebuilt = DataType.from_dict(d)
        self.assertIsInstance(rebuilt, UnionType)
        self.assertEqual(len(rebuilt.members), 2)
        # Structural sameness — Python's ``int | None`` survives the trip.
        self.assertEqual(rebuilt.to_pyhint(), u.to_pyhint())


class TestUnionTypeMerge(ArrowTestCase):

    def test_merge_concatenates_and_dedupes(self):
        a = UnionType(members=(IntegerType(), NullType()))
        b = UnionType(members=(StringType(), NullType()))
        merged = a.merge_with(b)
        self.assertIsInstance(merged, UnionType)
        # ``IntegerType``, ``NullType``, ``StringType`` — three unique members.
        self.assertEqual(len(merged.members), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
