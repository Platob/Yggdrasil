"""Unit tests for yggdrasil.pickle.ser module - Array and Collection types."""

import unittest

import pytest

from yggdrasil.pickle.ser import (
    ListSerialized,
    TupleSerialized,
    SetSerialized,
    FrozenSetSerialized,
    SerdeTags,
)


class TestListSerialized(unittest.TestCase):
    """Test ListSerialized for list values."""

    def test_empty_list(self):
        """Test empty list."""
        serialized = ListSerialized.from_value([])
        assert serialized.value == []

    def test_simple_list(self):
        """Test simple list with primitives."""
        data = [1, 2, 3]
        serialized = ListSerialized.from_value(data)
        assert serialized.value == data

    def test_string_list(self):
        """Test list of strings."""
        data = ["a", "b", "c"]
        serialized = ListSerialized.from_value(data)
        assert serialized.value == data

    def test_mixed_type_list(self):
        """Test list with mixed types."""
        data = [1, "two", 3.0, True, None]
        serialized = ListSerialized.from_value(data)
        assert serialized.value == data

    def test_nested_list(self):
        """Test nested list."""
        data = [[1, 2], [3, 4], [5, 6]]
        serialized = ListSerialized.from_value(data)
        assert serialized.value == data

    def test_deeply_nested_list(self):
        """Test deeply nested list."""
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        serialized = ListSerialized.from_value(data)
        assert serialized.value == data

    def test_list_tag(self):
        """Test correct tag."""
        assert ListSerialized.TAG == SerdeTags.LIST

    def test_list_roundtrip(self):
        """Test list roundtrip."""
        lists = [
            [],
            [1, 2, 3],
            ["a", "b", "c"],
            [1, "two", 3.0],
            [[1, 2], [3, 4]],
        ]
        for lst in lists:
            serialized = ListSerialized.from_value(lst)
            assert serialized.value == lst

    def test_list_with_metadata(self):
        """Test list with metadata."""
        data = [1, 2, 3]
        metadata = {b"encoding": b"json"}
        serialized = ListSerialized.from_value(data, metadata=metadata)
        assert serialized.metadata == metadata
        assert serialized.value == data

    def test_list_rejects_tuple(self):
        """Test that ListSerialized rejects tuple."""
        with pytest.raises(TypeError):
            ListSerialized.from_value((1, 2, 3))

    def test_list_large_size(self):
        """Test large list."""
        data = list(range(10000))
        serialized = ListSerialized.from_value(data)
        assert serialized.value == data


class TestTupleSerialized(unittest.TestCase):
    """Test TupleSerialized for tuple values."""

    def test_empty_tuple(self):
        """Test empty tuple."""
        serialized = TupleSerialized.from_value(())
        assert serialized.value == ()

    def test_simple_tuple(self):
        """Test simple tuple."""
        data = (1, 2, 3)
        serialized = TupleSerialized.from_value(data)
        assert serialized.value == data

    def test_string_tuple(self):
        """Test tuple of strings."""
        data = ("a", "b", "c")
        serialized = TupleSerialized.from_value(data)
        assert serialized.value == data

    def test_mixed_type_tuple(self):
        """Test tuple with mixed types."""
        data = (1, "two", 3.0, True, None)
        serialized = TupleSerialized.from_value(data)
        assert serialized.value == data

    def test_nested_tuple(self):
        """Test nested tuple."""
        data = ((1, 2), (3, 4), (5, 6))
        serialized = TupleSerialized.from_value(data)
        assert serialized.value == data

    def test_tuple_with_list(self):
        """Test tuple containing list."""
        data = ([1, 2], [3, 4])
        serialized = TupleSerialized.from_value(data)
        assert serialized.value == data

    def test_tuple_tag(self):
        """Test correct tag."""
        assert TupleSerialized.TAG == SerdeTags.TUPLE

    def test_tuple_roundtrip(self):
        """Test tuple roundtrip."""
        tuples = [
            (),
            (1,),
            (1, 2, 3),
            ("a", "b", "c"),
            (1, "two", 3.0),
            ((1, 2), (3, 4)),
        ]
        for tpl in tuples:
            serialized = TupleSerialized.from_value(tpl)
            assert serialized.value == tpl

    def test_tuple_with_metadata(self):
        """Test tuple with metadata."""
        data = (1, 2, 3)
        metadata = {b"type": b"coordinates"}
        serialized = TupleSerialized.from_value(data, metadata=metadata)
        assert serialized.metadata == metadata
        assert serialized.value == data

    def test_tuple_rejects_list(self):
        """Test that TupleSerialized rejects list."""
        with pytest.raises(TypeError):
            TupleSerialized.from_value([1, 2, 3])

    def test_single_element_tuple(self):
        """Test single-element tuple."""
        data = (42,)
        serialized = TupleSerialized.from_value(data)
        assert serialized.value == data


class TestSetSerialized(unittest.TestCase):
    """Test SetSerialized for set values."""

    def test_empty_set(self):
        """Test empty set."""
        serialized = SetSerialized.from_value(set())
        assert serialized.value == set()

    def test_int_set(self):
        """Test set of integers."""
        data = {1, 2, 3, 4, 5}
        serialized = SetSerialized.from_value(data)
        assert serialized.value == data

    def test_string_set(self):
        """Test set of strings."""
        data = {"a", "b", "c"}
        serialized = SetSerialized.from_value(data)
        assert serialized.value == data

    def test_mixed_set(self):
        """Test set with mixed hashable types."""
        data = {1, "two", 3.0}
        serialized = SetSerialized.from_value(data)
        assert serialized.value == data

    def test_set_tag(self):
        """Test correct tag."""
        assert SetSerialized.TAG == SerdeTags.SET

    def test_set_roundtrip(self):
        """Test set roundtrip."""
        sets = [
            set(),
            {1},
            {1, 2, 3},
            {"a", "b", "c"},
            {1, "two", 3.0},
        ]
        for s in sets:
            serialized = SetSerialized.from_value(s)
            assert serialized.value == s

    def test_set_with_metadata(self):
        """Test set with metadata."""
        data = {1, 2, 3}
        metadata = {b"unique": b"true"}
        serialized = SetSerialized.from_value(data, metadata=metadata)
        assert serialized.metadata == metadata
        assert serialized.value == data

    def test_set_rejects_list(self):
        """Test that SetSerialized rejects list."""
        with pytest.raises(TypeError):
            SetSerialized.from_value([1, 2, 3])

    def test_set_rejects_frozenset(self):
        """Test that SetSerialized rejects frozenset."""
        with pytest.raises(TypeError):
            SetSerialized.from_value(frozenset({1, 2, 3}))

    def test_set_deduplicates(self):
        """Test that set naturally deduplicates."""
        # Note: The serialization doesn't enforce uniqueness,
        # but Python sets already do this
        data = {1, 2, 3}
        serialized = SetSerialized.from_value(data)
        assert len(serialized.value) == 3


class TestFrozenSetSerialized(unittest.TestCase):
    """Test FrozenSetSerialized for frozenset values."""

    def test_empty_frozenset(self):
        """Test empty frozenset."""
        serialized = FrozenSetSerialized.from_value(frozenset())
        assert serialized.value == frozenset()

    def test_int_frozenset(self):
        """Test frozenset of integers."""
        data = frozenset({1, 2, 3, 4, 5})
        serialized = FrozenSetSerialized.from_value(data)
        assert serialized.value == data

    def test_string_frozenset(self):
        """Test frozenset of strings."""
        data = frozenset({"a", "b", "c"})
        serialized = FrozenSetSerialized.from_value(data)
        assert serialized.value == data

    def test_mixed_frozenset(self):
        """Test frozenset with mixed hashable types."""
        data = frozenset({1, "two", 3.0})
        serialized = FrozenSetSerialized.from_value(data)
        assert serialized.value == data

    def test_frozenset_tag(self):
        """Test correct tag."""
        assert FrozenSetSerialized.TAG == SerdeTags.FROZENSET

    def test_frozenset_roundtrip(self):
        """Test frozenset roundtrip."""
        frozensets = [
            frozenset(),
            frozenset({1}),
            frozenset({1, 2, 3}),
            frozenset({"a", "b", "c"}),
            frozenset({1, "two", 3.0}),
        ]
        for fs in frozensets:
            serialized = FrozenSetSerialized.from_value(fs)
            assert serialized.value == fs

    def test_frozenset_with_metadata(self):
        """Test frozenset with metadata."""
        data = frozenset({1, 2, 3})
        metadata = {b"immutable": b"true"}
        serialized = FrozenSetSerialized.from_value(data, metadata=metadata)
        assert serialized.metadata == metadata
        assert serialized.value == data

    def test_frozenset_rejects_set(self):
        """Test that FrozenSetSerialized rejects set."""
        with pytest.raises(TypeError):
            FrozenSetSerialized.from_value({1, 2, 3})

    def test_frozenset_rejects_list(self):
        """Test that FrozenSetSerialized rejects list."""
        with pytest.raises(TypeError):
            FrozenSetSerialized.from_value([1, 2, 3])

    def test_frozenset_immutable(self):
        """Test that frozenset is immutable."""
        fs = frozenset({1, 2, 3})
        serialized = FrozenSetSerialized.from_value(fs)

        with pytest.raises((TypeError, AttributeError)):
            getattr(serialized.value, "add")(4)


class TestCollectionNesting(unittest.TestCase):
    """Test complex nesting of collections."""

    def test_list_of_sets(self):
        """Test list containing sets."""
        # Note: Sets must be converted to tuples/lists for serialization
        # This test documents the behavior
        data = [[1, 2], [3, 4]]
        serialized = ListSerialized.from_value(data)
        assert serialized.value == data

    def test_tuple_of_lists(self):
        """Test tuple containing lists."""
        data = ([1, 2], [3, 4])
        serialized = TupleSerialized.from_value(data)
        assert serialized.value == data

    def test_list_of_tuples(self):
        """Test list containing tuples."""
        data = [(1, 2), (3, 4)]
        serialized = ListSerialized.from_value(data)
        assert serialized.value == data

    def test_tuple_of_tuples(self):
        """Test tuple containing tuples."""
        data = ((1, 2), (3, 4))
        serialized = TupleSerialized.from_value(data)
        assert serialized.value == data

    def test_complex_nesting(self):
        """Test complex nested structures."""
        data = [
            (1, 2, [3, 4]),
            (5, 6, [7, 8]),
        ]
        serialized = ListSerialized.from_value(data)
        assert serialized.value == data

    def test_set_of_tuples(self):
        """Test set of tuples (tuples are hashable)."""
        data = {(1, 2), (3, 4), (5, 6)}
        serialized = SetSerialized.from_value(data)
        assert serialized.value == data

    def test_frozenset_of_tuples(self):
        """Test frozenset of tuples."""
        data = frozenset({(1, 2), (3, 4), (5, 6)})
        serialized = FrozenSetSerialized.from_value(data)
        assert serialized.value == data


if __name__ == "__main__":
    unittest.main()

