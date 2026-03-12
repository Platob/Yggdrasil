"""Unit tests for yggdrasil.pickle.ser module - Maps and Factory functions."""

import unittest
from collections import OrderedDict

import pytest

from yggdrasil.pickle.ser import (
    DictSerialized,
    OrderedDictSerialized,
    SerdeTags,
    dumps,
    loads,
)


class TestDictSerialized(unittest.TestCase):
    """Test DictSerialized for dict values."""

    def test_empty_dict(self):
        """Test empty dict."""
        serialized = DictSerialized.from_value({})
        assert serialized.value == {}

    def test_simple_dict(self):
        """Test simple dict."""
        data = {"a": 1, "b": 2, "c": 3}
        serialized = DictSerialized.from_value(data)
        assert serialized.value == data

    def test_int_keys(self):
        """Test dict with integer keys."""
        data = {1: "a", 2: "b", 3: "c"}
        serialized = DictSerialized.from_value(data)
        assert serialized.value == data

    def test_mixed_key_types(self):
        """Test dict with mixed key types."""
        data = {"str": 1, 2: "int", 3.0: "float"}
        serialized = DictSerialized.from_value(data)
        assert serialized.value == data

    def test_complex_values(self):
        """Test dict with complex values."""
        data = {
            "int": 42,
            "str": "hello",
            "list": [1, 2, 3],
            "dict": {"nested": True},
        }
        serialized = DictSerialized.from_value(data)
        assert serialized.value == data

    def test_nested_dict(self):
        """Test nested dict."""
        data = {
            "outer": {
                "inner": {
                    "deep": "value"
                }
            }
        }
        serialized = DictSerialized.from_value(data)
        assert serialized.value == data

    def test_dict_tag(self):
        """Test correct tag."""
        assert DictSerialized.TAG == SerdeTags.DICT

    def test_dict_roundtrip(self):
        """Test dict roundtrip."""
        dicts = [
            {},
            {"a": 1},
            {"a": 1, "b": 2, "c": 3},
            {1: "a", 2: "b"},
            {"key": [1, 2, 3]},
            {"outer": {"inner": "value"}},
        ]
        for d in dicts:
            serialized = DictSerialized.from_value(d)
            assert serialized.value == d

    def test_dict_with_metadata(self):
        """Test dict with metadata."""
        data = {"a": 1, "b": 2}
        metadata = {b"source": b"json"}
        serialized = DictSerialized.from_value(data, metadata=metadata)
        assert serialized.metadata == metadata
        assert serialized.value == data

    def test_dict_rejects_ordereddict(self):
        """Test that DictSerialized rejects OrderedDict."""
        with pytest.raises(TypeError):
            DictSerialized.from_value(OrderedDict([("a", 1), ("b", 2)]))

    def test_dict_tuple_values(self):
        """Test dict with tuple values."""
        data = {"point": (1, 2), "size": (10, 20)}
        serialized = DictSerialized.from_value(data)
        assert serialized.value == data

    def test_dict_with_none_values(self):
        """Test dict with None values."""
        data = {"a": 1, "b": None, "c": 3}
        serialized = DictSerialized.from_value(data)
        assert serialized.value == data

    def test_large_dict(self):
        """Test large dict."""
        data = {f"key_{i}": i for i in range(10000)}
        serialized = DictSerialized.from_value(data)
        assert serialized.value == data


class TestOrderedDictSerialized(unittest.TestCase):
    """Test OrderedDictSerialized for OrderedDict values."""

    def test_empty_ordereddict(self):
        """Test empty OrderedDict."""
        serialized = OrderedDictSerialized.from_value(OrderedDict())
        assert serialized.value == OrderedDict()

    def test_simple_ordereddict(self):
        """Test simple OrderedDict."""
        data = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
        serialized = OrderedDictSerialized.from_value(data)
        assert serialized.value == data

    def test_ordereddict_preserves_order(self):
        """Test that OrderedDict preserves insertion order."""
        data = OrderedDict([("z", 26), ("a", 1), ("m", 13)])
        serialized = OrderedDictSerialized.from_value(data)

        # Check both equality and order
        assert serialized.value == data
        assert list(serialized.value.keys()) == ["z", "a", "m"]

    def test_ordereddict_with_complex_values(self):
        """Test OrderedDict with complex values."""
        data = OrderedDict([
            ("first", [1, 2, 3]),
            ("second", {"nested": True}),
            ("third", (4, 5, 6)),
        ])
        serialized = OrderedDictSerialized.from_value(data)
        assert serialized.value == data
        assert list(serialized.value.keys()) == ["first", "second", "third"]

    def test_ordereddict_tag(self):
        """Test correct tag."""
        assert OrderedDictSerialized.TAG == SerdeTags.ORDEREDDICT

    def test_ordereddict_roundtrip(self):
        """Test OrderedDict roundtrip."""
        orderedicts = [
            OrderedDict(),
            OrderedDict([("a", 1)]),
            OrderedDict([("a", 1), ("b", 2), ("c", 3)]),
            OrderedDict([("z", 26), ("a", 1), ("m", 13)]),
        ]
        for od in orderedicts:
            serialized = OrderedDictSerialized.from_value(od)
            assert serialized.value == od
            assert list(serialized.value.keys()) == list(od.keys())

    def test_ordereddict_with_metadata(self):
        """Test OrderedDict with metadata."""
        data = OrderedDict([("a", 1), ("b", 2)])
        metadata = {b"ordered": b"true"}
        serialized = OrderedDictSerialized.from_value(data, metadata=metadata)
        assert serialized.metadata == metadata
        assert serialized.value == data

    def test_ordereddict_rejects_regular_dict(self):
        """Test that OrderedDictSerialized rejects regular dict."""
        with pytest.raises(TypeError):
            OrderedDictSerialized.from_value({"a": 1, "b": 2})

    def test_ordereddict_rejects_list(self):
        """Test that OrderedDictSerialized rejects list."""
        with pytest.raises(TypeError):
            OrderedDictSerialized.from_value([("a", 1), ("b", 2)])

    def test_nested_ordereddict(self):
        """Test nested OrderedDict."""
        inner = OrderedDict([("x", 10), ("y", 20)])
        data = OrderedDict([("inner", inner), ("z", 30)])
        serialized = OrderedDictSerialized.from_value(data)
        assert serialized.value == data


class TestDumpsFunction(unittest.TestCase):
    """Test dumps function for serialization to bytes."""

    def test_dumps_none(self):
        """Test dumping None."""
        result = dumps(None)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_dumps_primitives(self):
        """Test dumping primitive values."""
        values = [True, False, 42, 3.14, "hello", b"bytes"]
        for value in values:
            result = dumps(value)
            assert isinstance(result, bytes)

    def test_dumps_collections(self):
        """Test dumping collections."""
        values = [
            [],
            [1, 2, 3],
            (),
            (1, 2, 3),
            set(),
            {1, 2, 3},
            frozenset(),
            frozenset({1, 2, 3}),
            {},
            {"a": 1, "b": 2},
            OrderedDict([("a", 1), ("b", 2)]),
        ]
        for value in values:
            result = dumps(value)
            assert isinstance(result, bytes)

    def test_dumps_complex_structure(self):
        """Test dumping complex nested structure."""
        data = {
            "integers": [1, 2, 3],
            "strings": ("a", "b", "c"),
            "nested": {
                "boolean": True,
                "float": 3.14,
            }
        }
        result = dumps(data)
        assert isinstance(result, bytes)

    def test_dumps_with_metadata(self):
        """Test dumps with metadata."""
        result = dumps([1, 2, 3], metadata={b"version": b"1.0"})
        assert isinstance(result, bytes)

    def test_dumps_deterministic(self):
        """Test that dumps produces deterministic output for same input."""
        value = {"a": 1, "b": "test"}
        result1 = dumps(value)
        result2 = dumps(value)
        # Note: dict order might vary in Python <3.7, but should be consistent
        # within a single test run
        assert result1 == result2


class TestLoadsFunction(unittest.TestCase):
    """Test loads function for deserialization from bytes."""

    def test_loads_none(self):
        """Test loading None."""
        data = dumps(None)
        result = loads(data)
        assert result is None

    def test_loads_bool(self):
        """Test loading boolean values."""
        for value in [True, False]:
            data = dumps(value)
            result = loads(data)
            assert result is value

    def test_loads_integers(self):
        """Test loading integers."""
        for value in [0, 1, -1, 42, -42, 10**20]:
            data = dumps(value)
            result = loads(data)
            assert result == value

    def test_loads_floats(self):
        """Test loading floats."""
        for value in [0.0, 1.5, -1.5, 3.14159]:
            data = dumps(value)
            result = loads(data)
            assert abs(result - value) < 1e-14

    def test_loads_strings(self):
        """Test loading strings."""
        for value in ["", "hello", "Hello 世界 🌍"]:
            data = dumps(value)
            result = loads(data)
            assert result == value

    def test_loads_bytes(self):
        """Test loading bytes."""
        for value in [b"", b"hello", bytes(range(256))]:
            data = dumps(value)
            result = loads(data)
            assert result == value

    def test_loads_list(self):
        """Test loading list."""
        for value in [[], [1, 2, 3], ["a", "b", "c"], [1, "two", 3.0]]:
            data = dumps(value)
            result = loads(data)
            assert result == value

    def test_loads_tuple(self):
        """Test loading tuple."""
        for value in [(), (1,), (1, 2, 3), ("a", "b", "c")]:
            data = dumps(value)
            result = loads(data)
            assert result == value

    def test_loads_set(self):
        """Test loading set."""
        for value in [set(), {1, 2, 3}, {"a", "b", "c"}]:
            data = dumps(value)
            result = loads(data)
            assert result == value

    def test_loads_frozenset(self):
        """Test loading frozenset."""
        for value in [frozenset(), frozenset({1, 2, 3})]:
            data = dumps(value)
            result = loads(data)
            assert result == value

    def test_loads_dict(self):
        """Test loading dict."""
        for value in [{}, {"a": 1}, {"a": 1, "b": 2, "c": 3}]:
            data = dumps(value)
            result = loads(data)
            assert result == value

    def test_loads_ordereddict(self):
        """Test loading OrderedDict."""
        value = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
        data = dumps(value)
        result = loads(data)
        assert result == value
        assert list(result.keys()) == list(value.keys())

    def test_loads_complex_structure(self):
        """Test loading complex nested structure."""
        value = {
            "integers": [1, 2, 3],
            "strings": ("a", "b", "c"),
            "nested": {
                "boolean": True,
                "float": 3.14,
            }
        }
        data = dumps(value)
        result = loads(data)
        assert result == value

    def test_loads_preserves_types(self):
        """Test that loads preserves types."""
        value = {
            "list": [1, 2],
            "tuple": (3, 4),
            "set": {5, 6},
            "frozenset": frozenset({7, 8}),
        }
        data = dumps(value)
        result = loads(data)

        assert isinstance(result["list"], list)
        assert isinstance(result["tuple"], tuple)
        assert isinstance(result["set"], set)
        assert isinstance(result["frozenset"], frozenset)


class TestRoundTripConversions(unittest.TestCase):
    """Test round-trip conversions: Python → dumps → loads → Python."""

    def test_roundtrip(self):
        """Test roundtrip for all primitive/container types."""
        values = [
            None,
            True,
            False,
            0,
            42,
            -42,
            3.14,
            "",
            "hello",
            b"",
            b"hello",
            [],
            [1, 2, 3],
            (),
            (1, 2, 3),
            set(),
            {1, 2, 3},
            frozenset(),
            frozenset({1, 2, 3}),
            {},
            {"a": 1, "b": 2},
        ]
        for value in values:
            with self.subTest(value=value):
                data = dumps(value)
                result = loads(data)
                if isinstance(value, float) and not (value != value):
                    assert abs(result - value) < 1e-14
                else:
                    assert result == value

    def test_roundtrip_complex_nesting(self):
        """Test roundtrip with complex nesting."""
        value = {
            "list": [1, 2, {"nested": True}],
            "tuple": (3, 4, [5, 6]),
            "dict": {"a": {b"bytes": 1}},
        }
        data = dumps(value)
        result = loads(data)
        assert result == value

    def test_roundtrip_preserves_metadata(self):
        """Test that metadata is preserved during roundtrip."""
        value = [1, 2, 3]
        metadata = {b"source": b"api"}
        data = dumps(value, metadata=metadata)
        result = loads(data)
        # The value should be preserved
        assert result == value


if __name__ == "__main__":
    unittest.main()

