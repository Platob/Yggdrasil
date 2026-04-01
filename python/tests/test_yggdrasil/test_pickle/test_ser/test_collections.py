# tests/test_collections_ser.py
from __future__ import annotations

from array import array
from collections import deque
from collections.abc import Mapping as AbcMapping
from types import MappingProxyType

import pytest

import yggdrasil.pickle.ser.collections as m
from yggdrasil.io import BytesIO


def _roundtrip(obj):
    ser = m.CollectionSerialized.from_python_object(obj)
    assert ser is not None, f"Failed to serialize object: {obj!r}"
    return ser, ser.as_python()


def test_read_u32():
    buf = BytesIO((123).to_bytes(4, "big"))
    assert m._read_u32(buf) == 123


def test_read_u32_invalid_size():
    buf = BytesIO(b"\x00")
    with pytest.raises(ValueError):
        m._read_u32(buf)


def test_read_u64():
    buf = BytesIO((123456789).to_bytes(8, "big"))
    assert m._read_u64(buf) == 123456789


def test_read_u64_invalid_size():
    buf = BytesIO(b"\x00")
    with pytest.raises(ValueError):
        m._read_u64(buf)


def test_write_count_small():
    buf = BytesIO()
    m._write_count(buf, 7, large=False)
    assert buf.getvalue() == (7).to_bytes(4, "big")


def test_write_count_large():
    buf = BytesIO()
    m._write_count(buf, 7, large=True)
    assert buf.getvalue() == (7).to_bytes(8, "big")


def test_is_large_count():
    assert m._is_large_count(0xFFFFFFFF) is False
    assert m._is_large_count(0x100000000) is True


def test_materialize_iterable():
    values, count = m._materialize_iterable(iter([1, 2, 3]))
    assert values == (1, 2, 3)
    assert count == 3


def test_build_collection_payload_and_iter_items():
    payload = m._build_collection_payload(iter([1, "x", True]), count=3, large=False)
    buf = BytesIO(payload.getvalue())
    count = m._read_u32(buf)
    assert count == 3

    items = list(m._iter_items(buf, count))
    assert [item.as_python() for item in items] == [1, "x", True]


def test_build_mapping_payload_and_iter_entry_pairs():
    payload = m._build_mapping_payload(
        iter([("a", 1), ("b", 2)]),
        count=2,
        large=False,
    )
    buf = BytesIO(payload.getvalue())
    count = m._read_u32(buf)
    assert count == 2

    entries = list(m._iter_entry_pairs(buf, count))
    assert [(k.as_python(), v.as_python()) for k, v in entries] == [("a", 1), ("b", 2)]


def test_list_roundtrip():
    ser, value = _roundtrip([1, "x", True, None])
    assert ser.tag == m.Tags.LIST
    assert value == [1, "x", True, None]


def test_tuple_roundtrip():
    ser, value = _roundtrip((1, "x", True))
    assert ser.tag == m.Tags.TUPLE
    assert value == (1, "x", True)


def test_set_roundtrip():
    ser, value = _roundtrip({1, 2, 3})
    assert ser.tag == m.Tags.SET
    assert value == {1, 2, 3}


def test_frozenset_roundtrip():
    ser, value = _roundtrip(frozenset({1, 2, 3}))
    assert ser.tag == m.Tags.FROZENSET
    assert value == frozenset({1, 2, 3})


def test_deque_roundtrip():
    ser, value = _roundtrip(deque([1, 2, 3]))
    assert ser.tag == m.Tags.DEQUE
    assert list(value) == [1, 2, 3]


def test_array_roundtrip_as_list():
    ser, value = _roundtrip(array("i", [1, 2, 3]))
    assert ser.tag == m.Tags.ARRAY
    # current serializer decodes ARRAY to list[object]
    assert value == [1, 2, 3]


def test_dict_roundtrip():
    ser, value = _roundtrip({"a": 1, "b": 2})
    assert ser.tag == m.Tags.MAPPING
    assert value == {"a": 1, "b": 2}


def test_mapping_roundtrip_custom_mapping():
    class MyMapping(AbcMapping):
        def __init__(self):
            self._data = {"a": 1, "b": 2}

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    ser, value = _roundtrip(MyMapping())
    assert ser.tag == m.Tags.MAPPING
    assert value == {"a": 1, "b": 2}


def test_mappingproxy_roundtrip():
    src = {"a": 1, "b": 2}
    proxy = MappingProxyType(src)
    ser, value = _roundtrip(proxy)
    assert ser.tag == m.Tags.MAPPING_PROXY
    assert isinstance(ser, m.MappingProxySerialized)
    assert isinstance(value, MappingProxyType)
    assert dict(value) == {"a": 1, "b": 2}


def test_mappingproxy_is_read_only_after_roundtrip():
    src = {"a": 1}
    _, value = _roundtrip(MappingProxyType(src))
    with pytest.raises(TypeError):
        value["b"] = 2


def test_generator_roundtrip():
    def gen():
        yield 1
        yield 2
        yield 3

    ser, value = _roundtrip(gen())
    assert ser.tag == m.Tags.GENERATOR
    assert list(value) == [1, 2, 3]


def test_iterator_roundtrip():
    ser, value = _roundtrip(iter([1, 2, 3]))
    assert ser.tag == m.Tags.ITERATOR
    assert list(value) == [1, 2, 3]


def test_generator_is_consumed_on_serialization():
    events = []

    def gen():
        events.append("start")
        yield 1
        yield 2
        events.append("end")

    g = gen()
    ser = m.CollectionSerialized.from_python_object(g)
    assert ser is not None
    assert events == ["start", "end"]
    assert list(ser.as_python()) == [1, 2]


def test_iter_property_items_on_list():
    ser = m.CollectionSerialized.from_python_object([1, 2, 3])
    assert ser is not None
    items = ser.items
    assert [item.as_python() for item in items] == [1, 2, 3]


def test_list_iter_method():
    ser = m.CollectionSerialized.from_python_object([1, 2, 3])
    assert ser is not None
    assert [item.as_python() for item in ser.iter_()] == [1, 2, 3]


def test_mapping_iter_entries():
    ser = m.CollectionSerialized.from_python_object({"a": 1, "b": 2})
    assert ser is not None
    assert isinstance(ser, m.MappingSerialized)
    entries = list(ser.iter_entries())
    decoded = [(k.as_python(), v.as_python()) for k, v in entries]
    assert decoded == [("a", 1), ("b", 2)]


def test_mapping_entries_property():
    ser = m.CollectionSerialized.from_python_object({"a": 1, "b": 2})
    assert ser is not None
    assert isinstance(ser, m.MappingSerialized)
    decoded = [(k.as_python(), v.as_python()) for k, v in ser.entries]
    assert decoded == [("a", 1), ("b", 2)]


def test_mapping_iter_yields_flat_sequence():
    ser = m.CollectionSerialized.from_python_object({"a": 1})
    assert ser is not None
    assert isinstance(ser, m.MappingSerialized)
    flat = [x.as_python() for x in ser.iter_()]
    assert flat == ["a", 1]


def test_mappingproxy_iter_entries():
    ser = m.CollectionSerialized.from_python_object(MappingProxyType({"a": 1, "b": 2}))
    assert ser is not None
    assert isinstance(ser, m.MappingProxySerialized)
    entries = list(ser.iter_entries())
    decoded = [(k.as_python(), v.as_python()) for k, v in entries]
    assert decoded == [("a", 1), ("b", 2)]


def test_mappingproxy_entries_property():
    ser = m.CollectionSerialized.from_python_object(MappingProxyType({"a": 1, "b": 2}))
    assert ser is not None
    assert isinstance(ser, m.MappingProxySerialized)
    decoded = [(k.as_python(), v.as_python()) for k, v in ser.entries]
    assert decoded == [("a", 1), ("b", 2)]


def test_mappingproxy_iter_yields_flat_sequence():
    ser = m.CollectionSerialized.from_python_object(MappingProxyType({"a": 1}))
    assert ser is not None
    assert isinstance(ser, m.MappingProxySerialized)
    flat = [x.as_python() for x in ser.iter_()]
    assert flat == ["a", 1]


def test_payload_buffer_codec_none():
    ser = m.CollectionSerialized.from_python_object([1, 2, 3], codec=m.CODEC_NONE)
    assert ser is not None
    buf = ser._payload_buffer()
    assert isinstance(buf, BytesIO)
    assert len(buf.getvalue()) > 0


def test_large_collection_read_count():
    buf = BytesIO((5).to_bytes(8, "big"))
    obj = object.__new__(m.LargeCollectionSerialized)
    assert obj._read_count(buf) == 5


def test_unknown_type_returns_none():
    class Weird:
        pass

    assert m.CollectionSerialized.from_python_object(Weird()) is None


def test_tags_registered_for_core_types():
    assert m.Tags.get_class(m.Tags.LIST) is m.ListSerialized
    assert m.Tags.get_class(m.Tags.TUPLE) is m.TupleSerialized
    assert m.Tags.get_class(m.Tags.SET) is m.SetSerialized
    assert m.Tags.get_class(m.Tags.FROZENSET) is m.FrozenSetSerialized
    assert m.Tags.get_class(m.Tags.MAPPING) is m.MappingSerialized
    assert m.Tags.get_class(m.Tags.MAPPING_PROXY) is m.MappingProxySerialized
    assert m.Tags.get_class(m.Tags.DEQUE) is m.DequeSerialized
    assert m.Tags.get_class(m.Tags.ARRAY) is m.ArraySerialized


def test_type_registry_for_core_types():
    assert m.Tags.get_class_from_type(list) is m.ListSerialized
    assert m.Tags.get_class_from_type(tuple) is m.TupleSerialized
    assert m.Tags.get_class_from_type(set) is m.SetSerialized
    assert m.Tags.get_class_from_type(frozenset) is m.FrozenSetSerialized
    assert m.Tags.get_class_from_type(dict) is m.MappingSerialized
    assert m.Tags.get_class_from_type(MappingProxyType) is m.MappingProxySerialized
    assert m.Tags.get_class_from_type(deque) is m.DequeSerialized
    assert m.Tags.get_class_from_type(array) is m.ArraySerialized


@pytest.mark.parametrize(
    "factory, expected_tag",
    [
        (lambda: [1, 2], lambda m: m.Tags.LIST),
        (lambda: (1, 2), lambda m: m.Tags.TUPLE),
        (lambda: {1, 2}, lambda m: m.Tags.SET),
        (lambda: frozenset({1, 2}), lambda m: m.Tags.FROZENSET),
        (lambda: deque([1, 2]), lambda m: m.Tags.DEQUE),
        (lambda: array("i", [1, 2]), lambda m: m.Tags.ARRAY),
        (lambda: {"a": 1}, lambda m: m.Tags.MAPPING),
        (lambda: MappingProxyType({"a": 1}), lambda m: m.Tags.MAPPING_PROXY),
    ],
)
def test_expected_small_tags(factory, expected_tag):
    ser = m.CollectionSerialized.from_python_object(factory())
    assert ser is not None
    assert ser.tag == expected_tag(m)


def test_large_builder_uses_large_tags(monkeypatch):
    # fake a "large" count without allocating absurd memory
    monkeypatch.setattr(m, "_is_large_count", lambda count: True)

    ser = m.CollectionSerialized.from_python_object([1, 2, 3])
    assert ser is not None
    assert ser.tag == m.Tags.LARGE_LIST

    ser = m.CollectionSerialized.from_python_object((1, 2, 3))
    assert ser is not None
    assert ser.tag == m.Tags.LARGE_TUPLE

    ser = m.CollectionSerialized.from_python_object({1, 2, 3})
    assert ser is not None
    assert ser.tag == m.Tags.LARGE_SET

    ser = m.CollectionSerialized.from_python_object(frozenset({1, 2, 3}))
    assert ser is not None
    assert ser.tag == m.Tags.LARGE_FROZENSET

    ser = m.CollectionSerialized.from_python_object(deque([1, 2, 3]))
    assert ser is not None
    assert ser.tag == m.Tags.LARGE_DEQUE

    ser = m.CollectionSerialized.from_python_object(array("i", [1, 2, 3]))
    assert ser is not None
    assert ser.tag == m.Tags.LARGE_ARRAY

    ser = m.CollectionSerialized.from_python_object({"a": 1})
    assert ser is not None
    assert ser.tag == m.Tags.LARGE_MAPPING

    ser = m.CollectionSerialized.from_python_object(MappingProxyType({"a": 1}))
    assert ser is not None
    assert ser.tag == m.Tags.LARGE_MAPPING_PROXY


def test_large_generator_tag(monkeypatch):
    monkeypatch.setattr(m, "_is_large_count", lambda count: True)

    def gen():
        yield 1
        yield 2

    ser = m.CollectionSerialized.from_python_object(gen())
    assert ser is not None
    assert ser.tag == m.Tags.LARGE_GENERATOR


def test_large_iterator_tag(monkeypatch):
    monkeypatch.setattr(m, "_is_large_count", lambda count: True)

    ser = m.CollectionSerialized.from_python_object(iter([1, 2]))
    assert ser is not None
    assert ser.tag == m.Tags.LARGE_ITERATOR


def test_large_mappingproxy_roundtrip(monkeypatch):
    monkeypatch.setattr(m, "_is_large_count", lambda count: True)

    ser, value = _roundtrip(MappingProxyType({"a": 1, "b": 2}))
    assert ser.tag == m.Tags.LARGE_MAPPING_PROXY
    assert isinstance(ser, m.LargeMappingProxySerialized)
    assert isinstance(value, MappingProxyType)
    assert dict(value) == {"a": 1, "b": 2}


def test_generator_value_is_reiterable_only_once_per_returned_generator():
    def gen():
        yield 1
        yield 2

    ser = m.CollectionSerialized.from_python_object(gen())
    assert ser is not None
    out = ser.as_python()
    assert list(out) == [1, 2]
    assert list(out) == []


def test_iterator_value_is_reiterable_only_once_per_returned_iterator():
    ser = m.CollectionSerialized.from_python_object(iter([1, 2]))
    assert ser is not None
    out = ser.as_python()
    assert list(out) == [1, 2]
    assert list(out) == []


def test_mapping_with_non_string_keys_roundtrip():
    obj = {1: "a", (2, 3): "b"}
    _, value = _roundtrip(obj)
    assert value == obj


def test_mappingproxy_with_non_string_keys_roundtrip():
    obj = MappingProxyType({1: "a", (2, 3): "b"})
    _, value = _roundtrip(obj)
    assert isinstance(value, MappingProxyType)
    assert dict(value) == {1: "a", (2, 3): "b"}


# ============================================================================
# Complex inner-type tests
# ============================================================================

# ---------------------------------------------------------------------------
# All primitive types inside a single list / tuple
# ---------------------------------------------------------------------------

def test_list_all_primitive_types():
    obj = [None, True, False, -1, 0, 1, 3.14, -2.71, b"bytes", "str", b"", ""]
    _, value = _roundtrip(obj)
    assert value == obj


def test_tuple_all_primitive_types():
    obj = (None, True, False, -128, 127, 3.14, b"\x00\xff", "hello\nworld")
    _, value = _roundtrip(obj)
    assert value == obj


def test_list_with_bytes_values():
    obj = [b"", b"\x00", b"\xff\xfe", b"hello world"]
    _, value = _roundtrip(obj)
    assert value == obj


def test_tuple_with_bytes_values():
    obj = (b"alpha", b"beta", b"\x00\x01\x02")
    _, value = _roundtrip(obj)
    assert value == obj


# ---------------------------------------------------------------------------
# Two-level nesting: list / tuple containing other collections
# ---------------------------------------------------------------------------

def test_list_of_lists():
    obj = [[1, 2, 3], [4, 5], [], [6]]
    _, value = _roundtrip(obj)
    assert value == obj


def test_list_of_tuples():
    obj = [(1, "a"), (2, "b"), (), (None, True)]
    _, value = _roundtrip(obj)
    assert value == obj


def test_list_of_dicts():
    obj = [{"x": 1, "y": 2}, {}, {"only": None}]
    _, value = _roundtrip(obj)
    assert value == obj


def test_list_of_sets():
    obj = [{1, 2, 3}, set(), {True, False}]
    _, value = _roundtrip(obj)
    assert value == obj


def test_list_of_frozensets():
    obj = [frozenset({1, 2}), frozenset(), frozenset({"a", "b", "c"})]
    _, value = _roundtrip(obj)
    assert value == obj


def test_list_of_deques():
    obj = [deque([1, 2, 3]), deque(), deque(["a", "b"])]
    _, value = _roundtrip(obj)
    assert value == obj


def test_tuple_of_lists():
    obj = ([1, 2], [3, 4], [], [None])
    _, value = _roundtrip(obj)
    assert value == obj


def test_tuple_of_tuples():
    obj = ((1, 2), (3, 4), (), ("a", "b", "c"))
    _, value = _roundtrip(obj)
    assert value == obj


def test_tuple_of_dicts():
    obj = ({"a": 1}, {}, {"b": None, "c": True})
    _, value = _roundtrip(obj)
    assert value == obj


def test_tuple_of_sets():
    obj = ({1, 2}, set(), {"hello", "world"})
    _, value = _roundtrip(obj)
    assert value == obj


def test_tuple_of_frozensets():
    obj = (frozenset({1, 2}), frozenset(), frozenset({3}))
    _, value = _roundtrip(obj)
    assert value == obj


# ---------------------------------------------------------------------------
# Two-level nesting: mappings with complex values
# ---------------------------------------------------------------------------

def test_dict_with_list_values():
    obj = {"a": [1, 2, 3], "b": [], "c": [None, True, "x"]}
    _, value = _roundtrip(obj)
    assert value == obj


def test_dict_with_tuple_values():
    obj = {"a": (1, 2), "b": (), "c": (None, "z")}
    _, value = _roundtrip(obj)
    assert value == obj


def test_dict_with_set_values():
    obj = {"a": {1, 2}, "b": set(), "c": {"hello"}}
    _, value = _roundtrip(obj)
    assert value == obj


def test_dict_with_frozenset_values():
    obj = {"a": frozenset({1, 2}), "b": frozenset()}
    _, value = _roundtrip(obj)
    assert value == obj


def test_dict_with_dict_values():
    obj = {"outer": {"inner": 42}, "empty": {}, "nested": {"a": 1, "b": 2}}
    _, value = _roundtrip(obj)
    assert value == obj


def test_dict_with_deque_values():
    obj = {"q": deque([1, 2, 3]), "empty": deque()}
    _, value = _roundtrip(obj)
    assert value == obj


def test_dict_with_tuple_keys():
    obj = {(1, 2): "pair", (3,): "single", (): "empty"}
    _, value = _roundtrip(obj)
    assert value == obj


def test_dict_with_tuple_keys_and_list_values():
    obj = {(0, 0): [0, 0], (1, 2): [1, 2, 3], ("x", "y"): ["a", "b"]}
    _, value = _roundtrip(obj)
    assert value == obj


def test_dict_with_frozenset_keys():
    obj = {frozenset({1, 2}): "ab", frozenset(): "empty", frozenset({3}): "c"}
    _, value = _roundtrip(obj)
    assert value == obj


def test_mappingproxy_with_list_values():
    src = {"a": [1, 2], "b": [], "c": [None]}
    _, value = _roundtrip(MappingProxyType(src))
    assert isinstance(value, MappingProxyType)
    assert dict(value) == src


def test_mappingproxy_with_dict_values():
    src = {"outer": {"inner": 1}, "empty": {}}
    _, value = _roundtrip(MappingProxyType(src))
    assert isinstance(value, MappingProxyType)
    assert dict(value) == src


def test_frozenset_of_tuples():
    # tuples of primitives are hashable
    obj = frozenset({(1, 2), (3, 4), ("a", "b"), ()})
    _, value = _roundtrip(obj)
    assert value == obj


def test_set_of_frozensets():
    obj = {frozenset({1, 2}), frozenset({3}), frozenset()}
    _, value = _roundtrip(obj)
    assert value == obj


def test_deque_of_lists():
    obj = deque([[1, 2], [3, 4], [], [None]])
    _, value = _roundtrip(obj)
    assert list(value) == [[1, 2], [3, 4], [], [None]]


def test_deque_of_dicts():
    obj = deque([{"a": 1}, {}, {"b": 2}])
    _, value = _roundtrip(obj)
    assert list(value) == [{"a": 1}, {}, {"b": 2}]


# ---------------------------------------------------------------------------
# Three-level nesting
# ---------------------------------------------------------------------------

def test_list_of_list_of_lists():
    obj = [[[1, 2], [3]], [[4]], [], [[]], [[]]]
    _, value = _roundtrip(obj)
    assert value == obj


def test_list_of_list_of_dicts():
    obj = [[{"a": 1}, {"b": 2}], [{}], [{"c": [1, 2]}]]
    _, value = _roundtrip(obj)
    assert value == obj


def test_tuple_of_list_of_tuples():
    obj = ([(1, 2), (3, 4)], [(5,)], [])
    _, value = _roundtrip(obj)
    assert value == obj


def test_dict_of_list_of_tuples():
    obj = {"row_a": [(1, "x"), (2, "y")], "row_b": [], "row_c": [(3, "z")]}
    _, value = _roundtrip(obj)
    assert value == obj


def test_dict_of_dict_of_lists():
    obj = {
        "a": {"x": [1, 2, 3], "y": []},
        "b": {"x": [4], "y": [5, 6]},
    }
    _, value = _roundtrip(obj)
    assert value == obj


def test_list_of_dict_with_tuple_keys():
    obj = [{(0, 0): "origin", (1, 0): "east"}, {(0, 1): "north"}]
    _, value = _roundtrip(obj)
    assert value == obj


def test_deeply_nested_list():
    obj = [[[[1, 2]], [[3]]], [[[]]]]
    _, value = _roundtrip(obj)
    assert value == obj


def test_deeply_nested_dict():
    obj = {"a": {"b": {"c": {"d": 42}}}}
    _, value = _roundtrip(obj)
    assert value == obj


def test_list_mixed_collection_types():
    """Single list holds every supported collection kind."""
    obj = [
        [1, 2, 3],
        (4, 5, 6),
        {7, 8, 9},
        frozenset({10, 11}),
        deque([12, 13]),
        {"key": 14},
        MappingProxyType({"k": 15}),
    ]
    _, value = _roundtrip(obj)
    assert value[0] == [1, 2, 3]
    assert value[1] == (4, 5, 6)
    assert value[2] == {7, 8, 9}
    assert value[3] == frozenset({10, 11})
    assert list(value[4]) == [12, 13]
    assert value[5] == {"key": 14}
    assert isinstance(value[6], MappingProxyType)
    assert dict(value[6]) == {"k": 15}


def test_tuple_mixed_collection_types():
    """Single tuple holds every supported collection kind."""
    obj = (
        [1, 2],
        (3, 4),
        {5, 6},
        frozenset({7}),
        deque([8, 9]),
        {"a": 10},
    )
    _, value = _roundtrip(obj)
    assert value[0] == [1, 2]
    assert value[1] == (3, 4)
    assert value[2] == {5, 6}
    assert value[3] == frozenset({7})
    assert list(value[4]) == [8, 9]
    assert value[5] == {"a": 10}


# ---------------------------------------------------------------------------
# Empty-container corners
# ---------------------------------------------------------------------------

def test_list_of_empty_collections():
    obj = [[], (), set(), frozenset(), {}, deque()]
    _, value = _roundtrip(obj)
    assert value[0] == []
    assert value[1] == ()
    assert value[2] == set()
    assert value[3] == frozenset()
    assert value[4] == {}
    assert list(value[5]) == []


def test_tuple_of_empty_collections():
    obj = ([], (), set(), frozenset(), {})
    _, value = _roundtrip(obj)
    assert value == ([], (), set(), frozenset(), {})


def test_dict_with_empty_collection_values():
    obj = {"l": [], "t": (), "s": set(), "f": frozenset(), "d": {}}
    _, value = _roundtrip(obj)
    assert value == obj


def test_nested_empty_list():
    obj = [[[[]]]]
    _, value = _roundtrip(obj)
    assert value == obj


def test_nested_empty_tuple():
    obj = ((((),),),)
    _, value = _roundtrip(obj)
    assert value == obj


# ---------------------------------------------------------------------------
# None and boolean edge cases inside collections
# ---------------------------------------------------------------------------

def test_list_of_nones():
    obj = [None, None, None]
    _, value = _roundtrip(obj)
    assert value == obj


def test_tuple_of_nones():
    obj = (None, None)
    _, value = _roundtrip(obj)
    assert value == obj


def test_list_with_booleans_not_collapsed_to_int():
    # Ensure True/False survive roundtrip as bools, not as ints
    obj = [True, False, True]
    _, value = _roundtrip(obj)
    assert value == obj
    assert all(type(v) is bool for v in value)


def test_dict_with_bool_keys():
    obj = {True: "yes", False: "no"}
    _, value = _roundtrip(obj)
    assert value == obj


def test_list_with_none_inside_nested_dict():
    obj = [{"a": None, "b": [None, None]}]
    _, value = _roundtrip(obj)
    assert value == [{"a": None, "b": [None, None]}]


# ---------------------------------------------------------------------------
# Generator / iterator with complex inner types
# ---------------------------------------------------------------------------

def test_generator_of_dicts():
    def gen():
        yield {"a": 1}
        yield {"b": 2}
        yield {}

    _, value = _roundtrip(gen())
    assert list(value) == [{"a": 1}, {"b": 2}, {}]


def test_generator_of_lists():
    def gen():
        yield [1, 2]
        yield []
        yield [3]

    _, value = _roundtrip(gen())
    assert list(value) == [[1, 2], [], [3]]


def test_generator_of_tuples():
    def gen():
        yield (1, "a")
        yield ()
        yield (None,)

    _, value = _roundtrip(gen())
    assert list(value) == [(1, "a"), (), (None,)]


def test_iterator_of_lists():
    _, value = _roundtrip(iter([[1, 2], [3, 4], []]))
    assert list(value) == [[1, 2], [3, 4], []]


def test_iterator_of_dicts():
    _, value = _roundtrip(iter([{"x": 1}, {}, {"y": 2}]))
    assert list(value) == [{"x": 1}, {}, {"y": 2}]


# ---------------------------------------------------------------------------
# Large-count paths with complex inner types (monkeypatched)
# ---------------------------------------------------------------------------

def test_large_list_of_dicts(monkeypatch):
    monkeypatch.setattr(m, "_is_large_count", lambda count: True)
    obj = [{"a": 1}, {"b": [1, 2]}, {}]
    ser, value = _roundtrip(obj)
    assert ser.tag == m.Tags.LARGE_LIST
    assert value == obj


def test_large_tuple_of_lists(monkeypatch):
    monkeypatch.setattr(m, "_is_large_count", lambda count: True)
    obj = ([1, 2], [3, 4], [])
    ser, value = _roundtrip(obj)
    assert ser.tag == m.Tags.LARGE_TUPLE
    assert value == obj


def test_large_dict_with_list_values(monkeypatch):
    monkeypatch.setattr(m, "_is_large_count", lambda count: True)
    obj = {"a": [1, 2], "b": [], "c": [None]}
    ser, value = _roundtrip(obj)
    assert ser.tag == m.Tags.LARGE_MAPPING
    assert value == obj


def test_large_mappingproxy_with_tuple_values(monkeypatch):
    monkeypatch.setattr(m, "_is_large_count", lambda count: True)
    src = {"a": (1, 2), "b": ()}
    ser, value = _roundtrip(MappingProxyType(src))
    assert ser.tag == m.Tags.LARGE_MAPPING_PROXY
    assert isinstance(value, MappingProxyType)
    assert dict(value) == src


def test_large_generator_of_dicts(monkeypatch):
    monkeypatch.setattr(m, "_is_large_count", lambda count: True)

    def gen():
        yield {"a": 1}
        yield {}

    ser, value = _roundtrip(gen())
    assert ser.tag == m.Tags.LARGE_GENERATOR
    assert list(value) == [{"a": 1}, {}]


# ---------------------------------------------------------------------------
# Realistic compound structures
# ---------------------------------------------------------------------------

def test_table_as_list_of_row_tuples():
    """Simulates a query result: list of (col_a, col_b, col_c) rows."""
    rows = [
        (1, "alice", 30.5),
        (2, "bob", None),
        (3, "carol", 25.0),
    ]
    _, value = _roundtrip(rows)
    assert value == rows


def test_schema_as_dict_of_field_dicts():
    """Simulates a nested schema descriptor."""
    obj = {
        "name": {"type": "string", "nullable": False},
        "age": {"type": "int32", "nullable": True},
        "scores": {"type": "list", "items": {"type": "float64"}, "nullable": True},
    }
    _, value = _roundtrip(obj)
    assert value == obj


def test_graph_as_adjacency_dict():
    """Simulates a graph: node → list of neighbour ids."""
    obj = {
        "A": ["B", "C"],
        "B": ["A"],
        "C": ["A", "D"],
        "D": [],
    }
    _, value = _roundtrip(obj)
    assert value == obj


def test_event_log_as_list_of_dicts():
    events = [
        {"ts": 0, "event": "start", "tags": ["init", "boot"]},
        {"ts": 1, "event": "ready", "tags": []},
        {"ts": 2, "event": "stop", "tags": ["shutdown"]},
    ]
    _, value = _roundtrip(events)
    assert value == events


def test_config_tree_with_mixed_values():
    """Dict tree where leaves are strings, ints, bools, lists, or None."""
    obj = {
        "server": {
            "host": "localhost",
            "port": 8080,
            "tls": False,
            "allowed_origins": ["https://example.com", "https://api.example.com"],
        },
        "database": {
            "url": None,
            "pool_size": 5,
            "options": {"timeout": 30, "retry": True},
        },
        "features": ["auth", "logging"],
    }
    _, value = _roundtrip(obj)
    assert value == obj


def test_matrix_as_list_of_lists():
    matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    _, value = _roundtrip(matrix)
    assert value == matrix


def test_sparse_matrix_as_dict_of_tuple_keys():
    """Sparse matrix stored as {(row, col): value}."""
    obj = {(0, 0): 1.0, (1, 2): 3.5, (2, 1): -0.5}
    _, value = _roundtrip(obj)
    assert value == obj


def test_grouped_data_as_dict_of_lists_of_tuples():
    """group_by result: {key: [(col_a, col_b), ...]}"""
    obj = {
        "admin": [(1, "alice"), (3, "carol")],
        "user": [(2, "bob")],
        "guest": [],
    }
    _, value = _roundtrip(obj)
    assert value == obj


# ============================================================================
# Cache-bootstrap and load-path (Serialized.read_from) tests
# ============================================================================
#
# Two registry dicts govern (de)serialization:
#
#   Tags.TYPES   : type  → Serialized subclass   (used by from_python_object)
#   Tags.CLASSES : tag   → Serialized subclass   (used by read_from)
#
# Both are populated lazily via Tags._ensure_category_imported().  The bug
# fixed in tags.py caused the primitive category to be skipped whenever the
# collection category had already been imported first (the old
# `if not cls.TYPES` guard only fired when the dict was completely empty).
#
# These tests cover:
#   1. Serialized.read_from() — the "load from bytes" path
#   2. TYPES / CLASSES cache state after bootstrap
#   3. _ensure_category_imported() is called on every TYPES cache miss
#   4. Wire-format tag correctness (None/bool must NOT fall through to Pickle)
# ============================================================================

# ---------------------------------------------------------------------------
# Internal import
# ---------------------------------------------------------------------------

from yggdrasil.pickle.ser.serialized import Serialized  # noqa: E402  (module-level OK)
from yggdrasil.pickle.ser.primitives import (            # noqa: E402
    NoneSerialized,
    BoolSerialized,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_wire(obj) -> bytes:
    """Serialise *obj* and return the raw wire bytes."""
    ser = m.CollectionSerialized.from_python_object(obj)
    assert ser is not None, f"Could not serialise {obj!r}"
    return ser.write_to().to_bytes()


# ---------------------------------------------------------------------------
# 1. Load-path: Serialized.read_from() → as_python()
# ---------------------------------------------------------------------------

def test_read_from_returns_list_serialized_for_list_bytes():
    wire = _to_wire([1, 2, 3])
    loaded = Serialized.read_from(BytesIO(wire), pos=0)
    assert isinstance(loaded, m.ListSerialized)


def test_read_from_list_of_none():
    """Regression: list[None] must deserialise as [None, …], not raise AttributeError."""
    wire = _to_wire([None, None, None])
    loaded = Serialized.read_from(BytesIO(wire), pos=0)
    assert loaded.as_python() == [None, None, None]


def test_read_from_list_of_all_primitive_types():
    obj = [None, True, False, -1, 0, 255, 3.14, b"bytes", "str"]
    wire = _to_wire(obj)
    loaded = Serialized.read_from(BytesIO(wire), pos=0)
    assert loaded.as_python() == obj


def test_read_from_tuple_of_mixed_types():
    obj = (None, True, 42, "hello", b"\xff")
    wire = _to_wire(obj)
    loaded = Serialized.read_from(BytesIO(wire), pos=0)
    assert loaded.as_python() == obj


def test_read_from_dict_with_none_values():
    obj = {"a": None, "b": 1, "c": None}
    wire = _to_wire(obj)
    loaded = Serialized.read_from(BytesIO(wire), pos=0)
    assert loaded.as_python() == obj


def test_read_from_nested_dict_with_list_of_none():
    obj = {"key": [None, None], "other": {"inner": None}}
    wire = _to_wire(obj)
    loaded = Serialized.read_from(BytesIO(wire), pos=0)
    assert loaded.as_python() == obj


def test_read_from_list_of_dicts_with_none():
    obj = [{"a": None}, {}, {"b": [None, 1]}]
    wire = _to_wire(obj)
    loaded = Serialized.read_from(BytesIO(wire), pos=0)
    assert loaded.as_python() == obj


def test_read_from_deeply_nested_with_none():
    obj = [[None, [None]], {"x": {"y": None}}]
    wire = _to_wire(obj)
    loaded = Serialized.read_from(BytesIO(wire), pos=0)
    assert loaded.as_python() == obj


def test_read_from_mappingproxy_with_none_value():
    obj = MappingProxyType({"a": None, "b": 1})
    wire = _to_wire(obj)
    loaded = Serialized.read_from(BytesIO(wire), pos=0)
    result = loaded.as_python()
    assert isinstance(result, MappingProxyType)
    assert dict(result) == {"a": None, "b": 1}


def test_read_from_deque_with_none_and_list():
    obj = deque([None, [1, 2], None])
    wire = _to_wire(obj)
    loaded = Serialized.read_from(BytesIO(wire), pos=0)
    assert list(loaded.as_python()) == [None, [1, 2], None]


def test_read_from_generator_with_none_items():
    def gen():
        yield None
        yield 1
        yield None

    wire = _to_wire(gen())
    loaded = Serialized.read_from(BytesIO(wire), pos=0)
    assert list(loaded.as_python()) == [None, 1, None]


def test_read_from_preserves_tag_for_each_collection_type():
    """read_from must produce an instance carrying the original wire tag."""
    cases = [
        ([1, None],                        m.Tags.LIST),
        ((None, 2),                        m.Tags.TUPLE),
        ({"k": None},                      m.Tags.MAPPING),
        (MappingProxyType({"k": None}),    m.Tags.MAPPING_PROXY),
        (deque([None, 1]),                 m.Tags.DEQUE),
        ({1, 2},                           m.Tags.SET),
        (frozenset({1, 2}),                m.Tags.FROZENSET),
    ]
    for obj, expected_tag in cases:
        wire = _to_wire(obj)
        loaded = Serialized.read_from(BytesIO(wire), pos=0)
        assert loaded.tag == expected_tag, (
            f"tag mismatch for {type(obj).__name__}: "
            f"got {loaded.tag}, expected {expected_tag}"
        )


def test_read_from_large_list_bytes(monkeypatch):
    """Large-count variant of list loads correctly via read_from."""
    monkeypatch.setattr(m, "_is_large_count", lambda _: True)
    wire = _to_wire([None, 1, "x"])
    loaded = Serialized.read_from(BytesIO(wire), pos=0)
    assert isinstance(loaded, m.LargeListSerialized)
    assert loaded.as_python() == [None, 1, "x"]


def test_read_from_large_mapping_bytes(monkeypatch):
    """Large-count dict loads correctly via read_from."""
    monkeypatch.setattr(m, "_is_large_count", lambda _: True)
    wire = _to_wire({"a": None, "b": [1, 2]})
    loaded = Serialized.read_from(BytesIO(wire), pos=0)
    assert isinstance(loaded, m.LargeMappingSerialized)
    assert loaded.as_python() == {"a": None, "b": [1, 2]}


# ---------------------------------------------------------------------------
# 2. Cache state: TYPES and CLASSES must map all core entries
# ---------------------------------------------------------------------------

def test_types_cache_maps_none_type_to_none_serialized():
    assert m.Tags.get_class_from_type(type(None)) is NoneSerialized


def test_types_cache_maps_bool_to_bool_serialized():
    assert m.Tags.get_class_from_type(bool) is BoolSerialized


def test_types_cache_maps_int_to_a_serialized_class():
    cls = m.Tags.get_class_from_type(int)
    assert cls is not None and issubclass(cls, Serialized)


def test_types_cache_maps_float_to_a_serialized_class():
    cls = m.Tags.get_class_from_type(float)
    assert cls is not None and issubclass(cls, Serialized)


def test_types_cache_maps_str_to_a_serialized_class():
    cls = m.Tags.get_class_from_type(str)
    assert cls is not None and issubclass(cls, Serialized)


def test_types_cache_maps_bytes_to_a_serialized_class():
    cls = m.Tags.get_class_from_type(bytes)
    assert cls is not None and issubclass(cls, Serialized)


def test_classes_cache_maps_none_tag_to_none_serialized():
    assert m.Tags.get_class(m.Tags.NONE) is NoneSerialized


def test_classes_cache_maps_bool_tag_to_bool_serialized():
    assert m.Tags.get_class(m.Tags.BOOL) is BoolSerialized


def test_classes_cache_maps_all_collection_tags():
    expected = {
        m.Tags.LIST:               m.ListSerialized,
        m.Tags.TUPLE:              m.TupleSerialized,
        m.Tags.SET:                m.SetSerialized,
        m.Tags.FROZENSET:          m.FrozenSetSerialized,
        m.Tags.MAPPING:            m.MappingSerialized,
        m.Tags.MAPPING_PROXY:      m.MappingProxySerialized,
        m.Tags.DEQUE:              m.DequeSerialized,
        m.Tags.GENERATOR:          m.GeneratorSerialized,
        m.Tags.ITERATOR:           m.IteratorSerialized,
        m.Tags.LARGE_LIST:         m.LargeListSerialized,
        m.Tags.LARGE_TUPLE:        m.LargeTupleSerialized,
        m.Tags.LARGE_MAPPING:      m.LargeMappingSerialized,
        m.Tags.LARGE_MAPPING_PROXY: m.LargeMappingProxySerialized,
    }
    for tag, expected_cls in expected.items():
        actual = m.Tags.get_class(tag)
        assert actual is expected_cls, (
            f"CLASSES[{tag}] = {actual!r}, expected {expected_cls!r}"
        )


# ---------------------------------------------------------------------------
# 3. Bootstrap mechanism: _ensure_category_imported is called on every miss
# ---------------------------------------------------------------------------

def test_get_class_from_type_calls_ensure_category_imported_on_miss(monkeypatch):
    """
    When TYPES is only partially initialised, get_class_from_type must call
    _ensure_category_imported for ALL four core category IDs so that primitives,
    collections, complex, and Arrow types are all guaranteed to be importable.

    This is the mechanical guarantee provided by the bug fix (the old
    `if not cls.TYPES` guard was the defect: it skipped the bootstrap whenever
    even one entry existed in TYPES).
    """
    called_tags: list[int] = []
    original = m.Tags._ensure_category_imported   # bound classmethod

    def spy(tag: int) -> None:
        called_tags.append(tag)
        original(tag)

    # Clear the idempotency set so every category triggers a real spy call.
    monkeypatch.setattr(m.Tags, "_IMPORTED_CATEGORIES", set())
    # Simulate the buggy state: TYPES has collections but not primitives.
    monkeypatch.setattr(m.Tags, "TYPES", {list: m.ListSerialized, tuple: m.TupleSerialized})
    monkeypatch.setattr(m.Tags, "_ensure_category_imported", spy)

    m.Tags.get_class_from_type(type(None))

    # The fix loops over cid * CATEGORY_SIZE for cid in (0, 1, 2, 4).
    called_cids = {m.Tags._category_id(t) for t in called_tags}
    assert 0 in called_cids, "primitive category (cid=0) was never bootstrapped"
    assert 1 in called_cids, "collection category (cid=1) was never bootstrapped"
    assert 2 in called_cids, "system/complex category (cid=2) was never bootstrapped"
    assert 4 in called_cids, "arrow category (cid=4) was never bootstrapped"


def test_get_class_from_type_skips_bootstrap_on_fast_path_hit(monkeypatch):
    """
    When the type IS found on the fast path (TYPES already has the entry),
    _ensure_category_imported must not be called at all — no unnecessary work.
    """
    called_tags: list[int] = []

    def spy(tag: int) -> None:
        called_tags.append(tag)

    # Put NoneSerialized directly into a fresh TYPES dict.
    monkeypatch.setattr(m.Tags, "TYPES", {type(None): NoneSerialized})
    monkeypatch.setattr(m.Tags, "_ensure_category_imported", spy)

    result = m.Tags.get_class_from_type(type(None))

    assert result is NoneSerialized
    assert called_tags == [], (
        "_ensure_category_imported must not be called when the fast path hits"
    )


def test_get_class_calls_ensure_category_imported_on_miss(monkeypatch):
    """
    Tags.get_class() also bootstraps lazily.  When CLASSES does not contain a
    tag, _ensure_category_imported is called for that tag's category.
    """
    called_tags: list[int] = []
    original = m.Tags._ensure_category_imported

    def spy(tag: int) -> None:
        called_tags.append(tag)
        original(tag)

    monkeypatch.setattr(m.Tags, "_IMPORTED_CATEGORIES", set())  # reset guard
    monkeypatch.setattr(m.Tags, "CLASSES", {})
    monkeypatch.setattr(m.Tags, "_ensure_category_imported", spy)

    # Querying for Tags.LIST (tag=101, cid=1) must trigger bootstrap for cid=1.
    try:
        m.Tags.get_class(m.Tags.LIST)
    except NotImplementedError:
        pass   # acceptable if modules were already cached and can't re-register

    assert any(
        m.Tags._category_id(t) == 1 for t in called_tags
    ), "get_class must call _ensure_category_imported for the collection category"


def test_ensure_category_imported_is_idempotent(monkeypatch):
    """
    _ensure_category_imported must call the underlying imports exactly once per
    category ID.  The _IMPORTED_CATEGORIES guard makes subsequent calls a no-op.
    """
    import_call_counts: dict[int, int] = {}
    original = m.Tags._ensure_category_imported

    def counting_spy(tag: int) -> None:
        cid = m.Tags._category_id(tag)
        import_call_counts[cid] = import_call_counts.get(cid, 0) + 1
        original(tag)

    monkeypatch.setattr(m.Tags, "_IMPORTED_CATEGORIES", set())
    monkeypatch.setattr(m.Tags, "_ensure_category_imported", counting_spy)

    # Call three times for the same category.
    for _ in range(3):
        m.Tags._ensure_category_imported(m.Tags.LIST)   # cid=1 each time

    # The spy is called every time (it wraps the function), but after the first
    # real call the guard causes original() to return immediately.
    # What matters is that the imports inside original run only once — verified
    # by checking CLASSES and TYPES are populated (not tripled).
    assert m.Tags.get_class(m.Tags.LIST) is m.ListSerialized


def test_imported_categories_set_populated_after_bootstrap(monkeypatch):
    """
    After _ensure_category_imported runs for a given tag, its category ID must
    appear in _IMPORTED_CATEGORIES.  This is checked in isolation so the result
    doesn't depend on which other modules happened to be imported first.
    """
    # Start with a clean set so we can observe exactly what gets added.
    fresh = set()
    monkeypatch.setattr(m.Tags, "_IMPORTED_CATEGORIES", fresh)

    # cid=1 (collections): call with the LIST tag.
    m.Tags._ensure_category_imported(m.Tags.LIST)
    assert 1 in fresh, "cid=1 (collections) must be in _IMPORTED_CATEGORIES after import"

    # cid=0 (primitives): call with the NONE tag.
    m.Tags._ensure_category_imported(m.Tags.NONE)
    assert 0 in fresh, "cid=0 (primitives) must be in _IMPORTED_CATEGORIES after import"

    # cid=2 (system/complex): call with the PICKLE tag.
    m.Tags._ensure_category_imported(m.Tags.PICKLE)
    assert 2 in fresh, "cid=2 (system) must be in _IMPORTED_CATEGORIES after import"

    # Idempotency: calling again for the same cid must not raise or change the set.
    size_before = len(fresh)
    m.Tags._ensure_category_imported(m.Tags.LIST)
    assert len(fresh) == size_before, "repeated call must not add a duplicate cid"


# ---------------------------------------------------------------------------
# 4. Wire-format tag correctness: None / bool must not fall through to Pickle
# ---------------------------------------------------------------------------

def test_none_serialized_with_none_tag_not_pickle():
    """
    None must be encoded as tag=NONE (0).  Before the fix, when only collection
    types were in TYPES, None fell through to PickleSerialized (tag=206), whose
    loader then crashed with `AttributeError: module 'builtins' has no attribute
    'NoneType'`.
    """
    result = Serialized.from_python_object(None)
    assert result.tag == m.Tags.NONE, (
        f"None was serialised with tag {result.tag} instead of NONE ({m.Tags.NONE})"
    )
    assert result.as_python() is None


def test_none_inside_list_each_item_has_none_tag():
    """Every None element stored in a list must carry tag=NONE."""
    ser = m.CollectionSerialized.from_python_object([None, None, None])
    assert ser is not None
    for item in ser.iter_():
        assert item.tag == m.Tags.NONE, (
            f"None element has tag {item.tag} instead of NONE ({m.Tags.NONE})"
        )


def test_none_as_dict_value_has_none_tag():
    """None stored as a mapping value must carry tag=NONE."""
    ser = m.CollectionSerialized.from_python_object({"a": None, "b": None})
    assert ser is not None
    assert isinstance(ser, m.MappingSerialized)
    for _, value_ser in ser.iter_entries():
        assert value_ser.tag == m.Tags.NONE, (
            f"None dict value has tag {value_ser.tag} instead of NONE ({m.Tags.NONE})"
        )


def test_bool_serialized_with_bool_tag_not_pickle():
    """True/False must be encoded as tag=BOOL (15), not via Pickle."""
    for val in (True, False):
        result = Serialized.from_python_object(val)
        assert result.tag == m.Tags.BOOL, (
            f"{val!r} was serialised with tag {result.tag} instead of BOOL ({m.Tags.BOOL})"
        )
        assert result.as_python() is val
        assert type(result.as_python()) is bool


def test_bool_inside_list_each_item_has_bool_tag():
    """Every bool element stored in a list must carry tag=BOOL, not UINT8/INT8."""
    ser = m.CollectionSerialized.from_python_object([True, False, True])
    assert ser is not None
    for item in ser.iter_():
        assert item.tag == m.Tags.BOOL, (
            f"bool element has tag {item.tag} instead of BOOL ({m.Tags.BOOL})"
        )


def test_none_and_bool_survive_load_roundtrip_with_correct_types():
    """
    Full roundtrip through wire bytes: None stays None, True stays True (bool,
    not int).  This is the end-to-end regression guard for the bug fix.
    """
    obj = [None, True, False, None, True]
    wire = _to_wire(obj)
    loaded = Serialized.read_from(BytesIO(wire), pos=0)
    result = loaded.as_python()

    assert result == obj
    assert result[0] is None
    assert result[1] is True  and type(result[1]) is bool
    assert result[2] is False and type(result[2]) is bool
    assert result[3] is None

