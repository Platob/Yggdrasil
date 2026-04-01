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