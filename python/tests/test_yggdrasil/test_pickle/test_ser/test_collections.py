from __future__ import annotations

from yggdrasil.pickle.ser.collections import (
    LargeListSerialized,
    LargeMappingSerialized,
    ListSerialized,
    MappingSerialized,
    SetSerialized,
    TupleSerialized,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


def _pack_count_u32(n: int) -> bytes:
    return n.to_bytes(4, "big", signed=False)


def _pack_count_u64(n: int) -> bytes:
    return n.to_bytes(8, "big", signed=False)


def _ser_bytes(tag: int, payload: bytes) -> bytes:
    return Serialized.build(tag=tag, data=payload).write_to().to_bytes()


def test_list_iter_is_lazy_and_typed() -> None:
    item1 = _ser_bytes(Tags.UINT32, (1).to_bytes(4, "big", signed=False))
    item2 = _ser_bytes(Tags.UTF8_STRING, b"abc")

    payload = _pack_count_u32(2) + item1 + item2
    ser = Serialized.build(tag=Tags.LIST, data=payload)

    assert isinstance(ser, ListSerialized)

    it = ser.iter_()
    first = next(it)
    second = next(it)

    assert first.as_python() == 1
    assert second.as_python() == "abc"


def test_list_items_materialize_serialized_children() -> None:
    item1 = _ser_bytes(Tags.BOOL, b"\x01")
    item2 = _ser_bytes(Tags.BYTES, b"xyz")

    payload = _pack_count_u32(2) + item1 + item2
    ser = Serialized.build(tag=Tags.LIST, data=payload)

    items = ser.items

    assert len(items) == 2
    assert items[0].as_python() is True
    assert items[1].as_python() == b"xyz"


def test_list_value_materializes_python_values() -> None:
    item1 = _ser_bytes(Tags.BOOL, b"\x01")
    item2 = _ser_bytes(Tags.BYTES, b"xyz")

    payload = _pack_count_u32(2) + item1 + item2
    ser = Serialized.build(tag=Tags.LIST, data=payload)

    assert isinstance(ser, ListSerialized)
    assert ser.as_python() == [True, b"xyz"]


def test_tuple_value() -> None:
    item1 = _ser_bytes(Tags.INT64, (-2).to_bytes(8, "big", signed=True))
    item2 = _ser_bytes(Tags.UTF8_STRING, b"yo")

    payload = _pack_count_u32(2) + item1 + item2
    ser = Serialized.build(tag=Tags.TUPLE, data=payload)

    assert isinstance(ser, TupleSerialized)
    assert ser.as_python() == (-2, "yo")


def test_set_value() -> None:
    item1 = _ser_bytes(Tags.UINT32, (1).to_bytes(4, "big", signed=False))
    item2 = _ser_bytes(Tags.UINT32, (2).to_bytes(4, "big", signed=False))

    payload = _pack_count_u32(2) + item1 + item2
    ser = Serialized.build(tag=Tags.SET, data=payload)

    assert isinstance(ser, SetSerialized)
    assert ser.as_python() == {1, 2}


def test_mapping_iter_entries() -> None:
    key1 = _ser_bytes(Tags.UTF8_STRING, b"a")
    val1 = _ser_bytes(Tags.UINT32, (10).to_bytes(4, "big", signed=False))
    key2 = _ser_bytes(Tags.UTF8_STRING, b"b")
    val2 = _ser_bytes(Tags.UINT32, (20).to_bytes(4, "big", signed=False))

    payload = _pack_count_u32(2) + key1 + val1 + key2 + val2
    ser = Serialized.build(tag=Tags.MAPPING, data=payload)

    assert isinstance(ser, MappingSerialized)

    entries = list(ser.iter_entries())
    assert len(entries) == 2
    assert entries[0][0].as_python() == "a"
    assert entries[0][1].as_python() == 10
    assert entries[1][0].as_python() == "b"
    assert entries[1][1].as_python() == 20


def test_mapping_iter_flattens_wire_order() -> None:
    key = _ser_bytes(Tags.UTF8_STRING, b"x")
    val = _ser_bytes(Tags.BOOL, b"\x01")

    payload = _pack_count_u32(1) + key + val
    ser = Serialized.build(tag=Tags.MAPPING, data=payload)

    flat = list(ser.iter_())
    assert len(flat) == 2
    assert flat[0].as_python() == "x"
    assert flat[1].as_python() is True


def test_mapping_value() -> None:
    key = _ser_bytes(Tags.UTF8_STRING, b"answer")
    val = _ser_bytes(Tags.UINT32, (42).to_bytes(4, "big", signed=False))

    payload = _pack_count_u32(1) + key + val
    ser = Serialized.build(tag=Tags.MAPPING, data=payload)

    assert isinstance(ser, MappingSerialized)
    assert ser.as_python() == {"answer": 42}


def test_large_list_uses_u64_count() -> None:
    item = _ser_bytes(Tags.UINT32, (7).to_bytes(4, "big", signed=False))
    payload = _pack_count_u64(1) + item

    ser = Serialized.build(tag=Tags.LARGE_LIST, data=payload)

    assert isinstance(ser, LargeListSerialized)
    assert ser.as_python() == [7]


def test_large_mapping_uses_u64_count() -> None:
    key = _ser_bytes(Tags.UTF8_STRING, b"k")
    val = _ser_bytes(Tags.BOOL, b"\x01")
    payload = _pack_count_u64(1) + key + val

    ser = Serialized.build(tag=Tags.LARGE_MAPPING, data=payload)

    assert isinstance(ser, LargeMappingSerialized)
    assert ser.as_python() == {"k": True}


def test_collection_write_to_roundtrip() -> None:
    item1 = _ser_bytes(Tags.UINT32, (5).to_bytes(4, "big", signed=False))
    item2 = _ser_bytes(Tags.UTF8_STRING, b"ok")
    payload = _pack_count_u32(2) + item1 + item2

    original = Serialized.build(tag=Tags.LIST, data=payload)
    buf = original.write_to()
    reread = Serialized.read_from(buf, pos=0)

    assert isinstance(reread, ListSerialized)
    assert reread.as_python() == [5, "ok"]


def test_large_collection_write_to_roundtrip() -> None:
    item = _ser_bytes(Tags.BOOL, b"\x01")
    payload = _pack_count_u64(1) + item

    original = Serialized.build(tag=Tags.LARGE_LIST, data=payload)
    buf = original.write_to()
    reread = Serialized.read_from(buf, pos=0)

    assert isinstance(reread, LargeListSerialized)
    assert reread.as_python() == [True]


def test_from_python_object_list_dispatches_list_serialized() -> None:
    ser = Serialized.from_python_object([1, "abc", True])

    assert isinstance(ser, ListSerialized)
    assert ser.as_python() == [1, "abc", True]


def test_from_python_object_tuple_dispatches_tuple_serialized() -> None:
    ser = Serialized.from_python_object((-2, "yo"))

    assert isinstance(ser, TupleSerialized)
    assert ser.as_python() == (-2, "yo")


def test_from_python_object_set_dispatches_set_serialized() -> None:
    ser = Serialized.from_python_object({1, 2})

    assert isinstance(ser, SetSerialized)
    assert ser.as_python() == {1, 2}


def test_from_python_object_mapping_dispatches_mapping_serialized() -> None:
    ser = Serialized.from_python_object({"answer": 42, "ok": True})

    assert isinstance(ser, MappingSerialized)
    assert ser.as_python() == {"answer": 42, "ok": True}


def test_from_python_object_nested_collections_roundtrip() -> None:
    obj = {
        "nums": [1, 2, 3],
        "pair": ("x", False),
        "nested": {"a": [True, b"z"]},
    }

    ser = Serialized.from_python_object(obj)

    assert isinstance(ser, MappingSerialized)
    assert ser.as_python() == obj


def test_from_python_object_collection_roundtrip_via_write_to() -> None:
    original = Serialized.from_python_object(
        {"items": [1, "x", False], "meta": {"k": b"v"}}
    )

    buf = original.write_to()
    reread = Serialized.read_from(buf, pos=0)

    assert reread.as_python() == {"items": [1, "x", False], "meta": {"k": b"v"}}


def test_from_python_object_list_children_are_typed_when_iterated() -> None:
    ser = Serialized.from_python_object([1, "abc", True])

    assert isinstance(ser, ListSerialized)

    children = list(ser.iter_())
    assert children[0].as_python() == 1
    assert children[1].as_python() == "abc"
    assert children[2].as_python() is True


def test_from_python_object_mapping_entries_are_typed_when_iterated() -> None:
    ser = Serialized.from_python_object({"a": 10, "b": False})

    assert isinstance(ser, MappingSerialized)

    got = {k.as_python(): v.as_python() for k, v in ser.iter_entries()}
    assert got == {"a": 10, "b": False}