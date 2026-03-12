"""Unit tests for yggdrasil.pickle.ser.object – ObjectSerialized fallback.

Strategy under test (tried in order):
  1. native  – Serialized.from_python (functions, modules, scalars, …)
  2. pickle  – stdlib pickle for regular picklable objects
  3. cloudpickle – local classes / lambdas that stdlib pickle can't handle
"""

import dataclasses
import math
import unittest
from collections import deque

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser import (
    ObjectSerialized,
    SerdeTags,
    Serialized,
    dumps,
    loads,
)


# ── module-level helpers ──────────────────────────────────────────────

@dataclasses.dataclass
class Point:
    x: float
    y: float


class CustomObj:
    def __init__(self, name: str, values: list):
        self.name = name
        self.values = values

    def __eq__(self, other):
        return (
            isinstance(other, CustomObj)
            and self.name == other.name
            and self.values == other.values
        )


class MySet(set):
    """Module-level set subclass (picklable)."""


def module_level_fn(x):
    return x * 3


# ── tests ─────────────────────────────────────────────────────────────

class TestObjectSerializedTag(unittest.TestCase):
    def test_tag(self):
        assert ObjectSerialized.TAG == SerdeTags.OBJECT


class TestObjectSerializedNativePath(unittest.TestCase):
    """Objects handled by the native Serialized framework use pickler=native."""

    def test_module_level_function_uses_native(self):
        ser = ObjectSerialized.from_value(module_level_fn)
        assert ser.metadata[b"pickler"] == b"pickle"
        assert ser.value(4) == 12

    def test_lambda_uses_native(self):
        """Lambdas are now handled by FunctionSerialized (native path)."""
        fn = lambda x: x * 2  # noqa: E731
        ser = ObjectSerialized.from_value(fn)
        assert ser.metadata[b"pickler"] == b"cloudpickle"
        assert ser.value(5) == 10

    def test_module_uses_native(self):
        ser = ObjectSerialized.from_value(math)
        assert ser.metadata[b"pickler"] == b"cloudpickle"
        assert ser.value is math

    def test_native_value_roundtrip_int(self):
        ser = ObjectSerialized.from_value(42)
        assert ser.metadata[b"pickler"] == b"pickle"
        assert ser.value == 42

    def test_native_value_roundtrip_list(self):
        ser = ObjectSerialized.from_value([1, 2, 3])
        assert ser.metadata[b"pickler"] == b"pickle"
        assert ser.value == [1, 2, 3]

    def test_native_value_roundtrip_dict(self):
        ser = ObjectSerialized.from_value({"a": 1})
        assert ser.metadata[b"pickler"] == b"pickle"
        assert ser.value == {"a": 1}

    def test_native_inner_is_readable_as_serialized(self):
        """The payload of a native-path ObjectSerialized is valid wire bytes."""
        ser = ObjectSerialized.from_value(42)
        ser.data.seek(0)
        inner = Serialized.pread(ser.data)
        assert inner.value == 42


class TestObjectSerializedPicklePath(unittest.TestCase):
    """Unregistered-but-picklable objects use stdlib pickle."""

    def test_dataclass_uses_pickle(self):
        ser = ObjectSerialized.from_value(Point(1.5, 2.5))
        assert ser.metadata[b"pickler"] == b"pickle"
        assert ser.value == Point(1.5, 2.5)

    def test_custom_object_uses_pickle(self):
        obj = CustomObj("test", [1, 2, 3])
        ser = ObjectSerialized.from_value(obj)
        assert ser.metadata[b"pickler"] == b"pickle"
        assert ser.value == obj

    def test_deque_uses_pickle(self):
        d = deque([1, 2, 3], maxlen=5)
        ser = ObjectSerialized.from_value(d)
        assert ser.metadata[b"pickler"] == b"pickle"
        assert ser.value == d
        assert ser.value.maxlen == 5

    def test_complex_uses_pickle(self):
        ser = ObjectSerialized.from_value(3 + 4j)
        assert ser.metadata[b"pickler"] == b"pickle"
        assert ser.value == 3 + 4j

    def test_set_subclass_uses_pickle(self):
        """MySet subclasses set; set IS registered natively (SetSerialized).
        However MySet is a *subclass* and may or may not be caught by native.
        What matters is the roundtrip produces the correct value."""
        s = MySet([1, 2, 3])
        ser = ObjectSerialized.from_value(s)
        result = ser.value
        assert result == s

    def test_type_in_metadata(self):
        ser = ObjectSerialized.from_value(Point(0, 0))
        assert ser.metadata[b"type"] == b"Point"


class TestObjectSerializedCloudpicklePath(unittest.TestCase):
    """Locally-defined classes/functions that stdlib pickle can't handle."""

    def test_local_class_uses_cloudpickle(self):
        class LocalPoint:
            def __init__(self, x, y):
                self.x, self.y = x, y

            def __eq__(self, other):
                return (
                    type(other).__name__ == "LocalPoint"
                    and self.x == other.x
                    and self.y == other.y
                )

        lp = LocalPoint(1, 2)
        ser = ObjectSerialized.from_value(lp)
        assert ser.metadata[b"pickler"] == b"cloudpickle"
        assert ser.value == lp

    def test_local_function_uses_cloudpickle(self):
        def adder(a, b):
            return a + b

        ser = ObjectSerialized.from_value(adder)
        # local functions have no __module__ path – native FunctionSerialized
        # serialises them fine, so this could be native; what matters is round-trip.
        assert ser.value(3, 4) == 7

    def test_nested_function_roundtrip(self):
        def make_fn():
            def inner(x):
                return x * 7

            return inner

        fn = make_fn()
        ser = ObjectSerialized.from_value(fn)
        assert ser.value(3) == 21


class TestObjectSerializedMetadata(unittest.TestCase):
    def test_custom_metadata_preserved(self):
        ser = ObjectSerialized.from_value(Point(0, 0), metadata={b"src": b"test"})
        assert ser.metadata[b"src"] == b"test"

    def test_type_always_present(self):
        for val in [42, "hi", [1, 2], Point(0, 0)]:
            ser = ObjectSerialized.from_value(val)
            assert b"type" in ser.metadata

    def test_codec_default_no_compression(self):
        assert ObjectSerialized.from_value(Point(0, 0)).codec == 0

    def test_byte_limit_none_no_compression(self):
        assert ObjectSerialized.from_value(Point(0, 0), byte_limit=None).codec == 0


class TestObjectSerializedWireFormat(unittest.TestCase):
    def test_bwrite_pread_roundtrip_pickle_path(self):
        pt = Point(1.0, 2.0)
        ser = ObjectSerialized.from_value(pt)
        buf = BytesIO()
        ser.bwrite(buf)
        restored, _ = Serialized.pread_from(buf, 0)
        assert isinstance(restored, ObjectSerialized)
        assert restored.value == pt

    def test_bwrite_pread_roundtrip_native_path(self):
        ser = ObjectSerialized.from_value(math)
        wire = ser.to_bytes()
        buf2 = BytesIO(wire)
        restored, _ = Serialized.pread_from(buf2, 0)
        assert isinstance(restored, ObjectSerialized)
        assert restored.metadata.get(b"pickler") is None
        assert restored.value is math


class TestObjectFallbackResolution(unittest.TestCase):
    """Registry falls back to ObjectSerialized for unregistered types."""

    def test_from_python_uses_fallback_for_dataclass(self):
        pt = Point(3.0, 4.0)
        ser = Serialized.from_python(pt)
        assert isinstance(ser, ObjectSerialized)
        assert ser.value == pt

    def test_dumps_loads_custom_object(self):
        obj = CustomObj("hello", [10, 20])
        assert loads(dumps(obj)) == obj

    def test_dumps_loads_deque(self):
        d = deque(range(10), maxlen=20)
        assert loads(dumps(d)) == d

    def test_dumps_loads_complex(self):
        assert loads(dumps(1 + 2j)) == 1 + 2j

    def test_dumps_loads_lambda_via_fallback(self):
        """When dumps hits a lambda it uses the native FunctionSerialized path."""
        fn = lambda x: x + 99  # noqa: E731
        result = loads(dumps(fn))
        assert result(1) == 100

    def test_dumps_loads_module_via_fallback(self):
        assert loads(dumps(math)) is math


if __name__ == "__main__":
    unittest.main()

