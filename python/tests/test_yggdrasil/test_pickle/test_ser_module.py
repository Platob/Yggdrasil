"""Unit tests for yggdrasil.pickle.ser.module – ModuleSerialized (by reference)."""

import importlib
import json
import math
import os
import unittest

import pytest

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser import (
    ModuleSerialized,
    SerdeTags,
    Serialized,
    dumps,
    loads,
)


class TestModuleSerialized(unittest.TestCase):
    """ModuleSerialized stores module name and re-imports on read."""

    # ── tag ───────────────────────────────────────────────────────

    def test_tag(self):
        assert ModuleSerialized.TAG == SerdeTags.MODULE

    # ── basic roundtrip ──────────────────────────────────────────

    def test_math_roundtrip(self):
        ser = ModuleSerialized.from_value(math)
        assert ser.value is math

    def test_os_roundtrip(self):
        ser = ModuleSerialized.from_value(os)
        assert ser.value is os

    def test_json_roundtrip(self):
        ser = ModuleSerialized.from_value(json)
        assert ser.value is json

    def test_submodule_roundtrip(self):
        """os.path is a sub-module; its __name__ is 'posixpath' or 'ntpath'."""
        ser = ModuleSerialized.from_value(os.path)
        restored = ser.value
        assert restored is os.path

    def test_package_roundtrip(self):
        """importlib is a package."""
        ser = ModuleSerialized.from_value(importlib)
        assert ser.value is importlib

    # ── payload is just the name ─────────────────────────────────

    def test_payload_is_module_name(self):
        ser = ModuleSerialized.from_value(math)
        assert ser.payload() == b"math"

    def test_payload_submodule(self):
        ser = ModuleSerialized.from_value(os.path)
        assert ser.payload() == os.path.__name__.encode("utf-8")

    # ── metadata ─────────────────────────────────────────────────

    def test_metadata_module_key(self):
        ser = ModuleSerialized.from_value(math)
        assert ser.metadata[b"module"] == b"math"

    def test_metadata_custom(self):
        ser = ModuleSerialized.from_value(math, metadata={b"env": b"prod"})
        assert ser.metadata[b"env"] == b"prod"
        assert ser.metadata[b"module"] == b"math"

    # ── type checks ──────────────────────────────────────────────

    def test_rejects_non_module(self):
        with pytest.raises(TypeError):
            ModuleSerialized.from_value("math")

    def test_rejects_none(self):
        with pytest.raises(TypeError):
            ModuleSerialized.from_value(None)

    def test_rejects_class(self):
        with pytest.raises(TypeError):
            ModuleSerialized.from_value(int)

    # ── codec (modules are tiny, never compressed) ───────────────

    def test_codec_none(self):
        ser = ModuleSerialized.from_value(math)
        assert ser.codec == 0

    # ── wire-format roundtrip ────────────────────────────────────

    def test_bwrite_pread_roundtrip(self):
        ser = ModuleSerialized.from_value(json)
        buf = BytesIO()
        ser.bwrite(buf)
        restored, _ = Serialized.pread_from(buf, 0)
        assert isinstance(restored, ModuleSerialized)
        assert restored.value is json

    # ── identity: deserialized module is the live sys.modules one ─

    def test_value_returns_live_module(self):
        """The returned module must be the exact same object in sys.modules."""
        import sys
        ser = ModuleSerialized.from_value(math)
        assert ser.value is sys.modules["math"]


class TestModuleRegistryResolution(unittest.TestCase):
    """Serialized.from_python dispatches modules to ModuleSerialized."""

    def test_from_python_uses_module_serializer(self):
        ser = Serialized.from_python(math)
        assert isinstance(ser, ModuleSerialized)
        assert ser.value is math

    def test_dumps_loads_roundtrip(self):
        raw = dumps(math)
        assert loads(raw) is math

    def test_dumps_loads_os(self):
        assert loads(dumps(os)) is os

    def test_dumps_loads_submodule(self):
        assert loads(dumps(os.path)) is os.path


if __name__ == "__main__":
    unittest.main()

