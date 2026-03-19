from __future__ import annotations

import io
import math
import sys
import zipfile
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
from yggdrasil.pickle.ser import Serialized, Tags
from yggdrasil.pickle.ser.complexs import (
    BaseExceptionSerialized,
    ClassSerialized,
    ComplexSerialized,
    DataclassSerialized,
    ModuleSerialized,
    _dump_dataclass_payload,
    _get_or_zip_module,
    _is_whitelisted_module,
    _load_dataclass_payload,
    _MODULE_ZIP_CACHE,
    _should_exclude_module_path,
    _MAX_MODULE_INLINE_BYTES,
)


class DemoClass:
    value = 123

    def mul(self, x: int) -> int:
        return x * 2


# ===================================================================
# ModuleSerialized – whitelist / name-only path
# ===================================================================

class TestModuleSerializedWhitelisted:
    """Whitelisted modules (stdlib, well-known) must use name-only payload."""

    def test_math_is_whitelisted(self):
        assert _is_whitelisted_module("math") is True

    def test_pytest_is_whitelisted(self):
        assert _is_whitelisted_module("pytest") is True

    def test_unknown_package_not_whitelisted(self):
        assert _is_whitelisted_module("_ygg_fake_pkg_xyz") is False

    def test_build_whitelisted_uses_name_only(self):
        ser = ModuleSerialized.build_module(math)

        assert isinstance(ser, ModuleSerialized)
        assert ser.tag == Tags.MODULE
        # name-only: no module_mode metadata
        meta = ser.metadata or {}
        assert meta.get(ModuleSerialized.M_MODULE_MODE) != ModuleSerialized._MODE_FULL

    def test_load_name_only_whitelisted_uses_install_true(self):
        """Whitelisted name-only payload calls runtime_import_module with install=True."""
        ser = ModuleSerialized.build_module(math)
        calls = []

        def fake_import(module_name, *, install, **kw):
            calls.append((module_name, install))
            return math

        with patch("yggdrasil.pickle.ser.complexs.runtime_import_module", fake_import):
            ser.as_python()

        assert calls, "runtime_import_module was not called"
        assert calls[0] == ("math", True)

    def test_load_name_only_non_whitelisted_uses_install_false(self, tmp_path):
        """Non-whitelisted name-only fallback calls runtime_import_module with install=False."""
        # Build a name-only payload for a non-whitelisted module by forcing
        # the size check to fail so build_module falls back to name-only.
        fake_mod = ModuleType("_ygg_nwl_mod")
        fake_mod.__file__ = str(tmp_path / "_ygg_nwl_mod" / "__init__.py")

        with patch(
            "yggdrasil.pickle.ser.complexs._module_dir_filtered_bytes",
            return_value=_MAX_MODULE_INLINE_BYTES + 1,
        ), patch(
            "yggdrasil.pickle.ser.complexs._get_module_root_path",
            return_value=tmp_path / "_ygg_nwl_mod",
        ):
            ser = ModuleSerialized.build_module(fake_mod)

        meta = ser.metadata or {}
        assert meta.get(ModuleSerialized.M_MODULE_MODE) != ModuleSerialized._MODE_FULL

        calls = []

        def fake_import(module_name, *, install, **kw):
            calls.append((module_name, install))
            return fake_mod

        with patch("yggdrasil.pickle.ser.complexs.runtime_import_module", fake_import):
            ser.as_python()

        assert calls, "runtime_import_module was not called"
        assert calls[0] == ("_ygg_nwl_mod", False)

    def test_value_roundtrip_whitelisted(self):
        ser = ModuleSerialized.build_module(math)
        mod = ser.as_python()

        assert mod is math
        assert mod.sqrt(9) == 3.0

    def test_write_to_roundtrip_whitelisted(self):
        original = ModuleSerialized.build_module(math)
        buf = original.write_to()
        reread = Serialized.read_from(buf, pos=0)

        assert isinstance(reread, ModuleSerialized)
        assert reread.tag == Tags.MODULE
        assert reread.as_python() is math

    def test_dispatches_from_python_object_whitelisted(self):
        ser = Serialized.from_python_object(math)

        assert isinstance(ser, ModuleSerialized)
        assert ser.as_python() is math


# ===================================================================
# ModuleSerialized – full-zip path (metadata mode)
# ===================================================================

class TestModuleSerializedFullZip:
    """Non-whitelisted, small modules must use the full-zip path with metadata."""

    def _make_fake_module(self, tmp_path: Path, name: str, source: str) -> ModuleType:
        """Write a tiny single-file module to tmp_path and import it."""
        pkg = tmp_path / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text(source, encoding="utf-8")
        if str(tmp_path) not in sys.path:
            sys.path.insert(0, str(tmp_path))
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, str(pkg / "__init__.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_small_non_whitelisted_uses_full_zip(self, tmp_path):
        mod = self._make_fake_module(tmp_path, "_ygg_tiny_mod", "ANSWER = 42\n")
        _MODULE_ZIP_CACHE.clear()

        ser = ModuleSerialized.build_module(mod)

        assert isinstance(ser, ModuleSerialized)
        assert ser.tag == Tags.MODULE
        meta = ser.metadata or {}
        assert meta.get(ModuleSerialized.M_MODULE_MODE) == ModuleSerialized._MODE_FULL

    def test_full_zip_value_fast_path_already_imported(self, tmp_path):
        """Module already in sys.modules → fast path, no extraction needed."""
        mod = self._make_fake_module(tmp_path, "_ygg_fast_mod", "ANSWER = 99\n")
        _MODULE_ZIP_CACHE.clear()

        ser = ModuleSerialized.build_module(mod)
        result = ser.as_python()

        assert isinstance(result, ModuleType)
        assert result.ANSWER == 99

    def test_full_zip_value_slow_path_extraction(self, tmp_path):
        """Module NOT in sys.modules → zip is extracted and module is imported."""
        name = "_ygg_extract_mod"
        mod = self._make_fake_module(tmp_path, name, "MAGIC = 7\n")
        _MODULE_ZIP_CACHE.clear()

        ser = ModuleSerialized.build_module(mod)

        # Now remove from sys.modules to force extraction on deserialize
        sys.modules.pop(name, None)
        sys.modules.pop(f"{name}.__init__", None)

        result = ser.as_python()

        assert isinstance(result, ModuleType)
        assert result.MAGIC == 7

    def test_write_to_roundtrip_full_zip(self, tmp_path):
        mod = self._make_fake_module(tmp_path, "_ygg_rt_mod", "VAL = 55\n")
        _MODULE_ZIP_CACHE.clear()

        original = ModuleSerialized.build_module(mod)
        buf = original.write_to()
        reread = Serialized.read_from(buf, pos=0)

        assert isinstance(reread, ModuleSerialized)
        assert reread.tag == Tags.MODULE
        meta = reread.metadata or {}
        assert meta.get(ModuleSerialized.M_MODULE_MODE) == ModuleSerialized._MODE_FULL
        result = reread.as_python()
        assert result.VAL == 55

    def test_oversized_module_falls_back_to_name_only(self, tmp_path):
        """If filtered size > 1 MB, fall back to name-only even without whitelist."""
        mod = self._make_fake_module(tmp_path, "_ygg_big_mod", "BIG = True\n")
        _MODULE_ZIP_CACHE.clear()

        with patch(
            "yggdrasil.pickle.ser.complexs._module_dir_filtered_bytes",
            return_value=_MAX_MODULE_INLINE_BYTES + 1,
        ):
            ser = ModuleSerialized.build_module(mod)

        meta = ser.metadata or {}
        assert meta.get(ModuleSerialized.M_MODULE_MODE) != ModuleSerialized._MODE_FULL

    def test_zip_error_falls_back_to_name_only(self, tmp_path):
        """If zipping fails, gracefully fall back to name-only."""
        mod = self._make_fake_module(tmp_path, "_ygg_err_mod", "ERR = True\n")
        _MODULE_ZIP_CACHE.clear()

        with patch(
            "yggdrasil.pickle.ser.complexs._get_or_zip_module",
            side_effect=OSError("zip failed"),
        ):
            ser = ModuleSerialized.build_module(mod)

        meta = ser.metadata or {}
        assert meta.get(ModuleSerialized.M_MODULE_MODE) != ModuleSerialized._MODE_FULL


# ===================================================================
# _get_or_zip_module – zip cache
# ===================================================================

class TestModuleZipCache:
    def test_cache_hit_returns_same_bytes(self, tmp_path):
        pkg = tmp_path / "_ygg_cache_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("X = 1\n", encoding="utf-8")

        _MODULE_ZIP_CACHE.clear()

        first = _get_or_zip_module("_ygg_cache_pkg", pkg)
        second = _get_or_zip_module("_ygg_cache_pkg", pkg)

        assert first is second  # exact same bytes object from cache

    def test_cache_miss_builds_valid_zip(self, tmp_path):
        pkg = tmp_path / "_ygg_zip_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("HELLO = 'world'\n", encoding="utf-8")

        _MODULE_ZIP_CACHE.clear()
        data = _get_or_zip_module("_ygg_zip_pkg", pkg)

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
        assert "__init__.py" in names

    def test_pycache_excluded_from_zip(self, tmp_path):
        pkg = tmp_path / "_ygg_excl_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("A = 1\n", encoding="utf-8")
        (pkg / "__pycache__").mkdir()
        (pkg / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00cache")

        _MODULE_ZIP_CACHE.clear()
        data = _get_or_zip_module("_ygg_excl_pkg", pkg)

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
        assert not any("__pycache__" in n for n in names)
        assert "__init__.py" in names


# ===================================================================
# _should_exclude_module_path
# ===================================================================

class TestShouldExcludeModulePath:
    def test_pycache_excluded(self):
        assert _should_exclude_module_path(("__pycache__", "foo.pyc")) is True

    def test_pyc_file_excluded(self):
        assert _should_exclude_module_path(("pkg", "mod.pyc")) is True

    def test_so_file_excluded(self):
        assert _should_exclude_module_path(("pkg", "_ext.so")) is True

    def test_egg_info_excluded(self):
        assert _should_exclude_module_path(("mypkg.egg-info", "PKG-INFO")) is True

    def test_git_excluded(self):
        assert _should_exclude_module_path((".git", "config")) is True

    def test_venv_excluded(self):
        assert _should_exclude_module_path((".venv", "lib")) is True

    def test_regular_python_file_not_excluded(self):
        assert _should_exclude_module_path(("pkg", "module.py")) is False

    def test_nested_regular_file_not_excluded(self):
        assert _should_exclude_module_path(("pkg", "sub", "utils.py")) is False


# ===================================================================
# ClassSerialized
# ===================================================================

def test_class_serialized_value_roundtrip() -> None:
    ser = ClassSerialized.build_class(DemoClass)

    assert isinstance(ser, ClassSerialized)
    cls = ser.as_python()

    assert cls is DemoClass
    assert cls.value == 123
    assert cls().mul(4) == 8


def test_class_serialized_write_to_roundtrip() -> None:
    original = ClassSerialized.build_class(DemoClass)

    buf = original.write_to()
    reread = Serialized.read_from(buf, pos=0)

    assert isinstance(reread, ClassSerialized)
    assert reread.tag == Tags.CLASS
    assert reread.as_python() is DemoClass


def test_serialized_from_python_object_dispatches_class() -> None:
    ser = Serialized.from_python_object(DemoClass)

    assert isinstance(ser, ClassSerialized)
    assert ser.as_python() is DemoClass


# ===================================================================
# BaseExceptionSerialized
# ===================================================================

class HttpError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(code, msg)
        self.code = code
        self.msg = msg


def test_base_exception_serialized_value_error_roundtrip() -> None:
    exc = ValueError("bad input")

    ser = BaseExceptionSerialized.build_exception(exc)

    assert isinstance(ser, BaseExceptionSerialized)

    got = ser.as_python()
    assert isinstance(got, ValueError)
    assert got.args == ("bad input",)


def test_base_exception_serialized_custom_exception_roundtrip() -> None:
    exc = HttpError(404, "missing")
    ser = BaseExceptionSerialized.build_exception(exc)

    got = ser.as_python()
    assert isinstance(got, HttpError)
    assert got.args == (404, "missing")
    assert got.code == 404
    assert got.msg == "missing"


def test_serialized_from_python_object_dispatches_base_exception() -> None:
    exc = RuntimeError("boom")

    ser = Serialized.from_python_object(exc)

    assert isinstance(ser, BaseExceptionSerialized)
    got = ser.as_python()
    assert isinstance(got, RuntimeError)
    assert got.args == ("boom",)


def test_base_exception_serialized_write_to_roundtrip() -> None:
    exc = KeyError("x")

    original = BaseExceptionSerialized.build_exception(exc)
    buf = original.write_to()
    reread = Serialized.read_from(buf, pos=0)

    assert isinstance(reread, BaseExceptionSerialized)
    got = reread.as_python()
    assert isinstance(got, KeyError)
    assert got.args == ("x",)


# ===================================================================
# Dataclass tests
# ===================================================================

@dataclass
class Point:
    x: int
    y: int


@dataclass
class Nested:
    name: str
    point: Point


@dataclass(frozen=True)
class FrozenPoint:
    x: int
    y: int


@dataclass
class WithNonInit:
    x: int
    y: int = field(init=False)

    def __post_init__(self) -> None:
        self.y = self.x * 10


@dataclass(slots=True)
class SlotPoint:
    x: int
    y: int


@dataclass
class StatefulPoint:
    x: int
    y: int

    def __getstate__(self):
        return {"x": self.x * 10, "y": self.y * 10}

    def __setstate__(self, state):
        self.x = state["x"] // 10
        self.y = state["y"] // 10


def sample_function(x: int, y: int = 2) -> int:
    return x + y


class BoomError(RuntimeError):
    pass


def test_dump_load_dataclass_payload_simple():
    obj = Point(1, 2)

    payload = _dump_dataclass_payload(obj)
    restored = _load_dataclass_payload(payload)

    assert isinstance(restored, Point)
    assert restored == obj
    assert restored is not obj


def test_dump_load_dataclass_payload_nested():
    obj = Nested(name="nika", point=Point(3, 4))

    payload = _dump_dataclass_payload(obj)
    restored = _load_dataclass_payload(payload)

    assert isinstance(restored, Nested)
    assert restored == obj
    assert isinstance(restored.point, Point)
    assert restored.point == Point(3, 4)


def test_dump_load_dataclass_payload_frozen():
    obj = FrozenPoint(5, 6)

    payload = _dump_dataclass_payload(obj)
    restored = _load_dataclass_payload(payload)

    assert isinstance(restored, FrozenPoint)
    assert restored == obj


def test_dump_load_dataclass_payload_non_init_field():
    obj = WithNonInit(7)
    assert obj.y == 70

    payload = _dump_dataclass_payload(obj)
    restored = _load_dataclass_payload(payload)

    assert isinstance(restored, WithNonInit)
    assert restored.x == 7
    assert restored.y == 70


def test_dump_load_dataclass_payload_extra_dict_state():
    obj = Point(10, 20)
    obj.label = "origin-ish"

    payload = _dump_dataclass_payload(obj)
    restored = _load_dataclass_payload(payload)

    assert isinstance(restored, Point)
    assert restored == Point(10, 20)
    assert restored.label == "origin-ish"


def test_dump_load_dataclass_payload_slots():
    obj = SlotPoint(8, 9)

    payload = _dump_dataclass_payload(obj)
    restored = _load_dataclass_payload(payload)

    assert isinstance(restored, SlotPoint)
    assert restored == obj


def test_dump_load_dataclass_payload_custom_getstate_setstate():
    obj = StatefulPoint(2, 3)

    payload = _dump_dataclass_payload(obj)
    restored = _load_dataclass_payload(payload)

    assert isinstance(restored, StatefulPoint)
    assert restored == StatefulPoint(2, 3)


def test_dataclass_serialized_build_and_restore():
    obj = Point(11, 22)

    ser = DataclassSerialized.build_dataclass(obj)
    restored = ser.as_python()

    assert isinstance(restored, Point)
    assert restored == obj
    assert restored is not obj


def test_dataclass_serialized_value_property():
    obj = Point(4, 5)

    ser = DataclassSerialized.build_dataclass(obj)
    restored = ser.value

    assert isinstance(restored, Point)
    assert restored == obj


def test_complex_serialized_from_python_object_uses_dataclass_serialized():
    obj = Point(100, 200)

    ser = ComplexSerialized.from_python_object(obj)

    assert ser is not None
    assert isinstance(ser, DataclassSerialized)
    restored = ser.as_python()
    assert restored == obj


def test_complex_serialized_dataclass_type_prefers_class_serialized():
    ser = ComplexSerialized.from_python_object(Point)

    assert ser is not None
    assert isinstance(ser, ClassSerialized)
    restored = ser.as_python()
    assert restored is Point


def test_complex_serialized_function_still_uses_function_serialized():
    from yggdrasil.pickle.ser.complexs import FunctionSerialized

    ser = ComplexSerialized.from_python_object(sample_function)

    assert ser is not None
    assert isinstance(ser, FunctionSerialized)
    restored = ser.as_python()
    assert restored(3, 4) == 7
    assert restored(3) == 5


def test_complex_serialized_exception_still_uses_base_exception_serialized():
    exc = BoomError("boom")
    exc.code = 500

    ser = ComplexSerialized.from_python_object(exc)

    assert ser is not None
    assert isinstance(ser, BaseExceptionSerialized)
    restored = ser.as_python()
    assert isinstance(restored, BoomError)
    assert restored.args == ("boom",)
    assert restored.code == 500


def test_complex_serialized_module_still_uses_module_serialized():
    ser = ComplexSerialized.from_python_object(pytest)

    assert ser is not None
    assert isinstance(ser, ModuleSerialized)
    restored = ser.as_python()
    assert isinstance(restored, ModuleType)
    assert restored.__name__ == pytest.__name__


def test_dataclass_serialized_roundtrip_with_nested_and_extra_state():
    obj = Nested(name="alpha", point=Point(1, 9))
    obj.tag = {"kind": "demo", "ok": True}

    ser = DataclassSerialized.build_dataclass(obj)
    restored = ser.as_python()

    assert isinstance(restored, Nested)
    assert restored == Nested(name="alpha", point=Point(1, 9))
    assert restored.tag == {"kind": "demo", "ok": True}


def test_dataclass_serialized_frozen_with_nested_value():
    @dataclass(frozen=True)
    class FrozenNested:
        point: Point
        name: str

    obj = FrozenNested(point=Point(2, 3), name="fp")

    ser = DataclassSerialized.build_dataclass(obj)
    restored = ser.as_python()

    assert is_dataclass(restored)
    assert restored.__class__.__name__ == "FrozenNested"
    assert restored.point == obj.point
    assert restored.name == obj.name


def test_complex_serialized_from_python_object_non_supported_returns_none():
    assert ComplexSerialized.from_python_object(12345) is None
    assert ComplexSerialized.from_python_object("hello") is None
    assert ComplexSerialized.from_python_object([1, 2, 3]) is None

