"""
Shared utilities and base serializer classes for the complex serialization layer.

This module provides:
- Validation helpers (_require_*)
- Object-state helpers (_dump_object_state / _restore_object_state)
- Module / class cache helpers
- Hashing helpers
- Annotation helpers
- Base Serialized sub-classes: ComplexSerialized, ModuleSerialized,
  ClassSerialized

FunctionSerialized / MethodSerialized live in callables.py.
DataclassSerialized lives in dataclasses.py.
BaseExceptionSerialized lives in exceptions.py.
complexs.py re-exports everything for backward compatibility.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, is_dataclass
from types import FunctionType, MethodType, ModuleType
from typing import ClassVar, Generic, Mapping

from yggdrasil.environ import runtime_import_module
from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.serialized import Serialized, T
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "ComplexSerialized",
    "ModuleSerialized",
    "ClassSerialized",
]

# ---------------------------------------------------------------------------
# wire-format constants
# ---------------------------------------------------------------------------

_BUILTINS_KEY = "__builtins__"
_FORMAT_VERSION = 1
_PYTHON_VERSION = tuple(sys.version_info[:3])

# ---------------------------------------------------------------------------
# module-level caches
# ---------------------------------------------------------------------------

_MODULE_CACHE: dict[str, ModuleType] = {}
_CLASS_CACHE: dict[tuple[str, str], type[object]] = {}

# ---------------------------------------------------------------------------
# object-state tags
# ---------------------------------------------------------------------------

_STATE_DEFAULT = "d"
_STATE_CUSTOM = "c"

# ---------------------------------------------------------------------------
# annotation payload tags
# ---------------------------------------------------------------------------

_ANN_VALUE = "v"
_ANN_REPR = "r"

# ---------------------------------------------------------------------------
# class-ref payload indices
# ---------------------------------------------------------------------------

_CLASS_REF_MODULE = 0
_CLASS_REF_QUALNAME = 1

# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------

def _require_dict(obj: object, *, name: str) -> dict[object, object]:
    if not isinstance(obj, dict):
        raise TypeError(f"{name} must be dict")
    return obj


def _require_tuple(obj: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(obj, tuple):
        raise TypeError(f"{name} must be tuple")
    return obj


def _require_list(obj: object, *, name: str) -> list[object]:
    if not isinstance(obj, list):
        raise TypeError(f"{name} must be list")
    return obj


def _require_str(obj: object, *, name: str) -> str:
    if not isinstance(obj, str):
        raise TypeError(f"{name} must be str")
    return obj


def _require_bytes(obj: object, *, name: str) -> bytes:
    if not isinstance(obj, (bytes, bytearray)):
        raise TypeError(f"{name} must be bytes")
    return bytes(obj)


def _require_tuple_len(
    obj: object,
    *,
    name: str,
    expected: int,
) -> tuple[object, ...]:
    out = _require_tuple(obj, name=name)
    if len(out) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(out)}")
    return out


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------

def _resolve_qualname(root: object, qualname: str) -> object:
    obj = root
    for part in qualname.split("."):
        if part == "<locals>":
            raise AttributeError("Cannot resolve local qualname segment '<locals>'")
        obj = getattr(obj, part)
    return obj


def _make_cell(value: object):
    return (lambda x: lambda: x)(value).__closure__[0]


def _serialize_nested(obj: object) -> bytes:
    try:
        return Serialized.from_python_object(obj).write_to().to_bytes()
    except AttributeError:
        raise TypeError(
            f"Object of type {type(obj).__name__} is not serializable as a nested payload"
        )


def _deserialize_nested(blob: bytes) -> object:
    return Serialized.read_from(BytesIO(blob), pos=0).as_python()


# ---------------------------------------------------------------------------
# object-state extraction / restoration
# ---------------------------------------------------------------------------

def _iter_slots(cls: type[object]) -> tuple[str, ...]:
    out: list[str] = []
    for base in reversed(cls.__mro__):
        slots = getattr(base, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name not in ("__dict__", "__weakref__"):
                out.append(name)
    return tuple(dict.fromkeys(out))


def _extract_object_state(obj: object) -> dict[str, object]:
    state: dict[str, object] = {}

    if hasattr(obj, "__dict__"):
        state.update(getattr(obj, "__dict__", {}))

    for name in _iter_slots(type(obj)):
        if name in state:
            continue
        try:
            state[name] = getattr(obj, name)
        except AttributeError:
            continue

    return state


def _get_declared_attr(cls: type[object], name: str) -> object | None:
    for base in cls.__mro__:
        if name in base.__dict__:
            return base.__dict__[name]
    return None


def _has_meaningful_custom_getstate(obj: object) -> bool:
    attr = _get_declared_attr(type(obj), "__getstate__")
    if attr is None:
        return False
    object_attr = getattr(object, "__getstate__", None)
    return attr is not object_attr


def _has_meaningful_custom_setstate(obj: object) -> bool:
    attr = _get_declared_attr(type(obj), "__setstate__")
    if attr is None:
        return False
    object_attr = getattr(object, "__setstate__", None)
    return attr is not object_attr


def _dump_object_state(obj: object) -> tuple[str, object]:
    if _has_meaningful_custom_getstate(obj):
        return (_STATE_CUSTOM, obj.__getstate__())
    return (_STATE_DEFAULT, _extract_object_state(obj))


def _restore_object_state(obj: object, payload: object) -> None:
    tag, value = _require_tuple_len(payload, name="Object state payload", expected=2)
    kind = _require_str(tag, name="Object state payload kind")

    if kind == _STATE_CUSTOM:
        if _has_meaningful_custom_setstate(obj):
            obj.__setstate__(value)
            return

        if isinstance(value, dict):
            for name, item in value.items():
                try:
                    object.__setattr__(obj, name, item)
                except Exception:
                    if hasattr(obj, "__dict__"):
                        obj.__dict__[name] = item
            return

        raise TypeError("Custom object state requires __setstate__ or dict-compatible state")

    if kind == _STATE_DEFAULT:
        state_obj = _require_dict(value, name="Default object state value")
        for name, item in state_obj.items():
            try:
                object.__setattr__(obj, name, item)
            except Exception:
                if hasattr(obj, "__dict__"):
                    obj.__dict__[name] = item
        return

    raise ValueError(f"Unsupported object state payload kind: {kind!r}")


# ---------------------------------------------------------------------------
# module / class detection and caching
# ---------------------------------------------------------------------------

def _module_file_contains_site_packages(module_name: str | None) -> bool:
    if not module_name:
        return False

    try:
        module = _module_cache_get_or_load(module_name)
    except Exception:
        return False

    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False

    path = module_file if isinstance(module_file, str) else str(module_file)
    path = path.replace("\\", "/")
    return "site-packages" in path or "site-packages" in path.lower()


def _should_reference_only_module(module_name: str | None) -> bool:
    return _module_file_contains_site_packages(module_name)


def _is_importable_class(cls: type[object]) -> bool:
    return "<locals>" not in cls.__qualname__


def _should_use_reference_only_for_class(cls: type[object]) -> bool:
    return _is_importable_class(cls) and _should_reference_only_module(
        getattr(cls, "__module__", None)
    )


def _module_cache_get_or_load(module_name: str) -> ModuleType:
    cached = _MODULE_CACHE.get(module_name)
    if cached is not None:
        return cached

    module = runtime_import_module(module_name, install=False)
    _MODULE_CACHE[module_name] = module
    return module


def _class_cache_get_or_load(module_name: str, qualname: str) -> type[object]:
    key = (module_name, qualname)
    cached = _CLASS_CACHE.get(key)
    if cached is not None:
        return cached

    module = _module_cache_get_or_load(module_name)
    obj = _resolve_qualname(module, qualname)

    if not isinstance(obj, type):
        raise TypeError(f"Resolved object is not a class: {module_name}.{qualname}")

    _CLASS_CACHE[key] = obj
    return obj


# ---------------------------------------------------------------------------
# hashing helpers
# ---------------------------------------------------------------------------

def _hash_bytes(data: bytes | None) -> str | None:
    if data is None:
        return None
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _hash_text(data: str | None) -> str | None:
    if data is None:
        return None
    return hashlib.blake2b(data.encode("utf-8"), digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# annotation helpers
# ---------------------------------------------------------------------------

def _safe_dump_annotation(annotation: object) -> tuple[str, object]:
    try:
        return (_ANN_VALUE, _serialize_nested(annotation))
    except Exception:
        return (_ANN_REPR, repr(annotation))


def _safe_load_annotation(payload: object) -> object:
    tag, value = _require_tuple_len(payload, name="Annotation payload", expected=2)
    kind = _require_str(tag, name="Annotation payload kind")

    if kind == _ANN_VALUE:
        blob = _require_bytes(value, name="Annotation payload value")
        return _deserialize_nested(blob)

    if kind == _ANN_REPR:
        return _require_str(value, name="Annotation payload value")

    raise ValueError(f"Unsupported annotation payload kind: {kind!r}")


# ---------------------------------------------------------------------------
# class-ref helpers
# ---------------------------------------------------------------------------

def _dump_class_ref(cls: type[object]) -> bytes:
    if "<locals>" in cls.__qualname__:
        raise TypeError(f"Cannot serialize non-importable local class reference: {cls!r}")
    return _serialize_nested((cls.__module__, cls.__qualname__))


def _load_class_ref(data: bytes) -> type[object]:
    payload = _require_tuple_len(_deserialize_nested(data), name="Class payload", expected=2)
    module_name = _require_str(payload[_CLASS_REF_MODULE], name="Class payload module")
    qualname = _require_str(payload[_CLASS_REF_QUALNAME], name="Class payload qualname")
    return _class_cache_get_or_load(module_name, qualname)


# ---------------------------------------------------------------------------
# base serializer classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ComplexSerialized(Serialized[T], Generic[T]):
    TAG: ClassVar[int]

    @property
    def value(self) -> T:
        raise NotImplementedError

    def as_python(self) -> T:
        return self.value

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        # Lazy imports to avoid circular dependencies with callables.py /
        # dataclasses.py / exceptions.py, which all import from this module.
        if isinstance(obj, MethodType):
            from yggdrasil.pickle.ser.callables import MethodSerialized  # noqa: PLC0415
            return MethodSerialized.build_method(obj, codec=codec)

        if isinstance(obj, FunctionType):
            from yggdrasil.pickle.ser.callables import FunctionSerialized  # noqa: PLC0415
            return FunctionSerialized.build_function(obj, codec=codec)

        if is_dataclass(obj) and not isinstance(obj, type):
            from yggdrasil.pickle.ser.dataclasses import DataclassSerialized  # noqa: PLC0415
            return DataclassSerialized.build_dataclass(obj, codec=codec)

        if isinstance(obj, BaseException):
            from yggdrasil.pickle.ser.exceptions import BaseExceptionSerialized  # noqa: PLC0415
            return BaseExceptionSerialized.build_exception(obj, codec=codec)

        if isinstance(obj, type):
            return ClassSerialized.build_class(obj, codec=codec)

        if isinstance(obj, ModuleType):
            return ModuleSerialized.build_module(obj, codec=codec)

        return None


@dataclass(frozen=True, slots=True)
class ModuleSerialized(ComplexSerialized[ModuleType]):
    TAG: ClassVar[int] = Tags.MODULE

    @property
    def value(self) -> ModuleType:
        module_name = self.decode().decode("utf-8")
        return _module_cache_get_or_load(module_name)

    @classmethod
    def build_module(
        cls,
        module: ModuleType,
        *,
        codec: int | None = None,
    ) -> Serialized[object]:
        return cls.build(
            tag=cls.TAG,
            data=module.__name__.encode("utf-8"),
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class ClassSerialized(ComplexSerialized[type[object]]):
    TAG: ClassVar[int] = Tags.CLASS

    @property
    def value(self) -> type[object]:
        return _load_class_ref(self.decode())

    @classmethod
    def build_class(
        cls,
        klass: type[object],
        *,
        codec: int | None = None,
    ) -> Serialized[object]:
        return cls.build(
            tag=cls.TAG,
            data=_dump_class_ref(klass),
            codec=codec,
        )

