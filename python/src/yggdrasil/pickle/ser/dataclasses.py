"""
Dataclass serialization.

This module provides:
- _dump_dataclass_payload / _load_dataclass_payload
- _dump_dataclass_class_payload / _load_dataclass_class_payload
- DataclassSerialized

All shared utilities live in libs.py.
complexs.py re-exports everything for backward compatibility.
"""

from __future__ import annotations

from dataclasses import (
    MISSING,
    dataclass,
    field as dataclass_field,
    fields,
    is_dataclass,
    make_dataclass,
)
from typing import ClassVar

from yggdrasil.pickle.ser.libs import (
    _FORMAT_VERSION,
    _STATE_DEFAULT,
    _class_cache_get_or_load,
    _deserialize_nested,
    _dump_object_state,
    _is_importable_class,
    _require_dict,
    _require_list,
    _require_str,
    _require_tuple,
    _require_tuple_len,
    _restore_object_state,
    _safe_dump_annotation,
    _safe_load_annotation,
    _serialize_nested,
    ComplexSerialized,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

__all__ = ["DataclassSerialized"]

# ---------------------------------------------------------------------------
# dataclass-specific cache
# ---------------------------------------------------------------------------

_LOCAL_DATACLASS_CACHE: dict[tuple[object, ...], type[object]] = {}

# ---------------------------------------------------------------------------
# dataclass class-payload kind flags
# ---------------------------------------------------------------------------

_DC_CLASS_REF = 0
_DC_CLASS_LOCAL = 1

# ---------------------------------------------------------------------------
# dataclass @dataclass() parameter bit-flags
# ---------------------------------------------------------------------------

_DC_REPR = 1 << 0
_DC_EQ = 1 << 1
_DC_ORDER = 1 << 2
_DC_UNSAFE_HASH = 1 << 3
_DC_FROZEN = 1 << 4
_DC_SLOTS = 1 << 5

# ---------------------------------------------------------------------------
# dataclass field bit-flags
# ---------------------------------------------------------------------------

_DCF_INIT = 1 << 0
_DCF_REPR = 1 << 1
_DCF_COMPARE = 1 << 2
_DCF_KW_ONLY = 1 << 3

# ---------------------------------------------------------------------------
# class-ref payload indices (local variant)
# ---------------------------------------------------------------------------

_DC_REF_KIND = 0
_DC_REF_MODULE = 1
_DC_REF_QUALNAME = 2

# ---------------------------------------------------------------------------
# local-class payload indices
# ---------------------------------------------------------------------------

_DC_LOCAL_KIND = 0
_DC_LOCAL_NAME = 1
_DC_LOCAL_QUALNAME = 2
_DC_LOCAL_MODULE = 3
_DC_LOCAL_FLAGS = 4
_DC_LOCAL_FIELDS = 5

# ---------------------------------------------------------------------------
# field descriptor payload indices
# ---------------------------------------------------------------------------

_DCF_NAME = 0
_DCF_ANNOTATION = 1
_DCF_FLAGS = 2
_DCF_HASH = 3
_DCF_METADATA = 4

# ---------------------------------------------------------------------------
# top-level dataclass-instance payload indices
# ---------------------------------------------------------------------------

_DC_PAYLOAD_VERSION = 0
_DC_PAYLOAD_CLASS = 1
_DC_PAYLOAD_INIT_VALUES = 2
_DC_PAYLOAD_NON_INIT_VALUES = 3
_DC_PAYLOAD_EXTRA_STATE = 4


# ---------------------------------------------------------------------------
# field-default helpers
# ---------------------------------------------------------------------------

def _field_has_explicit_default(f) -> bool:
    return getattr(f, "default", MISSING) is not MISSING


def _field_value_equals_default(f, value: object) -> bool:
    if not _field_has_explicit_default(f):
        return False

    try:
        return value == f.default
    except Exception:
        return False


# ---------------------------------------------------------------------------
# flag helpers
# ---------------------------------------------------------------------------

def _dataclass_param_flags(cls: type[object]) -> int:
    p = cls.__dataclass_params__
    flags = 0

    if p.repr:
        flags |= _DC_REPR
    if p.eq:
        flags |= _DC_EQ
    if p.order:
        flags |= _DC_ORDER
    if p.unsafe_hash:
        flags |= _DC_UNSAFE_HASH
    if p.frozen:
        flags |= _DC_FROZEN
    if "__slots__" in cls.__dict__:
        flags |= _DC_SLOTS

    return flags


def _field_flags(f) -> int:
    flags = 0
    if f.init:
        flags |= _DCF_INIT
    if f.repr:
        flags |= _DCF_REPR
    if f.compare:
        flags |= _DCF_COMPARE
    if getattr(f, "kw_only", False):
        flags |= _DCF_KW_ONLY
    return flags


def _flag_on(flags: int, bit: int) -> bool:
    return bool(flags & bit)


# ---------------------------------------------------------------------------
# dataclass class payload serialization
# ---------------------------------------------------------------------------

def _dump_dataclass_class_payload(cls: type[object]) -> tuple[object, ...]:
    if not is_dataclass(cls):
        raise TypeError(f"Expected dataclass type, got {cls!r}")

    if _is_importable_class(cls):
        return (
            _DC_CLASS_REF,
            cls.__module__,
            cls.__qualname__,
        )

    field_payloads: list[dict[str, object]] = []

    for f in fields(cls):
        field_payloads.append(
            {
                "name": f.name,
                "annotation": _safe_dump_annotation(f.type),
                "flags": _field_flags(f),
                "hash": f.hash,
                "metadata": dict(f.metadata) if f.metadata else None,
            }
        )

    return (
        _DC_CLASS_LOCAL,
        cls.__name__,
        cls.__qualname__,
        cls.__module__,
        _dataclass_param_flags(cls),
        field_payloads,
    )


def _load_dataclass_class_payload(payload: object) -> type[object]:
    data = _require_tuple(payload, name="Dataclass class payload")
    if not data:
        raise ValueError("Dataclass class payload must not be empty")

    kind = data[0]

    if kind == _DC_CLASS_REF:
        data = _require_tuple_len(data, name="Dataclass class ref payload", expected=3)
        module_name = _require_str(data[_DC_REF_MODULE], name="Dataclass class payload module")
        qualname = _require_str(data[_DC_REF_QUALNAME], name="Dataclass class payload qualname")
        cls = _class_cache_get_or_load(module_name, qualname)
        if not is_dataclass(cls):
            raise TypeError(f"Resolved class is not a dataclass: {module_name}.{qualname}")
        return cls

    if kind != _DC_CLASS_LOCAL:
        raise ValueError(f"Unsupported dataclass class payload kind: {kind!r}")

    data = _require_tuple_len(data, name="Dataclass local class payload", expected=6)

    name = _require_str(data[_DC_LOCAL_NAME], name="Dataclass local payload name")
    qualname = _require_str(data[_DC_LOCAL_QUALNAME], name="Dataclass local payload qualname")
    module_name = _require_str(data[_DC_LOCAL_MODULE], name="Dataclass local payload module")
    flags = data[_DC_LOCAL_FLAGS]
    if not isinstance(flags, int):
        raise TypeError("Dataclass local payload flags must be int")

    fields_payload = _require_list(data[_DC_LOCAL_FIELDS], name="Dataclass local payload fields")

    cache_key = (
        module_name,
        qualname,
        flags,
        repr(fields_payload),
    )
    cached = _LOCAL_DATACLASS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    spec = []
    for item in fields_payload:
        if isinstance(item, dict):
            field_name = _require_str(item.get("name"), name="Dataclass field name")
            annotation = _safe_load_annotation(item.get("annotation"))
            field_flags_value = item.get("flags")
            hash_flag = item.get("hash")
            metadata_obj = item.get("metadata")
        else:
            field_data = _require_tuple_len(item, name="Dataclass field payload", expected=5)
            field_name = _require_str(field_data[_DCF_NAME], name="Dataclass field name")
            annotation = _safe_load_annotation(field_data[_DCF_ANNOTATION])
            field_flags_value = field_data[_DCF_FLAGS]
            hash_flag = field_data[_DCF_HASH]
            metadata_obj = field_data[_DCF_METADATA]

        if not isinstance(field_flags_value, int):
            raise TypeError("Dataclass field flags must be int")

        if metadata_obj is None:
            metadata = {}
        else:
            metadata = _require_dict(metadata_obj, name="Dataclass field metadata")

        fld = dataclass_field(
            init=_flag_on(field_flags_value, _DCF_INIT),
            repr=_flag_on(field_flags_value, _DCF_REPR),
            hash=hash_flag,
            compare=_flag_on(field_flags_value, _DCF_COMPARE),
            kw_only=_flag_on(field_flags_value, _DCF_KW_ONLY),
            metadata=metadata,
        )
        spec.append((field_name, annotation, fld))

    namespace = {
        "__module__": module_name,
        "__qualname__": qualname,
    }

    cls = make_dataclass(
        cls_name=name,
        fields=spec,
        namespace=namespace,
        repr=_flag_on(flags, _DC_REPR),
        eq=_flag_on(flags, _DC_EQ),
        order=_flag_on(flags, _DC_ORDER),
        unsafe_hash=_flag_on(flags, _DC_UNSAFE_HASH),
        frozen=_flag_on(flags, _DC_FROZEN),
        slots=_flag_on(flags, _DC_SLOTS),
    )

    _LOCAL_DATACLASS_CACHE[cache_key] = cls
    return cls


# ---------------------------------------------------------------------------
# dataclass-instance payload serialization
# ---------------------------------------------------------------------------

def _dump_dataclass_payload(obj: object) -> bytes:
    try:
        obj_fields = fields(obj)
    except Exception:
        if not is_dataclass(obj) or isinstance(obj, type):
            raise TypeError(
                f"DataclassSerialized requires a dataclass instance, got {type(obj)!r}"
            )
        raise

    cls = type(obj)
    init_values: dict[str, object] = {}
    non_init_values: dict[str, object] = {}

    field_names = {f.name for f in obj_fields}

    for f in obj_fields:
        value = getattr(obj, f.name)

        if _field_value_equals_default(f, value):
            continue

        if f.init:
            init_values[f.name] = value
        # Skip heavy non init values
        # else:
        #     non_init_values[f.name] = value

    raw_state = _dump_object_state(obj)

    if raw_state[0] == _STATE_DEFAULT:
        default_state = _require_dict(raw_state[1], name="Dataclass default state")
        extra_state_payload: tuple[str, object] = (
            _STATE_DEFAULT,
            {
                key: value
                for key, value in default_state.items()
                if key not in field_names
            },
        )
    else:
        extra_state_payload = raw_state

    payload = (
        _FORMAT_VERSION,
        _dump_dataclass_class_payload(cls),
        init_values,
        non_init_values,
        extra_state_payload,
    )
    return _serialize_nested(payload)


def _load_dataclass_payload(data: bytes) -> object:
    payload = _require_tuple_len(
        _deserialize_nested(data), name="Dataclass payload", expected=5
    )

    version = payload[_DC_PAYLOAD_VERSION]
    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported dataclass payload version: {version!r}")

    class_payload = payload[_DC_PAYLOAD_CLASS]
    cls_obj = _load_dataclass_class_payload(class_payload)
    if not isinstance(cls_obj, type) or not is_dataclass(cls_obj):
        raise TypeError("Decoded dataclass class is not a dataclass type")

    init_values = _require_dict(payload[_DC_PAYLOAD_INIT_VALUES], name="Dataclass init_values")
    non_init_values = _require_dict(
        payload[_DC_PAYLOAD_NON_INIT_VALUES], name="Dataclass non_init_values"
    )
    extra_state_payload = payload[_DC_PAYLOAD_EXTRA_STATE]

    try:
        obj = cls_obj(**init_values)
    except Exception:
        init_values = {
            key: value
            for key, value in init_values.items()
            if key in cls_obj.__dataclass_fields__ and cls_obj.__dataclass_fields__[key].init
        }
        obj = cls_obj(**init_values)

    for name, value in non_init_values.items():
        object.__setattr__(obj, name, value)

    _restore_object_state(obj, extra_state_payload)
    return obj


# ---------------------------------------------------------------------------
# serializer class
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DataclassSerialized(ComplexSerialized[object]):
    TAG: ClassVar[int] = Tags.DATACLASS

    @property
    def value(self) -> object:
        return _load_dataclass_payload(self.decode())

    @classmethod
    def build_dataclass(
        cls,
        obj: object,
        *,
        codec: int | None = None,
    ) -> Serialized[object]:
        return cls.build(
            tag=cls.TAG,
            data=_dump_dataclass_payload(obj),
            codec=codec,
        )


Tags.register_class(DataclassSerialized, tag=DataclassSerialized.TAG)
DataclassSerialized = Tags.get_class(Tags.DATACLASS) or DataclassSerialized


