from __future__ import annotations

from typing import Mapping

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.serialized import Serialized

__all__ = [
    "_safe_dump_annotation",
    "_safe_load_annotation",
    "dump_function_annotations",
    "load_function_annotations",
]

_ANN_VALUE = "v"
_ANN_REPR = "r"
_ANN_MAP_VERSION = 1


def _serialize_nested(obj: object) -> bytes:
    return Serialized.from_python_object(obj).write_to().to_bytes()


def _deserialize_nested(blob: bytes) -> object:
    return Serialized.read_from(BytesIO(blob), pos=0).as_python()


def _safe_repr(obj: object) -> str:
    try:
        return repr(obj)
    except Exception:
        obj_type = type(obj)
        return f"<unrepresentable {obj_type.__module__}.{obj_type.__qualname__}>"


def _safe_dump_annotation(annotation: object) -> tuple[str, object]:
    try:
        return (_ANN_VALUE, _serialize_nested(annotation))
    except Exception:
        return (_ANN_REPR, _safe_repr(annotation))


def _safe_load_annotation(payload: object) -> object:
    if not isinstance(payload, tuple) or len(payload) != 2:
        return _safe_repr(payload)

    kind, value = payload

    if kind == _ANN_VALUE:
        try:
            if isinstance(value, (bytes, bytearray)):
                return _deserialize_nested(bytes(value))
        except Exception:
            pass
        return _safe_repr(value)

    if kind == _ANN_REPR:
        return value if isinstance(value, str) else _safe_repr(value)

    return _safe_repr(payload)


def dump_function_annotations(
    annotations: Mapping[object, object] | None,
) -> tuple[int, dict[str, tuple[str, object]]]:
    """Encode function annotations with per-entry fallback.

    Any annotation entry that cannot be nested-serialized is downgraded to a
    string representation so function payload construction never fails because
    of annotation recursion/complexity.
    """
    if not annotations:
        return (_ANN_MAP_VERSION, {})

    out: dict[str, tuple[str, object]] = {}
    for key, value in annotations.items():
        if not isinstance(key, str):
            continue
        out[key] = _safe_dump_annotation(value)

    return (_ANN_MAP_VERSION, out)


def load_function_annotations(payload: object) -> dict[str, object]:
    """Decode function annotations payload.

    Supports both the current encoded format and legacy raw ``dict`` payloads.
    Any malformed entry is downgraded to a string representation instead of
    raising, keeping function reconstruction resilient.
    """
    if isinstance(payload, dict):
        return {k: v for k, v in payload.items() if isinstance(k, str)}

    if not isinstance(payload, tuple) or len(payload) != 2:
        return {}

    version, entries = payload
    if version != _ANN_MAP_VERSION or not isinstance(entries, dict):
        return {}

    out: dict[str, object] = {}
    for key, value in entries.items():
        if not isinstance(key, str):
            continue
        out[key] = _safe_load_annotation(value)

    return out

