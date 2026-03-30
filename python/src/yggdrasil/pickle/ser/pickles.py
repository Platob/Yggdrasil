from __future__ import annotations

import pickle
import struct
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from yggdrasil.pickle.ser.libs import (
    _dump_object_state,
    _resolve_qualname,
    _restore_object_state,
)
from yggdrasil.environ import runtime_import_module
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "AnyObjectSerialized",
    "PickleSerialized",
    "StdlibPickleSerialized",
    "DillSerialized",
    "CloudPickleSerialized",
]

_ANY_VERSION = 1
_U32 = struct.Struct(">I")


@dataclass(frozen=True, slots=True)
class PickleSerialized(Serialized[Any]):
    """Base class for pickle-family serialized Python objects."""

    TAG: ClassVar[int]

    def loads(self) -> Any:
        raise NotImplementedError

    @property
    def value(self) -> Any:
        return self.loads()

    def as_python(self) -> Any:
        return self.value

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        try:
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            return Serialized.build(tag=Tags.PICKLE, data=data, metadata=metadata, codec=codec)
        except Exception:
            return None


@dataclass(frozen=True, slots=True)
class AnyObjectSerialized(Serialized[Any]):
    """
    Catch-all serializer for arbitrary Python objects.

    Wire payload layout:
        [version:1]
        [module_len:4][module bytes]
        [qualname_len:4][qualname bytes]
        [state_wire_bytes...]
    """

    TAG: ClassVar[int] = Tags.ANY_OBJECT

    @staticmethod
    def _read_u32(blob: bytes, offset: int) -> tuple[int, int]:
        if offset + _U32.size > len(blob):
            raise ValueError("Invalid ANY_OBJECT payload: truncated length prefix")
        return _U32.unpack_from(blob, offset)[0], offset + _U32.size

    @staticmethod
    def _read_slice(blob: bytes, offset: int, size: int) -> tuple[bytes, int]:
        if size < 0:
            raise ValueError("Invalid ANY_OBJECT payload: negative field size")
        end = offset + size
        if end > len(blob):
            raise ValueError("Invalid ANY_OBJECT payload: truncated field")
        return blob[offset:end], end

    @property
    def value(self) -> Any:
        blob = self.decode()
        if len(blob) < 1 + _U32.size * 2:
            raise ValueError("Invalid ANY_OBJECT payload: too short")

        version = blob[0]
        if version != _ANY_VERSION:
            raise ValueError(f"Unsupported ANY_OBJECT payload version: {version!r}")

        offset = 1
        module_len, offset = self._read_u32(blob, offset)
        module_data, offset = self._read_slice(blob, offset, module_len)
        module_name = module_data.decode("utf-8")

        qualname_len, offset = self._read_u32(blob, offset)
        qualname_data, offset = self._read_slice(blob, offset, qualname_len)
        qualname = qualname_data.decode("utf-8")

        state_wire = blob[offset:]
        if not state_wire:
            raise ValueError("Invalid ANY_OBJECT payload: missing state payload")
        state_payload = pickle.loads(state_wire)

        module = runtime_import_module(module_name, install=False)
        cls = _resolve_qualname(module, qualname)
        if not isinstance(cls, type):
            raise TypeError(f"Decoded ANY_OBJECT class is not a type: {module_name}.{qualname}")

        obj = cls.__new__(cls)
        _restore_object_state(obj, state_payload)
        return obj

    def as_python(self) -> Any:
        return self.value

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        try:
            obj_cls = type(obj)
            module_name = obj_cls.__module__
            qualname = obj_cls.__qualname__
            if "<locals>" in qualname:
                return None

            state_payload = _dump_object_state(obj)
            state_wire = pickle.dumps(state_payload, protocol=pickle.HIGHEST_PROTOCOL)
            module_data = module_name.encode("utf-8")
            qualname_data = qualname.encode("utf-8")
            if len(module_data) > 0xFFFFFFFF or len(qualname_data) > 0xFFFFFFFF:
                return None

            payload = (
                bytes([_ANY_VERSION])
                + _U32.pack(len(module_data)) + module_data
                + _U32.pack(len(qualname_data)) + qualname_data
                + state_wire
            )
            return Serialized.build(tag=cls.TAG, data=payload, metadata=metadata, codec=codec)
        except Exception:
            return None


@dataclass(frozen=True, slots=True)
class StdlibPickleSerialized(PickleSerialized):
    TAG: ClassVar[int] = Tags.PICKLE

    def loads(self) -> Any:
        return pickle.loads(self.decode())


@dataclass(frozen=True, slots=True)
class DillSerialized(PickleSerialized):
    TAG: ClassVar[int] = Tags.DILL

    def loads(self) -> Any:
        try:
            import dill
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("dill is required to load DILL payloads") from exc
        return dill.loads(self.decode())


@dataclass(frozen=True, slots=True)
class CloudPickleSerialized(PickleSerialized):
    TAG: ClassVar[int] = Tags.CLOUDPICKLE

    def loads(self) -> Any:
        try:
            import cloudpickle
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("cloudpickle is required to load CLOUDPICKLE payloads") from exc
        return cloudpickle.loads(self.decode())


for cls in PickleSerialized.__subclasses__():
    Tags.register_class(cls, tag=cls.TAG)
Tags.register_class(AnyObjectSerialized, tag=AnyObjectSerialized.TAG)
