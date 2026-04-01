from __future__ import annotations

import pickle
import socket
import threading
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from yggdrasil.pickle.ser.libs import (
    _dump_object_state,
    _module_cache_get_or_load,
    _resolve_qualname,
    _restore_object_state,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "RuntimeResourceSerialized",
    "GenericObjectSerialized",
    "PickleSerialized",
    "StdlibPickleSerialized",
    "DillSerialized",
    "CloudPickleSerialized",
]


_GENERIC_OBJECT_VERSION = 1
_RESOURCE_VERSION = 1
_RESOURCE_LOCK = "thread_lock"
_RESOURCE_RLOCK = "thread_rlock"
_RESOURCE_SOCKET = "socket"

_THREAD_LOCK_TYPE = type(threading.Lock())
_THREAD_RLOCK_TYPE = type(threading.RLock())


@dataclass(frozen=True, slots=True)
class RuntimeResourceSerialized(Serialized[Any]):
    """
    Serializer for non-classical runtime resources.

    This targets objects that are typically unpicklable (thread locks/sockets)
    and restores semantically equivalent runtime instances.
    """

    TAG: ClassVar[int] = Tags.RUNTIME_RESOURCE

    def loads(self) -> Any:
        payload = pickle.loads(self.decode())

        if not isinstance(payload, tuple) or len(payload) != 3:
            raise TypeError("Runtime resource payload must be a 3-item tuple")

        version, kind, value = payload
        if version != _RESOURCE_VERSION:
            raise ValueError(f"Unsupported runtime resource payload version: {version!r}")
        if not isinstance(kind, str):
            raise TypeError("Runtime resource kind must be a string")

        if kind == _RESOURCE_LOCK:
            lock = threading.Lock()
            if bool(value):
                lock.acquire(blocking=False)
            return lock

        if kind == _RESOURCE_RLOCK:
            lock = threading.RLock()
            if bool(value):
                lock.acquire(blocking=False)
            return lock

        if kind == _RESOURCE_SOCKET:
            if not isinstance(value, tuple) or len(value) != 5:
                raise TypeError("Socket payload must be a 5-item tuple")
            family, sock_type, proto, timeout, is_blocking = value

            sock = socket.socket(family=family, type=sock_type, proto=proto)
            if timeout is None:
                sock.setblocking(is_blocking)
            else:
                sock.settimeout(float(timeout))
            return sock

        raise ValueError(f"Unsupported runtime resource kind: {kind!r}")

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
            if isinstance(obj, _THREAD_LOCK_TYPE):
                payload = (_RESOURCE_VERSION, _RESOURCE_LOCK, obj.locked())
            elif isinstance(obj, _THREAD_RLOCK_TYPE):
                payload = (_RESOURCE_VERSION, _RESOURCE_RLOCK, False)
            elif isinstance(obj, socket.socket):
                payload = (
                    _RESOURCE_VERSION,
                    _RESOURCE_SOCKET,
                    (obj.family, obj.type, obj.proto, obj.gettimeout(), obj.getblocking()),
                )
            else:
                return None

            data = pickle.dumps(payload, protocol=4)
            return Serialized.build(tag=cls.TAG, data=data, metadata=metadata, codec=codec)
        except Exception:
            return None


@dataclass(frozen=True, slots=True)
class GenericObjectSerialized(Serialized[Any]):
    """
    Structured fallback for importable Python objects.

    Unlike pickle opcode streams, this stores class identity + constructor args
    + object state using existing ygg nested serializations, which is generally
    more robust across Python runtime versions.
    """

    TAG: ClassVar[int] = Tags.GENERIC_OBJECT

    def loads(self) -> Any:
        payload = pickle.loads(self.decode())

        if not isinstance(payload, tuple) or len(payload) != 6:
            raise TypeError("Generic object payload must be a 6-item tuple")

        version, module_name, qualname, new_args, new_kwargs, state_payload = payload
        if version != _GENERIC_OBJECT_VERSION:
            raise ValueError(f"Unsupported generic object payload version: {version!r}")
        if not isinstance(module_name, str) or not isinstance(qualname, str):
            raise TypeError("Generic object payload module/qualname must be strings")
        if not isinstance(new_args, tuple):
            raise TypeError("Generic object payload new_args must be a tuple")
        if not isinstance(new_kwargs, dict):
            raise TypeError("Generic object payload new_kwargs must be a dict")

        klass = _resolve_qualname(_module_cache_get_or_load(module_name), qualname)
        if not isinstance(klass, type):
            raise TypeError("Generic object payload class reference did not resolve to a type")

        try:
            obj = klass.__new__(klass, *new_args, **new_kwargs)
        except Exception:
            obj = klass.__new__(klass)

        _restore_object_state(obj, state_payload)
        return obj

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
        klass = type(obj)
        module_name = getattr(klass, "__module__", "")
        qualname = getattr(klass, "__qualname__", "")

        if not module_name or not qualname or "<locals>" in qualname:
            return None

        # Typing constructs (TypeVar, ParamSpec, TypeVarTuple, …) use __reduce__
        # for pickling and their C-level __new__ requires positional arguments
        # that are not exposed via __getnewargs__/__getnewargs_ex__.  Let them
        # fall through to PickleSerialized which calls pickle.dumps directly.
        if module_name in ("typing", "typing_extensions"):
            return None

        try:
            new_args: tuple[object, ...] = ()
            new_kwargs: dict[str, object] = {}

            if hasattr(obj, "__getnewargs_ex__"):
                args_ex = obj.__getnewargs_ex__()
                if (
                    isinstance(args_ex, tuple)
                    and len(args_ex) == 2
                    and isinstance(args_ex[0], tuple)
                    and isinstance(args_ex[1], dict)
                ):
                    new_args = args_ex[0]
                    new_kwargs = args_ex[1]
            elif hasattr(obj, "__getnewargs__"):
                args = obj.__getnewargs__()
                if isinstance(args, tuple):
                    new_args = args

            state_payload = _dump_object_state(obj)
            payload = (
                _GENERIC_OBJECT_VERSION,
                module_name,
                qualname,
                new_args,
                new_kwargs,
                state_payload,
            )
            data = pickle.dumps(payload, protocol=4)
            return Serialized.build(tag=cls.TAG, data=data, metadata=metadata, codec=codec)
        except Exception:
            return None


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
        out = RuntimeResourceSerialized.from_python_object(
            obj,
            metadata=metadata,
            codec=codec,
        )
        if out is not None:
            return out

        out = GenericObjectSerialized.from_python_object(
            obj,
            metadata=metadata,
            codec=codec,
        )
        if out is not None:
            return out

        try:
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            return Serialized.build(tag=Tags.PICKLE, data=data, metadata=metadata, codec=codec)
        except Exception:
            pass

        try:
            import dill  # type: ignore

            data = dill.dumps(obj)
            return Serialized.build(tag=Tags.DILL, data=data, metadata=metadata, codec=codec)
        except Exception:
            pass

        try:
            import cloudpickle  # type: ignore

            data = cloudpickle.dumps(obj)
            return Serialized.build(tag=Tags.CLOUDPICKLE, data=data, metadata=metadata, codec=codec)
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


Tags.register_class(RuntimeResourceSerialized, tag=RuntimeResourceSerialized.TAG)
Tags.register_class(GenericObjectSerialized, tag=GenericObjectSerialized.TAG)
for cls in PickleSerialized.__subclasses__():
    Tags.register_class(cls, tag=cls.TAG)
