from __future__ import annotations

import pickle
import socket
import threading
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from yggdrasil.pickle.ser.libs import (
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


_GENERIC_OBJECT_VERSION = 1   # legacy v1: (ver, module, qualname, new_args, new_kwargs, state)
_GENERIC_OBJECT_V2 = 2        # v2: (ver, fn_module, fn_qualname, args_bytes, state_bytes, list_bytes, dict_bytes)
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

    Uses the standard ``__reduce_ex__`` protocol (v2 payload) so C-extension
    and Cython types that supply custom ``__reduce__`` implementations (e.g.
    ``pandas.Timestamp``) are handled correctly.

    Payload versions
    ----------------
    v1 (legacy): (version, module, qualname, new_args, new_kwargs, state_payload)
        Written by old code; still decoded on load.
    v2 (current): (version, fn_module, fn_qualname, args_bytes,
                   state_bytes, list_bytes, dict_bytes)
        Stores the reduction callable by module+qualname and the args/state as
        inner pickle blobs so class references embedded in args survive intact.
    """

    TAG: ClassVar[int] = Tags.GENERIC_OBJECT

    def loads(self) -> Any:
        payload = pickle.loads(self.decode())

        if not isinstance(payload, tuple) or len(payload) not in (6, 7):
            raise TypeError(
                f"Generic object payload must be a 6- or 7-item tuple, got {len(payload)}"
            )

        version = payload[0]

        # ------------------------------------------------------------------
        # v1 legacy path
        # ------------------------------------------------------------------
        if version == _GENERIC_OBJECT_VERSION:
            _, module_name, qualname, new_args, new_kwargs, state_payload = payload
            if not isinstance(module_name, str) or not isinstance(qualname, str):
                raise TypeError("Generic object payload module/qualname must be strings")
            if not isinstance(new_args, tuple):
                raise TypeError("Generic object payload new_args must be a tuple")
            if not isinstance(new_kwargs, dict):
                raise TypeError("Generic object payload new_kwargs must be a dict")

            klass = _resolve_qualname(_module_cache_get_or_load(module_name), qualname)
            if not isinstance(klass, type):
                raise TypeError(
                    "Generic object payload class reference did not resolve to a type"
                )

            try:
                obj = klass.__new__(klass, *new_args, **new_kwargs)
            except Exception:
                obj = klass.__new__(klass)

            _restore_object_state(obj, state_payload)
            return obj

        # ------------------------------------------------------------------
        # v2 reduce-based path
        # ------------------------------------------------------------------
        if version == _GENERIC_OBJECT_V2:
            if len(payload) != 7:
                raise TypeError("V2 generic object payload must have 7 items")
            _, fn_module, fn_qualname, args_bytes, state_bytes, list_bytes, dict_bytes = payload

            fn = _resolve_qualname(_module_cache_get_or_load(fn_module), fn_qualname)
            reduce_args = pickle.loads(args_bytes)
            obj = fn(*reduce_args)

            if state_bytes is not None:
                state = pickle.loads(state_bytes)
                if state is not None:
                    if hasattr(obj, "__setstate__"):
                        obj.__setstate__(state)
                    elif isinstance(state, dict):
                        try:
                            obj.__dict__.update(state)
                        except AttributeError:
                            for k, v in state.items():
                                try:
                                    object.__setattr__(obj, k, v)
                                except Exception:
                                    pass

            if list_bytes is not None:
                items = pickle.loads(list_bytes)
                if items is not None:
                    for item in items:
                        obj.append(item)  # type: ignore[union-attr]

            if dict_bytes is not None:
                items = pickle.loads(dict_bytes)
                if items is not None:
                    obj.update(items)  # type: ignore[union-attr]

            return obj

        raise ValueError(f"Unsupported generic object payload version: {version!r}")

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

        # Typing constructs (TypeVar, ParamSpec, …) use __reduce__ for pickling
        # but their C-level __new__ is not generally reconstructable this way.
        if module_name in ("typing", "typing_extensions"):
            return None

        try:
            rv = obj.__reduce_ex__(4)  # type: ignore[union-attr]
        except Exception:
            return None

        # String result means "import this name" — not our job.
        if isinstance(rv, str):
            return None

        if not isinstance(rv, tuple) or len(rv) < 2:
            return None

        fn = rv[0]
        reduce_args: tuple[object, ...] = rv[1] if rv[1] is not None else ()
        state = rv[2] if len(rv) > 2 else None
        # rv[3] / rv[4] are iterators for list/dict subclasses — consume eagerly
        list_items = list(rv[3]) if len(rv) > 3 and rv[3] is not None else None
        dict_items = list(rv[4]) if len(rv) > 4 and rv[4] is not None else None

        fn_module = getattr(fn, "__module__", None)
        fn_qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)

        if not fn_module or not fn_qualname or "<locals>" in fn_qualname:
            return None

        try:
            args_bytes = pickle.dumps(reduce_args, protocol=4)
            state_bytes = pickle.dumps(state, protocol=4) if state is not None else None
            list_bytes = pickle.dumps(list_items, protocol=4) if list_items is not None else None
            dict_bytes = pickle.dumps(dict_items, protocol=4) if dict_items is not None else None

            payload = (
                _GENERIC_OBJECT_V2,
                fn_module,
                fn_qualname,
                args_bytes,
                state_bytes,
                list_bytes,
                dict_bytes,
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

RuntimeResourceSerialized = Tags.get_class(Tags.RUNTIME_RESOURCE) or RuntimeResourceSerialized
GenericObjectSerialized = Tags.get_class(Tags.GENERIC_OBJECT) or GenericObjectSerialized
StdlibPickleSerialized = Tags.get_class(Tags.PICKLE) or StdlibPickleSerialized
DillSerialized = Tags.get_class(Tags.DILL) or DillSerialized
CloudPickleSerialized = Tags.get_class(Tags.CLOUDPICKLE) or CloudPickleSerialized

