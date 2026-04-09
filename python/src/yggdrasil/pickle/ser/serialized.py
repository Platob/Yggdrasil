from __future__ import annotations

import inspect
import logging
import threading
from abc import ABC
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Generic, Mapping, Optional, TypeVar

from yggdrasil.io import BytesIO
from yggdrasil.io.buffer.bytes_view import BytesIOView
from yggdrasil.pickle.ser.codec import (
    DEFAULT_CODEC,
    codec_name,
    compress_bytes,
    decompress_bytes,
)
from yggdrasil.pickle.ser.constants import CODEC_NONE, COMPRESS_THRESHOLD
from yggdrasil.pickle.ser.header import Header
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "T",
    "Serialized",
]

T = TypeVar("T", bound=object)

# Guard against recursive serializer dispatch chains like:
# Serialized.from_python_object -> SomeSubclass.from_python_object
# -> Serialized.from_python_object -> ...
_SERIALIZE_GUARD = threading.local()


def _restore_serialized_from_wire(payload: bytes) -> "Serialized[object]":
    """Rebuild a ``Serialized`` instance from its wire-format bytes."""
    return Serialized.read_from(BytesIO(payload), pos=0)


def _guard_key(obj: object) -> tuple[int, type]:
    return id(obj), type(obj)


def _get_guard_stack() -> set[tuple[int, type]]:
    stack = getattr(_SERIALIZE_GUARD, "active", None)
    if stack is None:
        stack = set()
        _SERIALIZE_GUARD.active = stack
    return stack


@dataclass(frozen=True, slots=True)
class Serialized(ABC, Generic[T]):
    head: Header
    data: BytesIOView

    _cached_obj: Optional[T] = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
        hash=False,
    )

    def __new__(
        cls,
        head: Header | None = None,
        data: BytesIOView | None = None,
    ):
        # ``pickle``/``copy`` may call ``__new__`` with no constructor args.
        # Keep zero-arg construction valid for dataclass instance restoration.
        if head is None or data is None:
            return object.__new__(cls)

        if cls is Serialized:
            target_cls = cls._resolve_type(head.tag)
            return object.__new__(target_cls)

        return object.__new__(cls)

    @staticmethod
    def _resolve_type(tag: int) -> type["Serialized[object]"]:
        found = Tags.get_class(tag)
        if found is None:
            return Serialized
        if not isinstance(found, type) or not issubclass(found, Serialized):
            raise TypeError(f"Tag {tag} resolved to invalid serializer class: {found!r}")
        return found

    @property
    def tag(self) -> int:
        return self.head.tag

    @property
    def codec(self) -> int:
        return self.head.codec

    @property
    def size(self) -> int:
        return self.head.size

    @property
    def metadata(self) -> dict[bytes, bytes] | None:
        return self.head.metadata

    @property
    def codec_label(self) -> str:
        return codec_name(self.codec)

    def to_bytes(self) -> bytes:
        return self.data.to_bytes()

    def decode(self) -> bytes:
        return decompress_bytes(self.to_bytes(), self.codec)

    # Subclasses should override this, not as_python().
    def _as_python_uncached(self) -> T:
        return self.decode()  # type: ignore[return-value]

    def as_python(self) -> T:
        return self.as_cache_python()

    def as_cache_python(self) -> T:
        cached = self._cached_obj
        if cached is None:
            cached = self._as_python_uncached()
            object.__setattr__(self, "_cached_obj", cached)
        return cached

    def write_to(self, buffer: BytesIO | None = None) -> BytesIO:
        return self.head.write_to(self.to_bytes(), buffer=buffer)

    def __reduce_ex__(self, protocol: int):
        """
        Prefer Yggdrasil wire-format for pickle payloads.

        This is compact and robust across Python versions because restoration
        goes through ``Serialized.read_from`` (tag-based dispatch). If writing
        wire bytes fails for any reason, fall back to standard constructor-based
        pickling.
        """
        try:
            return _restore_serialized_from_wire, (self.write_to().to_bytes(),)
        except Exception:
            return self.__class__, (self.head, self.data)

    @staticmethod
    def module_and_name(obj: Any, *, fallback: str = "") -> tuple[str, str]:
        """
        Return (module, qualname) for obj.

        Robust across Python objects, including many C-extension / PyArrow objects
        where __module__ may be missing or misleading.
        """
        if obj is None:
            return "builtins", fallback or "None"

        if isinstance(obj, ModuleType):
            mod = getattr(obj, "__name__", None) or "builtins"
            return mod, fallback or getattr(obj, "__name__", "module")

        try:
            unwrapped = inspect.unwrap(obj)  # type: ignore[arg-type]
        except Exception:
            unwrapped = obj

        try:
            if inspect.isclass(unwrapped):
                cls = unwrapped
            else:
                cls = getattr(unwrapped, "__objclass__", None) or type(unwrapped)
        except Exception:
            cls = type(unwrapped)

        def _safe_getattr(x: Any, name: str, default: Any = None) -> Any:
            try:
                return getattr(x, name, default)
            except Exception:
                return default

        def _qualname(x: Any) -> str:
            return (
                _safe_getattr(x, "__qualname__")
                or _safe_getattr(x, "__name__")
                or _safe_getattr(type(x), "__qualname__")
                or _safe_getattr(type(x), "__name__")
                or fallback
            )

        mod = _safe_getattr(unwrapped, "__module__")
        qual = _safe_getattr(unwrapped, "__qualname__") or _safe_getattr(unwrapped, "__name__")

        if not mod:
            mod = _safe_getattr(cls, "__module__")
        if not qual:
            qual = _qualname(unwrapped)

        if not mod or mod == "builtins":
            cls_mod = _safe_getattr(_safe_getattr(unwrapped, "__class__"), "__module__")
            if cls_mod and cls_mod != "builtins":
                mod = cls_mod

        dotted_candidate = None
        for cand in (
            _safe_getattr(cls, "__qualname__"),
            _safe_getattr(cls, "__name__"),
            qual,
        ):
            if isinstance(cand, str) and "." in cand:
                dotted_candidate = cand
                break

        if dotted_candidate:
            left, right = dotted_candidate.rsplit(".", 1)
            if (not mod) or mod == "builtins":
                mod = left
                qual = right

        if not mod:
            mod = "builtins"
        if not qual:
            qual = fallback or "unknown"

        return mod, qual

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "Serialized[object]":
        if isinstance(obj, Serialized):
            return obj

        key = _guard_key(obj)
        active = _get_guard_stack()
        if key in active:
            raise RecursionError(
                f"Recursive serializer dispatch detected for object type {type(obj)!r}"
            )

        active.add(key)
        try:
            is_obj = not isinstance(obj, type)
            if is_obj:
                found = Tags.get_class_from_type(type(obj))
                if found is not None:
                    result = found.from_python_object(obj, metadata=metadata, codec=codec)
                    if result is not None:
                        return result

            if isinstance(obj, (logging.Logger, logging.Handler, logging.Formatter, logging.LogRecord)):
                from yggdrasil.pickle.ser.logging import LoggingSerialized as _LoggingSerialized

                out = _LoggingSerialized.from_python_object(obj, metadata=metadata, codec=codec)
                if out is not None:
                    if is_obj:
                        Tags.register_class(out.__class__, pytype=type(obj))
                    return out

            mod, name = cls.module_and_name(obj, fallback="unknown")

            if mod.startswith("yggdrasil."):
                if mod.startswith("yggdrasil.data"):
                    from yggdrasil.pickle.ser.data import DataSerialized

                    out = DataSerialized.from_python_object(obj, metadata=metadata, codec=codec)
                    if out is not None:
                        if is_obj:
                            Tags.register_class(out.__class__, pytype=type(obj))
                        return out

                if mod.startswith("yggdrasil.io"):
                    from yggdrasil.pickle.ser.http_ import HttpSerialized

                    out = HttpSerialized.from_python_object(obj, metadata=metadata, codec=codec)
                    if out is not None:
                        if is_obj:
                            Tags.register_class(out.__class__, pytype=type(obj))
                        return out

                    from yggdrasil.pickle.ser.media import (
                        CodecSerialized as _CS,
                        MediaTypeSerialized as _MTS,
                        MimeTypeSerialized as _MiTS,
                    )

                    for _ser_cls in (_MTS, _MiTS, _CS):
                        out = _ser_cls.from_python_object(obj, metadata=metadata, codec=codec)
                        if out is not None:
                            if is_obj:
                                Tags.register_class(out.__class__, pytype=type(obj))
                            return out

                if mod.startswith("yggdrasil.databricks"):
                    from yggdrasil.pickle.ser.databricks import DatabricksSerialized

                    out = DatabricksSerialized.from_python_object(obj, metadata=metadata, codec=codec)
                    if out is not None:
                        if is_obj:
                            Tags.register_class(out.__class__, pytype=type(obj))
                        return out

            if mod.startswith("databricks.sdk"):
                from yggdrasil.pickle.ser.databricks import DatabricksSerialized

                out = DatabricksSerialized.from_python_object(obj, metadata=metadata, codec=codec)
                if out is not None:
                    if is_obj:
                        Tags.register_class(out.__class__, pytype=type(obj))
                    return out

            if mod.startswith("pyarrow"):
                from yggdrasil.pickle.ser.pyarrow import ArrowSerialized

                out = ArrowSerialized.from_python_object(obj, metadata=metadata, codec=codec)
                if out is not None:
                    if is_obj:
                        Tags.register_class(out.__class__, pytype=type(obj))
                    return out

            elif mod.startswith("pandas"):
                from yggdrasil.pickle.ser.pandas import PandasSerialized

                out = PandasSerialized.from_python_object(obj, metadata=metadata, codec=codec)
                if out is not None:
                    if is_obj:
                        Tags.register_class(out.__class__, pytype=type(obj))
                    return out

            elif mod.startswith("polars"):
                from yggdrasil.pickle.ser.polars import PolarsSerialized

                out = PolarsSerialized.from_python_object(obj, metadata=metadata, codec=codec)
                if out is not None:
                    if is_obj:
                        Tags.register_class(out.__class__, pytype=type(obj))
                    return out

            elif mod.startswith("pyspark"):
                from yggdrasil.pickle.ser.pyspark import PySparkSerialized

                out = PySparkSerialized.from_python_object(obj, metadata=metadata, codec=codec)
                if out is not None:
                    if is_obj:
                        Tags.register_class(out.__class__, pytype=type(obj))
                    return out

            elif mod == "logging" or mod.startswith("logging."):
                from yggdrasil.pickle.ser.logging import LoggingSerialized

                out = LoggingSerialized.from_python_object(obj, metadata=metadata, codec=codec)
                if out is not None:
                    if is_obj:
                        Tags.register_class(out.__class__, pytype=type(obj))
                    return out

            from yggdrasil.pickle.ser.complexs import ComplexSerialized

            out = ComplexSerialized.from_python_object(obj, metadata=metadata, codec=codec)
            if out is not None:
                if is_obj:
                    Tags.register_class(out.__class__, pytype=type(obj))
                return out

            from yggdrasil.pickle.ser.pickles import PickleSerialized

            out = PickleSerialized.from_python_object(obj, metadata=metadata, codec=codec)
            if out is not None:
                if is_obj:
                    Tags.register_class(out.__class__, pytype=type(obj))
                return out

            raise ValueError(f"Cannot serialize object of type {type(obj)} (module: {mod})")
        finally:
            active.remove(key)

    @classmethod
    def build(
        cls,
        *,
        tag: int,
        data: bytes | bytearray | memoryview,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
        compress_threshold: int = COMPRESS_THRESHOLD,
    ) -> "Serialized[object]":
        raw = bytes(memoryview(data))

        if codec is None:
            if len(raw) >= compress_threshold:
                codec = DEFAULT_CODEC
                encoded = compress_bytes(raw, codec)
                if len(encoded) >= len(raw):
                    codec = CODEC_NONE
                    encoded = raw
            else:
                codec = CODEC_NONE
                encoded = raw
        else:
            encoded = compress_bytes(raw, codec)

        head = Header.build(
            tag=tag,
            codec=codec,
            size=len(encoded),
            metadata=metadata,
        )
        buf = head.write_to(encoded)
        payload = head.payload_view(buf)

        if cls is Serialized:
            target = Tags.get_class(tag)
            if target is None or not isinstance(target, type) or not issubclass(target, Serialized) or target is Serialized:
                raise NotImplementedError(
                    f"Tag {tag} resolved to invalid serializer class: {target!r}, "
                    "install more recent version with uv pip install ygg[data,databricks,pickle]>=0.6.21"
                )
        elif cls is Serialized:
            raise TypeError(f"Tag {tag} resolved to invalid serializer class: {cls!r}")
        else:
            target = cls

        return target(head=head, data=payload)

    @classmethod
    def read_from(
        cls,
        buffer: BytesIO,
        *,
        pos: int | None = None,
    ) -> "Serialized[object]":
        head = Header.read_from(buffer, pos=pos)
        found = Tags.get_class(head.tag)

        if found is not None:
            if not isinstance(found, type) or not issubclass(found, Serialized):
                raise TypeError(
                    f"Tag {head.tag} resolved to invalid serializer class: {found!r}"
                )
            return found(head=head, data=head.payload_view(buffer))

        return cls(head=head, data=head.payload_view(buffer))