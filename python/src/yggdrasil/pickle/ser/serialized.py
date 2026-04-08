from __future__ import annotations

import inspect
import logging
from abc import ABC
from dataclasses import dataclass, field
from types import ModuleType
from typing import Generic, Mapping, TypeVar, Any, Optional

from yggdrasil.io import BytesIO
from yggdrasil.io.buffer.bytes_view import BytesIOView
from yggdrasil.pickle.ser.codec import DEFAULT_CODEC, codec_name, compress_bytes, decompress_bytes
from yggdrasil.pickle.ser.constants import CODEC_NONE, COMPRESS_THRESHOLD
from yggdrasil.pickle.ser.header import Header
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "T",
    "Serialized",
]

T = TypeVar("T", bound=object)


def _restore_serialized_from_wire(payload: bytes) -> "Serialized[object]":
    """Rebuild a ``Serialized`` instance from its wire-format bytes."""
    return Serialized.read_from(BytesIO(payload), pos=0)


@dataclass(frozen=True, slots=True)
class Serialized(ABC, Generic[T]):
    head: Header
    data: BytesIOView

    _cached_obj: Optional[T] = field(
        init=False, default=None,
        repr=False, compare=False, hash=False
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
        from yggdrasil.pickle.ser.tags import Tags
        return Tags.get_class(tag) or Serialized

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

    def as_python(self) -> T:
        return self.decode()  # type: ignore[return-value]

    def as_cache_python(self):
        if self._cached_obj is not None:
            return self._cached_obj

        obj = self.as_python()
        object.__setattr__(self, "_cached_obj", obj)
        return self._cached_obj

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

        Notes
        -----
        - For instances, we primarily identify the type (cls = type(obj)).
        - For method descriptors, we use __objclass__ when available.
        - For extension types, we may parse a dotted qualname (e.g. "pyarrow.lib.Table")
          to recover a better module path.
        """
        if obj is None:
            return "builtins", fallback or "None"

        if isinstance(obj, ModuleType):
            return (
                getattr(obj, "__name__", None) or "builtins",
                fallback or getattr(obj, "__name__", "module"),
            )

        try:
            unwrapped = inspect.unwrap(obj)  # type: ignore[arg-type]
        except Exception:
            unwrapped = obj

        cls = None
        if inspect.isclass(unwrapped):
            cls = unwrapped
        else:
            cls = getattr(unwrapped, "__objclass__", None) or type(unwrapped)

        def _qualname(x: Any) -> str:
            return (
                getattr(x, "__qualname__", None)
                or getattr(x, "__name__", None)
                or getattr(type(x), "__qualname__", None)
                or getattr(type(x), "__name__", None)
                or fallback
            )

        mod = getattr(unwrapped, "__module__", None)
        qual = getattr(unwrapped, "__qualname__", None) or getattr(unwrapped, "__name__", None)

        if not mod:
            mod = getattr(cls, "__module__", None)
        if not qual:
            qual = _qualname(unwrapped)

        if not mod or mod == "builtins":
            cls_mod = getattr(getattr(unwrapped, "__class__", None), "__module__", None)
            if cls_mod and cls_mod != "builtins":
                mod = cls_mod

        dotted_candidate = None
        for cand in (
            getattr(cls, "__qualname__", None),
            getattr(cls, "__name__", None),
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
            qual = fallback

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

        mod, _ = cls.module_and_name(obj, fallback="unknown")

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
                MediaTypeSerialized as _MTS,
                MimeTypeSerialized as _MiTS,
                CodecSerialized as _CS,
            )
            for _ser_cls in (_MTS, _MiTS, _CS):
                out = _ser_cls.from_python_object(obj, metadata=metadata, codec=codec)
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

        if cls is Serialized:
            from yggdrasil.pickle.ser.tags import Tags

            found = Tags.get_class(head.tag)
            if found is not None:
                return found.read_from(buf, pos=0)
            return cls.read_from(buf, pos=0)

        return cls.read_from(buf, pos=0)

    @classmethod
    def read_from(
        cls,
        buffer: BytesIO,
        *,
        pos: int | None = None,
    ) -> "Serialized[object]":
        from yggdrasil.pickle.ser.tags import Tags

        head = Header.read_from(buffer, pos=pos)
        found = Tags.get_class(head.tag)

        if found is not None:
            return found(head=head, data=head.payload_view(buffer))
        return cls(head=head, data=head.payload_view(buffer))
