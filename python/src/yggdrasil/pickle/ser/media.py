"""Pickle serializers for :class:`MediaType`, :class:`MimeType`, and :class:`Codec`.

Wire layout
-----------
``MimeTypeSerialized``
    payload = UTF-8 encoded MimeType name (e.g. ``b"JSON"``)

``CodecSerialized``
    payload = UTF-8 encoded Codec name (e.g. ``b"gzip"``)

``MediaTypeSerialized``
    payload = UTF-8 encoded ``"<mime_name>"`` or ``"<mime_name>+<codec_name>"``
    (e.g. ``b"PARQUET+gzip"`` or ``b"JSON"``)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping

from yggdrasil.io.enums.codec import Codec
from yggdrasil.io.enums.media_type import MediaType
from yggdrasil.io.enums.mime_type import MimeType
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "MediaTypeSerialized",
    "MimeTypeSerialized",
    "CodecSerialized",
]


# ============================================================================
# MimeType
# ============================================================================

@dataclass(frozen=True, slots=True)
class MimeTypeSerialized(Serialized[MimeType]):
    """
    Serialize a :class:`MimeType` as its ``.name`` attribute (e.g. ``"JSON"``).

    Deserialization reconstructs via ``MimeType._BY_NAME`` registry lookup.
    """

    TAG: ClassVar[int] = Tags.MIME_TYPE

    @property
    def value(self) -> MimeType:
        name = self.decode().decode("utf-8")
        mt = MimeType._BY_NAME.get(name.lower())
        if mt is None:
            raise ValueError(f"Unknown MimeType name: {name!r}")
        return mt

    def as_python(self) -> MimeType:
        return self.value

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,  # wire compression codec
    ) -> Serialized[object] | None:
        if not isinstance(obj, MimeType):
            return None

        payload = obj.name.encode("utf-8")
        return cls.build(tag=cls.TAG, data=payload, metadata=metadata, codec=codec)


# ============================================================================
# Codec
# ============================================================================

@dataclass(frozen=True, slots=True)
class CodecSerialized(Serialized[Codec]):
    """
    Serialize a :class:`Codec` as its ``.name`` attribute (e.g. ``"gzip"``).

    Deserialization reconstructs via ``Codec.parse``.
    """

    TAG: ClassVar[int] = Tags.CODEC

    @property
    def value(self) -> Codec:
        name = self.decode().decode("utf-8")
        result = Codec.parse(name)
        if result is None:
            raise ValueError(f"Unknown Codec name: {name!r}")
        return result

    def as_python(self) -> Codec:
        return self.value

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,  # wire compression codec
    ) -> Serialized[object] | None:
        if not isinstance(obj, Codec):
            return None

        payload = obj.name.encode("utf-8")
        return cls.build(tag=cls.TAG, data=payload, metadata=metadata, codec=codec)


# ============================================================================
# MediaType
# ============================================================================

_SEPARATOR = b"+"


@dataclass(frozen=True, slots=True)
class MediaTypeSerialized(Serialized[MediaType]):
    """
    Serialize a :class:`MediaType` as ``"<mime_name>"`` or ``"<mime_name>+<codec_name>"``.

    Examples::

        MediaType(MimeType.JSON)             → b"JSON"
        MediaType(MimeType.PARQUET, GZIP)    → b"PARQUET+gzip"
    """

    TAG: ClassVar[int] = Tags.MEDIA_TYPE

    @property
    def value(self) -> MediaType:
        raw = self.decode().decode("utf-8")

        if "+" in raw:
            mime_name, codec_name = raw.split("+", 1)
        else:
            mime_name = raw
            codec_name = None

        mt = MimeType._BY_NAME.get(mime_name.lower())
        if mt is None:
            raise ValueError(f"Unknown MimeType name: {mime_name!r}")

        c = None
        if codec_name:
            c = Codec.parse(codec_name)
            if c is None:
                raise ValueError(f"Unknown Codec name: {codec_name!r}")

        return MediaType(mime_type=mt, codec=c)

    def as_python(self) -> MediaType:
        return self.value

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,  # wire compression codec (not MediaType.codec)
    ) -> Serialized[object] | None:
        if not isinstance(obj, MediaType):
            return None

        wire: bytes = obj.mime_type.name.encode("utf-8")
        if obj.codec is not None:
            wire = wire + _SEPARATOR + obj.codec.name.encode("utf-8")

        return cls.build(tag=cls.TAG, data=wire, metadata=metadata, codec=codec)


# ============================================================================
# registration
# ============================================================================

for _cls in (MimeTypeSerialized, CodecSerialized, MediaTypeSerialized):
    Tags.register_class(_cls, tag=_cls.TAG)

Tags.register_class(MimeTypeSerialized, pytype=MimeType)
Tags.register_class(MediaTypeSerialized, pytype=MediaType)

# Codec is abstract — register all concrete subclasses
for _codec in Codec.all():
    Tags.register_class(CodecSerialized, pytype=type(_codec))

