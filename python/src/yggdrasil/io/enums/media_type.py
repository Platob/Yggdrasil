# yggdrasil/io/enums/media_type.py
# Patch the codec-inner-sniff fallbacks in parse_bytes() and parse_io()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Union

from .codec import Codec
from .mime_type import MimeType

__all__ = ["MediaType"]


@dataclass(frozen=True, slots=True)
class MediaType:
    mime_type: MimeType
    codec: Codec | None = None

    def __post_init__(self):
        if self.mime_type.is_codec:
            codec = Codec.from_mime(self.mime_type)
            object.__setattr__(self, "mime_type", MimeType.OCTET_STREAM)
            object.__setattr__(self, "codec", codec)

    def __repr__(self) -> str:
        if self.codec is None:
            return f"<MediaType {self.mime_type.value}>"
        return f"<MediaType {self.mime_type.value} + {self.codec.name}>"

    @classmethod
    def parse(
        cls,
        obj: Union[
            "MediaType", MimeType, Codec,
            tuple[str, str],
            str,
            bytes, bytearray, memoryview,
            Path,
            IO[bytes]
        ],
        *,
        default: "MediaType | None" = None
    ) -> "MediaType":
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, tuple) and len(obj) == 2:
            mime, codec = obj
            mt = MimeType.parse(mime)
            if mt is None:
                return default or cls(mime_type=MimeType.OCTET_STREAM)
            c = Codec.parse(codec) if codec is not None else None
            return cls(mime_type=mt, codec=c)

        if isinstance(obj, MimeType):
            return cls(mime_type=obj, codec=None)

        if isinstance(obj, Codec):
            return cls(mime_type=MimeType.OCTET_STREAM, codec=obj)

        if isinstance(obj, str):
            return cls.parse_str(obj, default=default)

        if isinstance(obj, (bytes, bytearray, memoryview)):
            return cls.parse_bytes(bytes(obj), default=default)

        if isinstance(obj, Path):
            return cls.parse_str(str(obj), default=default)

        if hasattr(obj, "read"):
            return cls.parse_io(obj, default=default)  # type: ignore[arg-type]

        return default or cls(mime_type=MimeType.OCTET_STREAM)

    @classmethod
    def parse_str(cls, s: str, default: "MediaType | None" = None) -> "MediaType":
        if not s:
            return default or cls(mime_type=MimeType.OCTET_STREAM)

        raw = s.strip()

        # compound first
        outer: Codec | None = None
        inner_str = raw

        if "+" in raw:
            left, right = raw.rsplit("+", 1)
            maybe = Codec.parse(right.strip(), default=None)
            if maybe is not None:
                outer = maybe
                inner_str = left.strip()

        if outer is None:
            p = Path(raw)
            if p.suffixes:
                last = p.suffixes[-1].lstrip(".").lower()
                maybe = Codec.parse(last, default=None)
                if maybe is not None:
                    outer = maybe
                    inner_str = str(p.with_suffix(""))

        if outer is not None:
            inner = MimeType.parse_str(inner_str, default=None)
            if inner is None:
                # if codec exists but inner unknown -> octet-stream
                return cls(mime_type=MimeType.OCTET_STREAM, codec=outer)
            return cls(mime_type=inner, codec=outer)

        # plain
        mt = MimeType.parse_str(raw, default=None)
        if mt is None:
            return default or cls(mime_type=MimeType.OCTET_STREAM)

        if mt.is_codec:
            return cls(mime_type=mt, codec=Codec.from_mime(mt))

        return cls(mime_type=mt, codec=None)

    @classmethod
    def _sniff_inner_from_codec_bytes(cls, codec: Codec, data: bytes) -> MimeType | None:
        try:
            head, tail = codec.read_start_end(data, n_start=256, n_end=256)
        except Exception:
            return None

        inner = MimeType.parse_magic(head, default=None)
        if inner is not None:
            return inner
        return MimeType.parse_magic(tail, default=None)

    @classmethod
    def _sniff_inner_from_codec_io(cls, codec: Codec, fh: IO[bytes]) -> MimeType | None:
        try:
            head, tail = codec.read_start_end(fh, n_start=256, n_end=256)
        except Exception:
            return None

        inner = MimeType.parse_magic(head, default=None)
        if inner is not None:
            return inner
        return MimeType.parse_magic(tail, default=None)

    @classmethod
    def parse_bytes(cls, data: bytes, default: "MediaType | None" = None) -> "MediaType":
        c = Codec.parse(data, default=None)
        if c is not None:
            inner = cls._sniff_inner_from_codec_bytes(c, data)
            if inner is None:
                # NEW: if codec exists but inner unknown -> octet-stream
                return cls(mime_type=MimeType.OCTET_STREAM, codec=c)
            return cls(mime_type=inner, codec=c)

        mt = MimeType.parse_magic(data, default=None)
        if mt is None:
            return default or cls(mime_type=MimeType.OCTET_STREAM)
        return cls(mime_type=mt, codec=None)

    @classmethod
    def parse_io(cls, data: IO[bytes], default: "MediaType | None" = None) -> "MediaType":
        c = Codec.parse(data, default=None)
        if c is not None:
            inner = cls._sniff_inner_from_codec_io(c, data)
            if inner is None:
                # NEW: if codec exists but inner unknown -> octet-stream
                return cls(mime_type=MimeType.OCTET_STREAM, codec=c)
            return cls(mime_type=inner, codec=c)

        header = MimeType.peek(data, 256)
        mt = MimeType.parse_magic(header, default=None)
        if mt is None:
            return default or cls(mime_type=MimeType.OCTET_STREAM)
        return cls(mime_type=mt, codec=None)

    @property
    def is_octet(self):
        return self.mime_type == MimeType.OCTET_STREAM

    def full_mime_type(self):
        if self.codec is None:
            return self.mime_type

        return MimeType(
            name=self.mime_type.name + "_" + self.codec.name,
            value=self.mime_type.value + "+" + self.codec.mime_type.name.lower(),
            extensions=(self.full_extension,),
            is_codec=False,
            is_tabular=self.mime_type.is_tabular
        )

    @property
    def full_extension(self):
        if self.codec is None:
            return self.mime_type.extension

        return "%s.%s" % (
            self.mime_type.extension,
            self.codec.extension
        )