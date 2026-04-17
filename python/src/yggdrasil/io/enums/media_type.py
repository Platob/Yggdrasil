# yggdrasil/io/enums/media_type.py
# Patch the codec-inner-sniff fallbacks in parse_bytes() and parse_io()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Union

from .codec import Codec
from .mime_type import MimeType, MimeTypes

__all__ = ["MediaType", "MediaTypes"]


@dataclass(frozen=True, slots=True)
class MediaType:
    mime_type: MimeType
    codec: Codec | None = None

    def __post_init__(self):
        if self.mime_type.is_codec:
            codec = Codec.from_mime(self.mime_type)
            object.__setattr__(self, "mime_type", MimeTypes.OCTET_STREAM)
            object.__setattr__(self, "codec", codec)

    def __repr__(self) -> str:
        if self.codec is None:
            return f"MediaType({self.mime_type!r})"
        return f"MediaType({self.mime_type!r} + {self.codec!r})"

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
        codec: Codec | None = None,
        default: "MediaType | None" = None
    ) -> "MediaType":
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, tuple) and len(obj) == 2:
            mime, codec = obj
            mt = MimeType.parse(mime)
            if mt is None:
                return default or cls(mime_type=MimeTypes.OCTET_STREAM)
            c = Codec.parse(codec) if codec is not None else None
            return cls(mime_type=mt, codec=c)

        if isinstance(obj, MimeType):
            return cls(mime_type=obj, codec=None)

        if isinstance(obj, Codec):
            return cls(mime_type=MimeTypes.OCTET_STREAM, codec=obj)

        if isinstance(obj, str):
            if "+" in obj:
                # handle MIME type forms like "application/json+gzip"
                base, _, codec_part = obj.rpartition("+")
                inner = cls.parse(base)
                codec = codec or Codec.parse(codec_part)

                if codec is not None:
                    if inner is None:
                        if default is None:
                            return cls(mime_type=MimeTypes.OCTET_STREAM, codec=codec)
                        return default.with_codec(codec)
                    return cls(mime_type=inner.mime_type, codec=codec)

                obj = base

        mt = MimeType.parse(obj)

        if mt is None:
            return default or cls(mime_type=MimeTypes.OCTET_STREAM, codec=codec)

        if mt.is_codec:
            from ..buffer import BytesIO
            codec = codec or Codec.from_mime(mt)
            inner = None

            if isinstance(obj, (str, Path)):
                obj = str(obj)

                if "." in obj:
                    for ext in codec.extensions:
                        if obj.endswith(ext):
                            skip = 1 + len(ext) # account for the dot before the extension
                            obj = obj[:-skip]
                            inner = cls.parse(obj)
                            break
            else:
                buff = BytesIO(obj, copy=False)
                inner = cls.parse(codec.read_start_end(buff), default=default)

            if inner is None:
                if default is None:
                    return cls(mime_type=MimeTypes.OCTET_STREAM, codec=codec)
                return default.with_codec(codec)

            assert not inner.mime_type.is_codec, "Inner MIME type cannot be a codec"
            return cls(mime_type=inner.mime_type, codec=codec)

        return cls(mime_type=mt, codec=codec)

    @property
    def is_octet(self):
        return self.mime_type == MimeTypes.OCTET_STREAM

    @property
    def is_json(self):
        return self.mime_type == MimeTypes.JSON

    def full_mime_type(self, concat_codec: bool = True) -> MimeType:
        if not concat_codec or self.codec is None:
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

    def with_codec(self, codec: Codec) -> "MediaType":
        return MediaType(mime_type=self.mime_type, codec=codec)

    def without_codec(self) -> "MediaType":
        return MediaType(mime_type=self.mime_type, codec=None)


class MediaTypes:
    PARQUET = MediaType(mime_type=MimeTypes.PARQUET, codec=None)
    JSON = MediaType(mime_type=MimeTypes.JSON, codec=None)