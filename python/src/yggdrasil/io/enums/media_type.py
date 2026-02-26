from __future__ import annotations

import io
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import IO, Any, ClassVar, Optional

from .codec import Codec, _peek

__all__ = ["MediaType", "MimeType", "MediaTypes"]


class MimeType(str, Enum):
    # Columnar / analytics
    PARQUET = "application/vnd.apache.parquet"
    PARQUET_DELTA = "application/vnd.apache.parquet+delta"
    ARROW_FILE = "application/vnd.apache.arrow.file"
    ARROW_STREAM = "application/vnd.apache.arrow.stream"
    IPC = "application/vnd.apache.arrow.file"
    FEATHER = "application/vnd.apache.arrow.file"
    ORC = "application/vnd.apache.orc"
    AVRO = "application/avro"
    ICEBERG = "application/vnd.apache.iceberg"
    DELTA = "application/vnd.delta"

    # Text / semi-structured
    JSON = "application/json"
    NDJSON = "application/x-ndjson"
    JSONL = "application/x-ndjson"
    CSV = "text/csv"
    TSV = "text/tab-separated-values"
    XML = "application/xml"
    HTML = "text/html"
    PLAIN = "text/plain"
    YAML = "application/yaml"
    TOML = "application/toml"

    # Binary serialisation
    MSGPACK = "application/msgpack"
    PROTOBUF = "application/x-protobuf"
    FLATBUFFERS = "application/x-flatbuffers"
    CBOR = "application/cbor"
    BSON = "application/bson"
    PICKLE = "application/x-python-pickle"
    NUMPY = "application/x-npy"
    NUMPY_ARCHIVE = "application/x-npz"

    # Document / container
    PDF = "application/pdf"
    ZIP = "application/zip"
    TAR = "application/x-tar"
    SQLITE = "application/x-sqlite3"
    HDF5 = "application/x-hdf5"
    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    # Image
    PNG = "image/png"
    JPEG = "image/jpeg"
    GIF = "image/gif"
    WEBP = "image/webp"
    TIFF = "image/tiff"
    BMP = "image/bmp"

    # Fallback
    OCTET_STREAM = "application/octet-stream"


@dataclass(frozen=True, slots=True)
class MediaType:
    mime: str
    codec: Codec | None = None

    @property
    def full_mime(self) -> str:
        if self.codec is None:
            return self.mime
        return self.mime + "+" + self.codec.value

    @classmethod
    def of(
        cls,
        mime: str | MimeType | "MediaType",
        *,
        codec: Codec | str | None = None,
    ) -> "MediaType":
        """Construct MediaType from a mime-ish input + optional codec."""
        if isinstance(mime, MediaType):
            base = mime.mime
        elif isinstance(mime, MimeType):
            base = mime.value
        else:
            base = str(mime)

        if isinstance(codec, str):
            codec = Codec(codec)

        return cls(mime=base, codec=codec)

    def without_codec(self) -> "MediaType":
        if self.codec is None:
            return self
        return replace(self, codec=None)

    def with_codec(self, codec: Codec | str | None) -> "MediaType":
        if isinstance(codec, str):
            codec = Codec(codec)
        return replace(self, codec=codec)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.mime == other
        if isinstance(other, MediaType):
            return self.mime == other.mime and self.codec == other.codec
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.mime, self.codec))

    # ---- detection table hook
    _MAGIC: ClassVar[list[tuple[int, bytes, str]]]

    @classmethod
    def from_io(cls, src: "IO[bytes] | BytesIO") -> "MediaType":
        fh: IO[bytes] = src.buffer() if hasattr(src, "buffer") else src  # type: ignore[union-attr]

        codec = Codec.from_io(fh)
        if codec is not None:
            return cls(mime=MimeType.OCTET_STREAM.value, codec=codec)

        header = _peek(fh, 512)
        for offset, magic, mime in cls._MAGIC:
            end = offset + len(magic)
            if len(header) >= end and header[offset:end] == magic:
                return cls(mime=mime, codec=None)

        try:
            text_head = header.decode("utf-8", errors="strict").lstrip()
        except UnicodeDecodeError:
            return cls(mime=MimeType.OCTET_STREAM.value, codec=None)

        non_empty = [ln.strip() for ln in text_head.split("\n") if ln.strip()]
        first = non_empty[0] if non_empty else ""
        second = non_empty[1] if len(non_empty) > 1 else ""

        if first.startswith("{") and second.startswith("{"):
            return cls(mime=MimeType.NDJSON.value, codec=None)
        if first.startswith(("{", "[")):
            return cls(mime=MimeType.JSON.value, codec=None)
        if first.startswith("<?xml") or first.startswith("<xml"):
            return cls(mime=MimeType.XML.value, codec=None)
        if first.lower().startswith(("<!doctype html", "<html")):
            return cls(mime=MimeType.HTML.value, codec=None)
        if first.startswith("---"):
            return cls(mime=MimeType.YAML.value, codec=None)
        if "=" in first and not first.startswith("#") and not first.startswith("{"):
            return cls(mime=MimeType.TOML.value, codec=None)
        if "\t" in first:
            return cls(mime=MimeType.TSV.value, codec=None)
        if "," in first and first and first[0].isalnum():
            return cls(mime=MimeType.CSV.value, codec=None)

        return cls(mime=MimeType.OCTET_STREAM.value, codec=None)

    @classmethod
    def from_bytes(cls, data: bytes) -> "MediaType":
        return cls.from_io(io.BytesIO(data))

    @classmethod
    def parse_any(cls, obj: Any, default: Optional["MediaType"] = None) -> "MediaType":
        if isinstance(obj, cls):
            return obj
        if obj is None:
            if default is None:
                raise ValueError
            return cls.parse_any(default)
        if isinstance(obj, str):
            if obj.startswith(("application/", "text/", "image/")):
                return cls.parse_mime(obj)
            if "/" not in obj and "\\" not in obj:
                return cls.parse_extension(obj)
        path = Path(obj)
        return cls.parse_extension(path.name.split(".", 1)[-1])

    @classmethod
    def parse_extension(cls, ext: str) -> "MediaType":
        parts = ext.lstrip(".").lower().split(".")

        codec: Codec | None = None
        if len(parts) > 1 and parts[-1] in _EXT_TO_CODEC:
            codec = _EXT_TO_CODEC[parts[-1]]
            parts = parts[:-1]

        mime = _EXT_TO_MIME.get(".".join(parts), MimeType.OCTET_STREAM.value)
        return cls(mime=mime, codec=codec)

    @classmethod
    def parse_mime(cls, mime: str, *, codec: Codec | str | None = None) -> "MediaType":
        if isinstance(codec, str):
            codec = Codec(codec)

        base = mime.strip().lower().split(";", 1)[0].strip()

        _CODEC_MIME: dict[str, Codec] = {
            "application/gzip": Codec.GZIP,
            "application/x-gzip": Codec.GZIP,
            "application/zstd": Codec.ZSTD,
            "application/x-zstd": Codec.ZSTD,
            "application/x-lz4": Codec.LZ4,
            "application/x-bzip2": Codec.BZIP2,
            "application/x-xz": Codec.XZ,
            "application/x-snappy-framed": Codec.SNAPPY,
        }
        if base in _CODEC_MIME:
            return cls(mime=MimeType.OCTET_STREAM.value, codec=codec or _CODEC_MIME[base])

        inferred_codec: Codec | None = None
        format_base = base
        if "+" in base:
            format_part, codec_part = base.rsplit("+", 1)
            _SUFFIX_TO_CODEC: dict[str, Codec] = {
                "gzip": Codec.GZIP,
                "gz": Codec.GZIP,
                "zstd": Codec.ZSTD,
                "zst": Codec.ZSTD,
                "lz4": Codec.LZ4,
                "bzip2": Codec.BZIP2,
                "bz2": Codec.BZIP2,
                "xz": Codec.XZ,
                "snappy": Codec.SNAPPY,
            }
            if codec_part in _SUFFIX_TO_CODEC:
                inferred_codec = _SUFFIX_TO_CODEC[codec_part]
                format_base = format_part

        all_known = {m.value for m in MimeType}
        resolved_mime = format_base if format_base in all_known else MimeType.OCTET_STREAM.value
        return cls(mime=resolved_mime, codec=codec or inferred_codec)

    def __str__(self) -> str:
        return self.mime

    def __repr__(self) -> str:
        codec_part = f", codec={self.codec!r}" if self.codec else ""
        return f"MediaType({self.mime!r}{codec_part})"

    @property
    def extension(self) -> str:
        """Suggested file extension for this MIME type (without leading dot).

        Returns "bin" for unknown types.
        Does NOT include the codec wrapper extension (e.g. ".zst").
        """
        return _MIME_TO_EXT.get(self.mime, "bin")

    @property
    def codec_extension(self) -> str:
        """File extension for the outer codec, or empty string when uncompressed."""
        return _CODEC_EXT.get(self.codec, "") if self.codec else ""

    @property
    def full_extension(self) -> str:
        """Combined extension including the codec suffix when present."""
        ext = self.extension
        cext = self.codec_extension
        return f"{ext}.{cext}" if cext else ext


class MediaTypes:
    """Singleton MediaType instances for reuse and dispatch."""
    # Columnar / analytics
    PARQUET = MediaType.of(MimeType.PARQUET)
    PARQUET_DELTA = MediaType.of(MimeType.PARQUET_DELTA)
    ARROW_FILE = MediaType.of(MimeType.ARROW_FILE)
    ARROW_STREAM = MediaType.of(MimeType.ARROW_STREAM)
    IPC = MediaType.of(MimeType.IPC)
    FEATHER = MediaType.of(MimeType.FEATHER)
    ORC = MediaType.of(MimeType.ORC)
    AVRO = MediaType.of(MimeType.AVRO)
    ICEBERG = MediaType.of(MimeType.ICEBERG)
    DELTA = MediaType.of(MimeType.DELTA)

    # Text / semi-structured
    JSON = MediaType.of(MimeType.JSON)
    NDJSON = MediaType.of(MimeType.NDJSON)
    JSONL = MediaType.of(MimeType.JSONL)
    CSV = MediaType.of(MimeType.CSV)
    TSV = MediaType.of(MimeType.TSV)
    XML = MediaType.of(MimeType.XML)
    HTML = MediaType.of(MimeType.HTML)
    PLAIN = MediaType.of(MimeType.PLAIN)
    YAML = MediaType.of(MimeType.YAML)
    TOML = MediaType.of(MimeType.TOML)

    # Binary serialisation
    MSGPACK = MediaType.of(MimeType.MSGPACK)
    PROTOBUF = MediaType.of(MimeType.PROTOBUF)
    FLATBUFFERS = MediaType.of(MimeType.FLATBUFFERS)
    CBOR = MediaType.of(MimeType.CBOR)
    BSON = MediaType.of(MimeType.BSON)
    PICKLE = MediaType.of(MimeType.PICKLE)
    NUMPY = MediaType.of(MimeType.NUMPY)
    NUMPY_ARCHIVE = MediaType.of(MimeType.NUMPY_ARCHIVE)

    # Document / container
    PDF = MediaType.of(MimeType.PDF)
    ZIP = MediaType.of(MimeType.ZIP)
    TAR = MediaType.of(MimeType.TAR)
    SQLITE = MediaType.of(MimeType.SQLITE)
    HDF5 = MediaType.of(MimeType.HDF5)
    XLSX = MediaType.of(MimeType.XLSX)
    DOCX = MediaType.of(MimeType.DOCX)

    # Image
    PNG = MediaType.of(MimeType.PNG)
    JPEG = MediaType.of(MimeType.JPEG)
    GIF = MediaType.of(MimeType.GIF)
    WEBP = MediaType.of(MimeType.WEBP)
    TIFF = MediaType.of(MimeType.TIFF)
    BMP = MediaType.of(MimeType.BMP)

    # Fallback
    OCTET_STREAM = MediaType.of(MimeType.OCTET_STREAM)

    @classmethod
    def all(cls) -> tuple[MediaType, ...]:
        return tuple(v for v in cls.__dict__.values() if isinstance(v, MediaType))


# Magic-byte dispatch table (store MIME STRINGS, not MediaType objects)
MediaType._MAGIC = [
    (0, b"PAR1", MediaTypes.PARQUET.mime),
    (0, b"ARROW1\x00\x00", MediaTypes.ARROW_FILE.mime),
    (0, b"\xff\xff\xff\xff", MediaTypes.ARROW_STREAM.mime),
    (0, b"ORC", MediaTypes.ORC.mime),
    (0, b"Obj\x01", MediaTypes.AVRO.mime),

    (0, b"\x93NUMPY", MediaTypes.NUMPY.mime),
    (0, b"SQLite format 3\x00", MediaTypes.SQLITE.mime),
    (0, b"\x89HDF\r\n\x1a\n", MediaTypes.HDF5.mime),

    (0, b"%PDF", MediaTypes.PDF.mime),
    (0, b"PK\x03\x04", MediaTypes.ZIP.mime),
    (257, b"ustar", MediaTypes.TAR.mime),

    (0, b"\x89PNG\r\n\x1a\n", MediaTypes.PNG.mime),
    (0, b"\xff\xd8\xff", MediaTypes.JPEG.mime),
    (0, b"GIF87a", MediaTypes.GIF.mime),
    (0, b"GIF89a", MediaTypes.GIF.mime),
    (0, b"RIFF", MediaTypes.WEBP.mime),
    (0, b"II\x2a\x00", MediaTypes.TIFF.mime),
    (0, b"MM\x00\x2a", MediaTypes.TIFF.mime),
    (0, b"BM", MediaTypes.BMP.mime),
]

# ---------------------------------------------------------------------------
# Extension ↔ MIME lookup tables (must be defined before MediaType.parse_* runs)
# ---------------------------------------------------------------------------

_EXT_TO_MIME: dict[str, str] = {
    # Columnar
    "parquet":  MimeType.PARQUET.value,
    "pq":       MimeType.PARQUET.value,
    "arrow":    MimeType.ARROW_FILE.value,
    "arrows":   MimeType.ARROW_STREAM.value,
    "ipc":      MimeType.ARROW_FILE.value,
    "feather":  MimeType.ARROW_FILE.value,
    "orc":      MimeType.ORC.value,
    "avro":     MimeType.AVRO.value,
    # Text
    "json":     MimeType.JSON.value,
    "ndjson":   MimeType.NDJSON.value,
    "jsonl":    MimeType.NDJSON.value,
    "csv":      MimeType.CSV.value,
    "tsv":      MimeType.TSV.value,
    "txt":      MimeType.PLAIN.value,
    "xml":      MimeType.XML.value,
    "html":     MimeType.HTML.value,
    "htm":      MimeType.HTML.value,
    "yaml":     MimeType.YAML.value,
    "yml":      MimeType.YAML.value,
    "toml":     MimeType.TOML.value,
    # Binary serialisation
    "msgpack":  MimeType.MSGPACK.value,
    "proto":    MimeType.PROTOBUF.value,
    "pb":       MimeType.PROTOBUF.value,
    "fbs":      MimeType.FLATBUFFERS.value,
    "cbor":     MimeType.CBOR.value,
    "bson":     MimeType.BSON.value,
    "pkl":      MimeType.PICKLE.value,
    "pickle":   MimeType.PICKLE.value,
    "npy":      MimeType.NUMPY.value,
    "npz":      MimeType.NUMPY_ARCHIVE.value,
    # Containers / docs
    "pdf":      MimeType.PDF.value,
    "zip":      MimeType.ZIP.value,
    "tar":      MimeType.TAR.value,
    "sqlite":   MimeType.SQLITE.value,
    "db":       MimeType.SQLITE.value,
    "h5":       MimeType.HDF5.value,
    "hdf5":     MimeType.HDF5.value,
    "xlsx":     MimeType.XLSX.value,
    "docx":     MimeType.DOCX.value,
    # Images
    "png":      MimeType.PNG.value,
    "jpg":      MimeType.JPEG.value,
    "jpeg":     MimeType.JPEG.value,
    "gif":      MimeType.GIF.value,
    "webp":     MimeType.WEBP.value,
    "tiff":     MimeType.TIFF.value,
    "tif":      MimeType.TIFF.value,
    "bmp":      MimeType.BMP.value,
}

_EXT_TO_CODEC: dict[str, Codec] = {
    "gz": Codec.GZIP,
    "gzip": Codec.GZIP,
    "zst": Codec.ZSTD,
    "zstd": Codec.ZSTD,
    "lz4": Codec.LZ4,
    "bz2": Codec.BZIP2,
    "bzip2": Codec.BZIP2,
    "xz": Codec.XZ,
    "lzma": Codec.XZ,
    "snappy": Codec.SNAPPY,
    "sz": Codec.SNAPPY,
}

_MIME_TO_EXT: dict[str, str] = {
    MimeType.PARQUET.value: "parquet",
    MimeType.PARQUET_DELTA.value: "parquet",
    MimeType.ARROW_FILE.value: "arrow",
    MimeType.ARROW_STREAM.value: "arrows",
    MimeType.ORC.value: "orc",
    MimeType.AVRO.value: "avro",
    MimeType.ICEBERG.value: "iceberg",
    MimeType.DELTA.value: "delta",
    MimeType.JSON.value: "json",
    MimeType.NDJSON.value: "ndjson",
    MimeType.CSV.value: "csv",
    MimeType.TSV.value: "tsv",
    MimeType.XML.value: "xml",
    MimeType.HTML.value: "html",
    MimeType.PLAIN.value: "txt",
    MimeType.YAML.value: "yaml",
    MimeType.TOML.value: "toml",
    MimeType.MSGPACK.value: "msgpack",
    MimeType.PROTOBUF.value: "proto",
    MimeType.FLATBUFFERS.value: "fbs",
    MimeType.CBOR.value: "cbor",
    MimeType.BSON.value: "bson",
    MimeType.PICKLE.value: "pkl",
    MimeType.NUMPY.value: "npy",
    MimeType.NUMPY_ARCHIVE.value: "npz",
    MimeType.PDF.value: "pdf",
    MimeType.ZIP.value: "zip",
    MimeType.TAR.value: "tar",
    MimeType.SQLITE.value: "sqlite",
    MimeType.HDF5.value: "h5",
    MimeType.XLSX.value: "xlsx",
    MimeType.DOCX.value: "docx",
    MimeType.PNG.value: "png",
    MimeType.JPEG.value: "jpg",
    MimeType.GIF.value: "gif",
    MimeType.WEBP.value: "webp",
    MimeType.TIFF.value: "tiff",
    MimeType.BMP.value: "bmp",
    MimeType.OCTET_STREAM.value: "bin",
}

_CODEC_EXT: dict[Codec, str] = {
    Codec.GZIP: "gz",
    Codec.ZSTD: "zst",
    Codec.LZ4: "lz4",
    Codec.BZIP2: "bz2",
    Codec.XZ: "xz",
    Codec.SNAPPY: "snappy",
}