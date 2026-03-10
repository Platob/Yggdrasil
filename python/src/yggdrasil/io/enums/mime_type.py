from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, IO, Mapping

__all__ = ["MimeType"]

MagicMatcher = Callable[[bytes], bool]


# ----------------------------
# Magic helpers
# ----------------------------
def magic_prefix(prefix: bytes) -> MagicMatcher:
    return lambda b: b.startswith(prefix)


def magic_riff_webp(b: bytes) -> bool:
    return b.startswith(b"RIFF") and len(b) >= 12 and b[8:12] == b"WEBP"


def magic_parquet(b: bytes) -> bool:
    # header only; footer exists too but we usually peek small buffers
    return b.startswith(b"PAR1")


# ----------------------------
# MimeType
# ----------------------------
@dataclass(frozen=True, slots=True)
class MimeType:
    """
    Dataclass MIME descriptor + registries.

    - extensions: dotless, lower-case keys (we normalize on registration anyway)
    - magics: ordered matchers (usually prefix checks; can be any predicate)
    - is_codec: compression / wrapper formats (gzip/zstd/...)
    - is_tabular: row/tabular-ish formats (csv/tsv/ndjson/jsonl)
    """

    name: str
    value: str
    extensions: tuple[str, ...] = ()
    magics: tuple[MagicMatcher, ...] = ()

    is_codec: bool = False
    is_tabular: bool = False

    _BY_NAME: ClassVar[dict[str, "MimeType"]] = {}
    _BY_VALUE: ClassVar[dict[str, "MimeType"]] = {}
    _EXT_MAP: ClassVar[dict[str, "MimeType"]] = {}
    _MAGIC_ORDER: ClassVar[list["MimeType"]] = []

    # -------- registration --------
    @classmethod
    def define(cls, mt: "MimeType") -> "MimeType":
        """
        Register a MimeType instance in the global registries.

        Public on purpose (avoids "protected access" lint noise).
        """
        cls._BY_NAME[mt.name.lower()] = mt
        cls._BY_VALUE[mt.value.lower()] = mt

        for ext in mt.extensions:
            cls._EXT_MAP[ext.lower().lstrip(".")] = mt

        if mt.magics:
            cls._MAGIC_ORDER.append(mt)

        return mt

    # -------- basic helpers --------
    @staticmethod
    def peek(fh: IO[bytes], n: int) -> bytes:
        """Read *n* bytes from *fh* without advancing its cursor."""
        pos = fh.tell()
        try:
            return fh.read(n)
        finally:
            fh.seek(pos)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}.{self.name}: {self.value!r}>"

    # -------- lax resolution --------
    @classmethod
    def get(cls, value: object) -> "MimeType | None":
        """
        Lax lookup:
          1) exact MIME value (case-insensitive)
          2) name (case-insensitive)
          3) stripped common prefixes then name match
        """
        if not isinstance(value, str):
            return None

        s = value.strip().lower()

        hit = cls._BY_VALUE.get(s)
        if hit is not None:
            return hit

        hit = cls._BY_NAME.get(s)
        if hit is not None:
            return hit

        for prefix in ("application/", "text/", "image/", "audio/", "video/"):
            if s.startswith(prefix):
                return cls._BY_NAME.get(s[len(prefix) :])

        return None

    # -------- parsing entrypoint --------
    @classmethod
    def parse(cls, obj: Any, default: "MimeType | None" = None) -> "MimeType | None":
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, str):
            return cls.parse_str(obj, default=default)

        if isinstance(obj, (bytes, bytearray, memoryview)):
            return cls.parse_magic(obj, default=default)

        if isinstance(obj, Path):
            return cls.parse_str(str(obj), default=default)

        if hasattr(obj, "read"):
            obj = cls.peek(obj, 64)

        return cls.parse_magic(obj, default=default)

    # -------- magic parsing --------
    @classmethod
    def parse_magic(cls, magic: bytes, default: "MimeType | None" = None) -> "MimeType | None":
        if not magic:
            return default

        # ensure bytes (memoryview-safe)
        if not isinstance(magic, (bytes, bytearray)):
            magic = bytes(magic)

        # ordered scan (define codecs first)
        for mt in cls._MAGIC_ORDER:
            for matcher in mt.magics:
                try:
                    if matcher(magic):
                        return mt
                except Exception:
                    # matcher bug shouldn't kill parsing
                    continue

        # weak signals (fallback territory)
        stripped = magic.lstrip()
        if stripped.startswith(b"{") or stripped.startswith(b"["):
            return cls.NDJSON if b"\n{" in stripped else cls.JSON
        if stripped.startswith(b"<"):
            return cls.XML  # could also be HTML; indistinguishable here

        return default

    # -------- string parsing --------
    @classmethod
    def parse_str(cls, value: str, default: "MimeType | None" = None) -> "MimeType | None":
        """
        Resolution order:
          1) extension map (fast)
          2) lax get() by value/name/prefix-stripped
          3) tiny JSON literal heuristic
        """
        candidate = value.strip()

        # path-ish: use suffix
        if "." in candidate or "/" in candidate or "\\" in candidate:
            p = Path(candidate)
            ext = p.suffix.lstrip(".").lower()
            if ext:
                hit = cls._EXT_MAP.get(ext)
                if hit is not None:
                    return hit

        # bare ext or ".ext"
        bare = candidate.lstrip(".").lower()
        hit = cls._EXT_MAP.get(bare)
        if hit is not None:
            return hit

        mt = cls.get(candidate)
        if mt is not None:
            return mt

        if candidate.startswith("{") or candidate.startswith("["):
            return cls.NDJSON if "\n{" in candidate else cls.JSON

        return default

    # -------- runtime extension registry --------
    @classmethod
    def register_extension(
        cls,
        ext: str,
        mime: "MimeType | str",
        *,
        overwrite: bool = False,
    ) -> None:
        key = ext.lstrip(".").lower()

        resolved: MimeType | None
        if isinstance(mime, MimeType):
            resolved = mime
        else:
            resolved = cls.get(mime) or cls.parse_str(mime)

        if resolved is None:
            raise ValueError(f"register_extension: {mime!r} does not resolve to a known MimeType")

        if key in cls._EXT_MAP and not overwrite:
            raise KeyError(
                f"register_extension: extension {ext!r} is already registered as "
                f"{cls._EXT_MAP[key]!r}; pass overwrite=True to replace it"
            )

        cls._EXT_MAP[key] = resolved

    @classmethod
    def register_extensions(
        cls,
        mapping: Mapping[str, "MimeType | str"],
        *,
        overwrite: bool = False,
    ) -> None:
        # validate all entries first (all-or-nothing)
        resolved: list[tuple[str, MimeType]] = []
        for ext, mime in mapping.items():
            key = ext.lstrip(".").lower()

            mt: MimeType | None
            if isinstance(mime, MimeType):
                mt = mime
            else:
                mt = cls.get(mime) or cls.parse_str(mime)

            if mt is None:
                raise ValueError(
                    f"register_extensions: {mime!r} (key {ext!r}) does not resolve to a known MimeType"
                )

            if key in cls._EXT_MAP and not overwrite:
                raise KeyError(
                    f"register_extensions: extension {ext!r} is already registered as "
                    f"{cls._EXT_MAP[key]!r}; pass overwrite=True to replace it"
                )

            resolved.append((key, mt))

        for key, mt in resolved:
            cls._EXT_MAP[key] = mt

    @classmethod
    def extensions_for(cls, mime: "MimeType | str") -> list[str]:
        target: MimeType | None
        if isinstance(mime, MimeType):
            target = mime
        else:
            target = cls.get(mime) or cls.parse_str(mime)

        if target is None:
            return []

        return sorted(k for k, v in cls._EXT_MAP.items() if v is target)

    @classmethod
    def registered_extensions(cls) -> dict[str, "MimeType"]:
        return dict(cls._EXT_MAP)

    @property
    def extension(self) -> str:
        return self.extensions[0]


# ----------------------------
# Registry: definitions
# Order matters for parse_magic: codecs first.
# ----------------------------

# --- Compression / codecs ---
MimeType.GZIP = MimeType.define(
    MimeType(
        "GZIP",
        "application/gzip",
        extensions=("gz", "gzip", "tgz"),
        magics=(magic_prefix(b"\x1f\x8b"),),
        is_codec=True,
    )
)
MimeType.ZSTD = MimeType.define(
    MimeType(
        "ZSTD",
        "application/zstd",
        extensions=("zst", "zstd"),
        magics=(magic_prefix(b"\x28\xb5\x2f\xfd"),),
        is_codec=True,
    )
)
MimeType.BROTLI = MimeType.define(
    MimeType("BROTLI", "application/x-brotli", extensions=("br", "brotli"), is_codec=True)
)
MimeType.LZ4 = MimeType.define(
    MimeType(
        "LZ4",
        "application/x-lz4",
        extensions=("lz4",),
        magics=(magic_prefix(b"\x04\x22\x4d\x18"),),
        is_codec=True,
    )
)
MimeType.SNAPPY = MimeType.define(
    MimeType("SNAPPY", "application/x-snappy", extensions=("snappy", "sz"), is_codec=True)
)
MimeType.BZ2 = MimeType.define(
    MimeType(
        "BZ2",
        "application/x-bzip2",
        extensions=("bz2", "bzip2", "tbz2"),
        magics=(magic_prefix(b"\x42\x5a\x68"),),
        is_codec=True,
    )
)
MimeType.XZ = MimeType.define(
    MimeType(
        "XZ",
        "application/x-xz",
        extensions=("xz", "txz"),
        magics=(magic_prefix(b"\xfd\x37\x7a\x58\x5a\x00"),),
        is_codec=True,
    )
)
MimeType.ZLIB = MimeType.define(
    MimeType(
        "ZLIB",
        "application/zlib",
        extensions=("zlib",),
        magics=(magic_prefix(b"\x78\x01"), magic_prefix(b"\x78\x9c"), magic_prefix(b"\x78\xda")),
        is_codec=True,
    )
)
MimeType.LZMA = MimeType.define(MimeType("LZMA", "application/x-lzma", extensions=("lzma",), is_codec=True))
MimeType.ZZIP = MimeType.define(MimeType("ZZIP", "application/x-compress", extensions=("z", "Z"), is_codec=True))

# --- Containers / docs ---
MimeType.ZIP = MimeType.define(
    MimeType(
        "ZIP",
        "application/zip",
        extensions=("zip",),
        magics=(magic_prefix(b"PK\x03\x04"),),
    )
)
MimeType.PDF = MimeType.define(
    MimeType(
        "PDF",
        "application/pdf",
        extensions=("pdf",),
        magics=(magic_prefix(b"%PDF-"),),
    )
)
MimeType.TAR = MimeType.define(
    MimeType(
        "TAR",
        "application/x-tar",
        extensions=("tar",),
        magics=(magic_prefix(b"\x75\x73\x74\x61\x72"),),
    )
)
MimeType.SQLITE = MimeType.define(
    MimeType(
        "SQLITE",
        "application/x-sqlite3",
        extensions=("db", "sqlite", "sqlite3"),
        magics=(magic_prefix(b"SQLite format 3\x00"),),
    )
)
MimeType.HDF5 = MimeType.define(
    MimeType(
        "HDF5",
        "application/x-hdf5",
        extensions=("h5", "hdf5", "he5"),
        magics=(magic_prefix(b"\x89HDF\r\n\x1a\n"),),
    )
)
MimeType.XLSX = MimeType.define(
    MimeType(
        "XLSX",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        extensions=("xlsx", "xls"),
    )
)
MimeType.DOCX = MimeType.define(
    MimeType(
        "DOCX",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        extensions=("docx",),
    )
)

# --- Columnar / analytics ---
MimeType.PARQUET = MimeType.define(
    MimeType(
        "PARQUET",
        "application/vnd.apache.parquet",
        extensions=("parquet", "pq"),
        magics=(magic_parquet,),
    )
)
MimeType.PARQUET_DELTA = MimeType.define(
    MimeType("PARQUET_DELTA", "application/vnd.apache.parquet+delta")
)
MimeType.IPC = MimeType.define(
    MimeType(
        "ARROW_IPC",
        "application/vnd.apache.arrow.file",
        extensions=("ipc", "feather"),
        magics=(magic_prefix(b"ARROW1"),),
    )
)
MimeType.IPC_STREAM = MimeType.define(
    MimeType(
        "ARROW_IPC_STREAM",
        "application/vnd.apache.arrow.stream",
        extensions=("arrow", "arrows"),
    )
)

MimeType.ORC = MimeType.define(
    MimeType(
        "ORC",
        "application/vnd.apache.orc",
        extensions=("orc",),
        magics=(magic_prefix(b"ORC"),),
    )
)
MimeType.AVRO = MimeType.define(
    MimeType(
        "AVRO",
        "application/avro",
        extensions=("avro",),
        magics=(magic_prefix(b"Obj\x01"),),
    )
)
MimeType.ICEBERG = MimeType.define(
    MimeType("ICEBERG", "application/vnd.apache.iceberg", extensions=("iceberg",))
)
MimeType.DELTA = MimeType.define(
    MimeType("DELTA", "application/vnd.delta", extensions=("delta", "deltatable"))
)

# --- Text / semi-structured ---
MimeType.JSON = MimeType.define(MimeType("JSON", "application/json", extensions=("json",)))
MimeType.NDJSON = MimeType.define(
    MimeType(
        "NDJSON",
        "application/ld+json",
        extensions=("jsonld", "ldjson"),
        is_tabular=True,
    )
)
MimeType.CSV = MimeType.define(
    MimeType("CSV", "text/csv", extensions=("csv",), is_tabular=True)
)
MimeType.TSV = MimeType.define(
    MimeType("TSV", "text/tab-separated-values", extensions=("tsv",), is_tabular=True)
)
MimeType.XML = MimeType.define(MimeType("XML", "application/xml", extensions=("xml",)))
MimeType.HTML = MimeType.define(MimeType("HTML", "text/html", extensions=("html", "htm")))
MimeType.PLAIN = MimeType.define(MimeType("PLAIN", "text/plain", extensions=("txt", "text")))
MimeType.YAML = MimeType.define(MimeType("YAML", "application/yaml", extensions=("yaml", "yml")))
MimeType.TOML = MimeType.define(MimeType("TOML", "application/toml", extensions=("toml",)))

# --- Binary serialisation ---
MimeType.MSGPACK = MimeType.define(MimeType("MSGPACK", "application/msgpack", extensions=("msgpack", "mpk")))
MimeType.PROTOBUF = MimeType.define(MimeType("PROTOBUF", "application/x-protobuf", extensions=("pb", "proto", "protobuf")))
MimeType.FLATBUFFERS = MimeType.define(MimeType("FLATBUFFERS", "application/x-flatbuffers", extensions=("bin", "fbs")))
MimeType.CBOR = MimeType.define(MimeType("CBOR", "application/cbor", extensions=("cbor",)))
MimeType.BSON = MimeType.define(MimeType("BSON", "application/bson", extensions=("bson",)))
MimeType.PICKLE = MimeType.define(MimeType("PICKLE", "application/x-python-pickle", extensions=("pkl", "pickle")))
MimeType.NUMPY = MimeType.define(
    MimeType(
        "NUMPY",
        "application/x-npy",
        extensions=("npy",),
        magics=(magic_prefix(b"\x93NUMPY"),),
    )
)
MimeType.NUMPY_ARCHIVE = MimeType.define(MimeType("NUMPY_ARCHIVE", "application/x-npz", extensions=("npz",)))

# --- Images ---
MimeType.PNG = MimeType.define(
    MimeType(
        "PNG",
        "image/png",
        extensions=("png",),
        magics=(magic_prefix(b"\x89PNG\r\n\x1a\n"),),
    )
)
MimeType.JPEG = MimeType.define(
    MimeType(
        "JPEG",
        "image/jpeg",
        extensions=("jpg", "jpeg"),
        magics=(magic_prefix(b"\xff\xd8\xff"),),
    )
)
MimeType.GIF = MimeType.define(
    MimeType(
        "GIF",
        "image/gif",
        extensions=("gif",),
        magics=(magic_prefix(b"GIF87a"), magic_prefix(b"GIF89a")),
    )
)
MimeType.WEBP = MimeType.define(
    MimeType(
        "WEBP",
        "image/webp",
        extensions=("webp",),
        magics=(magic_riff_webp,),
    )
)
MimeType.TIFF = MimeType.define(
    MimeType(
        "TIFF",
        "image/tiff",
        extensions=("tif", "tiff"),
        magics=(magic_prefix(b"II*\x00"), magic_prefix(b"MM\x00*")),
    )
)
MimeType.BMP = MimeType.define(
    MimeType(
        "BMP",
        "image/bmp",
        extensions=("bmp",),
        magics=(magic_prefix(b"BM"),),
    )
)

# --- Fallback ---
MimeType.OCTET_STREAM = MimeType.define(
    MimeType("OCTET_STREAM", "application/octet-stream")
)

# Seed container riders (XLSX/DOCX are ZIP containers)
# Put these after all defines so we don't care about define order.
MimeType.register_extensions({"xlsx": MimeType.XLSX, "xls": MimeType.XLSX, "docx": MimeType.DOCX}, overwrite=True)