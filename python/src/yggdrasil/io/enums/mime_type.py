from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, IO, Mapping, Union, Iterable, Iterator

__all__ = ["MimeType", "MimeTypes"]

from yggdrasil.lazy_imports import path_class, bytes_io_class

MagicMatcher = Callable[[bytes], bool]


# ----------------------------
# Default-handling sentinel
# ----------------------------
# ``...`` (Ellipsis) as the module-level sentinel for "no default was
# supplied; raise on miss." Mirrors :meth:`dict.pop` semantics — a
# passed ``None`` is a valid fallback (caller wants a soft miss), but
# omitting the argument altogether signals "I want to know about
# resolution failures."
#
# Exposed as :data:`_RAISE` so callers who want to make the raise
# explicit can pass it literally: ``MimeType.from_(x, default=_RAISE)``
# is equivalent to leaving it out.
_RAISE = ...


def _miss(default: Any, reason: str) -> Any:
    """Central miss handler — either returns *default* or raises.

    Every ``from_*`` failure path routes through here so the sentinel
    check lives in one place. When *default* is the ``...`` sentinel,
    raises :class:`ValueError` with *reason*. Otherwise returns the
    default as-is (including ``None``, an explicit MimeType, or any
    other value the caller chose).
    """
    if default is _RAISE:
        raise ValueError(f"MimeType resolution failed: {reason}")
    return default


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

    - extensions: dotless, lower-case keys
    - magics: ordered matchers
    - is_codec: compression / wrapper formats
    - is_tabular: row/tabular-ish formats
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

    @classmethod
    def define(cls, mt: "MimeType") -> "MimeType":
        cls._BY_NAME[mt.name.lower()] = mt
        cls._BY_VALUE[mt.value.lower()] = mt

        for ext in mt.extensions:
            cls._EXT_MAP[ext.lower().lstrip(".")] = mt

        if mt.magics:
            cls._MAGIC_ORDER.append(mt)

        return mt

    @property
    def is_any_bytes(self):
        return self.value == MimeTypes.OCTET_STREAM.value

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    @classmethod
    def get(cls, value: object) -> "MimeType | None":
        """Pure lookup — never raises. Returns ``None`` on miss.

        Kept as the low-level "resolve by name/value" entry point:
        :meth:`from_` / :meth:`from_str` / :meth:`from_magic` layer
        the default-handling contract on top.
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

        for prefix in (
            "application/", "text/", "image/", "audio/", "video/",
            "inode/"
        ):
            if s.startswith(prefix):
                return cls._BY_NAME.get(s[len(prefix):])

        return None

    # ------------------------------------------------------------------
    # Default-handling contract (shared by from_, from_magic, from_str)
    # ------------------------------------------------------------------
    # All three public parse entry points use the same sentinel:
    #
    # - ``default`` **omitted** → raise :class:`ValueError` on miss.
    # - ``default=None`` → return ``None`` on miss.
    # - ``default=<MimeType>`` → return that on miss.
    #
    # Mirrors :meth:`dict.pop`: the absent / present distinction gives
    # callers a way to opt into strict mode without the awkwardness of
    # ``MaybeNone | MimeType`` return types at every call site.
    # ------------------------------------------------------------------

    @classmethod
    def parse_many(cls, obj: Any) -> list["MimeType"]:
        """Resolve *obj* to a flat, deduped list of :class:`MimeType`.

        Accepts anything :meth:`from_` accepts, plus:

        - iterables (list, tuple, set, generator) of any supported input
        - Accept-header strings: ``"application/json, text/csv;q=0.8"``
        - composite ``format+codec`` strings: ``"application/csv+gzip"``,
          ``"parquet+zstd"``, ``"trades.parquet.zst"``
        - wildcard strings: ``"*/*"``, ``"text/*"``, ``"image/*"``
        - ``None`` → ``[]``

        Order: first-seen wins, deduped by identity. For composites, the
        base format is emitted first, then the codec (codec wraps format).

        Never raises on an unresolvable element — unknowns are dropped
        silently. For strict per-element resolution, call :meth:`from_`.
        """
        seen: dict[int, "MimeType"] = {}

        def emit(mt: "MimeType | None") -> None:
            if mt is None:
                return
            seen.setdefault(id(mt), mt)

        def resolve_scalar(token: str) -> "MimeType | None":
            token = token.strip()
            if not token:
                return None
            return cls.get(token) or cls.from_str(token, default=None)

        def walk_str(s: str) -> None:
            candidate = s.strip()
            if not candidate:
                return

            # Accept-header split: "application/json, text/csv;q=0.8"
            if "," in candidate:
                for part in candidate.split(","):
                    walk_str(part)
                return

            # Strip parameters: "text/csv; charset=utf-8; q=0.8"
            if ";" in candidate:
                candidate = candidate.split(";", 1)[0].strip()
                if not candidate:
                    return

            lower = candidate.lower()

            # Wildcards.
            if lower in ("*/*", "*"):
                emit(cls._BY_NAME.get("octet_stream"))
                return
            if lower.endswith("/*"):
                prefix = lower[:-1]  # keep trailing '/'
                for value, mt in cls._BY_VALUE.items():
                    if value.startswith(prefix):
                        emit(mt)
                return

            # Try full string first — keeps registered composites intact
            # (PARQUET_DELTA = "application/vnd.apache.parquet+delta",
            #  NDJSON = "application/ld+json").
            mt = cls.get(candidate)
            if mt is not None:
                emit(mt)
                return

            # "format+codec" composite. Only accept the split if the tail
            # actually names a codec — otherwise "application/ld+json"
            # would split on '+' before _BY_VALUE caught it above.
            if "+" in candidate:
                base, _, codec = candidate.rpartition("+")
                base_mt = resolve_scalar(base)
                codec_mt = resolve_scalar(codec)
                if codec_mt is not None and codec_mt.is_codec:
                    if base_mt is not None:
                        emit(base_mt)
                    emit(codec_mt)
                    return

            # Dotted extension chain: "data.csv.gz", "trades.parquet.zst".
            # If the last suffix is a codec and the penultimate is a known
            # non-codec format, emit both in wrapping order.
            if "." in candidate:
                suffixes = [p.lstrip(".").lower() for p in Path(candidate).suffixes]
                if len(suffixes) >= 2:
                    tail = cls._EXT_MAP.get(suffixes[-1])
                    base = cls._EXT_MAP.get(suffixes[-2])
                    if (tail is not None and tail.is_codec
                        and base is not None and not base.is_codec):
                        emit(base)
                        emit(tail)
                        return

            # Fallback: single-value resolution.
            emit(cls.from_str(candidate, default=None))

        def walk(x: Any) -> None:
            if x is None:
                return
            if isinstance(x, cls):
                emit(x)
                return

            # Iterables — but not scalars that happen to be iterable
            # (str, bytes/bytearray/memoryview magic buffers, Path, IO).
            if not isinstance(x, (str, bytes, bytearray, memoryview)):
                _Path = path_class()
                is_path = _Path.is_pathish(x)
                is_io = hasattr(x, "read") and hasattr(x, "seek")
                if not is_path and not is_io and isinstance(x, (list, tuple, set, frozenset, Iterable, Iterator)):
                    try:
                        for item in x:
                            walk(item)
                    except TypeError:
                        pass
                    else:
                        return

            if isinstance(x, str):
                walk_str(x)
                return

            # bytes / IO / Path / PathLike — soft miss.
            emit(cls.from_(x, default=None))

        walk(obj)
        return list(seen.values())

    @classmethod
    def from_(cls, obj: Any, default: "MimeType | None" = _RAISE) -> "MimeType | None":
        if isinstance(obj, cls):
            return obj

        _Path = path_class()
        if _Path.is_pathish(obj):
            # ``Path.infer_media_type`` returns a MediaType (never None
            # — it has its own default-handling), but its ``default``
            # kwarg expects a MediaType or None, not our sentinel. Map
            # the sentinel to None for the forwarded call, then check
            # the result and route through _miss when it falls back to
            # a default-ish value.
            mt = _Path.from_(obj).infer_media_type(default=None)

            if mt is not None:
                return mt.mime_type

        if isinstance(obj, str):
            return cls.from_str(obj, default=default)

        return cls.from_magic(obj, default=default)

    @classmethod
    def from_magic(
        cls,
        magic: Union[bytes, bytearray, memoryview, IO[bytes], str, Path],
        default: "MimeType | None" = _RAISE,
    ) -> "MimeType | None":
        """Resolve by sniffing magic bytes from *magic*.

        Accepts raw bytes/memoryview, a BytesIO, or anything the
        buffer class can wrap. Reads the first 64 bytes and walks
        the registered magic matchers in definition order.

        :param default: see :class:`MimeType` class docstring for the
            shared default-handling contract.
        :raises ValueError: on a miss when *default* was not supplied.
        """
        if not magic:
            return _miss(default, "empty magic buffer")

        if not isinstance(magic, (bytes, bytearray)):
            BytesIO = bytes_io_class()

            if isinstance(magic, BytesIO):
                magic = bytes(magic.head(64))
            else:
                with BytesIO(magic, copy=False).view(pos=0) as b:
                    magic = bytes(b.read(64))

        for mt in cls._MAGIC_ORDER:
            for matcher in mt.magics:
                try:
                    if matcher(magic):
                        return mt
                except Exception:
                    continue

        # Structural text-format sniffers for common non-magic formats.
        if magic.startswith(b"{") or magic.startswith(b"["):
            return MimeTypes.NDJSON if b"}\n{" in magic else MimeTypes.JSON
        if magic.startswith(b"<"):
            return MimeTypes.XML

        return _miss(default, f"no magic match for {magic[:16]!r}")

    @classmethod
    def from_str(cls, value: str, default: "MimeType | None" = _RAISE) -> "MimeType | None":
        """Resolve a :class:`str` — path-like, bare extension, or mime value.

        Tries, in order:

        - If the string looks path-like (contains ``.``, ``/``, or
          ``\\``), take its suffix as an extension key.
        - Strip leading dots and try the bare form as an extension key.
        - Fall back to :meth:`get` (name / mime-value lookup).
        - Structural sniff on leading ``{`` / ``[``.

        :param default: see :class:`MimeType` class docstring for the
            shared default-handling contract.
        :raises ValueError: on a miss when *default* was not supplied.
        """
        candidate = value.strip()

        if "." in candidate or "/" in candidate or "\\" in candidate:
            p = Path(candidate)
            ext = p.suffix.lstrip(".").lower()
            if ext:
                hit = cls._EXT_MAP.get(ext)
                if hit is not None:
                    return hit

        bare = candidate.lstrip(".").lower()
        hit = cls._EXT_MAP.get(bare)
        if hit is not None:
            return hit

        mt = cls.get(candidate)
        if mt is not None:
            return mt

        if candidate.startswith("{") or candidate.startswith("["):
            return MimeTypes.NDJSON if "\n{" in candidate else MimeTypes.JSON

        return _miss(default, f"no match for {value!r}")

    # ------------------------------------------------------------------
    # Extension-map mutators
    # ------------------------------------------------------------------
    # These call :meth:`from_str` internally to resolve string inputs
    # to a MimeType. We pass ``default=None`` explicitly so the old
    # None-on-miss cascade (``cls.get(mime) or cls.from_str(mime)``)
    # keeps working — these mutators want to raise their OWN error
    # messages ("did not resolve to a known MimeType"), not inherit
    # the generic one from _miss.
    # ------------------------------------------------------------------

    @classmethod
    def register_extension(
        cls,
        ext: str,
        mime: "MimeType | str",
        *,
        overwrite: bool = False,
    ) -> None:
        key = ext.lstrip(".").lower()

        if isinstance(mime, MimeType):
            resolved = mime
        else:
            resolved = cls.get(mime) or cls.from_str(mime, default=None)

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
        resolved: list[tuple[str, MimeType]] = []

        for ext, mime in mapping.items():
            key = ext.lstrip(".").lower()

            if isinstance(mime, MimeType):
                mt = mime
            else:
                mt = cls.get(mime) or cls.from_str(mime, default=None)

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
        if isinstance(mime, MimeType):
            target = mime
        else:
            target = cls.get(mime) or cls.from_str(mime, default=None)

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
# Namespace for declared mime constants
# ----------------------------
class MimeTypes:
    """Singleton MIME definitions."""

    # --- Compression / codecs ---
    GZIP = MimeType.define(
        MimeType(
            "GZIP",
            "application/gzip",
            extensions=("gz", "gzip", "tgz"),
            magics=(magic_prefix(b"\x1f\x8b"),),
            is_codec=True,
        )
    )
    ZSTD = MimeType.define(
        MimeType(
            "ZSTD",
            "application/zstd",
            extensions=("zst", "zstd"),
            magics=(magic_prefix(b"\x28\xb5\x2f\xfd"),),
            is_codec=True,
        )
    )
    BROTLI = MimeType.define(
        MimeType("BROTLI", "application/x-brotli", extensions=("br", "brotli"), is_codec=True)
    )
    LZ4 = MimeType.define(
        MimeType(
            "LZ4",
            "application/x-lz4",
            extensions=("lz4",),
            magics=(magic_prefix(b"\x04\x22\x4d\x18"),),
            is_codec=True,
        )
    )
    SNAPPY = MimeType.define(
        MimeType("SNAPPY", "application/x-snappy", extensions=("snappy", "sz"), is_codec=True)
    )
    BZ2 = MimeType.define(
        MimeType(
            "BZ2",
            "application/x-bzip2",
            extensions=("bz2", "bzip2", "tbz2"),
            magics=(magic_prefix(b"\x42\x5a\x68"),),
            is_codec=True,
        )
    )
    XZ = MimeType.define(
        MimeType(
            "XZ",
            "application/x-xz",
            extensions=("xz", "txz"),
            magics=(magic_prefix(b"\xfd\x37\x7a\x58\x5a\x00"),),
            is_codec=True,
        )
    )
    ZLIB = MimeType.define(
        MimeType(
            "ZLIB",
            "application/zlib",
            extensions=("zlib",),
            magics=(magic_prefix(b"\x78\x01"), magic_prefix(b"\x78\x9c"), magic_prefix(b"\x78\xda")),
            is_codec=True,
        )
    )
    LZMA = MimeType.define(
        MimeType("LZMA", "application/x-lzma", extensions=("lzma",), is_codec=True)
    )
    ZZIP = MimeType.define(
        MimeType("ZZIP", "application/x-compress", extensions=("z", "Z"), is_codec=True)
    )

    # --- Containers / docs ---
    ZIP = MimeType.define(
        MimeType(
            "ZIP",
            "application/zip",
            extensions=("zip",),
            magics=(magic_prefix(b"PK\x03\x04"),),
        )
    )

    ZIP_ENTRY = MimeType.define(
        MimeType(
            "ZIP_ENTRY",
            "application/zip-entry",
            extensions=("zipentry",),
            magics=(magic_prefix(b"PK\x01\x02"),),
        )
    )

    PDF = MimeType.define(
        MimeType(
            "PDF",
            "application/pdf",
            extensions=("pdf",),
            magics=(magic_prefix(b"%PDF-"),),
        )
    )
    TAR = MimeType.define(
        MimeType(
            "TAR",
            "application/x-tar",
            extensions=("tar",),
            magics=(magic_prefix(b"\x75\x73\x74\x61\x72"),),
        )
    )
    SQLITE = MimeType.define(
        MimeType(
            "SQLITE",
            "application/x-sqlite3",
            extensions=("db", "sqlite", "sqlite3"),
            magics=(magic_prefix(b"SQLite format 3\x00"),),
        )
    )
    HDF5 = MimeType.define(
        MimeType(
            "HDF5",
            "application/x-hdf5",
            extensions=("h5", "hdf5", "he5"),
            magics=(magic_prefix(b"\x89HDF\r\n\x1a\n"),),
        )
    )
    XLSX = MimeType.define(
        MimeType(
            "XLSX",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            extensions=("xlsx", "xls"),
        )
    )
    DOCX = MimeType.define(
        MimeType(
            "DOCX",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            extensions=("docx",),
        )
    )

    # --- Columnar / analytics ---
    PARQUET = MimeType.define(
        MimeType(
            "PARQUET",
            "application/vnd.apache.parquet",
            extensions=("parquet", "pq"),
            magics=(magic_parquet,),
        )
    )
    PARQUET_DELTA = MimeType.define(
        MimeType("PARQUET_DELTA", "application/vnd.apache.parquet+delta")
    )
    ARROW_IPC = MimeType.define(
        MimeType(
            "ARROW_IPC",
            "application/vnd.apache.arrow.file",
            extensions=("ipc", "feather", "arrow", "arrows"),
            magics=(magic_prefix(b"ARROW1"),),
        )
    )
    ORC = MimeType.define(
        MimeType(
            "ORC",
            "application/vnd.apache.orc",
            extensions=("orc",),
            magics=(magic_prefix(b"ORC"),),
        )
    )
    AVRO = MimeType.define(
        MimeType(
            "AVRO",
            "application/avro",
            extensions=("avro",),
            magics=(magic_prefix(b"Obj\x01"),),
        )
    )
    ICEBERG = MimeType.define(
        MimeType("ICEBERG", "application/vnd.apache.iceberg", extensions=("iceberg",))
    )
    DELTA = MimeType.define(
        MimeType("DELTA", "application/vnd.delta", extensions=("delta", "deltatable"))
    )

    # --- Text / semi-structured ---
    JSON = MimeType.define(MimeType("JSON", "application/json", extensions=("json",)))
    NDJSON = MimeType.define(
        MimeType(
            "NDJSON",
            "application/ld+json",
            extensions=("jsonld", "ldjson"),
            is_tabular=True,
        )
    )
    CSV = MimeType.define(MimeType("CSV", "text/csv", extensions=("csv",), is_tabular=True))
    TSV = MimeType.define(
        MimeType("TSV", "text/tab-separated-values", extensions=("tsv",), is_tabular=True)
    )
    XML = MimeType.define(MimeType("XML", "application/xml", extensions=("xml",)))
    HTML = MimeType.define(MimeType("HTML", "text/html", extensions=("html", "htm")))
    PLAIN = MimeType.define(MimeType("PLAIN", "text/plain", extensions=("txt", "text")))
    YAML = MimeType.define(MimeType("YAML", "application/yaml", extensions=("yaml", "yml")))
    TOML = MimeType.define(MimeType("TOML", "application/toml", extensions=("toml",)))

    # --- Binary serialisation ---
    MSGPACK = MimeType.define(
        MimeType("MSGPACK", "application/msgpack", extensions=("msgpack", "mpk"))
    )
    PROTOBUF = MimeType.define(
        MimeType("PROTOBUF", "application/x-protobuf", extensions=("pb", "proto", "protobuf"))
    )
    FLATBUFFERS = MimeType.define(
        MimeType("FLATBUFFERS", "application/x-flatbuffers", extensions=("bin", "fbs"))
    )
    CBOR = MimeType.define(MimeType("CBOR", "application/cbor", extensions=("cbor",)))
    BSON = MimeType.define(MimeType("BSON", "application/bson", extensions=("bson",)))
    PICKLE = MimeType.define(
        MimeType("PICKLE", "application/x-python-pickle", extensions=("pkl", "pickle"))
    )
    NUMPY = MimeType.define(
        MimeType(
            "NUMPY",
            "application/x-npy",
            extensions=("npy",),
            magics=(magic_prefix(b"\x93NUMPY"),),
        )
    )
    NUMPY_ARCHIVE = MimeType.define(
        MimeType("NUMPY_ARCHIVE", "application/x-npz", extensions=("npz",))
    )

    # --- Images ---
    PNG = MimeType.define(
        MimeType(
            "PNG",
            "image/png",
            extensions=("png",),
            magics=(magic_prefix(b"\x89PNG\r\n\x1a\n"),),
        )
    )
    JPEG = MimeType.define(
        MimeType(
            "JPEG",
            "image/jpeg",
            extensions=("jpg", "jpeg"),
            magics=(magic_prefix(b"\xff\xd8\xff"),),
        )
    )
    GIF = MimeType.define(
        MimeType(
            "GIF",
            "image/gif",
            extensions=("gif",),
            magics=(magic_prefix(b"GIF87a"), magic_prefix(b"GIF89a")),
        )
    )
    WEBP = MimeType.define(
        MimeType(
            "WEBP",
            "image/webp",
            extensions=("webp",),
            magics=(magic_riff_webp,),
        )
    )
    TIFF = MimeType.define(
        MimeType(
            "TIFF",
            "image/tiff",
            extensions=("tif", "tiff"),
            magics=(magic_prefix(b"II*\x00"), magic_prefix(b"MM\x00*")),
        )
    )
    BMP = MimeType.define(
        MimeType(
            "BMP",
            "image/bmp",
            extensions=("bmp",),
            magics=(magic_prefix(b"BM"),),
        )
    )

    # --- Filesystem containers ---
    FOLDER = MimeType.define(
        MimeType(
            "FOLDER",
            # `inode/directory` is what `file --mime-type` returns on
            # Unix. Not IANA-registered but the de facto convention.
            # No extensions (directories don't have them) and no magic
            # (not a byte stream) — resolution goes through the
            # path-class via is_dir_sink.
            "inode/directory",
        )
    )

    PARTITIONED_FOLDER = MimeType.define(
        MimeType(
            "PARTITIONED_FOLDER",
            "inode/directory+partitioned",
        )
    )

    DELTA_FOLDER = MimeType.define(
        MimeType(
            "DELTA_FOLDER",
            "inode/directory+delta",
        )
    )

    STATEMENT_RESULT = MimeType.define(
        MimeType(
            "STATEMENT_RESULT",
            "application/vnd.statement.result",
        )
    )

    DATABRICKS_STATEMENT_RESULT = MimeType.define(
        MimeType(
            "DATABRICKS_STATEMENT",
            "application/vnd.databricks.statement",
        )
    )

    SPARK_SQL_STATEMENT = MimeType.define(
        MimeType(
            "SPARK_SQL_STATEMENT",
            "application/vnd.databricks.spark.sql",
        )
    )

    # Databricks
    DATABRICKS_UNITY_CATALOG_TABLE = MimeType.define(
        MimeType(
            "DATABRICKS_TABLE",
            "application/vnd.databricks.uc.table",
        )
    )

    # Kafka — a topic, addressed by broker + topic name. Streaming
    # tabular source; values can be any of the registered codecs
    # (JSON, Avro, Protobuf, plain bytes).
    KAFKA_TOPIC = MimeType.define(
        MimeType(
            "KAFKA_TOPIC",
            "application/vnd.apache.kafka.topic",
            is_tabular=True,
        )
    )

    # --- Fallback ---
    OCTET_STREAM = MimeType.define(
        MimeType("OCTET_STREAM", "application/octet-stream")
    )


# Seed container riders (XLSX/DOCX are ZIP containers)
MimeType.register_extensions(
    {
        "xlsx": MimeTypes.XLSX,
        "xls": MimeTypes.XLSX,
        "docx": MimeTypes.DOCX,
    },
    overwrite=True,
)