# yggdrasil/io/enums/media_type.py
"""MIME-type inference from magic bytes and text heuristics.

This module provides :class:`MediaType`, a frozen dataclass that pairs a
canonical MIME string with an optional outer :class:`~.codec.Codec`.  It is
the primary format descriptor used throughout the yggdrasil I/O layer —
every buffer, reader, and writer accepts or returns a :class:`MediaType` so
that format and compression decisions are made once at the boundary and
propagated cleanly downstream.

Detection pipeline
------------------
:meth:`MediaType.from_io` applies the following steps in order:

1. **Codec sniff** — peek the first 16 bytes and test against
   :attr:`Codec._MAGIC`.  If a compression header is found, return
   ``(OCTET_STREAM, codec)``; the inner format cannot be determined without
   decompression.
2. **Binary magic table** — walk :attr:`MediaType._MAGIC`; first
   ``(offset, magic)`` match wins.
3. **Text heuristics** — if the header is valid UTF-8, apply lightweight
   pattern matching for NDJSON, JSON, TSV, CSV, and MessagePack text.
4. **Fallback** — ``"application/octet-stream"``.

Well-known constants
--------------------
Every supported format is a :class:`ClassVar[str]` on the class so call
sites can reference ``MediaType.PARQUET`` rather than hard-coding MIME
strings, and equality checks work with both string literals and constants::

    mt == MediaType.PARQUET          # True
    mt == "application/vnd.apache.parquet"  # also True

Typical usage
-------------
Detection::

    mt = MediaType.from_io(buf)
    mt = MediaType.from_bytes(raw)

Manual construction::

    mt = MediaType.of(MediaType.PARQUET)
    mt = MediaType.of(MediaType.PARQUET, codec=Codec.ZSTD)

Codec stripping (for format-dispatch branches)::

    fmt = mt.without_codec()        # same mime, codec=None
    if fmt == MediaType.PARQUET: ...
"""

from __future__ import annotations

import io
from dataclasses import dataclass, replace
from typing import IO, ClassVar

from .codec import Codec, _peek

__all__ = ["MediaType"]


@dataclass(frozen=True, slots=True)
class MediaType:
    """Inferred MIME type for a binary payload.

    A :class:`MediaType` is immutable and hashable — it is safe to use as a
    dict key or set member.

    Attributes
    ----------
    mime:
        Canonical MIME type string, e.g.
        ``"application/vnd.apache.parquet"``.  Defaults to
        ``"application/octet-stream"`` when the format is unrecognised.
    codec:
        Outer compression :class:`Codec` detected *before* format sniffing,
        or ``None`` for uncompressed payloads.

    Notes
    -----
    Format detection is purely magic-byte based — no attempt is made to
    fully parse or validate the payload.  Text formats (JSON, CSV, TSV,
    NDJSON) are identified via lightweight UTF-8 heuristics on the first
    32 bytes when no binary magic matches.
    """

    mime:  str
    codec: Codec | None = None

    # ------------------------------------------------------------------
    # Well-known MIME strings — columnar / analytics
    # ------------------------------------------------------------------

    PARQUET:        ClassVar[str] = "application/vnd.apache.parquet"
    PARQUET_DELTA:  ClassVar[str] = "application/vnd.apache.parquet+delta"
    ARROW_FILE:     ClassVar[str] = "application/vnd.apache.arrow.file"
    ARROW_STREAM:   ClassVar[str] = "application/vnd.apache.arrow.stream"
    # IPC and FEATHER are widely-used aliases for the Arrow file format.
    IPC:            ClassVar[str] = "application/vnd.apache.arrow.file"
    FEATHER:        ClassVar[str] = "application/vnd.apache.arrow.file"
    ORC:            ClassVar[str] = "application/vnd.apache.orc"
    AVRO:           ClassVar[str] = "application/avro"
    ICEBERG:        ClassVar[str] = "application/vnd.apache.iceberg"
    DELTA:          ClassVar[str] = "application/vnd.delta"

    # ------------------------------------------------------------------
    # Well-known MIME strings — text / semi-structured
    # ------------------------------------------------------------------

    JSON:           ClassVar[str] = "application/json"
    NDJSON:         ClassVar[str] = "application/x-ndjson"
    JSONL:          ClassVar[str] = "application/x-ndjson"   # alias
    CSV:            ClassVar[str] = "text/csv"
    TSV:            ClassVar[str] = "text/tab-separated-values"
    XML:            ClassVar[str] = "application/xml"
    HTML:           ClassVar[str] = "text/html"
    PLAIN:          ClassVar[str] = "text/plain"
    YAML:           ClassVar[str] = "application/yaml"
    TOML:           ClassVar[str] = "application/toml"

    # ------------------------------------------------------------------
    # Well-known MIME strings — binary serialisation
    # ------------------------------------------------------------------

    MSGPACK:        ClassVar[str] = "application/msgpack"
    PROTOBUF:       ClassVar[str] = "application/x-protobuf"
    FLATBUFFERS:    ClassVar[str] = "application/x-flatbuffers"
    CBOR:           ClassVar[str] = "application/cbor"
    BSON:           ClassVar[str] = "application/bson"
    PICKLE:         ClassVar[str] = "application/x-python-pickle"
    NUMPY:          ClassVar[str] = "application/x-npy"
    NUMPY_ARCHIVE:  ClassVar[str] = "application/x-npz"

    # ------------------------------------------------------------------
    # Well-known MIME strings — document / container
    # ------------------------------------------------------------------

    PDF:            ClassVar[str] = "application/pdf"
    ZIP:            ClassVar[str] = "application/zip"
    TAR:            ClassVar[str] = "application/x-tar"
    SQLITE:         ClassVar[str] = "application/x-sqlite3"
    HDF5:           ClassVar[str] = "application/x-hdf5"
    XLSX:           ClassVar[str] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    DOCX:           ClassVar[str] = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    # ------------------------------------------------------------------
    # Well-known MIME strings — image
    # ------------------------------------------------------------------

    PNG:            ClassVar[str] = "image/png"
    JPEG:           ClassVar[str] = "image/jpeg"
    GIF:            ClassVar[str] = "image/gif"
    WEBP:           ClassVar[str] = "image/webp"
    TIFF:           ClassVar[str] = "image/tiff"
    BMP:            ClassVar[str] = "image/bmp"

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    OCTET_STREAM:   ClassVar[str] = "application/octet-stream"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def of(cls, mime: str, *, codec: Codec | str | None = None) -> "MediaType":
        """Construct a :class:`MediaType` from a MIME string and optional codec.

        Prefer this over the raw constructor for call sites that may receive
        a string codec value — the coercion from ``str`` to :class:`Codec`
        is handled here.

        Parameters
        ----------
        mime:
            MIME type string, e.g. ``MediaType.PARQUET``.
        codec:
            A :class:`Codec` member, its string value (e.g. ``"zstd"``), or
            ``None``.

        Returns
        -------
        MediaType

        Examples
        --------
        >>> MediaType.of(MediaType.PARQUET, codec="zstd")
        MediaType('application/vnd.apache.parquet', codec=<Codec.ZSTD: 'zstd'>)
        """
        if isinstance(codec, str):
            codec = Codec(codec)
        return cls(mime=mime, codec=codec)

    def without_codec(self) -> "MediaType":
        """Return a copy of this instance with ``codec`` set to ``None``.

        Used in format-dispatch branches where the codec has already been
        handled (e.g. after calling :meth:`~.codec.Codec.open`)::

            fmt = mt.without_codec()
            if fmt == MediaType.PARQUET:
                df = pl.read_parquet(src)

        Returns
        -------
        MediaType
            A new frozen :class:`MediaType` with identical ``mime`` and
            ``codec=None``.  Returns ``self`` when ``codec`` is already
            ``None`` (no allocation).
        """
        if self.codec is None:
            return self
        return replace(self, codec=None)

    def with_codec(self, codec: Codec | str | None) -> "MediaType":
        """Return a copy of this instance with *codec* replaced.

        Parameters
        ----------
        codec:
            New :class:`Codec` value, its string representation, or ``None``
            to strip compression.

        Returns
        -------
        MediaType

        Examples
        --------
        >>> MediaType.of(MediaType.CSV).with_codec("zstd")
        MediaType('text/csv', codec=<Codec.ZSTD: 'zstd'>)
        """
        if isinstance(codec, str):
            codec = Codec(codec)
        return replace(self, codec=codec)

    # ------------------------------------------------------------------
    # Equality — support comparison against plain MIME strings
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Compare against another :class:`MediaType` or a plain MIME string.

        When *other* is a ``str`` only the :attr:`mime` field is compared,
        making ``mt == MediaType.PARQUET`` equivalent to
        ``mt.mime == MediaType.PARQUET`` and avoiding the need to strip
        codecs before format checks.

        Parameters
        ----------
        other:
            A :class:`MediaType` instance or a MIME string.

        Examples
        --------
        >>> mt = MediaType(MediaType.PARQUET, codec=Codec.ZSTD)
        >>> mt == MediaType.PARQUET          # True — mime-only comparison
        True
        >>> mt == MediaType(MediaType.PARQUET)   # True — full comparison
        True
        >>> mt == MediaType(MediaType.CSV)        # False
        False
        """
        if isinstance(other, str):
            return self.mime == other
        if isinstance(other, MediaType):
            return self.mime == other.mime and self.codec == other.codec
        return NotImplemented

    def __hash__(self) -> int:
        # Consistent with __eq__: two MediaType instances with the same mime
        # and codec hash identically.  String comparison bypasses codec so
        # hashing must still include codec to avoid spurious collisions in
        # dicts/sets that hold full MediaType keys.
        return hash((self.mime, self.codec))

    # ------------------------------------------------------------------
    # Predicates — format family
    # ------------------------------------------------------------------

    @property
    def is_parquet(self) -> bool:
        """``True`` for Parquet and Delta Lake (Parquet-based) payloads."""
        return self.mime in (self.PARQUET, self.PARQUET_DELTA)

    @property
    def is_arrow(self) -> bool:
        """``True`` for both Arrow IPC file and Arrow IPC stream formats."""
        return self.mime in (self.ARROW_FILE, self.ARROW_STREAM)

    @property
    def is_ipc(self) -> bool:
        """Alias for :attr:`is_arrow` — IPC and Feather are the same wire format."""
        return self.is_arrow

    @property
    def is_tabular(self) -> bool:
        """``True`` for any column / row format routinely used in data pipelines."""
        return self.mime in (
            self.PARQUET,
            self.PARQUET_DELTA,
            self.ARROW_FILE,
            self.ARROW_STREAM,
            self.ORC,
            self.AVRO,
            self.CSV,
            self.TSV,
            self.NDJSON,
            self.ICEBERG,
            self.XLSX,
            self.HDF5,
        )

    @property
    def is_json_like(self) -> bool:
        """``True`` for JSON and newline-delimited JSON / JSONL."""
        return self.mime in (self.JSON, self.NDJSON)

    @property
    def is_text(self) -> bool:
        """``True`` for human-readable text formats."""
        return self.mime in (
            self.JSON,
            self.NDJSON,
            self.CSV,
            self.TSV,
            self.XML,
            self.HTML,
            self.PLAIN,
            self.YAML,
            self.TOML,
        )

    @property
    def is_binary_serialisation(self) -> bool:
        """``True`` for efficient binary wire formats (not columnar analytics)."""
        return self.mime in (
            self.MSGPACK,
            self.PROTOBUF,
            self.FLATBUFFERS,
            self.CBOR,
            self.BSON,
            self.PICKLE,
        )

    @property
    def is_image(self) -> bool:
        """``True`` for recognised raster image formats."""
        return self.mime in (
            self.PNG,
            self.JPEG,
            self.GIF,
            self.WEBP,
            self.TIFF,
            self.BMP,
        )

    @property
    def is_archive(self) -> bool:
        """``True`` for container / archive formats (ZIP, TAR, HDF5, SQLite)."""
        return self.mime in (self.ZIP, self.TAR, self.HDF5, self.SQLITE)

    @property
    def is_apache(self) -> bool:
        """``True`` for any format in the Apache data ecosystem."""
        return self.mime in (
            self.PARQUET,
            self.PARQUET_DELTA,
            self.ARROW_FILE,
            self.ARROW_STREAM,
            self.ORC,
            self.AVRO,
            self.ICEBERG,
        )

    @property
    def is_compressed(self) -> bool:
        """``True`` when an outer compression codec was detected."""
        return self.codec is not None

    @property
    def is_unknown(self) -> bool:
        """``True`` when the format could not be identified."""
        return self.mime == self.OCTET_STREAM and self.codec is None

    @property
    def is_polars_readable(self) -> bool:
        """``True`` when :meth:`~BytesIO.read_polars` can handle this format.

        Does **not** account for the codec — a compressed Parquet buffer
        (``is_compressed=True``) is still polars-readable after decompression.
        """
        return self.without_codec().mime in (
            self.PARQUET,
            self.PARQUET_DELTA,
            self.ARROW_FILE,
            self.ARROW_STREAM,
            self.ORC,
            self.AVRO,
            self.CSV,
            self.TSV,
            self.JSON,
            self.NDJSON,
            self.XLSX,
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def extension(self) -> str:
        """Suggested file extension for this MIME type (without leading dot).

        Returns ``"bin"`` for ``application/octet-stream``.  The codec
        wrapper extension (e.g. ``".zst"``) is **not** included — use
        :attr:`full_extension` for that.

        Examples
        --------
        >>> MediaType.of(MediaType.PARQUET).extension
        'parquet'
        >>> MediaType.of(MediaType.CSV).extension
        'csv'
        """
        return _MIME_TO_EXT.get(self.mime, "bin")

    @property
    def codec_extension(self) -> str:
        """File extension for the outer codec, or empty string when ``codec`` is ``None``.

        Examples
        --------
        >>> MediaType.of(MediaType.PARQUET, codec="zstd").codec_extension
        'zst'
        """
        return _CODEC_EXT.get(self.codec, "") if self.codec else ""

    @property
    def full_extension(self) -> str:
        """Combined extension including the codec suffix when present.

        Examples
        --------
        >>> MediaType.of(MediaType.PARQUET, codec="zstd").full_extension
        'parquet.zst'
        >>> MediaType.of(MediaType.CSV).full_extension
        'csv'
        """
        ext = self.extension
        codec_ext = self.codec_extension
        return f"{ext}.{codec_ext}" if codec_ext else ext

    @classmethod
    def from_extension(cls, ext: str) -> "MediaType":
        """Construct a :class:`MediaType` from a file extension.

        The extension may optionally include a leading dot and a trailing
        codec suffix (e.g. ``".parquet.zst"`` or ``"csv.gz"``).

        Parameters
        ----------
        ext:
            File extension string.  Case-insensitive.

        Returns
        -------
        MediaType
            Best-guess :class:`MediaType`.  Falls back to
            :attr:`OCTET_STREAM` when the extension is not recognised.

        Examples
        --------
        >>> MediaType.from_extension(".parquet.zst")
        MediaType('application/vnd.apache.parquet', codec=<Codec.ZSTD: 'zstd'>)
        >>> MediaType.from_extension("csv")
        MediaType('text/csv')
        """
        parts = ext.lstrip(".").lower().split(".")

        # Detect trailing codec suffix (e.g. "zst", "gz", "lz4").
        codec: Codec | None = None
        if len(parts) > 1 and parts[-1] in _EXT_TO_CODEC:
            codec = _EXT_TO_CODEC[parts[-1]]
            parts = parts[:-1]

        mime = _EXT_TO_MIME.get(".".join(parts), cls.OCTET_STREAM)
        return cls(mime=mime, codec=codec)

    @classmethod
    def from_mime(
        cls,
        mime: str,
        *,
        codec: "Codec | str | None" = None,
    ) -> "MediaType":
        """Construct a :class:`MediaType` from a raw MIME string.

        Unlike the bare constructor, this method:

        * Normalises the input (strips whitespace, lowercases).
        * Resolves ``content-type`` header values that carry a ``; charset=``
          or other parameter suffix (e.g.
          ``"application/json; charset=utf-8"`` → ``"application/json"``).
        * Attempts to extract an embedded codec from compound MIME strings
          such as ``"application/gzip"`` or ``"application/x-bzip2"`` when
          no explicit *codec* is provided.
        * Falls back to :attr:`OCTET_STREAM` for unrecognised strings rather
          than storing an arbitrary value that nothing else understands.

        Parameters
        ----------
        mime:
            Raw MIME type string.  May include parameter suffixes, codec
            aliases (``"application/gzip"``, ``"application/zstd"``), or
            compound labels (``"application/vnd.apache.parquet+zstd"``).
        codec:
            Override the codec.  Accepts a :class:`Codec` member, its
            string value, or ``None``.  When ``None`` the codec is inferred
            from *mime* itself if possible.

        Returns
        -------
        MediaType

        Examples
        --------
        >>> MediaType.from_mime("application/vnd.apache.parquet")
        MediaType('application/vnd.apache.parquet')

        >>> MediaType.from_mime("application/json; charset=utf-8")
        MediaType('application/json')

        >>> MediaType.from_mime("application/gzip")
        MediaType('application/octet-stream', codec=<Codec.GZIP: 'gzip'>)

        >>> MediaType.from_mime("application/vnd.apache.parquet+zstd")
        MediaType('application/vnd.apache.parquet', codec=<Codec.ZSTD: 'zstd'>)

        >>> MediaType.from_mime("text/csv", codec="gzip")
        MediaType('text/csv', codec=<Codec.GZIP: 'gzip'>)
        """
        if isinstance(codec, str):
            codec = Codec(codec)

        # ── Normalise: strip parameters ("; charset=utf-8", "; boundary=…")
        base = mime.strip().lower().split(";")[0].strip()

        # ── Resolve pure-codec MIME aliases → (OCTET_STREAM, codec) ──────────
        # These MIME types *are* the compression — there is no inner format.
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
            resolved_codec = codec if codec is not None else _CODEC_MIME[base]
            return cls(mime=cls.OCTET_STREAM, codec=resolved_codec)

        # ── Resolve compound "format+codec" suffix (e.g. "parquet+zstd") ──────
        # Walk the known format table first; if the base ends with "+<codec>"
        # split it off and try to match the format prefix.
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

        # ── Validate against the known MIME constant set ───────────────────────
        _ALL_KNOWN: frozenset[str] = frozenset({
            cls.PARQUET, cls.PARQUET_DELTA, cls.ARROW_FILE, cls.ARROW_STREAM,
            cls.ORC, cls.AVRO, cls.ICEBERG, cls.DELTA,
            cls.JSON, cls.NDJSON, cls.CSV, cls.TSV,
            cls.XML, cls.HTML, cls.PLAIN, cls.YAML, cls.TOML,
            cls.MSGPACK, cls.PROTOBUF, cls.FLATBUFFERS, cls.CBOR, cls.BSON,
            cls.PICKLE, cls.NUMPY, cls.NUMPY_ARCHIVE,
            cls.PDF, cls.ZIP, cls.TAR, cls.SQLITE, cls.HDF5, cls.XLSX, cls.DOCX,
            cls.PNG, cls.JPEG, cls.GIF, cls.WEBP, cls.TIFF, cls.BMP,
            cls.OCTET_STREAM,
        })

        resolved_mime = format_base if format_base in _ALL_KNOWN else cls.OCTET_STREAM
        resolved_codec = codec if codec is not None else inferred_codec
        return cls(mime=resolved_mime, codec=resolved_codec)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    # Internal magic-byte dispatch table: list[tuple[int, bytes, str]]
    # (byte_offset, magic_sequence, mime_string)  — populated after class body.
    _MAGIC: ClassVar[list]

    @classmethod
    def from_io(cls, src: "IO[bytes] | BytesIO") -> "MediaType":
        """Infer the MIME type (and outer codec) from *src*'s magic bytes.

        The caller's cursor position is **preserved** throughout.

        Parameters
        ----------
        src:
            Any seekable binary stream or :class:`~dynamic_buffer.BytesIO`.

        Returns
        -------
        MediaType
            Frozen :class:`MediaType`.  :attr:`codec` is set when an outer
            compression wrapper is detected; :attr:`mime` is
            ``"application/octet-stream"`` in that case because the inner
            format requires decompression to determine.

        Algorithm
        ---------
        1. Peek first 16 bytes → test :attr:`Codec._MAGIC`.
        2. If compressed → ``(OCTET_STREAM, codec)`` (inner unknown).
        3. Peek first 32 bytes → walk :attr:`_MAGIC` table.
        4. If no binary match → decode as UTF-8 and apply text heuristics.
        5. Fallback → ``(OCTET_STREAM, None)``.

        Examples
        --------
        >>> MediaType.from_io(io.BytesIO(b"PAR1" + b"\\x00" * 28)).mime
        'application/vnd.apache.parquet'

        >>> mt = MediaType.from_io(io.BytesIO(b"\\x1f\\x8b\\x00" * 10))
        >>> mt.codec, mt.is_compressed
        (<Codec.GZIP: 'gzip'>, True)
        """
        fh: IO[bytes] = src.buffer() if hasattr(src, "buffer") else src  # type: ignore[union-attr]

        # ── 1. Outer codec detection ───────────────────────────────────────────
        codec: Codec | None = Codec.from_io(fh)
        if codec is not None:
            return cls(mime=cls.OCTET_STREAM, codec=codec)

        # ── 2. Binary magic table ──────────────────────────────────────────────
        header = _peek(fh, 512)

        for offset, magic, mime in cls._MAGIC:
            end = offset + len(magic)
            if len(header) >= end and header[offset:end] == magic:
                return cls(mime=mime, codec=None)

        # ── 3. Text heuristics ────────────────────────────────────────────────
        try:
            text_head = header.decode("utf-8", errors="strict").lstrip()
        except UnicodeDecodeError:
            return cls(mime=cls.OCTET_STREAM, codec=None)

        non_empty = [ln.strip() for ln in text_head.split("\n") if ln.strip()]
        first  = non_empty[0] if non_empty else ""
        second = non_empty[1] if len(non_empty) > 1 else ""

        # NDJSON: two or more consecutive JSON-object lines.
        if first.startswith("{") and second.startswith("{"):
            return cls(mime=cls.NDJSON, codec=None)

        # JSON object or array literal.
        if first.startswith(("{", "[")):
            return cls(mime=cls.JSON, codec=None)

        # XML / HTML.
        if first.startswith("<?xml") or first.startswith("<xml"):
            return cls(mime=cls.XML, codec=None)
        if first.lower().startswith(("<!doctype html", "<html")):
            return cls(mime=cls.HTML, codec=None)

        # YAML front-matter or document marker.
        if first.startswith("---"):
            return cls(mime=cls.YAML, codec=None)

        # TOML: key = value pattern on the first line.
        if "=" in first and not first.startswith("#") and not first.startswith("{"):
            return cls(mime=cls.TOML, codec=None)

        # TSV: tab-separated first line.
        if "\t" in first:
            return cls(mime=cls.TSV, codec=None)

        # CSV: comma in an alphanumeric-starting first line.
        if "," in first and first[0].isalnum():
            return cls(mime=cls.CSV, codec=None)

        return cls(mime=cls.OCTET_STREAM, codec=None)

    @classmethod
    def from_bytes(cls, data: bytes) -> "MediaType":
        """Detect :class:`MediaType` from a raw ``bytes`` object.

        Convenience wrapper around :meth:`from_io` — no cursor management
        required from the caller.

        Parameters
        ----------
        data:
            Raw bytes whose leading bytes may carry format magic.

        Returns
        -------
        MediaType
        """
        return cls.from_io(io.BytesIO(data))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return self.mime

    def __repr__(self) -> str:
        codec_part = f", codec={self.codec!r}" if self.codec else ""
        return f"MediaType({self.mime!r}{codec_part})"


# ---------------------------------------------------------------------------
# Extension ↔ MIME lookup tables
# ---------------------------------------------------------------------------

_MIME_TO_EXT: dict[str, str] = {
    MediaType.PARQUET:        "parquet",
    MediaType.PARQUET_DELTA:  "parquet",
    MediaType.ARROW_FILE:     "arrow",
    MediaType.ARROW_STREAM:   "arrows",
    MediaType.ORC:            "orc",
    MediaType.AVRO:           "avro",
    MediaType.ICEBERG:        "iceberg",
    MediaType.JSON:           "json",
    MediaType.NDJSON:         "ndjson",
    MediaType.CSV:            "csv",
    MediaType.TSV:            "tsv",
    MediaType.XML:            "xml",
    MediaType.HTML:           "html",
    MediaType.PLAIN:          "txt",
    MediaType.YAML:           "yaml",
    MediaType.TOML:           "toml",
    MediaType.MSGPACK:        "msgpack",
    MediaType.PROTOBUF:       "proto",
    MediaType.FLATBUFFERS:    "fbs",
    MediaType.CBOR:           "cbor",
    MediaType.BSON:           "bson",
    MediaType.PICKLE:         "pkl",
    MediaType.NUMPY:          "npy",
    MediaType.NUMPY_ARCHIVE:  "npz",
    MediaType.PDF:            "pdf",
    MediaType.ZIP:            "zip",
    MediaType.TAR:            "tar",
    MediaType.SQLITE:         "sqlite",
    MediaType.HDF5:           "h5",
    MediaType.XLSX:           "xlsx",
    MediaType.DOCX:           "docx",
    MediaType.PNG:            "png",
    MediaType.JPEG:           "jpg",
    MediaType.GIF:            "gif",
    MediaType.WEBP:           "webp",
    MediaType.TIFF:           "tiff",
    MediaType.BMP:            "bmp",
    MediaType.OCTET_STREAM:   "bin",
}

_EXT_TO_MIME: dict[str, str] = {
    # Columnar
    "parquet":  MediaType.PARQUET,
    "pq":       MediaType.PARQUET,
    "arrow":    MediaType.ARROW_FILE,
    "arrows":   MediaType.ARROW_STREAM,
    "ipc":      MediaType.ARROW_FILE,
    "feather":  MediaType.ARROW_FILE,
    "orc":      MediaType.ORC,
    "avro":     MediaType.AVRO,
    # Text / semi-structured
    "json":     MediaType.JSON,
    "ndjson":   MediaType.NDJSON,
    "jsonl":    MediaType.NDJSON,
    "csv":      MediaType.CSV,
    "tsv":      MediaType.TSV,
    "txt":      MediaType.PLAIN,
    "xml":      MediaType.XML,
    "html":     MediaType.HTML,
    "htm":      MediaType.HTML,
    "yaml":     MediaType.YAML,
    "yml":      MediaType.YAML,
    "toml":     MediaType.TOML,
    # Binary serialisation
    "msgpack":  MediaType.MSGPACK,
    "proto":    MediaType.PROTOBUF,
    "pb":       MediaType.PROTOBUF,
    "fbs":      MediaType.FLATBUFFERS,
    "cbor":     MediaType.CBOR,
    "bson":     MediaType.BSON,
    "pkl":      MediaType.PICKLE,
    "pickle":   MediaType.PICKLE,
    "npy":      MediaType.NUMPY,
    "npz":      MediaType.NUMPY_ARCHIVE,
    # Documents / containers
    "pdf":      MediaType.PDF,
    "zip":      MediaType.ZIP,
    "tar":      MediaType.TAR,
    "sqlite":   MediaType.SQLITE,
    "db":       MediaType.SQLITE,
    "h5":       MediaType.HDF5,
    "hdf5":     MediaType.HDF5,
    "xlsx":     MediaType.XLSX,
    "docx":     MediaType.DOCX,
    # Images
    "png":      MediaType.PNG,
    "jpg":      MediaType.JPEG,
    "jpeg":     MediaType.JPEG,
    "gif":      MediaType.GIF,
    "webp":     MediaType.WEBP,
    "tiff":     MediaType.TIFF,
    "tif":      MediaType.TIFF,
    "bmp":      MediaType.BMP,
}

# Canonical single extension per codec — used for output (full_extension, codec_extension).
_CODEC_EXT: dict[Codec, str] = {
    Codec.GZIP:   "gz",
    Codec.ZSTD:   "zst",
    Codec.LZ4:    "lz4",
    Codec.BZIP2:  "bz2",
    Codec.XZ:     "xz",
    Codec.SNAPPY: "snappy",
}

# All recognised extensions / suffixes → codec — used for input (from_extension, from_mime).
# One-to-many inverse: every alias maps to the same Codec member.
_EXT_TO_CODEC: dict[str, Codec] = {
    # GZIP
    "gz":     Codec.GZIP,
    "gzip":   Codec.GZIP,
    # ZSTD
    "zst":    Codec.ZSTD,
    "zstd":   Codec.ZSTD,
    # LZ4
    "lz4":    Codec.LZ4,
    # BZIP2
    "bz2":    Codec.BZIP2,
    "bzip2":  Codec.BZIP2,
    # XZ / LZMA
    "xz":     Codec.XZ,
    "lzma":   Codec.XZ,
    # Snappy
    "snappy": Codec.SNAPPY,
    "sz":     Codec.SNAPPY,
}


# ---------------------------------------------------------------------------
# Magic-byte table
# ---------------------------------------------------------------------------
# Entries: (byte_offset, magic_sequence, mime_string)
# Checked in order — more specific / longer magic first.

MediaType._MAGIC = [
    # ── Apache columnar ────────────────────────────────────────────────────
    (0,  b"PAR1",              MediaType.PARQUET),
    (0,  b"ARROW1\x00\x00",   MediaType.ARROW_FILE),
    # Arrow IPC stream: continuous marker 0xFFFFFFFF followed by schema msg.
    (0,  b"\xff\xff\xff\xff",  MediaType.ARROW_STREAM),
    (0,  b"ORC",               MediaType.ORC),
    # Avro object container: "Obj\x01"
    (0,  b"Obj\x01",           MediaType.AVRO),

    # ── Binary serialisation ───────────────────────────────────────────────
    # MessagePack fixmap (0x8*) or fixarray (0x9*) or str/bin/ext/float/int.
    # CBOR: first byte 0x9f (indefinite array) or 0xa0–0xbf (map).
    # BSON: little-endian int32 document length; first 4 bytes < 0x01000000.
    # NumPy .npy: magic "\x93NUMPY"
    (0,  b"\x93NUMPY",         MediaType.NUMPY),
    # SQLite: "SQLite format 3\x00"
    (0,  b"SQLite format 3\x00", MediaType.SQLITE),
    # HDF5: "\x89HDF\r\n\x1a\n"
    (0,  b"\x89HDF\r\n\x1a\n", MediaType.HDF5),

    # ── Document / container ──────────────────────────────────────────────
    (0,  b"%PDF",              MediaType.PDF),
    # ZIP / XLSX / DOCX / JAR (all share PK\x03\x04 local file header).
    (0,  b"PK\x03\x04",       MediaType.ZIP),
    # TAR ustar: magic at offset 257
    (257, b"ustar",            MediaType.TAR),

    # ── Image formats ─────────────────────────────────────────────────────
    (0,  b"\x89PNG\r\n\x1a\n", MediaType.PNG),
    (0,  b"\xff\xd8\xff",      MediaType.JPEG),
    (0,  b"GIF87a",            MediaType.GIF),
    (0,  b"GIF89a",            MediaType.GIF),
    (0,  b"RIFF",              MediaType.WEBP),   # refined by bytes 8-11 = "WEBP"
    (0,  b"II\x2a\x00",        MediaType.TIFF),   # little-endian TIFF
    (0,  b"MM\x00\x2a",        MediaType.TIFF),   # big-endian TIFF
    (0,  b"BM",                MediaType.BMP),
]