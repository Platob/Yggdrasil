"""MIME / media-type registry.

PARITY: ported to JS/TS at ``nextjs/src/lib/yggdrasil/enums/mimeType.ts``.
This module is the reference; keep the two in sync (see the "Cross-language
parity" rule in the repo CLAUDE.md).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, IO, Mapping, Union, Iterable, Iterator

__all__ = ["MimeType", "MimeTypes"]

from yggdrasil.lazy_imports import io_class

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
class _PrefixMatcher:
    """Callable wrapper around :meth:`bytes.startswith` that exposes
    its *prefix* for static introspection.

    :meth:`MimeType.define` reads :attr:`prefix` at registration time
    and slots the owning MimeType into the :data:`_MAGIC_BY_FIRST_BYTE`
    fast-path index, so the hot :meth:`MimeType.from_magic` loop can
    skip the matchers whose first byte doesn't match the head.
    """

    __slots__ = ("prefix",)

    def __init__(self, prefix: bytes) -> None:
        self.prefix = prefix

    def __call__(self, b: bytes) -> bool:
        return b.startswith(self.prefix)


def magic_prefix(prefix: bytes) -> MagicMatcher:
    return _PrefixMatcher(prefix)


def magic_riff_webp(b: bytes) -> bool:
    return b.startswith(b"RIFF") and len(b) >= 12 and b[8:12] == b"WEBP"


class _OffsetMatcher:
    """Match a fixed signature at a fixed byte *offset* (not a prefix).

    Some formats put their magic mid-header — tar's ``ustar`` lives at
    byte 257, not 0. These can't ride the first-byte prefix index, so
    :meth:`MimeType.define` files them on the dynamic-matcher list. They
    only fire when the sniff buffer is long enough to reach the offset.
    """

    __slots__ = ("offset", "sig")

    def __init__(self, offset: int, sig: bytes) -> None:
        self.offset = offset
        self.sig = sig

    def __call__(self, b: bytes) -> bool:
        return b[self.offset:self.offset + len(self.sig)] == self.sig


def magic_at(offset: int, sig: bytes) -> MagicMatcher:
    return _OffsetMatcher(offset, sig)


# Bytes peeked from the head of a source when sniffing by magic. Big
# enough to reach mid-header signatures (tar's ``ustar`` at offset 257)
# while staying a single small read.
_MAGIC_PEEK = 512


# Header only; footer exists too but we usually peek small buffers.
# Wrapped through :class:`_PrefixMatcher` so it lands on the
# first-byte fast path alongside the other pure-prefix matchers.
magic_parquet = _PrefixMatcher(b"PAR1")


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
    - is_tabular: row/tabular-ish formats (read into a frame)
    - is_blob: opaque single-file payload — straight byte IO, no row
      structure (images, pdf, archives, pickle, …). Mutually exclusive
      with ``is_tabular``; codecs and directory/connector mimes are
      neither.
    """

    name: str
    value: str
    extensions: tuple[str, ...] = ()
    magics: tuple[MagicMatcher, ...] = ()

    is_codec: bool = False
    is_tabular: bool = False
    is_blob: bool = False

    _BY_NAME: ClassVar[dict[str, "MimeType"]] = {}
    _BY_VALUE: ClassVar[dict[str, "MimeType"]] = {}
    _EXT_MAP: ClassVar[dict[str, "MimeType"]] = {}
    _MAGIC_ORDER: ClassVar[list["MimeType"]] = []
    # First-byte → (prefix, mime) candidates. Populated by :meth:`define`
    # for any matcher built via :func:`magic_prefix` (i.e. any matcher
    # that exposes a static :attr:`_PrefixMatcher.prefix`). The hot
    # :meth:`from_magic` loop probes this in O(1) instead of walking
    # every registered prefix matcher linearly.
    _MAGIC_PREFIX_INDEX: ClassVar[dict[int, list[tuple[bytes, "MimeType"]]]] = {}
    # Matchers that don't reduce to a static prefix (RIFF/WEBP) — these
    # still walk linearly, but the list is tiny.
    _MAGIC_DYNAMIC: ClassVar[list[tuple[MagicMatcher, "MimeType"]]] = []

    @classmethod
    def define(cls, mt: "MimeType") -> "MimeType":
        cls._BY_NAME[mt.name.lower()] = mt
        cls._BY_VALUE[mt.value.lower()] = mt

        for ext in mt.extensions:
            cls._EXT_MAP[ext.lower().lstrip(".")] = mt

        if mt.magics:
            cls._MAGIC_ORDER.append(mt)
            for matcher in mt.magics:
                prefix = getattr(matcher, "prefix", None)
                if isinstance(prefix, (bytes, bytearray)) and prefix:
                    cls._MAGIC_PREFIX_INDEX.setdefault(prefix[0], []).append(
                        (bytes(prefix), mt)
                    )
                else:
                    cls._MAGIC_DYNAMIC.append((matcher, mt))

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
            # (e.g. PARQUET_DELTA = "application/vnd.apache.parquet+delta").
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
                from yggdrasil.url import URL as _URL
                is_path = _URL.is_pathish(x)
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

        # Strings: hand straight to :meth:`from_str`. ``from_str``
        # already covers path-shaped strings via ``Path(candidate).suffix``
        # — routing through :class:`URL` for every string was a ~10x
        # detour for plain extension / mime-value inputs.
        if isinstance(obj, str):
            return cls.from_str(obj, default=default)

        # Path-shaped inputs short-circuit through URL: extension and
        # scheme tell us the mime without even constructing a Path
        # holder. Maps the sentinel to ``None`` for the forwarded call
        # so URL's own default handling stays consistent.
        from yggdrasil.url import URL as _URL
        if _URL.is_pathish(obj):
            mt = _URL.from_(obj).infer_media_type(default=None)
            if mt is not None:
                return mt.mime_type

        return cls.from_magic(obj, default=default)

    @classmethod
    def from_magic(
        cls,
        magic: Union[bytes, bytearray, memoryview, IO[bytes], str, Path],
        default: "MimeType | None" = _RAISE,
    ) -> "MimeType | None":
        """Resolve by sniffing magic bytes from *magic*.

        Accepts raw bytes/memoryview, an IO, or anything the
        buffer class can wrap. Reads the first 64 bytes and walks
        the registered magic matchers in definition order.

        :param default: see :class:`MimeType` class docstring for the
            shared default-handling contract.
        :raises ValueError: on a miss when *default* was not supplied.
        """
        if not magic:
            return _miss(default, "empty magic buffer")

        if not isinstance(magic, (bytes, bytearray)):
            IO = io_class()

            if isinstance(magic, IO):
                magic = bytes(magic.pread(_MAGIC_PEEK, 0))
            elif hasattr(magic, "read") and hasattr(magic, "seek"):
                # Stdlib-style file-like — read head, restore cursor so
                # the caller's IO comes out exactly as it went in.
                fh = magic
                saved = fh.tell()
                try:
                    fh.seek(0)
                    magic = fh.read(_MAGIC_PEEK)
                finally:
                    fh.seek(saved)
            else:
                # Path / PathLike — peek the head with a plain open (the
                # IO-holder wrap is legacy and over-heavy for a magic sniff).
                # Anything unreadable is a soft miss, not a crash.
                try:
                    with open(os.fspath(magic), "rb") as fh:
                        magic = fh.read(_MAGIC_PEEK)
                except Exception:
                    return _miss(default, "could not read magic bytes")

        # An empty head (e.g. a zero-byte file) has no magic to match.
        if not magic:
            return _miss(default, "empty magic buffer")

        # Fast path: probe the first-byte → prefix index. Almost every
        # registered magic is a fixed byte prefix (PNG / GZIP / PARQUET /
        # …) so a single ``magic[0]`` dispatch reduces the per-call cost
        # from "walk 25 matchers" to "walk 0-2 matchers sharing that
        # leading byte." :meth:`bytes.startswith` is C-level and can't
        # raise on a ``bytes`` argument, so no try/except is needed here.
        candidates = cls._MAGIC_PREFIX_INDEX.get(magic[0])
        if candidates is not None:
            for prefix, mt in candidates:
                if magic.startswith(prefix):
                    return mt

        # Dynamic matchers (RIFF/WEBP, …) — short list, still walked
        # linearly. Keep the try/except: these are arbitrary callables.
        for matcher, mt in cls._MAGIC_DYNAMIC:
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

        - Direct lookup against the lower-cased input (covers
          ``"text/csv"`` / ``"json"`` / ``".csv"`` without paying a
          :class:`pathlib.Path` allocation).
        - If the string looks path-like (contains ``/`` or ``\\``),
          take its suffix as an extension key.
        - Fall back to :meth:`get` (name / mime-value lookup with the
          ``application/`` / ``text/`` / … prefix stripping).
        - Structural sniff on leading ``{`` / ``[``.

        :param default: see :class:`MimeType` class docstring for the
            shared default-handling contract.
        :raises ValueError: on a miss when *default* was not supplied.
        """
        # Fast path: most callers pass already-stripped lowercase input
        # (``"text/csv"`` / ``"csv"`` / ``".parquet"`` / ``"application/json"``).
        # A single dict probe each against ``_BY_VALUE`` / ``_EXT_MAP``
        # resolves them without normalising or allocating a ``Path``.
        hit = cls._BY_VALUE.get(value)
        if hit is not None:
            return hit
        bare = value[1:] if value.startswith(".") else value
        hit = cls._EXT_MAP.get(bare)
        if hit is not None:
            return hit

        candidate = value.strip()
        lower = candidate.lower()

        hit = cls._BY_VALUE.get(lower)
        if hit is not None:
            return hit

        bare = lower.lstrip(".")
        hit = cls._EXT_MAP.get(bare)
        if hit is not None:
            return hit

        if "/" in lower or "\\" in lower:
            # ``Path(candidate).suffix`` worked but allocates a full
            # PurePath holder for what is a trailing-dot scan. The
            # inline form lifts ``MimeType.from_str('/data/trades.csv')``
            # from ~2.3us to a few hundred ns and matches
            # :attr:`URL.extensions`' string-level convention.
            #
            # Last path segment, then last suffix only — preserves
            # ``pathlib.PurePath.suffix`` semantics (leading dotfile
            # with no other dots → no suffix; trailing slash ignored).
            seg = lower
            slash = max(seg.rfind("/"), seg.rfind("\\"))
            if slash != -1:
                seg = seg[slash + 1:]
            if seg.endswith("/") or seg.endswith("\\"):
                seg = seg[:-1]
            if seg.startswith("."):
                seg = seg[1:]
            dot = seg.rfind(".")
            if dot != -1:
                ext = seg[dot + 1:]
                if ext:
                    hit = cls._EXT_MAP.get(ext)
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
        MimeType("ZZIP", "application/x-compress", extensions=("z",), is_codec=True)
    )

    # ------------------------------------------------------------------
    # Archives & multi-file containers — opaque single files on disk
    # ------------------------------------------------------------------
    ZIP = MimeType.define(
        MimeType(
            "ZIP",
            "application/zip",
            extensions=("zip",),
            magics=(magic_prefix(b"PK\x03\x04"),),
            is_blob=True,
        )
    )
    ZIP_ENTRY = MimeType.define(
        MimeType(
            "ZIP_ENTRY",
            "application/zip-entry",
            extensions=("zipentry",),
            magics=(magic_prefix(b"PK\x01\x02"),),
            is_blob=True,
        )
    )
    TAR = MimeType.define(
        MimeType(
            "TAR",
            "application/x-tar",
            extensions=("tar",),
            # ``ustar`` sits at byte 257 of the header block, not the start.
            magics=(magic_at(257, b"ustar"),),
            is_blob=True,
        )
    )
    SEVEN_ZIP = MimeType.define(
        MimeType(
            "SEVEN_ZIP",
            "application/x-7z-compressed",
            extensions=("7z",),
            magics=(magic_prefix(b"7z\xbc\xaf\x27\x1c"),),
            is_blob=True,
        )
    )
    RAR = MimeType.define(
        MimeType(
            "RAR",
            "application/vnd.rar",
            extensions=("rar",),
            magics=(magic_prefix(b"Rar!\x1a\x07"),),
            is_blob=True,
        )
    )

    # ------------------------------------------------------------------
    # Documents / office
    # ------------------------------------------------------------------
    PDF = MimeType.define(
        MimeType(
            "PDF",
            "application/pdf",
            extensions=("pdf",),
            magics=(magic_prefix(b"%PDF-"),),
            is_blob=True,
        )
    )
    XLSX = MimeType.define(
        MimeType(
            "XLSX",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            extensions=("xlsx", "xls"),
            is_tabular=True,
        )
    )
    DOCX = MimeType.define(
        MimeType(
            "DOCX",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            extensions=("docx",),
            is_blob=True,
        )
    )

    # ------------------------------------------------------------------
    # Embedded stores — single-file databases
    # ------------------------------------------------------------------
    SQLITE = MimeType.define(
        MimeType(
            "SQLITE",
            # IANA-registered value (was the de-facto ``application/x-sqlite3``).
            "application/vnd.sqlite3",
            extensions=("db", "sqlite", "sqlite3"),
            magics=(magic_prefix(b"SQLite format 3\x00"),),
            is_blob=True,
        )
    )
    HDF5 = MimeType.define(
        MimeType(
            "HDF5",
            "application/x-hdf5",
            extensions=("h5", "hdf5", "he5"),
            magics=(magic_prefix(b"\x89HDF\r\n\x1a\n"),),
            is_blob=True,
        )
    )

    # --- Columnar / analytics ---
    PARQUET = MimeType.define(
        MimeType(
            "PARQUET",
            "application/vnd.apache.parquet",
            extensions=("parquet", "pq"),
            magics=(magic_parquet,),
            is_tabular=True,
        )
    )
    PARQUET_DELTA = MimeType.define(
        MimeType("PARQUET_DELTA", "application/vnd.apache.parquet+delta", is_tabular=True)
    )
    ARROW_IPC = MimeType.define(
        MimeType(
            "ARROW_IPC",
            "application/vnd.apache.arrow.file",
            extensions=("ipc", "feather", "arrow", "arrows"),
            magics=(magic_prefix(b"ARROW1"),),
            is_tabular=True,
        )
    )
    # Arrow IPC *stream* framing — distinct from the file format above
    # (no ``ARROW1`` header). This is what the node's inter-process /
    # HTTP transport emits (``application/vnd.apache.arrow.stream``).
    ARROW_STREAM = MimeType.define(
        MimeType(
            "ARROW_STREAM",
            "application/vnd.apache.arrow.stream",
            is_tabular=True,
        )
    )
    ORC = MimeType.define(
        MimeType(
            "ORC",
            "application/vnd.apache.orc",
            extensions=("orc",),
            magics=(magic_prefix(b"ORC"),),
            is_tabular=True,
        )
    )
    AVRO = MimeType.define(
        MimeType(
            "AVRO",
            # vendor tree, consistent with PARQUET / ORC / ICEBERG.
            "application/vnd.apache.avro",
            extensions=("avro",),
            magics=(magic_prefix(b"Obj\x01"),),
            is_tabular=True,
        )
    )
    ICEBERG = MimeType.define(
        MimeType("ICEBERG", "application/vnd.apache.iceberg", extensions=("iceberg",), is_tabular=True)
    )
    DELTA = MimeType.define(
        MimeType("DELTA", "application/vnd.delta", extensions=("delta", "deltatable"), is_tabular=True)
    )

    # --- Text / semi-structured ---
    # JSON is tabular in this stack: the readers coerce it (and an array of
    # objects) into a frame, same as NDJSON below.
    JSON = MimeType.define(MimeType("JSON", "application/json", extensions=("json",), is_tabular=True))
    NDJSON = MimeType.define(
        MimeType(
            "NDJSON",
            "application/ld+json",
            extensions=("ndjson",),
            is_tabular=True,
        )
    )
    CSV = MimeType.define(MimeType("CSV", "text/csv", extensions=("csv",), is_tabular=True))
    TSV = MimeType.define(
        MimeType("TSV", "text/tab-separated-values", extensions=("tsv",), is_tabular=True)
    )
    # --- Text / markup (opaque single-file text) ---
    XML = MimeType.define(MimeType("XML", "application/xml", extensions=("xml",), is_blob=True))
    HTML = MimeType.define(MimeType("HTML", "text/html", extensions=("html", "htm"), is_blob=True))
    PLAIN = MimeType.define(MimeType("PLAIN", "text/plain", extensions=("txt", "text"), is_blob=True))
    MARKDOWN = MimeType.define(MimeType("MARKDOWN", "text/markdown", extensions=("md", "markdown"), is_blob=True))
    YAML = MimeType.define(MimeType("YAML", "application/yaml", extensions=("yaml", "yml"), is_blob=True))
    TOML = MimeType.define(MimeType("TOML", "application/toml", extensions=("toml",), is_blob=True))

    # --- Binary serialisation (opaque single-file payloads) ---
    MSGPACK = MimeType.define(
        MimeType("MSGPACK", "application/msgpack", extensions=("msgpack", "mpk"), is_blob=True)
    )
    PROTOBUF = MimeType.define(
        MimeType("PROTOBUF", "application/x-protobuf", extensions=("pb", "proto", "protobuf"), is_blob=True)
    )
    FLATBUFFERS = MimeType.define(
        # No ``bin`` extension: ``.bin`` is generic binary and must fall
        # through to OCTET_STREAM, not get claimed as a FlatBuffer.
        MimeType("FLATBUFFERS", "application/x-flatbuffers", extensions=("fbs",), is_blob=True)
    )
    CBOR = MimeType.define(MimeType("CBOR", "application/cbor", extensions=("cbor",), is_blob=True))
    BSON = MimeType.define(MimeType("BSON", "application/bson", extensions=("bson",), is_blob=True))
    PICKLE = MimeType.define(
        MimeType("PICKLE", "application/x-python-pickle", extensions=("pkl", "pickle"), is_blob=True)
    )
    NUMPY = MimeType.define(
        MimeType(
            "NUMPY",
            "application/x-npy",
            extensions=("npy",),
            magics=(magic_prefix(b"\x93NUMPY"),),
            is_blob=True,
        )
    )
    NUMPY_ARCHIVE = MimeType.define(
        MimeType("NUMPY_ARCHIVE", "application/x-npz", extensions=("npz",), is_blob=True)
    )

    # --- Images (opaque single-file payloads) ---
    PNG = MimeType.define(
        MimeType(
            "PNG",
            "image/png",
            extensions=("png",),
            magics=(magic_prefix(b"\x89PNG\r\n\x1a\n"),),
            is_blob=True,
        )
    )
    JPEG = MimeType.define(
        MimeType(
            "JPEG",
            "image/jpeg",
            extensions=("jpg", "jpeg"),
            magics=(magic_prefix(b"\xff\xd8\xff"),),
            is_blob=True,
        )
    )
    GIF = MimeType.define(
        MimeType(
            "GIF",
            "image/gif",
            extensions=("gif",),
            magics=(magic_prefix(b"GIF87a"), magic_prefix(b"GIF89a")),
            is_blob=True,
        )
    )
    WEBP = MimeType.define(
        MimeType(
            "WEBP",
            "image/webp",
            extensions=("webp",),
            magics=(magic_riff_webp,),
            is_blob=True,
        )
    )
    TIFF = MimeType.define(
        MimeType(
            "TIFF",
            "image/tiff",
            extensions=("tif", "tiff"),
            magics=(magic_prefix(b"II*\x00"), magic_prefix(b"MM\x00*")),
            is_blob=True,
        )
    )
    BMP = MimeType.define(
        MimeType(
            "BMP",
            "image/bmp",
            extensions=("bmp",),
            magics=(magic_prefix(b"BM"),),
            is_blob=True,
        )
    )
    SVG = MimeType.define(
        MimeType(
            "SVG",
            "image/svg+xml",
            extensions=("svg",),
            is_blob=True,
        )
    )
    ICO = MimeType.define(
        MimeType(
            "ICO",
            "image/x-icon",
            extensions=("ico",),
            magics=(magic_prefix(b"\x00\x00\x01\x00"),),
            is_blob=True,
        )
    )
    AVIF = MimeType.define(
        MimeType("AVIF", "image/avif", extensions=("avif",), is_blob=True)
    )
    HEIC = MimeType.define(
        MimeType("HEIC", "image/heic", extensions=("heic", "heif"), is_blob=True)
    )

    # --- Audio (opaque single-file payloads) ---
    MP3 = MimeType.define(
        MimeType(
            "MP3",
            "audio/mpeg",
            extensions=("mp3",),
            magics=(magic_prefix(b"ID3"),),
            is_blob=True,
        )
    )
    WAV = MimeType.define(
        MimeType("WAV", "audio/wav", extensions=("wav",), is_blob=True)
    )
    FLAC = MimeType.define(
        MimeType(
            "FLAC",
            "audio/flac",
            extensions=("flac",),
            magics=(magic_prefix(b"fLaC"),),
            is_blob=True,
        )
    )
    OGG = MimeType.define(
        MimeType(
            "OGG",
            "audio/ogg",
            extensions=("ogg", "oga"),
            magics=(magic_prefix(b"OggS"),),
            is_blob=True,
        )
    )
    AAC = MimeType.define(
        MimeType("AAC", "audio/aac", extensions=("aac",), is_blob=True)
    )

    # --- Video (opaque single-file payloads) ---
    MP4 = MimeType.define(
        MimeType("MP4", "video/mp4", extensions=("mp4", "m4v"), is_blob=True)
    )
    WEBM = MimeType.define(
        MimeType("WEBM", "video/webm", extensions=("webm",), is_blob=True)
    )
    MKV = MimeType.define(
        MimeType("MKV", "video/x-matroska", extensions=("mkv",), is_blob=True)
    )
    MOV = MimeType.define(
        MimeType("MOV", "video/quicktime", extensions=("mov",), is_blob=True)
    )
    AVI = MimeType.define(
        MimeType("AVI", "video/x-msvideo", extensions=("avi",), is_blob=True)
    )

    # --- Filesystem containers ---
    DIRECTORY = MimeType.define(
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

    HTTP_RESPONSE = MimeType.define(
        MimeType(
            "HTTP_RESPONSE",
            # Envelope mime for a yggdrasil :class:`Response` row — the
            # deterministic single-row metadata projection (status,
            # headers, body bytes, identity hashes, …) defined by
            # :data:`RESPONSE_ARROW_SCHEMA`. Used as the fallback mime
            # when the response body's own Content-Type doesn't
            # resolve to a registered tabular leaf, so the row still
            # reads through the Tabular surface as a one-row table.
            "application/vnd.yggdrasil.http-response",
            is_tabular=True,
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
    DATABRICKS_UNITY_CATALOG_CATALOG = MimeType.define(
        MimeType(
            "DATABRICKS_CATALOG",
            "application/vnd.databricks.uc.catalog",
        )
    )
    DATABRICKS_UNITY_CATALOG_SCHEMA = MimeType.define(
        MimeType(
            "DATABRICKS_SCHEMA",
            "application/vnd.databricks.uc.schema",
        )
    )

    # --- Databricks ``DataSourceFormat`` connectors ---------------------
    # One :class:`MimeType` per Databricks ``DataSourceFormat`` enum value
    # that doesn't already map to a generic columnar mime (DELTA / PARQUET
    # / AVRO / ORC / CSV / JSON / ICEBERG / TEXT live above). Names match
    # the SDK enum values so :meth:`MimeType.from_` resolves a Databricks
    # format string in one hop, and the values describe the underlying
    # system (``application/vnd.mysql`` not ``application/vnd.databricks
    # .mysql``) so non-Databricks integrations targeting the same source
    # can share the categorization.
    HIVE = MimeType.define(
        MimeType("HIVE", "application/vnd.apache.hive", is_tabular=True)
    )
    DELTASHARING = MimeType.define(
        MimeType("DELTASHARING", "application/vnd.delta.sharing", is_tabular=True)
    )
    DELTA_UNIFORM_HUDI = MimeType.define(
        MimeType(
            "DELTA_UNIFORM_HUDI",
            "application/vnd.delta.uniform.hudi",
            is_tabular=True,
        )
    )
    DELTA_UNIFORM_ICEBERG = MimeType.define(
        MimeType(
            "DELTA_UNIFORM_ICEBERG",
            "application/vnd.delta.uniform.iceberg",
            is_tabular=True,
        )
    )
    UNITY_CATALOG = MimeType.define(
        MimeType(
            "UNITY_CATALOG",
            # Foreign UC entry — distinct from the table-itself mime
            # (``application/vnd.databricks.uc.table``) which carries
            # the table resource. ``uc.foreign`` is the proxy view that
            # points outside Databricks-managed storage.
            "application/vnd.databricks.uc.foreign",
            is_tabular=True,
        )
    )
    DATABRICKS_FORMAT = MimeType.define(
        MimeType(
            "DATABRICKS_FORMAT", "application/vnd.databricks", is_tabular=True
        )
    )
    DATABRICKS_ROW_STORE_FORMAT = MimeType.define(
        MimeType(
            "DATABRICKS_ROW_STORE_FORMAT",
            "application/vnd.databricks.row_store",
            is_tabular=True,
        )
    )
    VECTOR_INDEX_FORMAT = MimeType.define(
        MimeType(
            "VECTOR_INDEX_FORMAT",
            "application/vnd.databricks.vector_index",
            is_tabular=True,
        )
    )
    BIGQUERY_FORMAT = MimeType.define(
        MimeType(
            "BIGQUERY_FORMAT",
            "application/vnd.google.bigquery",
            is_tabular=True,
        )
    )
    MONGODB_FORMAT = MimeType.define(
        MimeType("MONGODB_FORMAT", "application/vnd.mongodb", is_tabular=True)
    )
    MYSQL_FORMAT = MimeType.define(
        MimeType("MYSQL_FORMAT", "application/vnd.mysql", is_tabular=True)
    )
    NETSUITE_FORMAT = MimeType.define(
        MimeType("NETSUITE_FORMAT", "application/vnd.netsuite", is_tabular=True)
    )
    ORACLE_FORMAT = MimeType.define(
        MimeType("ORACLE_FORMAT", "application/vnd.oracle", is_tabular=True)
    )
    POSTGRESQL_FORMAT = MimeType.define(
        MimeType(
            "POSTGRESQL_FORMAT", "application/vnd.postgresql", is_tabular=True
        )
    )
    REDSHIFT_FORMAT = MimeType.define(
        MimeType(
            "REDSHIFT_FORMAT",
            "application/vnd.amazon.redshift",
            is_tabular=True,
        )
    )
    SALESFORCE_FORMAT = MimeType.define(
        MimeType(
            "SALESFORCE_FORMAT", "application/vnd.salesforce", is_tabular=True
        )
    )
    SALESFORCE_DATA_CLOUD_FORMAT = MimeType.define(
        MimeType(
            "SALESFORCE_DATA_CLOUD_FORMAT",
            "application/vnd.salesforce.data_cloud",
            is_tabular=True,
        )
    )
    SNOWFLAKE_FORMAT = MimeType.define(
        MimeType(
            "SNOWFLAKE_FORMAT", "application/vnd.snowflake", is_tabular=True
        )
    )
    SQLDW_FORMAT = MimeType.define(
        MimeType(
            "SQLDW_FORMAT",
            "application/vnd.microsoft.sqldw",
            is_tabular=True,
        )
    )
    SQLSERVER_FORMAT = MimeType.define(
        MimeType(
            "SQLSERVER_FORMAT",
            "application/vnd.microsoft.sqlserver",
            is_tabular=True,
        )
    )
    TERADATA_FORMAT = MimeType.define(
        MimeType("TERADATA_FORMAT", "application/vnd.teradata", is_tabular=True)
    )
    WORKDAY_RAAS_FORMAT = MimeType.define(
        MimeType(
            "WORKDAY_RAAS_FORMAT",
            "application/vnd.workday.raas",
            is_tabular=True,
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

    # --- Fallback (generic opaque bytes) ---
    OCTET_STREAM = MimeType.define(
        MimeType("OCTET_STREAM", "application/octet-stream", is_blob=True)
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