# yggdrasil/io/enums/media_type.py
"""Format + codec wrapper resolution.

Where :class:`MimeType` resolves a single MIME identifier (string,
extension, magic bytes), :class:`MediaType` resolves the *combination*
of an inner format and an outer wrapper codec — what you actually need
to dispatch reads/writes.

Two-stage sniffing
------------------

For bytes-shaped and file-like inputs the resolution is two-stage.
The outer magic identifies the wrapper (``b"\\x1f\\x8b..."`` → GZIP).
When the wrapper is a codec we then peek a head-window of the
decompressed payload and sniff *that* — so a gzip-wrapped parquet
returns ``MediaType(PARQUET, codec=GZIP)`` rather than the lossy
``MediaType(OCTET_STREAM, codec=GZIP)`` that comes from sniffing the
outer mime alone.

Two entry points:

- :meth:`MediaType.from_magic` — bytes / bytearray / memoryview.
  No cursor concerns; sniffs in-memory.
- :meth:`MediaType.from_io` — file-like with ``read`` + ``seek``.
  Seeks the source freely during sniffing (one read at position 0
  for the outer head; if a codec is identified, another seek to 0
  to drive a streaming decompressor for the inner head). The
  caller's cursor is captured on entry and restored on exit, so
  the IO is left exactly where it came in regardless of how much
  internal seeking happened.

Both entry points fall back to ``MediaType(OCTET_STREAM, codec=<wrapper>)``
when the wrapper is identified but the inner sniff fails — losing
nothing the caller already had.
"""

from __future__ import annotations

import io as _stdio
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, Union, Any

from yggdrasil.io.url import URL
from yggdrasil.lazy_imports import path_class

from yggdrasil.data.enums.codec import Codec
from yggdrasil.data.enums.mime_type import MimeType, MimeTypes

__all__ = ["MediaType", "MediaTypes"]


# Outer-magic peek window. :meth:`MimeType.from_magic` reads 64 bytes
# internally for bytes inputs; we match that for the IO path so a magic
# that resolves on raw bytes also resolves on an IO of the same content.
_OUTER_PEEK = 64

# Inner-sniff head window for codec-wrapped magic. 128 bytes is plenty
# for every magic in the registry — the longest known is SQLite at 16
# bytes, so we have ~8x headroom while keeping the decompression cheap
# for streaming codecs.
_INNER_PEEK = 128


@dataclass(frozen=True, slots=True)
class MediaType:
    mime_type: MimeType
    codec: Codec | None = None

    def __post_init__(self):
        object.__setattr__(self, "mime_type", MimeType.from_(self.mime_type))

        if self.mime_type.is_codec:
            codec = Codec.from_mime(self.mime_type)
            object.__setattr__(self, "mime_type", MimeTypes.OCTET_STREAM)
            object.__setattr__(self, "codec", codec)

    def __repr__(self) -> str:
        if self.codec is None:
            return self.mime_type.value
        return f"{self.mime_type.value!r} + {self.codec.mime_type.value!r}"

    @classmethod
    def from_(
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
        default: "MediaType" = ...
    ) -> "MediaType":
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, MimeType):
            return cls.from_mime(mime_type=obj, codec=codec)

        if isinstance(obj, Codec):
            return cls(mime_type=MimeTypes.OCTET_STREAM, codec=obj)

        if isinstance(obj, (bytes, bytearray, memoryview)):
            return cls.from_magic(obj, default=default)
        if hasattr(obj, "read") and hasattr(obj, "seek"):
            return cls.from_io(obj, default=default)
        if URL.is_pathish(obj):
            parsed = cls.from_path(obj, default=None)
            if parsed is not None:
                return parsed

        parsed = MimeType.parse_many(obj)
        return cls.from_many(parsed, default=default)

    @classmethod
    def from_mime(
        cls,
        mime_type: MimeType | None = None,
        codec: Codec | None = None,
        *,
        default: Any = ...
    ):
        mt = MimeType.from_(mime_type, default=None)
        codec_resolved = Codec.from_(codec, default=None)

        if mt is None and codec_resolved is None:
            if default is ...:
                raise ValueError(
                    f"Cannot parse {cls.__name__} from {mime_type!r}"
                )
            return default

        if mt is None:
            return cls(mime_type=MimeTypes.OCTET_STREAM, codec=codec_resolved)

        if mt.is_codec:
            # Caller passed a codec mime as the format; demote to
            # OCTET_STREAM and prefer the resolved codec field if it
            # carries a non-codec format... otherwise fold mt into codec.
            return cls(
                mime_type=MimeTypes.OCTET_STREAM,
                codec=codec_resolved or Codec.from_mime(mt),
            )
        return cls(mime_type=mt, codec=codec_resolved)

    @classmethod
    def from_many(
        cls,
        mime_types: Iterable[MimeType],
        default: "MediaType" = ...,
    ) -> "MediaType":
        """Compose a MediaType from an ordered mime list (e.g. URL extensions).

        Two conventions land here:

        - ``trades.csv.zst`` → ``["csv", "zst"]``: the codec is the
          *outer wrapper* (you have to unzst before you can parse
          csv) → ``MediaType(CSV, codec=ZSTD)``.
        - ``part-xxx.zstd.parquet`` → ``["zstd", "parquet"]``: the
          format is the *outer wrapper* and the codec is the parquet
          page-codec hint baked into the file. Parquet handles the
          decompression internally → ``MediaType(PARQUET)`` with no
          outer codec; setting one would route the read through a
          decompressor that doesn't belong on this byte stream.

        Order matters: the **last** mime decides. Last-is-codec
        promotes the codec to the wrapper slot; last-is-format keeps
        the format alone and drops earlier codec hints.
        """
        parsed = MimeType.parse_many(mime_types)

        if not parsed:
            if default is ...:
                raise ValueError(
                    f"Cannot parse {cls.__name__} from {mime_types!r}"
                )
            return default

        if len(parsed) == 1:
            return cls.from_mime(parsed[0], default=default)

        last = parsed[-1]
        if last.is_codec:
            # Codec is the outer wrapper. Pick the first non-codec
            # mime as the format if any, else codec-only payload.
            format_mime = next((m for m in parsed if not m.is_codec), None)
            return cls.from_mime(
                mime_type=format_mime,
                codec=last,
                default=default,
            )

        # Last is a format → format wins, codec extensions before it
        # are page-codec hints (parquet/orc/...) and don't belong in
        # the wrapper slot.
        return cls.from_mime(mime_type=last, default=default)

    # ------------------------------------------------------------------
    # Magic / IO sniffers — two-stage (outer wrapper + inner format)
    # ------------------------------------------------------------------

    @classmethod
    def from_magic(
        cls,
        buf: Union[bytes, bytearray, memoryview, IO[bytes]],
        *,
        default: "MediaType" = ...,
    ) -> "MediaType":
        """Sniff a :class:`MediaType` from raw bytes.

        Two-stage: outer magic identifies the wrapper; if the wrapper
        is a codec, decompress a head-window and sniff the inner
        format. See module docstring for the full story.

        Accepts bytes / bytearray / memoryview directly. For
        convenience, also accepts an IO[bytes] — in that case it
        delegates to :meth:`from_io`, which manages cursor save/restore.

        :param default: Fallback when the outer sniff finds nothing.
            ``...`` (default) raises :class:`ValueError`; any other
            value (including ``None``) is returned as-is.
        """
        if hasattr(buf, "read") and hasattr(buf, "seek"):
            return cls.from_io(buf, default=default)

        outer = MimeType.from_magic(buf, default=None)
        return cls._compose(buf, outer, default=default)

    @classmethod
    def from_path(
        cls,
        path: "Path",
        *,
        default: "MediaType" = ...,
    ):
        path = path_class().from_(path, default=None)
        if path is None:
            if default is ...:
                raise ValueError(
                    f"Cannot parse {cls.__name__} from {path!r}"
                )
            return default

        fast_parsed = cls.from_url(path.url, default=None)
        if fast_parsed is not None:
            return fast_parsed

        # Backend-aware "is this a directory sink?" probe — older
        # paths exposed ``is_dir_sink``; the new substrate uses
        # ``is_dir``. Fall through gracefully if neither is present
        # (a string-shaped pseudo-path) so we just attempt the
        # IO sniff next.
        is_dir_sink = getattr(path, "is_dir_sink", None)
        try:
            if callable(is_dir_sink) and is_dir_sink():
                return MediaType(MimeTypes.FOLDER)
        except Exception:
            pass
        try:
            if path.is_dir():
                return MediaType(MimeTypes.FOLDER)
        except Exception:
            pass

        # Magic-byte sniffing requires an existing file. Skip the open
        # when the path doesn't exist so we don't create an empty stub
        # on disk just to fail back to *default* (LocalPath._acquire
        # opens ``O_RDWR | O_CREAT`` regardless of the requested mode).
        try:
            path_exists = path.exists()
        except Exception:
            path_exists = True  # opaque backend — let the open below decide

        if not path_exists:
            if default is ...:
                raise ValueError(
                    f"Cannot parse {cls.__name__} from {path!r} (does not exist)"
                )
            return default

        try:
            with path.open(mode="rb") as f:
                return cls.from_io(f, default=default)
        except Exception as e:
            if default is ...:
                raise ValueError(
                    f"Cannot parse {cls.__name__} from {path!r}"
                ) from e
            return default

    @classmethod
    def from_url(
        cls,
        url: URL,
        default: Any = ...,
    ):
        url = URL.from_(url, default=None)

        if url is None:
            if default is ...:
                raise ValueError(
                    f"Cannot parse {cls.__name__} from {url!r}"
                )
            return default

        if not url.path or url.path == "/":
            return cls.from_mime(MimeTypes.FOLDER, default=default)

        return cls.from_many(
            url.extensions,
            default=default
        )

    @classmethod
    def from_io(
        cls,
        io_obj: IO[bytes],
        *,
        default: "MediaType" = ...,
    ) -> "MediaType":
        """Sniff a :class:`MediaType` from a file-like, seeking freely.

        Captures the cursor on entry and restores it on exit, so the
        caller's stream position is unaffected. Inside the call we
        seek freely:

        1. ``seek(0)`` and read the first :data:`_OUTER_PEEK` bytes
           for outer-magic resolution.
        2. If the outer mime is a codec, ``seek(0)`` again and drive
           the codec's streaming decompressor directly to read the
           first :data:`_INNER_PEEK` bytes of decoded payload.
        3. Sniff the decoded head for the inner mime.

        The codec resolution is one dict lookup
        (:meth:`Codec.from_mime`) — the outer magic loop already
        landed on the registered :class:`MimeType` singleton, so we
        reuse it as the key without re-resolving. For streaming
        codecs (gzip / zstd / lz4 / bz2 / xz / zlib / lzma) the
        decompressor is opened directly on the IO via
        :meth:`Codec._open_decompress_reader` — no BytesIO wrap, no
        full-buffer materialization. Non-streaming codecs (snappy,
        brotli) fall back to reading the compressed body and calling
        :meth:`Codec.decompress_bytes`.

        Decompression errors during the inner sniff are swallowed —
        the worst case is returning a less-specific MediaType (the
        outer codec is preserved). A separate caller that wants to
        validate decompressibility should call :meth:`Codec.decompress`
        directly.

        :param io_obj: Any file-like with ``read`` + ``seek`` (stdlib
            ``io.BytesIO``, an open file, our own ``BytesIO``, etc.).
        :param default: Fallback when the outer sniff finds nothing.
            ``...`` (default) raises :class:`ValueError`; any other
            value (including ``None``) is returned as-is.
        """
        saved = io_obj.tell()
        try:
            io_obj.seek(0)
            head = io_obj.read(_OUTER_PEEK)
            outer = MimeType.from_magic(head, default=None)

            if outer is None:
                if default is ...:
                    raise ValueError(
                        f"{cls.__name__}.from_io: no magic match"
                    )
                return default

            # Non-codec outer → done. __post_init__ does not flip
            # because outer.is_codec is False.
            if not outer.is_codec:
                return cls(mime_type=outer, codec=None)

            # Codec outer → resolve once via the registry's
            # mime→codec map (O(1)) and drive the inner sniff
            # directly off io_obj.
            codec = Codec.from_mime(outer, default=None)
            if codec is None:
                # Mime says codec but we have no impl registered
                # (e.g. BROTLI without the package). Best we can do
                # is the outer-only MediaType.
                return cls(mime_type=outer)

            inner_head = cls._read_inner_head_io(io_obj, codec)
            return cls._finalize_inner(inner_head, codec)
        finally:
            # Unconditional cursor restore — the whole point of from_io
            # is that the caller's stream is intact afterwards.
            try:
                io_obj.seek(saved)
            except Exception:
                pass

    @staticmethod
    def _read_inner_head_io(io_obj: IO[bytes], codec: Codec) -> bytes:
        """Read up to :data:`_INNER_PEEK` decompressed bytes from *io_obj*.

        Caller is inside :meth:`from_io`'s try/finally — we're free
        to seek(0) here and the outer restore will fix it up. The
        cursor it leaves us in does not matter.

        Streaming codecs: open the decompressor directly on io_obj
        and pull the head. Non-streaming codecs (snappy, brotli):
        read all compressed bytes and call ``decompress_bytes``,
        slicing the head.

        Returns ``b""`` on any decompression error — the caller
        treats an empty head as "inner unknown".
        """
        try:
            io_obj.seek(0)
        except Exception:
            return b""

        reader = codec._open_decompress_reader(io_obj)
        if reader is not None:
            try:
                with reader:
                    return reader.read(_INNER_PEEK)
            except Exception:
                return b""

        # Non-streaming fallback. Read all compressed bytes, decode,
        # slice the head. This materializes the full compressed body
        # in memory; in practice this only fires for snappy/brotli
        # blobs which are typically small.
        try:
            io_obj.seek(0)
            compressed = io_obj.read()
            if not compressed:
                return b""
            decoded = codec.decompress_bytes(compressed)
            return decoded[:_INNER_PEEK]
        except Exception:
            return b""

    @classmethod
    def _compose(
        cls,
        buf: Union[bytes, bytearray, memoryview],
        outer: MimeType | None,
        *,
        default: "MediaType",
    ) -> "MediaType":
        """Compose the final :class:`MediaType` from a bytes buffer +
        already-resolved outer mime.

        Bytes-side counterpart of :meth:`from_io`'s body. Diverges
        only in how the inner head is fetched: the bytes path uses
        :meth:`Codec.read_start_end` on the buffer (no cursor to
        manage — the buffer is immutable from our POV).
        """
        if outer is None:
            if default is ...:
                raise ValueError(f"{cls.__name__}.from_magic: no magic match")
            return default

        if not outer.is_codec:
            return cls(mime_type=outer, codec=None)

        codec = Codec.from_mime(outer, default=None)
        if codec is None:
            return cls(mime_type=outer)

        # Wrap the buffer in a stdlib BytesIO and reuse
        # :meth:`_read_inner_head_io` — same code path as :meth:`from_io`.
        # Going through :meth:`Codec.read_start_end` here would pay for
        # the full yggdrasil BytesIO acquire/release lifecycle plus a
        # tail-collection state machine we don't need (``n_end=0``).
        # A bare ``io.BytesIO`` is enough: streaming decompressors
        # (gzip / zstd / lz4 / bz2 / xz / lzma / zlib) only need a
        # file-like with ``read``, and the head sniff bounds the read
        # at :data:`_INNER_PEEK` bytes regardless of source size.
        bio = _stdio.BytesIO(bytes(buf) if isinstance(buf, memoryview) else buf)
        head = cls._read_inner_head_io(bio, codec)
        return cls._finalize_inner(head, codec)

    @classmethod
    def _finalize_inner(cls, head: bytes, codec: Codec) -> "MediaType":
        """Sniff *head* and assemble the final MediaType.

        Shared by :meth:`from_io` and :meth:`_compose`. Three
        possible outcomes:

        - Inner sniff yields a non-codec mime → ``MediaType(inner, codec=outer)``.
        - Inner sniff yields ``None`` (no magic match, empty head) →
          ``MediaType(OCTET_STREAM, codec=outer)``.
        - Inner sniff yields a codec mime → ``MediaType(OCTET_STREAM, codec=outer)``.
          We deliberately don't recurse into a second unwrapping
          layer; nested-codec blobs (gzip-of-zstd-of-parquet) are
          vanishingly rare. Letting a codec-mime inner reach
          :meth:`__post_init__` would silently overwrite ``codec``
          with the inner codec and drop the outer — strictly worse
          than OCTET_STREAM+outer.
        """
        inner = MimeType.from_magic(head, default=None) if head else None
        if inner is None or inner.is_codec:
            return cls(mime_type=MimeTypes.OCTET_STREAM, codec=codec)
        return cls(mime_type=inner, codec=codec)

    # ------------------------------------------------------------------
    # Property accessors
    # ------------------------------------------------------------------

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

    def with_mime_type(
        self,
        mime_type: MimeType
    ) -> "MediaType":
        return MediaType(mime_type=mime_type, codec=self.codec)

    def with_codec(self, codec: Codec) -> "MediaType":
        return MediaType(mime_type=self.mime_type, codec=codec)

    def without_codec(self) -> "MediaType":
        return MediaType(mime_type=self.mime_type, codec=None)


class MediaTypes:
    OCTET_STREAM = MediaType(mime_type=MimeTypes.OCTET_STREAM, codec=None)
    PARQUET = MediaType(mime_type=MimeTypes.PARQUET, codec=None)
    JSON = MediaType(mime_type=MimeTypes.JSON, codec=None)
    ARROW_IPC = MediaType(mime_type=MimeTypes.ARROW_IPC, codec=None)
    ZIP_ENTRY = MediaType(mime_type=MimeTypes.ZIP_ENTRY, codec=None)
