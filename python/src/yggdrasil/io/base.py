"""Format-buffer / arrow-stream context managers and IO helpers.

After the Holder ↔ IO merge the canonical :class:`IO` class lives in
:mod:`yggdrasil.io.holder` (storage + cursor + tabular handle in one
class). This module survives as the home for two things:

1. **Module-level helpers** used by the IO class and by callers
   poking at the temp-file spill convention: :func:`_mint_spill_path`,
   :func:`_as_byte_mv`, :func:`_local_path_for_handle`.
2. **Codec / pyarrow streaming context managers** returned by
   :meth:`IO._format_buffer` / :meth:`IO._format_input` /
   :meth:`IO.arrow_input_stream` / :meth:`IO.arrow_output_stream`.
   They live here rather than on the IO class so the codec /
   memory-map machinery can grow without bloating the class body.

:class:`IO` is re-exported from this module for backwards
compatibility — every prior ``from yggdrasil.io.base import IO``
call site keeps working unchanged.
"""

from __future__ import annotations

import os
import pathlib
import tempfile
import time
from typing import Any, Optional, TypeVar, Union

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.io.holder import IO

__all__ = ["IO", "BytesLike", "T", "O"]


T = TypeVar("T")
O = TypeVar("O", bound=CastOptions)

BytesLike = Union[bytes, bytearray, memoryview]


def _mint_spill_path(ext: str, ttl_seconds: int) -> pathlib.Path:
    """Mint a fresh temp file path under :func:`tempfile.gettempdir`.

    Filename layout (time-sortable):
    ``tmp-{start}-{end}-{seed}.{ext}``. Both timestamps are zero-
    padded to 12 digits so a lexical sort of the temp directory
    yields chronological order — useful for debugging and the
    cross-process janitor that reaps orphans oldest-first. The file
    itself is not created here — the caller writes to it.
    """
    seed = os.urandom(8).hex()
    start = int(time.time())
    end = start + max(0, int(ttl_seconds))
    name = f"tmp-{start:012d}-{end:012d}-{seed}.{ext}"
    return pathlib.Path(tempfile.gettempdir()) / name


def _as_byte_mv(data: BytesLike) -> memoryview:
    """Normalize bytes-like input to a 1-D unsigned-byte memoryview.

    pyarrow ``Buffer`` arrives as ``format='b'`` with itemsize 1,
    which trips bytearray slice assignment unless we cast. Centralize
    the cast/contiguity dance so every write path stays consistent.
    """
    mv = memoryview(data)
    if mv.format != "B" or mv.ndim != 1 or mv.itemsize != 1:
        mv = mv.cast("B")
    if not mv.c_contiguous:
        mv = memoryview(bytes(mv))
    return mv


def _local_path_for_handle(obj: Any) -> Optional[str]:
    """Return the on-disk path for a local file handle, or ``None``.

    Recognises real file handles (``open("...", "rb")``,
    ``pathlib.Path.open()``) by their string ``.name`` attribute —
    stdlib file objects expose the underlying path there. Filters
    out anonymous streams whose ``.name`` is an int fd (sockets,
    pipes) or a bracketed sentinel (``"<stdin>"``, ``"<fdopen>"``)
    and anything whose ``.name`` doesn't actually exist on disk.

    Used by :meth:`IO.from_` to scrap the drain-into-Memory step
    for live local files — the resulting :class:`LocalPath`
    reads from the file system on demand, so a multi-GB handle
    never gets materialised.
    """
    import os as _os

    name = getattr(obj, "name", None)
    if not isinstance(name, str):
        return None
    if name.startswith("<") and name.endswith(">"):
        return None
    try:
        if not _os.path.isfile(name):
            return None
    except (OSError, ValueError):
        return None
    return name



# ===========================================================================
# Codec writer / reader context managers
# ===========================================================================


class _FormatBufferContext:
    """Writer-side of :meth:`IO._format_buffer`.

    Caller does ``with bio._format_buffer() as buf: writer(buf)``;
    the yielded ``buf`` accepts raw format bytes. On exit:

    * No codec → ``buf is bio``; we just leave the bytes in place.
      (We pre-truncate so the writer sees an empty target.)
    * Codec set → ``buf`` is a fresh in-memory IO; on exit the bytes
      are compressed and committed to ``bio``.
    """

    def __init__(self, parent: "IO") -> None:
        self._parent = parent
        self._buf: "IO | None" = None
        self._codec = parent._codec()

    def __enter__(self) -> "IO":
        if self._codec is None:
            # Direct write path: pre-truncate so the leaf writer
            # opens onto an empty target.
            self._parent.seek(0)
            self._parent.truncate(0)
            self._buf = self._parent
            return self._parent
        # Codec path: scratch buffer; we compress on exit.
        self._buf = type(self._parent)()
        return self._buf

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._buf is None or exc_type is not None:
            return
        if self._codec is None:
            return
        # Compress scratch into the durable buffer.
        compressed = self._codec.compress(self._buf)
        try:
            payload = compressed.to_bytes()
        finally:
            try:
                compressed.close()
            except Exception:
                pass
        try:
            self._buf.close()
        except Exception:
            pass
        self._parent.seek(0)
        self._parent.truncate(0)
        self._parent.write(payload)


class _FormatInputContext:
    """Reader-side companion to :class:`_FormatBufferContext`.

    Resolves the cheapest pyarrow-friendly input source for the
    formatted bytes:

    - Local-path holder with no codec → :func:`pyarrow.memory_map`.
      The file lands in the kernel page cache once and every reader
      (Parquet, Arrow IPC, CSV, NDJSON) walks it without a copy.
    - Anything else → :meth:`IO._format_view` (a non-owning view of
      ``self`` when uncompressed, a decompressed in-memory IO when a
      codec is bound).

    The object yielded by ``__enter__`` is closed on ``__exit__`` —
    callers don't have to track which branch fired.
    """

    def __init__(self, parent: "IO") -> None:
        self._parent = parent
        self._mm: "Any | None" = None
        self._view: "IO | None" = None

    def __enter__(self) -> "Any":
        if self._parent._codec() is None:
            holder = self._parent._parent
            if holder is not None and getattr(holder, "is_local_path", False):
                full_path = getattr(holder, "full_path", None)
                if callable(full_path):
                    try:
                        self._mm = pa.memory_map(full_path(), "r")
                        return self._mm
                    except Exception:
                        # Fall through to the view; mmap failures
                        # (race with delete, fs that doesn't support
                        # mmap, sandbox restrictions) shouldn't break
                        # the read.
                        self._mm = None
        self._view = self._parent._format_view()
        return self._view

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
        if self._view is not None:
            try:
                self._view.close()
            except Exception:
                pass
            self._view = None


class _ArrowInputStreamContext:
    """Reader-side companion for :meth:`IO.arrow_input_stream`.

    Yields a real :class:`pa.NativeFile` over the buffer's payload,
    transparently decompressing first when the holder's media type
    carries a codec. Resolution:

    - Local-path holder + no codec → :func:`pyarrow.memory_map`
      (zero-copy :class:`pa.MemoryMappedFile`). The kernel pages the
      file in once and every reader walks it without a Python copy.
    - Codec-tagged holder → :meth:`Codec.decompress` into a scratch
      in-memory IO; the uncompressed bytes are then handed to a
      :class:`pa.BufferReader`.
    - Anything else → snapshot :meth:`IO.to_bytes` and wrap in a
      :class:`pa.BufferReader`.

    The stream and any scratch decompression buffer are closed on
    ``__exit__``.
    """

    def __init__(self, parent: "IO") -> None:
        self._parent = parent
        self._stream: "pa.NativeFile | None" = None
        self._scratch: "IO | None" = None
        self._mv: "memoryview | None" = None

    def __enter__(self) -> "pa.NativeFile":
        parent = self._parent
        codec = parent._codec()

        if codec is None:
            holder = parent._parent
            if holder is not None and getattr(holder, "is_local_path", False):
                full_path = getattr(holder, "full_path", None)
                if callable(full_path):
                    try:
                        self._stream = pa.memory_map(full_path(), "r")
                        return self._stream
                    except Exception:
                        # mmap can fail on sandboxed filesystems or if
                        # the file was deleted under us; fall back to
                        # the bytes snapshot path rather than escalate.
                        self._stream = None
            # Zero-copy snapshot: wrap the buffer's memoryview in a
            # pyarrow Buffer instead of ``to_bytes()``, which would copy
            # the whole payload into an intermediate ``bytes`` that
            # pyarrow then re-reads. ``read_mv(-1, 0)`` is a view into the
            # backing bytearray (and pyarrow keeps it alive for the
            # reader's lifetime), so a full Parquet / Arrow read goes
            # straight from the existing buffer into Arrow with no
            # intermediate full-object copy. Held on ``self`` so the
            # view outlives this method.
            self._mv = parent.read_mv(-1, 0)
            self._stream = pa.BufferReader(pa.py_buffer(self._mv))
            return self._stream

        # Codec path — decompress through the codec's streaming
        # roundtrip, then expose the uncompressed bytes as a NativeFile.
        self._scratch = codec.decompress(parent)
        self._stream = pa.BufferReader(self._scratch.to_bytes())
        return self._stream

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._scratch is not None:
            try:
                self._scratch.close()
            except Exception:
                pass
            self._scratch = None
        # Release the zero-copy view after the reader is closed.
        self._mv = None


class _ArrowOutputStreamContext:
    """Writer-side companion for :meth:`IO.arrow_output_stream`.

    Yields a :class:`pa.NativeFile` so format encoders
    (:class:`pa.ipc.RecordBatchFileWriter`,
    :func:`pa.parquet.ParquetWriter`, :func:`pa.csv.CSVWriter`, …) can
    stream directly into the cheapest available sink:

    - **Local-path holder, no codec** → :func:`pyarrow.OSFile` opened
      against the holder's filesystem path. The encoder writes pages
      straight to disk; no Python-side copy through an in-memory
      buffer. ``append=False`` truncates first; ``append=True`` opens
      with ``"ab"``.
    - **Anything else** (in-memory holder, remote path, codec-tagged
      buffer) → :class:`pa.BufferOutputStream`. On clean exit the
      accumulated bytes are bulk-committed to the parent IO through
      :meth:`IO._commit_format_payload`, which handles codec
      compression and the overwrite-vs-append disposition.

    On exception the sink is closed and the parent is left untouched
    — the caller's prior payload is not overwritten by a half-written
    encoder run.
    """

    def __init__(self, parent: "IO", *, append: bool = False) -> None:
        self._parent = parent
        self._append = bool(append)
        self._sink: "pa.NativeFile | None" = None
        # ``True`` when the sink writes straight to a local file —
        # nothing left to commit on exit; ``False`` when the sink is
        # an in-memory buffer that still needs a commit.
        self._direct: bool = False
        # Set to a temp path when the sink spills the encode to disk for a
        # holder that streams its upload (e.g. a Databricks VolumePath): the
        # encoded payload never lives whole in memory — it's written to this
        # file and then streamed to the backend on exit.
        self._stream_spill: "pathlib.Path | None" = None

    def __enter__(self) -> "pa.NativeFile":
        parent = self._parent
        if parent._codec() is None:
            holder = parent._parent
            if holder is not None and getattr(holder, "is_local_path", False):
                full_path = getattr(holder, "full_path", None)
                if callable(full_path):
                    try:
                        path_str = full_path()
                        # ``OSFile`` doesn't accept ``"ab"`` directly;
                        # we open in ``"wb"`` and seek to EOF for
                        # append, matching :meth:`_commit_format_payload`'s
                        # disposition.
                        if self._append:
                            self._sink = pa.OSFile(path_str, "ab")
                        else:
                            self._sink = pa.OSFile(path_str, "wb")
                        self._direct = True
                        # Local writes bypass the Python-side cursor, so
                        # invalidate the holder's cached stat — the next
                        # ``size`` / ``mtime`` probe re-reads from disk.
                        invalidate = getattr(
                            holder, "invalidate_singleton", None,
                        )
                        if callable(invalidate):
                            invalidate()
                        return self._sink
                    except Exception:
                        # Fall through to the in-memory path; OSFile
                        # failures (sandbox, missing parent dir, mode
                        # mismatch) shouldn't break the write.
                        self._sink = None
                        self._direct = False
            # Remote holder that streams its upload (VolumePath): spill the
            # encode to a temp file so the payload never materialises whole in
            # memory, then stream it to the backend on exit. Gated on an
            # explicit opt-in so every other remote backend (S3 via boto,
            # Workspace via the SDK) keeps the unchanged in-memory commit.
            if holder is not None and getattr(holder, "SUPPORTS_STREAMING_UPLOAD", False):
                try:
                    spill = _mint_spill_path("arrowup", 3600)
                    self._sink = pa.OSFile(str(spill), "wb")
                    self._stream_spill = spill
                    self._direct = False
                    return self._sink
                except Exception:
                    self._sink = None
                    self._stream_spill = None
                    self._direct = False
        self._sink = pa.BufferOutputStream()
        self._direct = False
        return self._sink

    def __exit__(self, exc_type, exc, tb) -> None:
        sink = self._sink
        self._sink = None
        direct = self._direct
        self._direct = False
        spill = self._stream_spill
        self._stream_spill = None
        if sink is None:
            return
        if exc_type is not None:
            try:
                sink.close()
            except Exception:
                pass
            if spill is not None:
                spill.unlink(missing_ok=True)
            return
        if spill is not None:
            # Encode landed on disk — stream the spill file to the backend
            # (the holder's streaming upload reads it in bounded chunks), then
            # drop the temp file. Memory peaks at one chunk, not the payload.
            try:
                sink.close()
            except Exception:
                pass
            try:
                from yggdrasil.path.local_path import LocalPath

                self._parent._commit_format_source(
                    LocalPath.from_(str(spill)), append=self._append,
                )
            finally:
                spill.unlink(missing_ok=True)
            return
        if direct:
            # Bytes already on disk — just close the file handle and
            # invalidate the cached stat one more time so the next
            # reader sees the post-write size.
            try:
                sink.close()
            except Exception:
                pass
            holder = self._parent._parent
            if holder is not None:
                invalidate = getattr(holder, "invalidate_singleton", None)
                if callable(invalidate):
                    invalidate()
            return
        try:
            payload = sink.getvalue()
        finally:
            try:
                sink.close()
            except Exception:
                pass
        self._parent._commit_format_payload(payload, append=self._append)
