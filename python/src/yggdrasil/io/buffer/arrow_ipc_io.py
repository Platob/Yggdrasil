"""Arrow IPC (Feather v2) I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Layouts
-------
* **File** (default) — schema header, body batches, footer at tail.
  Random access; cannot be byte-extended. APPEND/UPSERT require
  read-old-then-rewrite.
* **Stream** — schema header, body batches, EOS marker. Sequential
  access; APPEND can strip the old EOS and concatenate as long as the
  schema matches.

The reader auto-detects the layout (file first, stream fallback). The
writer picks based on :attr:`IPCOptions.layout`.

Transport-level compression (``MediaType.codec``) is handled by the
base class. Intra-file *body* compression (snappy/zstd/lz4 inside the
IPC file) is controlled by :attr:`IPCOptions.ipc_compression` —
renamed from ``compression`` to avoid colliding with the base
:class:`MediaOptions.compression` field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional

import pyarrow as pa
import pyarrow.ipc as ipc

from yggdrasil.io.enums import SaveMode
from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["IPCIO", "IPCOptions"]


_VALID_LAYOUTS = frozenset({"file", "stream"})
_VALID_BODY_CODECS = frozenset({"lz4", "lz4_frame", "zstd", None})


# ---------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------

@dataclass
class IPCOptions(MediaOptions):
    """Options for Arrow IPC I/O.

    Parameters
    ----------
    ipc_compression:
        Intra-file body compression codec used by
        :class:`pyarrow.ipc.IpcWriteOptions`. One of ``"lz4"``,
        ``"lz4_frame"``, ``"zstd"``, or ``None`` (no body compression).
        Default ``"zstd"``.

        Named ``ipc_compression`` (not ``compression``) to keep it
        distinct from the base :class:`MediaOptions.compression` field,
        which is the transport-level codec applied around the whole
        IPC stream.
    layout:
        On write: ``"file"`` (default) for the random-access file
        layout, or ``"stream"`` for the sequential stream layout.
        Ignored on read — the reader auto-detects.
    """

    ipc_compression: str | None = "zstd"
    layout: str = "file"

    def __post_init__(self) -> None:
        """Normalize and validate IPC-specific options."""
        super().__post_init__()

        if self.ipc_compression is not None:
            if not isinstance(self.ipc_compression, str):
                raise TypeError(
                    f"ipc_compression must be str|None, "
                    f"got {type(self.ipc_compression).__name__}"
                )
            lowered = self.ipc_compression.lower()
            if lowered not in {c for c in _VALID_BODY_CODECS if c is not None}:
                raise ValueError(
                    f"ipc_compression must be one of "
                    f"{sorted(c for c in _VALID_BODY_CODECS if c)} or None, "
                    f"got {self.ipc_compression!r}"
                )
            self.ipc_compression = lowered

        if not isinstance(self.layout, str):
            raise TypeError(f"layout must be str, got {type(self.layout).__name__}")
        self.layout = self.layout.lower()
        if self.layout not in _VALID_LAYOUTS:
            raise ValueError(
                f"layout must be one of {sorted(_VALID_LAYOUTS)}, got {self.layout!r}"
            )

    @classmethod
    def resolve(cls, *, options: "IPCOptions | None" = None, **overrides: Any) -> "IPCOptions":
        return cls.check_parameters(options=options, **overrides)


# ---------------------------------------------------------------------
# IPCIO
# ---------------------------------------------------------------------

@dataclass(slots=True)
class IPCIO(MediaIO[IPCOptions]):
    """Arrow IPC I/O backed by :mod:`pyarrow.ipc`."""

    @classmethod
    def check_options(
        cls,
        options: Optional[IPCOptions],
        *args,
        **kwargs,
    ) -> IPCOptions:
        return IPCOptions.check_parameters(options=options, **kwargs)

    # ------------------------------------------------------------------
    # Reader helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _open_ipc_reader(arrow_io: Any) -> tuple[Any, str]:
        """Open *arrow_io* as an IPC reader. Returns ``(reader, layout)``.

        Tries the file layout first (which has a magic marker), falls
        back to the stream layout on :class:`pa.ArrowInvalid` or
        :class:`pa.ArrowIOError`. Callers must rewind *arrow_io* via
        ``seek(0)`` between attempts — we do it here rather than
        expecting the caller to remember.
        """
        try:
            reader = ipc.open_file(arrow_io)
            return reader, "file"
        except (pa.ArrowInvalid, pa.ArrowIOError):
            arrow_io.seek(0)
            reader = ipc.open_stream(arrow_io)
            return reader, "stream"

    @staticmethod
    def _iter_reader_batches(reader: Any, layout: str) -> Iterator["pa.RecordBatch"]:
        """Yield batches from a file-layout or stream-layout reader."""
        if layout == "file":
            for i in range(reader.num_record_batches):
                yield reader.get_batch(i)
        else:
            # Stream layout iterates directly.
            for batch in reader:
                yield batch

    # ------------------------------------------------------------------
    # Core read protocol
    # ------------------------------------------------------------------

    def _read_arrow_batches(
        self,
        options: IPCOptions,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """Yield record batches from the IPC buffer.

        Auto-detects file vs stream layout. Applies column projection,
        batch_size rechunking, ignore_empty filtering, and the cast
        iterator in that order.
        """
        with self.open() as b:
            if b.buffer.size <= 0:
                return

            arrow_io = b.buffer.to_arrow_io("r")
            try:
                reader, layout = self._open_ipc_reader(arrow_io)
                try:
                    batches: Iterator[pa.RecordBatch] = self._iter_reader_batches(
                        reader, layout
                    )

                    if options.columns is not None:
                        batches = (
                            batch.select(options.columns) for batch in batches
                        )

                    batch_size = getattr(options, "batch_size", 0) or 0
                    if batch_size > 0:
                        batches = self._rechunk(batches, batch_size)

                    if options.ignore_empty:
                        batches = (batch for batch in batches if batch.num_rows > 0)

                    yield from options.cast.cast_iterator(batches)
                finally:
                    # ipc readers don't have a uniform close() API —
                    # file reader does, stream reader doesn't on older
                    # pyarrow. Best-effort.
                    close = getattr(reader, "close", None)
                    if close is not None:
                        try:
                            close()
                        except Exception:
                            pass
            finally:
                arrow_io.close()

    @staticmethod
    def _rechunk(
        batches: Iterator["pa.RecordBatch"],
        batch_size: int,
    ) -> Iterator["pa.RecordBatch"]:
        """Rechunk a batch iterator into chunks of at most *batch_size* rows.

        Small input batches get coalesced; large ones get split. Used
        when the caller wants output rows grouped differently from
        whatever the on-disk IPC writer chose.
        """
        pending: list[pa.RecordBatch] = []
        pending_rows = 0

        for batch in batches:
            if batch.num_rows == 0:
                continue
            pending.append(batch)
            pending_rows += batch.num_rows

            if pending_rows >= batch_size:
                combined = pa.Table.from_batches(pending)
                # Emit exact-sized slices; the tail (< batch_size) stays
                # in pending for the next iteration.
                offset = 0
                while combined.num_rows - offset >= batch_size:
                    slab = combined.slice(offset, batch_size)
                    yield from slab.to_batches(max_chunksize=batch_size)
                    offset += batch_size

                tail = combined.slice(offset)
                pending = list(tail.to_batches()) if tail.num_rows > 0 else []
                pending_rows = sum(b.num_rows for b in pending)

        if pending:
            combined = pa.Table.from_batches(pending)
            yield from combined.to_batches(max_chunksize=batch_size)

    # ------------------------------------------------------------------
    # Schema (header-only)
    # ------------------------------------------------------------------

    def _collect_arrow_schema(self, full: bool = False) -> "pyarrow.Schema":
        """Return the IPC schema from the file/stream header only."""
        del full

        with self.open() as b:
            if b.buffer.size <= 0:
                return pa.schema([])

            arrow_io = b.buffer.to_arrow_io("r")
            try:
                reader, _ = self._open_ipc_reader(arrow_io)
                try:
                    return reader.schema
                finally:
                    close = getattr(reader, "close", None)
                    if close is not None:
                        try:
                            close()
                        except Exception:
                            pass
            finally:
                arrow_io.close()

    # ------------------------------------------------------------------
    # Core write protocol
    # ------------------------------------------------------------------

    def _write_arrow_batches(
        self,
        batches: Iterator["pyarrow.RecordBatch"],
        options: IPCOptions,
    ) -> None:
        """Write record batches as IPC (file or stream layout).

        Save-mode dispatch:

        * OVERWRITE / AUTO — truncate, write fresh.
        * IGNORE / ERROR_IF_EXISTS — base-class guard.
        * APPEND / UPSERT — read existing batches, merge, rewrite.
          Stream layout could in principle be byte-concatenated after
          stripping the old EOS, but the APIs to do that cleanly aren't
          stable across pyarrow versions; read-then-rewrite is uniform
          and correct for both layouts.
        """
        with self.open() as b:
            if self.skip_write(options.mode):
                return

            # --- Route the new stream through the write-side cast ------
            # options.cast.cast_iterator wraps an iterator of batches and
            # applies the cast to each batch as it flows through. This is
            # the canonical streaming write entry point — symmetric with
            # how the read path uses it. Use this instead of
            # cast_arrow_tabular(batches) which cannot accept raw
            # generators (it tries Field.from_any(generator)).
            cast_batches = options.cast.cast_iterator(batches)

            # Peek once (post-cast) to resolve the schema for the IPC
            # writer. The writer's schema is the *target* (post-cast)
            # schema, not the source.
            peeked, target_schema = _peek_schema(cast_batches)
            if len(target_schema) == 0:
                # Empty new stream, nothing to write regardless of mode.
                # Don't open the writer (would fail on empty schema) and
                # don't touch the buffer.
                return

            # --- Merge with existing buffer on APPEND/UPSERT ------------
            existing_batches: list[pa.RecordBatch] = []
            if (
                options.mode in (SaveMode.APPEND, SaveMode.UPSERT)
                and b.buffer.size > 0
            ):
                existing_batches = list(self._read_existing_batches(b, options))

            # UPSERT on IPC: we don't have a natural row-identity concept
            # here without match_by. Support the same semantic as other
            # formats — anti-join on match_by — when it's set.
            if options.mode == SaveMode.UPSERT and existing_batches:
                match_by = _normalize_match_by(options.match_by)
                if not match_by:
                    raise ValueError(
                        "SaveMode.UPSERT requires options.match_by to be set"
                    )
                new_list = list(peeked)
                filtered_old = self._anti_join(
                    old_batches=existing_batches,
                    new_batches=new_list,
                    match_by=match_by,
                )
                combined: Iterator[pa.RecordBatch] = self._chain(
                    iter(filtered_old), iter(new_list)
                )
            elif options.mode == SaveMode.APPEND and existing_batches:
                combined = self._chain(iter(existing_batches), peeked)
            else:
                # OVERWRITE / AUTO / empty buffer.
                b.buffer.truncate(0)
                combined = peeked

            # --- Stream through the IPC writer --------------------------
            arrow_io = b.buffer.to_arrow_io("w")
            writer = None
            try:
                write_options = ipc.IpcWriteOptions(
                    compression=options.ipc_compression,
                    use_legacy_format=False,
                    use_threads=bool(getattr(options, "use_threads", True)),
                    allow_64bit=True,
                )

                if options.layout == "file":
                    writer = ipc.new_file(
                        arrow_io, target_schema, options=write_options
                    )
                else:
                    writer = ipc.new_stream(
                        arrow_io, target_schema, options=write_options
                    )

                for batch in combined:
                    if batch.num_rows == 0:
                        continue
                    writer.write_batch(batch)

                writer.close()
                writer = None
                b.mark_dirty()
            finally:
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                arrow_io.close()

    # ------------------------------------------------------------------
    # APPEND/UPSERT helpers
    # ------------------------------------------------------------------

    def _read_existing_batches(
        self,
        b: "MediaIO",
        options: IPCOptions,
    ) -> Iterator["pa.RecordBatch"]:
        """Read the existing IPC buffer into an iterator of batches.

        Used only on APPEND/UPSERT. Each batch is serialized through
        :func:`pa.ipc.serialize_record_batch` / deserialized into a
        fresh buffer, detaching it from any memory-mapped view of the
        source file. This matters on Windows where a lingering
        file-mapping against the buffer path blocks the subsequent
        write handle (``ERROR_USER_MAPPED_FILE`` / ``WinError 1224``).

        The serialize/deserialize trip costs one extra copy per row,
        but APPEND/UPSERT already pays a full rewrite cost so the
        marginal overhead is negligible.
        """
        import gc

        arrow_io = b.buffer.to_arrow_io("r")
        try:
            reader, layout = self._open_ipc_reader(arrow_io)
            try:
                for batch in self._iter_reader_batches(reader, layout):
                    # Force-copy away from any mmap view.
                    serialized = batch.serialize()
                    detached = pa.ipc.read_record_batch(serialized, batch.schema)
                    yield detached
            finally:
                close = getattr(reader, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass
        finally:
            arrow_io.close()
            # Encourage prompt mmap teardown on Windows.
            gc.collect()

    @staticmethod
    def _anti_join(
        *,
        old_batches: list["pa.RecordBatch"],
        new_batches: list["pa.RecordBatch"],
        match_by: tuple[str, ...],
    ) -> list["pa.RecordBatch"]:
        """Drop rows from *old_batches* whose match_by keys appear in new.

        Cheap Python-set implementation. For UPSERT loads where *new*
        is small, this is the right complexity; the big cost is always
        the rewrite of *old*.
        """
        if not old_batches:
            return []
        if not new_batches:
            return list(old_batches)

        # Verify columns exist on both sides.
        new_schema = new_batches[0].schema
        missing_new = [k for k in match_by if k not in new_schema.names]
        if missing_new:
            raise KeyError(
                f"match_by columns not found in new batches: {missing_new}"
            )

        old_schema = old_batches[0].schema
        missing_old = [k for k in match_by if k not in old_schema.names]
        if missing_old:
            # Nothing to match on; nothing to drop.
            return list(old_batches)

        new_keys: set[tuple] = set()
        for batch in new_batches:
            key_cols = [batch.column(k).to_pylist() for k in match_by]
            for i in range(batch.num_rows):
                new_keys.add(tuple(col[i] for col in key_cols))

        if not new_keys:
            return list(old_batches)

        out: list[pa.RecordBatch] = []
        for old_batch in old_batches:
            if old_batch.num_rows == 0:
                continue
            old_key_cols = [old_batch.column(k).to_pylist() for k in match_by]
            mask = [
                tuple(col[i] for col in old_key_cols) not in new_keys
                for i in range(old_batch.num_rows)
            ]
            if all(mask):
                out.append(old_batch)
            elif any(mask):
                out.append(old_batch.filter(pa.array(mask, type=pa.bool_())))
            # else: every row collides — drop.
        return out

    @staticmethod
    def _chain(*iters: Iterator["pa.RecordBatch"]) -> Iterator["pa.RecordBatch"]:
        for it in iters:
            yield from it


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _normalize_match_by(match_by: Any) -> tuple[str, ...]:
    """Return *match_by* as a tuple of column names, or ``()`` if unset."""
    if match_by is None or match_by is ...:
        return ()
    if isinstance(match_by, str):
        return (match_by,)
    return tuple(match_by)


def _peek_schema(
    batches: Iterator["pa.RecordBatch"],
) -> tuple[Iterator["pa.RecordBatch"], "pa.Schema"]:
    """Peek the first batch from *batches* to learn its schema.

    Returns ``(reconstructed_iter, schema)`` where
    ``reconstructed_iter`` yields every original batch (including the
    peeked one). If the iterator is empty, returns an empty iterator
    and ``pa.schema([])``.
    """
    try:
        first = next(batches)
    except StopIteration:
        return iter(()), pa.schema([])

    schema = first.schema

    def _chain():
        yield first
        yield from batches

    return _chain(), schema