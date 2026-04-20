"""Parquet I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Implementation principles
-------------------------

* **Streaming read.** :meth:`_read_arrow_batches` uses
  :class:`pyarrow.parquet.ParquetFile` with ``iter_batches`` so only the
  requested columns and one batch at a time are decoded. A full
  materialization only happens when the caller explicitly asks for a
  table via the base-class ``read_arrow_table``.
* **Streaming write.** :meth:`_write_arrow_batches` uses
  :class:`pyarrow.parquet.ParquetWriter` with ``write_batch`` so input
  batches flow to disk without being concatenated into a single table.
* **Footer-only schema.** :meth:`_collect_arrow_schema` reads just the
  Parquet footer via ``ParquetFile.schema_arrow``; no row group is
  decoded.
* **Spilled-file fast path.** When the buffer is backed by a file on
  disk (``self.holder.spilled``), reads memory-map the underlying path
  instead of copying bytes through Arrow's buffered pipeline.
* **Save-mode handling.** ``OVERWRITE`` truncates, ``IGNORE`` /
  ``ERROR_IF_EXISTS`` go through the base-class guard, and
  ``APPEND`` / ``UPSERT`` stream the existing file through the writer
  before the new batches — so peak memory stays at
  ``max(old_row_group, new_row_group)`` instead of
  ``old_table + new_table``.

Transport-level compression (gzip/zstd/lz4 applied to the whole
Parquet byte stream via ``MediaType.codec``) is handled by the base
class. Parquet's *internal* column compression is a separate concern
exposed here as :attr:`ParquetOptions.compression`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.io.enums import SaveMode
from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["ParquetOptions", "ParquetIO"]


_DEFAULT_COMPRESSION = "snappy"
_DEFAULT_ROW_GROUP_SIZE = 1_000_000
# How many rows to move through the writer per iter_batches chunk when
# rewriting an existing file for APPEND / UPSERT. This is the decode
# batch size, not the Parquet row group size.
_REWRITE_BATCH_SIZE = 64_000


@dataclass
class ParquetOptions(MediaOptions):
    """Options for Parquet I/O.

    Parameters
    ----------
    compression:
        Parquet-internal column compression codec. One of
        ``{"snappy", "gzip", "brotli", "lz4", "zstd", "none"}``. Defaults
        to ``"snappy"`` (fast, small, widely supported).
    compression_level:
        Optional compression level forwarded to the codec. ``None`` uses
        the codec default.
    row_group_size:
        Target rows per Parquet row group on write. Larger groups mean
        better compression but worse selective-read parallelism. Default
        1,000,000.
    use_dictionary:
        Enable dictionary encoding on string / binary columns. Default
        ``True``. Can be set per column by passing a list of column
        names.
    write_statistics:
        Whether to write per-row-group column statistics. Default
        ``True`` — required for predicate pushdown on read.
    """

    compression: str = _DEFAULT_COMPRESSION
    compression_level: int | None = None
    row_group_size: int = _DEFAULT_ROW_GROUP_SIZE
    use_dictionary: bool | Sequence[str] = True
    write_statistics: bool = True

    def __post_init__(self) -> None:
        """Normalize and validate Parquet-specific options.

        Compression is validated by the base :class:`MediaOptions`; the
        allowed codec set lives there. Here we only handle
        Parquet-specific fields.
        """
        super().__post_init__()

        if self.compression_level is not None and not isinstance(
            self.compression_level, int
        ):
            raise TypeError(
                "compression_level must be int|None, "
                f"got {type(self.compression_level).__name__}"
            )

        if not isinstance(self.row_group_size, int) or self.row_group_size <= 0:
            raise ValueError("row_group_size must be a positive int")

        if not isinstance(self.write_statistics, bool):
            raise TypeError(
                "write_statistics must be bool, "
                f"got {type(self.write_statistics).__name__}"
            )

        if isinstance(self.use_dictionary, bool):
            pass
        elif isinstance(self.use_dictionary, (list, tuple)):
            if not all(isinstance(name, str) for name in self.use_dictionary):
                raise TypeError("use_dictionary column list must contain only str")
            self.use_dictionary = tuple(self.use_dictionary)
        else:
            raise TypeError(
                "use_dictionary must be bool or Sequence[str], "
                f"got {type(self.use_dictionary).__name__}"
            )

    @classmethod
    def resolve(
        cls,
        *,
        options: "ParquetOptions | None" = None,
        **overrides: Any,
    ) -> "ParquetOptions":
        """Merge *overrides* into *options* (or a fresh default)."""
        return cls.check_parameters(options=options, **overrides)


@dataclass(slots=True)
class ParquetIO(MediaIO[ParquetOptions]):
    """Parquet I/O backed by :mod:`pyarrow.parquet`."""

    @classmethod
    def check_options(
        cls,
        options: Optional[ParquetOptions],
        *args,
        **kwargs,
    ) -> ParquetOptions:
        """Validate and merge caller-supplied options."""
        return ParquetOptions.check_parameters(options=options, **kwargs)

    # ------------------------------------------------------------------
    # Source selection: prefer memory-mapping a spilled file
    # ------------------------------------------------------------------

    def _open_parquet_file(
        self,
        options: ParquetOptions,
        *,
        allow_mmap: bool = True,
    ) -> tuple[pq.ParquetFile, Any]:
        """Open a :class:`pq.ParquetFile` on the current buffer.

        Returns ``(parquet_file, source)`` where *source* is the
        underlying pyarrow I/O object (``MemoryMappedFile`` or
        ``NativeFile``). Callers MUST close *source* when done — on
        Windows, failing to release a memory-mapped region against a
        spilled file prevents reopening that file for writing (see
        ``ERROR_USER_MAPPED_FILE`` / winerror 1224).

        When the underlying holder has been spilled to disk AND
        *allow_mmap* is ``True``, the source is a :func:`pa.memory_map`
        so pyarrow can read row groups zero-copy. Otherwise it is the
        buffer's Arrow I/O wrapper (``pa.OSFile`` / ``pa.BufferReader``).

        Parameters
        ----------
        allow_mmap:
            Pass ``False`` when the caller is going to reopen the same
            path for writing. Even after closing the ``ParquetFile`` and
            the source, pyarrow / the OS may not release the memory
            mapping fast enough for Windows to allow a subsequent
            ``pa.OSFile(path, "w")`` in the same process — causing
            ``WinError 1224``. The OSFile / BufferReader paths have no
            such issue.
        """
        holder = self.holder
        path = getattr(holder, "path", None)
        spilled = getattr(holder, "spilled", False)

        if allow_mmap and spilled and path is not None and self.codec is None:
            source = pa.memory_map(str(path), "r")
        else:
            source = self.buffer.to_arrow_io("r")

        try:
            pf = pq.ParquetFile(source, pre_buffer=True, buffer_size=0)
        except Exception:
            self._close_quietly(source)
            raise
        return pf, source

    # ------------------------------------------------------------------
    # Core read/write protocol
    # ------------------------------------------------------------------

    def _read_arrow_batches(
        self,
        options: ParquetOptions,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """Stream record batches from the Parquet buffer.

        Only the projected columns are decoded, one batch at a time.
        """
        with self.open() as b:
            if b.buffer.size <= 0:
                return

            pf, source = self._open_parquet_file(options)
            try:
                yield from self._iter_batches_from(pf, options)
            finally:
                self._close_quietly(pf)
                self._close_quietly(source)

    def _iter_batches_from(
        self,
        pf: pq.ParquetFile,
        options: ParquetOptions,
    ) -> Iterator["pa.RecordBatch"]:
        """Apply column projection, batch sizing, empty-batch filtering, cast."""
        iter_kwargs: dict[str, Any] = {
            "use_threads": bool(getattr(options, "use_threads", True)),
        }
        if options.columns is not None:
            # Filter to columns that actually exist in the footer — pyarrow
            # raises KeyError on unknown columns, which is usually too
            # strict for a framework wrapper.
            schema_names = set(pf.schema_arrow.names)
            iter_kwargs["columns"] = [c for c in options.columns if c in schema_names]

        batch_size = getattr(options, "batch_size", 0) or 0
        if batch_size > 0:
            iter_kwargs["batch_size"] = batch_size

        batches = pf.iter_batches(**iter_kwargs)

        if options.ignore_empty:
            batches = (batch for batch in batches if batch.num_rows > 0)

        yield from options.cast.cast_iterator(batches)

    def _collect_arrow_schema(self, full: bool = False) -> "pyarrow.Schema":
        """Return the Parquet schema from the footer — no row-group decode."""
        del full

        with self.open() as b:
            if b.buffer.size <= 0:
                return pa.schema([])

            options = self.check_options(options=None)
            pf, source = self._open_parquet_file(options)
            try:
                schema = pf.schema_arrow
                if options.columns is not None:
                    schema = pa.schema(
                        [schema.field(name) for name in options.columns if name in schema.names]
                    )
                return schema
            finally:
                self._close_quietly(pf)
                self._close_quietly(source)

    def _write_arrow_batches(
        self,
        batches: Iterator["pyarrow.RecordBatch"],
        options: ParquetOptions,
    ) -> None:
        """Stream record batches into a Parquet file inside the buffer.

        ``OVERWRITE`` truncates first, ``APPEND`` / ``UPSERT`` stream the
        existing file through the same writer before the new batches, so
        peak memory stays bounded at roughly one row group.
        """
        with self.open() as b:
            mode = options.mode
            # --- Save-mode dispatch --------------------------------------
            # skip_write handles IGNORE (returns True) and ERROR_IF_EXISTS
            # (raises IOError) for non-empty buffers.
            if self.skip_write(mode):
                return

            # For APPEND/UPSERT we must read the existing file AND release
            # all its handles (including any pa.memory_map from the read
            # fast path) BEFORE opening the buffer for write. On Windows,
            # pa.OSFile(path, "w") fails with ERROR_USER_MAPPED_FILE
            # (winerror 1224) if any memory-mapped region is still alive
            # against the same path. So we eagerly materialize old batches
            # into memory, then close both the reader and its source.
            # We also pass allow_mmap=False to skip pa.memory_map entirely
            # — on Windows, releasing the mapping fast enough for a
            # subsequent write is unreliable even after explicit close.
            old_batches: list["pa.RecordBatch"] = []
            if b.buffer.size > 0 and mode in (SaveMode.APPEND, SaveMode.UPSERT):
                existing_pf, existing_src = self._open_parquet_file(
                    options, allow_mmap=False
                )
                try:
                    old_batches = list(
                        existing_pf.iter_batches(
                            batch_size=_REWRITE_BATCH_SIZE,
                            use_threads=bool(getattr(options, "use_threads", True)),
                        )
                    )
                finally:
                    self._close_quietly(existing_pf)
                    self._close_quietly(existing_src)
                    # Drop local refs and force a GC cycle so any
                    # lingering pyarrow native handles against the path
                    # are released before pa.OSFile tries to reopen it
                    # for writing.
                    del existing_pf
                    del existing_src
                    import gc
                    gc.collect()
            # OVERWRITE / TRUNCATE / AUTO fall through.

            # Route the new stream through the write-side cast. This is
            # the canonical write entry point for options.cast — it
            # applies target-schema enforcement batch by batch without
            # materializing the whole stream. The ParquetWriter schema
            # is the cast target, not the input source.
            cast_batches = options.cast.cast_iterator(batches)

            # Peek once (post-cast) to resolve the schema for ParquetWriter.
            peeked, target_schema = _peek_schema(cast_batches)
            if len(target_schema) == 0:
                # Empty new stream. Nothing to write regardless of mode.
                # Don't open the writer (would fail on empty schema) and
                # don't touch the buffer.
                return

            # UPSERT needs an in-memory old-table lookup because anti-join
            # can't be streamed without a hash index. We still avoid
            # materializing the *new* side as a table — it's kept as a
            # list of already-cast batches, consumed once for key
            # extraction and then replayed through the writer.
            if mode == SaveMode.UPSERT and old_batches:
                match_by = _normalize_match_by(options.match_by)
                if not match_by:
                    raise ValueError(
                        "SaveMode.UPSERT requires options.match_by to be set"
                    )
                filtered_old_iter, new_stream = self._stream_upsert(
                    old_iter=iter(old_batches),
                    new_iter=peeked,
                    match_by=match_by,
                    target_schema=target_schema,
                )
                combined = self._chain(filtered_old_iter, new_stream)
            elif mode == SaveMode.APPEND and old_batches:
                combined = self._chain(iter(old_batches), peeked)
            else:
                # OVERWRITE (or empty buffer, or empty old file): truncate
                # first so a stale footer can't confuse a future reader if
                # the write fails halfway through.
                b.buffer.truncate(0)
                combined = peeked

            # --- Stream through the writer ------------------------------
            arrow_io = b.buffer.to_arrow_io("w")
            writer: pq.ParquetWriter | None = None
            try:
                writer = pq.ParquetWriter(
                    arrow_io,
                    schema=target_schema,
                    compression=options.compression,
                    compression_level=options.compression_level,
                    use_dictionary=options.use_dictionary,
                    write_statistics=options.write_statistics,
                )
                row_group_size = options.row_group_size
                pending_rows = 0
                pending: list[pa.RecordBatch] = []

                for batch in combined:
                    if batch.num_rows == 0:
                        continue
                    pending.append(batch)
                    pending_rows += batch.num_rows
                    if pending_rows >= row_group_size:
                        # row_group_size=row_group_size tells pyarrow to
                        # emit one row group per row_group_size rows — the
                        # default behavior coalesces small writes into a
                        # single row group regardless of how many
                        # write_table() calls we make.
                        writer.write_table(
                            pa.Table.from_batches(pending, schema=target_schema),
                            row_group_size=row_group_size,
                        )
                        pending.clear()
                        pending_rows = 0

                if pending:
                    writer.write_table(
                        pa.Table.from_batches(pending, schema=target_schema),
                        row_group_size=row_group_size,
                    )

                # writer.close() flushes the footer; must complete before
                # we mark the buffer dirty.
                writer.close()
                writer = None
                b.mark_dirty()
            finally:
                if writer is not None:
                    # Error path: close writer so file handle is freed,
                    # but do NOT mark_dirty — the footer may be missing.
                    self._close_quietly(writer)
                arrow_io.close()

    # ------------------------------------------------------------------
    # UPSERT helper
    # ------------------------------------------------------------------

    def _stream_upsert(
        self,
        *,
        old_iter: Iterator["pa.RecordBatch"],
        new_iter: Iterator["pa.RecordBatch"],
        match_by: tuple[str, ...],
        target_schema: pa.Schema,
    ) -> tuple[Iterator["pa.RecordBatch"], Iterator["pa.RecordBatch"]]:
        """Return ``(filtered_old, new)`` iterators for an UPSERT write.

        The new iterator is consumed once to extract match-keys into a
        Python set, then replayed from an in-memory batch list. The old
        iterator streams — rows whose match-keys appear on the new side
        are filtered out batch by batch, so peak memory is ``O(new_rows
        × len(match_by))`` plus one batch from each side at a time.

        This is the cheapest honest upsert without hash-join support.
        For UPSERT workloads on very wide new-side data, consider
        partitioned writes instead.
        """
        # Materialize the new side once so we can both hash its keys and
        # replay its rows. We don't build a Table — a list of batches is
        # enough and avoids the concat cost.
        new_batches: list[pa.RecordBatch] = list(new_iter)

        # Extract match-keys into a Python set of tuples. Projecting
        # first keeps the intermediate allocation bounded to the key
        # columns only, not the whole new-side payload.
        key_indices = [target_schema.get_field_index(c) for c in match_by]
        if any(idx < 0 for idx in key_indices):
            missing = [c for c, idx in zip(match_by, key_indices) if idx < 0]
            raise KeyError(
                f"match_by columns not found in target schema: {missing}"
            )

        new_keys: set[tuple] = set()
        for batch in new_batches:
            cols = [batch.column(idx).to_pylist() for idx in key_indices]
            for i in range(batch.num_rows):
                new_keys.add(tuple(col[i] for col in cols))

        def filtered_old() -> Iterator[pa.RecordBatch]:
            if not new_keys:
                yield from old_iter
                return
            for old_batch in old_iter:
                if old_batch.num_rows == 0:
                    continue
                old_cols = [old_batch.column(c).to_pylist() for c in match_by]
                mask = [
                    tuple(col[i] for col in old_cols) not in new_keys
                    for i in range(old_batch.num_rows)
                ]
                if all(mask):
                    yield old_batch
                elif any(mask):
                    yield old_batch.filter(pa.array(mask, type=pa.bool_()))
                # else: every row collided with the new side; drop.

        def replay_new() -> Iterator[pa.RecordBatch]:
            # Drain the cached list so the batches can be reclaimed as
            # they're written out.
            while new_batches:
                yield new_batches.pop(0)

        return filtered_old(), replay_new()

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _chain(
        *iters: Iterator["pa.RecordBatch"],
    ) -> Iterator["pa.RecordBatch"]:
        for it in iters:
            yield from it

    @staticmethod
    def _close_quietly(obj: Any) -> None:
        close = getattr(obj, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:
            pass


def _normalize_match_by(match_by: Any) -> tuple[str, ...]:
    """Return *match_by* as a tuple of column names, or ``()`` if unset."""
    if match_by is None or match_by is ...:
        return ()
    if isinstance(match_by, str):
        return (match_by,)
    return tuple(match_by)


def _peek_schema(
    batches: Iterator["pa.RecordBatch"],
) -> tuple[Iterator["pa.RecordBatch"], pa.Schema]:
    """Pull one batch from *batches* to learn its schema, then replay.

    Returns ``(iterator, schema)`` where *iterator* yields the peeked
    batch first, followed by the remainder. If the input is empty an
    empty schema is returned.
    """
    import itertools

    iterator = iter(batches)
    try:
        first = next(iterator)
    except StopIteration:
        return iter(()), pa.schema([])

    return itertools.chain((first,), iterator), first.schema