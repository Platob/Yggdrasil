"""Parquet I/O for :class:`PrimitiveIO`.

:class:`ParquetIO` is the concrete leaf for Parquet files. The
format is footer-indexed — readers parse the metadata block at
the end of the file once and use it to plan column reads. Writes
buffer row groups and finalize the footer on close.

Reads benefit from caching the parsed metadata across calls;
writes can't (the writer must own the file for the duration of
the write). Lifecycle, codec, and Mode resolution all live on
:class:`DataIO`. This leaf owns the cached metadata and the
format-specific options.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.enums import MimeTypes, Mode
from yggdrasil.lazy_imports import pyarrow_dataset_module
from yggdrasil.io.buffer.bytes_io import BytesIO

if TYPE_CHECKING:
    import pyarrow.dataset as pds


__all__ = ["ParquetIO", "ParquetOptions"]


# ---------------------------------------------------------------------------
# ParquetOptions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ParquetOptions(CastOptions):
    """:class:`CastOptions` extended with Parquet-specific knobs."""

    compression: "str | None" = "snappy"  # snappy | gzip | brotli | zstd | lz4 | None
    compression_level: "int | None" = None
    use_dictionary: bool = True
    write_statistics: bool = True
    row_group_size: "int | None" = None
    use_threads: bool = True


# ---------------------------------------------------------------------------
# ParquetIO
# ---------------------------------------------------------------------------


class ParquetIO(BytesIO):
    """A :class:`BytesIO` for Parquet files."""

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_mime_type(cls):
        return MimeTypes.PARQUET

    @classmethod
    def options_class(cls):
        return ParquetOptions

    _NATIVE_SCANNER_OK: ClassVar[bool] = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parquet_metadata: "pq.FileMetaData | None" = None

    def _drop_metadata(self) -> None:
        self._parquet_metadata = None

    def _before_release(self) -> None:
        self._drop_metadata()
        super()._before_release()

    # ==================================================================
    # Schema — cheap via footer
    # ==================================================================

    def _collect_schema(self, options: ParquetOptions) -> Schema:
        """Read the schema from the Parquet footer.

        Footer probe — must not move the buffer's byte cursor (a
        caller mid-stream over the same buffer would lose its place).
        Routes through a :meth:`BytesIO.view`: the view has its own
        cursor, so :class:`pq.ParquetFile`'s seeks land on the view
        and leave ``self._pos`` untouched. Codec'd buffers fall back
        to :meth:`_reading_context`, which materializes a
        decompressed sibling and yields it (also isolated from
        ``self``).
        """
        if self.is_empty():
            return Schema.empty()

        if self.codec is not None:
            with self._reading_context(options) as io:
                source = io.arrow_io(mode="rb")
                pf = pq.ParquetFile(source)
                return Schema.from_arrow(pf.schema_arrow)

        with self.view(pos=0) as v:
            source = v.arrow_io(mode="rb")
            pf = pq.ParquetFile(source)
            return Schema.from_arrow(pf.schema_arrow)

    # ==================================================================
    # Read path
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: ParquetOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches from the Parquet file.

        ``options.predicate`` is pushed down to
        :mod:`pyarrow.dataset` when possible — that path uses the
        Parquet footer's per-row-group min/max stats to skip whole
        row groups, then evaluates the residual filter row-wise.
        Two failure modes fall back to the unfiltered ``iter_batches``
        path, where the universal predicate filter at the TabularIO
        layer takes over instead:

        - The predicate references a column the parquet file
          doesn't have (Arrow raises on filter binding).
        - ``predicate.to_arrow()`` doesn't know how to compile the
          expression (rare for typed comparisons; possible for
          exotic UDF-shaped nodes).
        """
        if self.remaining_bytes() == 0:
            return

        with self._reading_context(options) as io:
            source = io.arrow_io(mode="rb")
            source.seek(0)
            with pq.ParquetFile(source) as pf:
                read_options = options.check_source(pf.schema_arrow)
                columns = read_options.select_source_column_names()
                batch_size = read_options.row_size or 65536
                predicate = getattr(read_options, "predicate", None)

                if predicate is not None:
                    pushed = self._iter_with_pushdown(
                        source=source,
                        schema=pf.schema_arrow,
                        predicate=predicate,
                        columns=columns,
                        batch_size=batch_size,
                        read_options=read_options,
                    )
                    if pushed is not None:
                        yield from pushed
                        return

                # No predicate, or pushdown unavailable — fall back
                # to the plain reader; the universal filter at the
                # TabularIO layer applies the predicate per batch
                # if it's still on the options.
                for batch in pf.iter_batches(
                    batch_size=batch_size,
                    use_threads=read_options.use_threads,
                    columns=columns,
                ):
                    yield read_options.cast_arrow_tabular(batch)

    def _iter_with_pushdown(
        self,
        *,
        source: "pa.NativeFile",
        schema: "pa.Schema",
        predicate: "object",
        columns: "list[str] | None",
        batch_size: int,
        read_options: "ParquetOptions",
    ) -> "Iterator[pa.RecordBatch] | None":
        """Try the :mod:`pyarrow.dataset` path with predicate pushdown.

        Returns an iterator on success, ``None`` when pushdown isn't
        applicable (predicate references a missing column, fails to
        compile to a :class:`pa.compute.Expression`, or the dataset
        API isn't importable). The caller falls back to the
        unfiltered path; the universal predicate filter then applies.
        """
        try:
            arrow_expr = predicate.to_arrow()
        except Exception:
            return None

        try:
            from yggdrasil.data.expr.nodes import free_columns
            cols = free_columns(predicate)
        except Exception:
            cols = ()
        schema_names = set(schema.names)
        if cols and not set(cols).issubset(schema_names):
            # Universal "missing column → accept everything" rule —
            # let the fallback path emit the data unfiltered.
            return None

        try:
            pds = pyarrow_dataset_module()
        except Exception:
            return None

        try:
            scanner = pds.dataset(
                source,
                format="parquet",
                schema=schema,
            ).scanner(
                filter=arrow_expr,
                columns=columns,
                batch_size=batch_size,
                use_threads=read_options.use_threads,
            )
        except Exception:
            return None

        return self._wrap_pushdown_batches(scanner, read_options)

    @staticmethod
    def _wrap_pushdown_batches(
        scanner: "pds.Scanner",
        read_options: "ParquetOptions",
    ) -> Iterator[pa.RecordBatch]:
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            yield read_options.cast_arrow_tabular(batch)

    # ==================================================================
    # Write path
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: ParquetOptions,
    ) -> None:
        """Persist Arrow record batches as a Parquet file.

        Pre-reserves buffer capacity per batch using ``batch.nbytes`` as
        an upper bound on output size. This forces the spill decision
        upfront on sizable writes so the ParquetWriter sink is the
        fast :class:`pa.OSFile` path (real fd) rather than the Python
        shim, and avoids mid-write bytearray reallocation churn.

        ``batch.nbytes`` over-estimates final Parquet size after
        compression / dictionary encoding, which is the safe direction:
        we may spill slightly earlier than strictly necessary, but we
        never under-reserve.
        """
        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return
        if action is Mode.APPEND:
            self.seek(0)
            return self._arrow_append_via_rewrite(batches, options)
        if action is Mode.UPSERT:
            self.seek(0)
            return self._arrow_upsert_via_rewrite(batches, options)
        if action is not Mode.OVERWRITE:
            raise NotImplementedError(
                f"{type(self).__name__}._write_arrow_batches handles "
                f"OVERWRITE / APPEND / UPSERT; got resolved action "
                f"{action!r}."
            )

        iterator = iter(batches)
        first = next(iterator, None)
        if first is None:
            return

        options = (
            options
            .check_source(first)
            .check_target(first)
        )
        schema = options.merged_schema.to_arrow_schema()
        first = options.cast_arrow_tabular(first)

        self._drop_metadata()

        lifecycle = options.copy(truncate_before_write=True)

        # Cap reserve target at one byte past spill threshold. Past that
        # point the buffer will be local-spilled and further reserves
        # are no-ops; pre-growing the bytearray to gigabytes would just
        # delay the spill we already know is coming.
        spill_cap = self._spill_bytes + 1
        cumulative = first.nbytes
        self.reserve(min(cumulative, spill_cap))

        with self._writing_context(lifecycle) as io:
            sink = io.arrow_io(mode="wb")
            with contextlib.ExitStack() as stack:
                stack.callback(sink.close)
                writer = pq.ParquetWriter(
                    sink,
                    schema,
                    compression=options.compression,
                    compression_level=options.compression_level,
                    use_dictionary=options.use_dictionary,
                    write_statistics=options.write_statistics,
                )

                try:
                    if first.num_rows > 0:
                        writer.write_batch(first)

                    for batch in iterator:
                        casting = options.check_source(batch)
                        batch = casting.cast_arrow_tabular(batch)
                        if batch.num_rows == 0:
                            continue
                        cumulative += batch.nbytes
                        # Idempotent for n <= current capacity, so the
                        # cost past the first crossing is ~free.
                        self.reserve(min(cumulative, spill_cap))
                        writer.write_batch(batch)
                finally:
                    writer.close()
        return None

    # ==================================================================
    # Native engine overrides
    # ==================================================================

    def _can_use_native_scanner(self, options: ParquetOptions) -> bool:
        """True iff parquet's native readers can be invoked directly.

        Path-bound local: hand the path to the reader (fastest —
        memory-mapped reads, native filter pushdown). Otherwise
        (in-memory, non-local path) route through a
        :meth:`BytesIO.view`: the reader sees a self-contained
        file-like over the buffer's bytes and the parent cursor
        stays untouched. Codec'd buffers and casting/filtering
        options still fall back to the generic Arrow batch reader.
        """
        if not type(self)._NATIVE_SCANNER_OK:
            return False
        if self.is_empty():
            return False
        if options.target_field is not None:
            return False
        if self.codec is not None:
            return False
        return True

    def _read_arrow_dataset(self, options: ParquetOptions) -> "pds.Dataset":
        if not self._can_use_native_scanner(options):
            return super()._read_arrow_dataset(options)

        pds = pyarrow_dataset_module()
        if self.path is not None and self.path.is_local:
            return pds.dataset(self.path.__fspath__(), format="parquet")

        # In-memory or non-local: ``pds.dataset`` doesn't accept
        # file-likes, so parse the footer through a view and wrap
        # the resulting table. The view's cursor isolates the read
        # from any caller mid-stream over ``self``.
        with self.view(pos=0) as v:
            source = v.arrow_io(mode="rb")
            table = pq.read_table(source)
        return pds.dataset(table)