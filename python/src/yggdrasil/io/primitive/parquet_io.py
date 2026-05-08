"""Parquet Tabular leaf over the new :class:`BytesIO` substrate.

:class:`ParquetFile` is a :class:`BytesIO` subclass that auto-registers
under :data:`MimeTypes.PARQUET`. The Parquet file format is
footer-indexed: readers parse the metadata block at the end of the
file once and use it to plan column reads. Writes buffer row groups
and finalize the footer on close.

The reworked memory-management model lets us pass ``self`` (or a
:meth:`view`) directly to :class:`pyarrow.parquet.ParquetWriter` and
:class:`pyarrow.parquet.ParquetFile`, so reads and writes don't have
to bounce through a :class:`pyarrow.BufferReader` /
:class:`pyarrow.BufferOutputStream` adapter.

Native engine dispatch
----------------------

When the holder is a local path, :meth:`_read_arrow_dataset` /
:meth:`_scan_polars_frame` / :meth:`_read_polars_frame` short-circuit
to the format-aware scanners (``pds.dataset(format="parquet")``,
``pl.scan_parquet``, ``pl.read_parquet``) — those push projection
and predicate filters into the Parquet reader at plan time, which
the generic Arrow batch shim can't do.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.lazy_imports import polars_module, pyarrow_dataset_module
from yggdrasil.io.bytes_io import BytesIO

if TYPE_CHECKING:
    import polars as pl
    import pyarrow.dataset as pds


__all__ = ["LazyParquetFile", "ParquetFile", "ParquetOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class ParquetOptions(CastOptions):
    """:class:`CastOptions` extended with Parquet-specific knobs."""

    compression: "str | None" = "snappy"  # snappy | gzip | brotli | zstd | lz4 | None
    compression_level: "int | None" = None
    use_dictionary: bool = True
    write_statistics: bool = True
    row_group_size: "int | None" = None
    use_threads: bool = True


class ParquetFile(BytesIO):
    """:class:`Tabular` leaf for Apache Parquet."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.PARQUET

    @classmethod
    def options_class(cls):
        return ParquetOptions

    # ==================================================================
    # Helpers
    # ==================================================================

    def _local_path_str(self) -> "str | None":
        """Backend-native path string when the holder is a local path,
        else ``None``.

        The native pyarrow / polars Parquet readers are dramatically
        faster than the file-like fallback when given a real path
        (memory-mapped reads, native predicate pushdown). Returning
        ``None`` opts out and the caller routes through the
        file-like view path.
        """
        holder = self._holder
        if holder is None:
            return None
        if not getattr(holder, "is_local_path", False):
            return None
        full_path = getattr(holder, "full_path", None)
        if full_path is None:
            return None
        return full_path()

    # ==================================================================
    # Schema — cheap via footer
    # ==================================================================

    def _collect_schema(self, options: ParquetOptions) -> Schema:
        """Read the schema from the Parquet footer.

        Routes through :meth:`view` so the parent cursor isn't moved
        — :class:`pq.ParquetFile` seeks to the footer to parse it
        and would otherwise leave ``self._pos`` pointing at the file
        end.
        """
        if self.size == 0:
            return Schema.empty()
        with self._format_input() as v:
            return Schema.from_arrow(pq.ParquetFile(v).schema_arrow)

    # ==================================================================
    # Read path
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: ParquetOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches from the Parquet file.

        Empty buffer short-circuits to no batches. The reader runs
        against a :meth:`view` so the parent cursor stays put, and
        ``options.row_size`` (when set) becomes the Parquet
        ``batch_size`` — otherwise pyarrow's default of 65536 rows
        per batch.
        """
        if self.size == 0:
            return

        batch_size = int(options.row_size or 65536)
        with self._format_input() as v:
            with pq.ParquetFile(v) as pf:
                for batch in pf.iter_batches(
                    batch_size=batch_size,
                    use_threads=options.use_threads,
                ):
                    yield batch

    # ==================================================================
    # Write path
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: ParquetOptions,
    ) -> None:
        """Persist Arrow record batches as a Parquet file.

        Mode dispatch:

        - **OVERWRITE / AUTO / TRUNCATE** — single
          :class:`pq.ParquetWriter` session over the buffer
          (truncated to zero before the writer opens).
        - **APPEND / UPSERT / MERGE** — read existing batches, chain
          incoming, recurse with OVERWRITE. Parquet's footer covers
          every row group, so a partial append still requires a full
          rewrite.
        - **IGNORE** — skip when non-empty.
        - **ERROR_IF_EXISTS** — raise when non-empty.
        """
        action = self._resolve_action(options.mode)

        if action is Mode.IGNORE:
            if self.size > 0:
                return
            action = Mode.OVERWRITE
        elif action is Mode.ERROR_IF_EXISTS:
            if self.size > 0:
                raise FileExistsError(
                    f"{type(self).__name__} buffer is non-empty "
                    f"({self.size} bytes); refusing to overwrite under "
                    f"mode={options.mode!r}."
                )
            action = Mode.OVERWRITE

        iterator = iter(batches)
        first = next(iterator, None)
        if first is None and action is Mode.OVERWRITE:
            self.seek(0)
            self.truncate(0)
            return
        if first is None:
            return

        if action is Mode.APPEND and self.size > 0:
            existing = list(self._read_arrow_batches(options))
            chained = iter([*existing, first, *iterator])
            return self._write_arrow_batches(
                chained, dataclasses.replace(options, mode=Mode.OVERWRITE),
            )

        # OVERWRITE — encode into a pure-Arrow in-memory sink, then
        # bulk-commit to self. ``pa.BufferOutputStream`` keeps the
        # row-group flushes in one contiguous Arrow buffer (no
        # syscalls, no Python BytesIO indirection); the leaf hands
        # the finished buffer to :meth:`BytesIO._commit_format_payload`,
        # which applies any codec and lays the bytes down in one
        # truncate+write against the durable holder.
        schema = first.schema

        sink = pa.BufferOutputStream()
        with contextlib.ExitStack() as stack:
            writer = pq.ParquetWriter(
                sink,
                schema,
                compression=options.compression,
                compression_level=options.compression_level,
                use_dictionary=options.use_dictionary,
                write_statistics=options.write_statistics,
            )
            stack.callback(writer.close)

            if first.num_rows > 0:
                writer.write_batch(first, row_group_size=options.row_group_size)
            for batch in iterator:
                if batch.num_rows > 0:
                    writer.write_batch(batch, row_group_size=options.row_group_size)

        self._commit_format_payload(sink.getvalue())

    def _resolve_action(self, mode: Mode) -> Mode:
        """Pick the disposition for a write call.

        Parquet has no native append story (one footer per file),
        so :data:`Mode.AUTO` resolves to :data:`Mode.OVERWRITE` and
        :data:`Mode.UPSERT` / :data:`Mode.MERGE` degrade to
        :data:`Mode.APPEND` (which still triggers a full rewrite).
        """
        if mode is Mode.AUTO or mode is Mode.OVERWRITE or mode is Mode.TRUNCATE:
            return Mode.OVERWRITE
        if mode is Mode.APPEND:
            return Mode.APPEND
        if mode is Mode.IGNORE:
            return Mode.IGNORE
        if mode is Mode.ERROR_IF_EXISTS:
            return Mode.ERROR_IF_EXISTS
        if mode is Mode.UPSERT or mode is Mode.MERGE:
            return Mode.APPEND
        return Mode.OVERWRITE

    # ==================================================================
    # Native engine overrides — push reads to format-aware scanners
    # ==================================================================

    def _read_arrow_dataset(self, options: ParquetOptions) -> "pds.Dataset":
        """Native :class:`pyarrow.dataset.Dataset` over the Parquet bytes."""
        pds = pyarrow_dataset_module()
        path = self._local_path_str()
        if path is not None:
            return pds.dataset(path, format="parquet")

        # In-memory / non-local — pyarrow.dataset doesn't accept a
        # file-like, so read the table through a view and wrap it.
        if self.size == 0:
            return pds.dataset(
                pa.table({}),
            )
        with self._format_input() as v:
            table = pq.read_table(v)
        return pds.dataset(table)

    def _scan_polars_frame(self, options: ParquetOptions) -> "pl.LazyFrame":
        """Native :func:`polars.scan_parquet` LazyFrame.

        For a local-path holder, hand polars the path so the rust
        scanner does its own predicate pushdown. Otherwise feed it a
        :meth:`view` — polars pulls the footer + planning bytes
        through the file-like at scan time and the view's cursor
        keeps the parent cursor untouched.
        """
        pl = polars_module()
        path = self._local_path_str()
        if path is not None:
            return pl.scan_parquet(path)
        with self._format_input() as v:
            return pl.scan_parquet(v)

    def _read_polars_frame(self, options: ParquetOptions) -> "pl.DataFrame":
        """Native :func:`polars.read_parquet` eager frame."""
        pl = polars_module()
        path = self._local_path_str()
        if path is not None:
            return pl.read_parquet(path, use_pyarrow=False)
        with self._format_input() as v:
            return pl.read_parquet(v, use_pyarrow=False)


# ----------------------------------------------------------------------
# Lazy wrapper — pushes column projection + predicate filters straight
# into the parquet reader for plans that only contain those ops.
# ----------------------------------------------------------------------

from yggdrasil.io.tabular.lazy import LazyTabular  # noqa: E402  (circular-safe)


class LazyParquetFile(LazyTabular):
    """:class:`LazyTabular` over a :class:`ParquetFile` source.

    Plans containing only :class:`Select` (bare column names) and
    :class:`Filter` (yggdrasil predicates that lift to
    :class:`pyarrow.compute.Expression`) push column projection and
    the AND-merged filter directly into a
    :class:`pyarrow.dataset.Dataset` scanner — so unselected columns
    are never read off disk and row groups whose footer statistics
    rule out the predicate are skipped. For in-memory holders the
    scanner can't take a file-like, so column projection still
    fires via :meth:`pyarrow.parquet.ParquetFile.iter_batches` and
    a remaining filter falls through to the base polars route.

    Plans containing :class:`GroupByAgg` / :class:`Apply` /
    :class:`Join` (or selectors that aren't bare column names)
    inherit the base :class:`LazyTabular` polars-LazyFrame path,
    which still pushes predicate / projection through
    ``pl.scan_parquet`` at the format level.
    """

    source_cls: ClassVar["type"] = ParquetFile

    @property
    def source(self) -> "ParquetFile":
        return self._source  # type: ignore[return-value]

    def _read_arrow_batches(
        self, options: ParquetOptions,
    ) -> Iterator[pa.RecordBatch]:
        spec = self._arrow_pushdown_spec()
        if spec is None:
            yield from super()._read_arrow_batches(options)
            return

        columns, filt = spec
        src = self._source
        if src.size == 0:
            return

        batch_size = int(getattr(options, "row_size", None) or 65536)
        path = src._local_path_str()
        if path is not None:
            pds = pyarrow_dataset_module()
            scanner = pds.dataset(path, format="parquet").scanner(
                columns=columns,
                filter=filt,
                batch_size=batch_size,
                use_threads=options.use_threads,
            )
            for batch in scanner.to_batches():
                if batch.num_rows > 0:
                    yield options.cast_arrow_tabular(batch)
            return

        # In-memory holder: pq.ParquetFile.iter_batches takes
        # ``columns=`` but its ``filters=`` slot wants DNF tuples
        # rather than pyarrow Expressions. Push the projection
        # only; if a filter is present, defer to the base path so
        # polars' scan_parquet can do its own predicate pushdown.
        if filt is not None:
            yield from super()._read_arrow_batches(options)
            return
        with src._format_input() as v:
            with pq.ParquetFile(v) as pf:
                for batch in pf.iter_batches(
                    batch_size=batch_size,
                    columns=columns,
                    use_threads=options.use_threads,
                ):
                    if batch.num_rows > 0:
                        yield options.cast_arrow_tabular(batch)
