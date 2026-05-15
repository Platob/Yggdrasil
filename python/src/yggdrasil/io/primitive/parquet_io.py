"""Parquet Tabular leaf over the new :class:`BytesIO` substrate.

:class:`ParquetIO` is a :class:`BytesIO` subclass that auto-registers
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

import dataclasses
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.arrow.ops import upsert_arrow_batches
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.lazy_imports import polars_module, pyarrow_dataset_module
from yggdrasil.io.base import IO

if TYPE_CHECKING:
    import polars as pl
    import pyarrow.dataset as pds


__all__ = ["ParquetIO", "ParquetOptions"]


#: Modes that read existing bytes, merge with the incoming stream,
#: and rewrite the file in one shot. Parquet's footer covers every
#: row group, so APPEND, UPSERT and MERGE all share the same
#: read-modify-rewrite shape — only the per-row dedup strategy
#: differs.
_MERGE_MODES = frozenset({Mode.APPEND, Mode.UPSERT, Mode.MERGE})


@dataclasses.dataclass(frozen=True, slots=True)
class ParquetOptions(CastOptions):
    """:class:`CastOptions` extended with Parquet-specific knobs."""

    compression: "str | None" = "snappy"  # snappy | gzip | brotli | zstd | lz4 | None
    compression_level: "int | None" = None
    use_dictionary: bool = True
    write_statistics: bool = True
    row_group_size: "int | None" = None
    use_threads: bool = True


class ParquetIO(IO[bytes, ParquetOptions]):
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

    @staticmethod
    def _projection_columns(
        options: ParquetOptions,
        available_names: "Iterable[str] | None",
    ) -> "list[str] | None":
        """Source-side column subset to push into the Parquet reader.

        Returns ``None`` (read every column) when no target is bound,
        when *available_names* is missing, or when the target asks for
        every column already on disk. Otherwise the intersection of
        target column names and *available_names*, preserving target
        order — the reader emits columns in the order requested, so the
        downstream :meth:`CastOptions.cast_arrow_tabular` lands on a
        batch shaped close to the merged schema and skips re-ordering.
        Columns the target asks for but the file doesn't carry are
        silently dropped here; the cast will fill them with nulls.
        """
        target = options.target
        if target is None or available_names is None:
            return None
        avail = available_names if isinstance(available_names, frozenset) else frozenset(available_names)
        selected = [n for n in target.names if n in avail]
        if not selected or len(selected) == len(avail):
            return None
        return selected

    # ==================================================================
    # Schema — cheap via footer
    # ==================================================================

    def _collect_schema(self, options: ParquetOptions) -> Schema:
        """Read the schema from the Parquet footer.

        Routes through :meth:`view` so the parent cursor isn't moved
        — :class:`pq.ParquetFile` seeks to the footer to parse it
        and would otherwise leave ``self._pos`` pointing at the file
        end. When the holder reports a known size (in-memory, or a
        path with a warm stat cache) we short-circuit on empty; for
        a cold remote path we attempt the read and translate
        FileNotFound / "File too small" errors into an empty schema
        rather than paying for an extra ``HeadObject`` /
        ``get_metadata`` round trip up front.
        """
        if options.target:
            return options.target

        if self.size_known and self.size == 0:
            return Schema.empty()
        try:
            with self.arrow_input_stream() as v:
                schema = Schema.from_arrow(pq.ParquetFile(v).schema_arrow)
                self._persist_schema(schema)
                return schema
        except (FileNotFoundError, pa.ArrowInvalid):
            return Schema.empty()

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

        When ``options.target`` is bound, the target column names are
        pushed into :meth:`pq.ParquetFile.iter_batches` as a
        ``columns=`` projection so the reader only decodes columns
        the caller actually wants — the cast on the way out still
        fills missing target columns from nulls. Each batch then
        funnels through :meth:`CastOptions.cast_arrow_tabular` so a
        bound ``target_field`` reshapes rows to the caller's schema.
        When no target is bound the cast is a passthrough.
        """
        if self.size_known and self.size == 0:
            return

        batch_size = int(options.row_size or 65536)
        try:
            stream_ctx = self.arrow_input_stream()
            stream = stream_ctx.__enter__()
        except FileNotFoundError:
            return
        try:
            try:
                pf = pq.ParquetFile(stream)
            except pa.ArrowInvalid:
                # Empty or truncated payload — same end state as the
                # ``size == 0`` short-circuit on the in-memory path.
                return
            with pf:
                columns = self._projection_columns(options, pf.schema_arrow.names)
                for batch in pf.iter_batches(
                    batch_size=batch_size,
                    use_threads=options.use_threads,
                    columns=columns,
                ):
                    yield options.cast_arrow_tabular(batch)
        finally:
            stream_ctx.__exit__(None, None, None)

    def _read_arrow_table(self, options: ParquetOptions) -> pa.Table:
        """Read the whole Parquet file as a :class:`pa.Table`.

        Overrides the base class's ``iter_batches`` →
        :func:`pa.Table.from_batches` shape with a single
        :meth:`pq.ParquetFile.read` call. The C++ reader fans out
        column decoding across :attr:`options.use_threads` worker
        threads in one shot — meaningfully faster on multi-column
        tables than streaming batch-by-batch and re-stitching on the
        Python side. ``options.target`` still drives the projection
        pushdown via :meth:`_projection_columns`.
        """
        if self.size_known and self.size == 0:
            return super()._read_arrow_table(options)

        try:
            with self.arrow_input_stream() as v:
                with pq.ParquetFile(v) as pf:
                    columns = self._projection_columns(options, pf.schema_arrow.names)
                    table = pf.read(
                        columns=columns,
                        use_threads=options.use_threads,
                    )
        except (FileNotFoundError, pa.ArrowInvalid):
            return super()._read_arrow_table(options)
        return options.cast_arrow_tabular(table)

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
        - **APPEND** — read existing batches, merge with incoming,
          recurse with OVERWRITE. With ``options.match_by_keys`` set
          incoming rows whose key tuple already exists are dropped
          (existing wins); without keys the streams are concatenated.
        - **UPSERT / MERGE** — same read-modify-rewrite, but with
          ``match_by`` set existing rows whose key matches the
          incoming stream are dropped (incoming wins). Without keys
          this collapses to plain APPEND — Parquet has no row-level
          identity at this layer.
        - **IGNORE** — skip when non-empty.
        - **ERROR_IF_EXISTS** — raise when non-empty.

        Key-aware merges are powered by
        :func:`yggdrasil.arrow.ops.upsert_arrow_batches`.
        """
        # Mode resolution. AUTO picks UPSERT when ``match_by``
        # is set (incoming wins on key conflict) or APPEND otherwise
        # — Parquet has no in-place append, so both end up doing the
        # same read-modify-rewrite, but the semantics line up with
        # what the caller was asking for. TRUNCATE collapses to
        # OVERWRITE; APPEND / UPSERT / MERGE keep their identity for
        # the merge branch below; IGNORE / ERROR_IF_EXISTS guard the
        # buffer.
        mode = options.mode
        if mode is Mode.AUTO:
            action = Mode.UPSERT if options.match_by_keys else Mode.APPEND
        elif mode is Mode.TRUNCATE:
            action = Mode.OVERWRITE
        elif mode in _MERGE_MODES:
            if self.size == 0:
                action = Mode.OVERWRITE
            else:
                action = mode
        elif mode in (
            Mode.IGNORE, Mode.ERROR_IF_EXISTS, Mode.OVERWRITE,
        ):
            action = mode
        else:
            action = Mode.OVERWRITE

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

        if action in _MERGE_MODES and self.size > 0:
            rewrite_options = options.with_target(self.collect_schema(options))
            existing = list(self._read_arrow_batches(rewrite_options))
            incoming: Iterator[pa.RecordBatch] = rewrite_options.cast_arrow_batch_iterator(iter([first, *iterator]))
            merged = upsert_arrow_batches(
                iter(existing),
                incoming,
                options.match_by_keys,
                Mode.APPEND if action is Mode.APPEND else Mode.UPSERT,
                memory_pool=options.arrow_memory_pool,
            )
            return self._write_arrow_batches(
                merged,
                rewrite_options.copy(
                    mode=Mode.OVERWRITE,
                    # remove already applied since cast_arrow_batch_iterator does it
                    target=None,
                    row_size=None,
                    byte_size=None
                ),
            )

        # OVERWRITE — drive the writer against the IO's
        # :meth:`arrow_output_stream`, which yields a
        # :class:`pa.BufferOutputStream` and bulk-commits the encoded
        # bytes (with codec compression when the holder's MediaType
        # carries one) on context exit.
        #
        # Bind the first batch's schema as the source so
        # :attr:`CastOptions.merged_schema` resolves to the writer
        # schema even when no target_field is set; each batch is
        # cast through :meth:`cast_arrow_tabular` so a bound target
        # reshapes the rows to the caller's schema before the
        # encoder sees them.
        write_options = options.check_source(first.schema)
        first_casted = write_options.cast_arrow_tabular(first)
        schema = write_options.merged.to_arrow_schema()

        # If the first batch came back as the same object the cast was a
        # full bypass (source schema already matched the target). The
        # batches downstream share the same source schema, so the cast
        # will keep bypassing — skip the per-batch dispatch entirely.
        bypass = first_casted is first

        with self.arrow_output_stream() as sink:
            with pq.ParquetWriter(
                sink,
                schema,
                compression=options.compression,
                compression_level=options.compression_level,
                use_dictionary=options.use_dictionary,
                write_statistics=options.write_statistics,
            ) as writer:
                if first_casted.num_rows > 0:
                    writer.write_batch(first_casted, row_group_size=options.row_group_size)
                if bypass:
                    for batch in iterator:
                        if batch.num_rows > 0:
                            writer.write_batch(batch, row_group_size=options.row_group_size)
                else:
                    for batch in iterator:
                        casted = write_options.cast_arrow_tabular(batch)
                        if casted.num_rows > 0:
                            writer.write_batch(casted, row_group_size=options.row_group_size)

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
        with self.arrow_input_stream() as v:
            table = pq.read_table(v)
        return pds.dataset(table)

    def _scan_polars_frame(self, options: ParquetOptions) -> "pl.LazyFrame":
        """Native :func:`polars.scan_parquet` LazyFrame.

        For a local-path holder, hand polars the path so the rust
        scanner does its own predicate pushdown. Otherwise feed it a
        :meth:`view` — polars pulls the footer + planning bytes
        through the file-like at scan time and the view's cursor
        keeps the parent cursor untouched.

        Bound ``options.target`` is applied via
        :meth:`CastOptions.cast_polars_tabular` so the lazy frame is
        already shaped to the caller's schema before ``.collect()`` —
        polars merges the cast into its plan, so projection pushdown
        still fires against the parquet file.
        """
        pl = polars_module()
        path = self._local_path_str()
        if path is not None:
            lf = pl.scan_parquet(path)
        else:
            with self.arrow_input_stream() as v:
                lf = pl.scan_parquet(v)
        return options.cast_polars_tabular(lf)

    def _read_polars_frame(self, options: ParquetOptions) -> "pl.DataFrame":
        """Native :func:`polars.read_parquet` eager frame.

        When ``options.target`` is bound, target column names are
        pushed to polars as a ``columns=`` projection (intersected
        with the parquet footer's column names so columns the target
        adds — which the file doesn't carry — don't trip polars'
        "column not in file" guard). The eager frame is then handed
        to :meth:`CastOptions.cast_polars_tabular` for the value-level
        coercion / fill.
        """
        pl = polars_module()
        path = self._local_path_str()
        columns: "list[str] | None" = None
        if options.target is not None:
            # Peek the footer once for the column-name intersection; the
            # parquet metadata read is the same cheap call ``collect_schema``
            # makes and dominates nothing in the read path.
            with self.arrow_input_stream() as v:
                schema_names = pq.ParquetFile(v).schema_arrow.names
            columns = self._projection_columns(options, schema_names)
        if path is not None:
            df = pl.read_parquet(path, columns=columns, use_pyarrow=False)
        else:
            with self.arrow_input_stream() as v:
                df = pl.read_parquet(v, columns=columns, use_pyarrow=False)
        return options.cast_polars_tabular(df)
