"""Parquet Tabular leaf over the new :class:`IO` substrate.

:class:`ParquetFile` is an :class:`IO` subclass that auto-registers
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
from yggdrasil.enums import MimeTypes, Mode
from yggdrasil.lazy_imports import polars_module, pyarrow_dataset_module
from yggdrasil.io.base import IO

if TYPE_CHECKING:
    import polars as pl
    import pyarrow.dataset as pds


#: Parquet footers with large metadata (many row groups / columns, big column
#: statistics) blow past pyarrow's default thrift deserialization limits and
#: fail to even open — "Couldn't deserialize thrift: TProtocolException:
#: Exceeded size limit". Open with generous limits so wide / heavily
#: row-grouped files read instead of erroring on the footer.
_THRIFT_LIMITS = {
    "thrift_string_size_limit": 1 << 30,      # 1 GiB strings (stats / schema)
    "thrift_container_size_limit": 1 << 28,   # 256M elements (row groups × cols)
}


__all__ = ["ParquetFile", "ParquetOptions"]


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


class ParquetFile(IO[bytes, ParquetOptions]):
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
        holder = self._parent
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
        wanted = options.read_columns()
        if wanted is None or available_names is None:
            return None
        avail = available_names if isinstance(available_names, frozenset) else frozenset(available_names)
        selected = [n for n in wanted if n in avail]
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
        if self.size_known and self.size == 0:
            return Schema.empty()
        try:
            with self.arrow_input_stream() as v:
                schema = Schema.from_arrow(pq.ParquetFile(v, **_THRIFT_LIMITS).schema_arrow)
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
        # Column projection over a backend that can range-read (e.g.
        # VolumePath / S3) seeks to the footer + just the projected
        # column chunks — fetch only those bytes instead of snapshotting
        # the whole object. A plain full read keeps the single-GET
        # snapshot (one big read beats many ranged round trips).
        holder = getattr(self, "_parent", None)
        ranged = options.target is not None and getattr(holder, "SUPPORTS_RANGED_RANDOM_ACCESS", False)
        try:
            stream_ctx = holder.arrow_random_access_file() if ranged else self.arrow_input_stream()
            stream = stream_ctx.__enter__()
        except FileNotFoundError:
            return
        try:
            try:
                pf = pq.ParquetFile(stream, pre_buffer=ranged, **_THRIFT_LIMITS)
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
                    yield options.cast_arrow_batch(batch)
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
        pushdown via :meth:`_projection_columns`; ``options.row_limit``
        is applied after the read so the slice still wins.
        """
        if self.size_known and self.size == 0:
            return super()._read_arrow_table(options)

        # Range-read the footer + projected column chunks when a target
        # projection is bound and the backend supports random access;
        # otherwise snapshot the whole object in one GET. See
        # :meth:`_read_arrow_batches` for the rationale.
        holder = getattr(self, "_parent", None)
        ranged = options.target is not None and getattr(holder, "SUPPORTS_RANGED_RANDOM_ACCESS", False)
        try:
            ctx = holder.arrow_random_access_file() if ranged else self.arrow_input_stream()
            with ctx as v:
                with pq.ParquetFile(v, pre_buffer=ranged, **_THRIFT_LIMITS) as pf:
                    columns = self._projection_columns(options, pf.schema_arrow.names)
                    table = pf.read(
                        columns=columns,
                        use_threads=options.use_threads,
                    )
        except (FileNotFoundError, pa.ArrowInvalid):
            return super()._read_arrow_table(options)
        table = options.cast_arrow_table(table)
        table = options.apply_post_read_table(table)
        if options.row_limit is not None and table.num_rows > options.row_limit:
            table = table.slice(0, options.row_limit)
        return table

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
        # Mode resolution. AUTO picks UPSERT when ``match_by`` is set
        # (incoming wins on key conflict) and otherwise defaults to
        # OVERWRITE — a bare write replaces the file, matching the JSON /
        # Excel / Zip leaves (a single file is "replace" by default, not
        # "grow"). TRUNCATE collapses to OVERWRITE; APPEND / UPSERT / MERGE
        # keep their identity for the merge branch below; IGNORE /
        # ERROR_IF_EXISTS guard the buffer.
        mode = options.mode
        _skip_existing = self.holder_is_overwrite
        if mode is Mode.AUTO:
            action = Mode.UPSERT if options.match_by_keys else Mode.OVERWRITE
        elif mode is Mode.TRUNCATE:
            action = Mode.OVERWRITE
        elif mode in _MERGE_MODES:
            if _skip_existing or not self.size_known or self.size == 0:
                action = Mode.OVERWRITE
            else:
                action = mode
        elif mode in (
            Mode.IGNORE, Mode.ERROR_IF_EXISTS, Mode.OVERWRITE,
        ):
            action = mode
        else:
            action = Mode.OVERWRITE

        _has_existing = not _skip_existing and self.size_known and self.size > 0
        if action is Mode.IGNORE:
            if _has_existing:
                return
            action = Mode.OVERWRITE
        elif action is Mode.ERROR_IF_EXISTS:
            if _has_existing:
                raise FileExistsError(
                    f"{type(self).__name__} buffer is non-empty "
                    f"({self.size} bytes); refusing to overwrite under "
                    f"mode={options.mode!r}."
                )
            action = Mode.OVERWRITE

        iterator = iter(batches)
        first = next(iterator, None)
        if first is None and action is Mode.OVERWRITE:
            # No incoming rows: ``to_batches()`` on a 0-row table yields an
            # empty iterator, dropping the schema. Replay a 0-row batch
            # carrying the bound schema through the writer below so the file
            # is a valid empty Parquet (footer + schema) rather than a
            # truncated 0-byte stub a reader can't parse.
            first = self._empty_overwrite_batch(options)
            if first is None:
                self.seek(0)
                self.truncate(0)
                return
        elif first is None:
            return

        if action in _MERGE_MODES and _has_existing:
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
        first_casted = write_options.cast_arrow_batch(first)
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
                        casted = write_options.cast_arrow_batch(batch)
                        if casted.num_rows > 0:
                            writer.write_batch(casted, row_group_size=options.row_group_size)

    def _write_arrow_table(self, table: pa.Table, options: ParquetOptions) -> None:
        """Write *table* directly via ``pq.write_table``.

        Bypasses the base ``_write_arrow_table`` →
        ``table.to_batches()`` → ``_write_arrow_batches`` round trip
        when the effective action is "replace the buffer wholesale"
        — ``pq.write_table`` lays out row groups internally with no
        per-batch Python dispatch, so a 1M-row table costs one
        C-level call instead of N ``writer.write_batch`` hops.

        The fast path runs when *any*:

        - The mode is explicitly ``OVERWRITE`` / ``TRUNCATE``, OR
        - The mode is ``AUTO`` with no ``match_by`` — a bare write
          replaces the buffer (see ``_write_arrow_batches``), OR
        - The buffer is empty (an OVERWRITE wins regardless of the
          caller's nominal mode — ``APPEND`` to empty *is* an
          ``OVERWRITE``, and ``IGNORE`` / ``ERROR_IF_EXISTS`` on
          empty also reduce to plain write).

        Every other shape (merge against existing rows, guarded
        ``IGNORE`` / ``ERROR_IF_EXISTS`` on non-empty) falls through
        to ``_write_arrow_batches`` where the read-modify-rewrite /
        size-check + raise / skip logic already lives.
        """
        mode = options.mode
        has_existing = (
            not self.holder_is_overwrite
            and self.size_known
            and self.size > 0
        )
        truly_overwrite = (
            mode in (Mode.OVERWRITE, Mode.TRUNCATE)
            or (mode is Mode.AUTO and not options.match_by_keys)
            or not has_existing
        )
        if not truly_overwrite:
            return self._write_arrow_batches(iter(table.to_batches()), options)

        casted = options.cast_arrow_table(table)
        write_options = options.check_source(casted.schema)
        schema = write_options.merged.to_arrow_schema()
        if casted.schema != schema:
            casted = casted.cast(schema)
        with self.arrow_output_stream() as sink:
            with pq.ParquetWriter(
                sink,
                schema,
                compression=options.compression,
                compression_level=options.compression_level,
                use_dictionary=options.use_dictionary,
                write_statistics=options.write_statistics,
            ) as writer:
                writer.write_table(casted, row_group_size=options.row_group_size)

    # Pandas index round-trip (tag each index level on write, set_index
    # on read) lives on the base :class:`Tabular` — Parquet and Arrow
    # IPC (and every other tabular leaf) share one implementation.

    # ==================================================================
    # Native engine overrides — push reads to format-aware scanners
    # ==================================================================

    def _read_arrow_dataset(self, options: ParquetOptions) -> "pds.SparkDataset":
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
            table = pq.read_table(v, **_THRIFT_LIMITS)
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
                schema_names = pq.ParquetFile(v, **_THRIFT_LIMITS).schema_arrow.names
            columns = self._projection_columns(options, schema_names)
        if path is not None:
            df = pl.read_parquet(path, columns=columns, use_pyarrow=False)
        else:
            with self.arrow_input_stream() as v:
                df = pl.read_parquet(v, columns=columns, use_pyarrow=False)
        return options.cast_polars_tabular(df)
