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
from yggdrasil.lazy_imports import (
    polars_module,
    pyarrow_dataset_module,
)
from yggdrasil.io.buffer.bytes_io import BytesIO

if TYPE_CHECKING:
    import polars as pl
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
        """Read the schema from the Parquet footer."""
        if self.is_empty():
            return Schema.empty()

        with self._reading_context(options.copy(reset_seek=True)) as io:
            source = io.arrow_io(mode="rb")
            pf = pq.ParquetFile(source)
            return Schema.from_arrow(pf.schema_arrow)

    # ==================================================================
    # Read path
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: ParquetOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches from the Parquet file."""
        if self.remaining_bytes() == 0:
            return

        with self._reading_context(options) as io:
            source = io.arrow_io(mode="rb")
            source.seek(0)
            with pq.ParquetFile(source) as pf:
                read_options = options.check_source(pf.schema_arrow)

                for batch in pf.iter_batches(
                    batch_size=read_options.row_size or 65536,
                    use_threads=read_options.use_threads,
                    columns=read_options.select_source_column_names(),
                ):
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
        if not type(self)._NATIVE_SCANNER_OK:
            return False
        if self.is_empty():
            return False
        if options.target_field is not None:
            return False
        if self.codec is not None:
            return False
        if self.path is None:
            return False
        if not self.path.is_local:
            return False
        return True

    def _read_arrow_dataset(self, options: ParquetOptions) -> "pds.Dataset":
        if not self._can_use_native_scanner(options):
            return super()._read_arrow_dataset(options)

        pds = pyarrow_dataset_module()
        return pds.dataset(self.path.__fspath__(), format="parquet")

    def _scan_polars_frame(self, options: ParquetOptions) -> "pl.LazyFrame":
        if not self._can_use_native_scanner(options):
            return super()._scan_polars_frame(options)

        pl = polars_module()
        return pl.scan_parquet(self.path.__fspath__())

    def _read_polars_frame(self, options: ParquetOptions) -> "pl.DataFrame":
        if not self._can_use_native_scanner(options):
            return super()._read_polars_frame(options)

        pl = polars_module()
        return pl.read_parquet(self.path.__fspath__(), use_pyarrow=False)