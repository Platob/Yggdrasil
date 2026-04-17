"""Parquet I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Uses :mod:`pyarrow.parquet` for both reading and writing.  Transport-level
compression (gzip, zstd, …) is handled transparently by the base class;
the ``compression`` option on :class:`ParquetOptions` controls the
*intra-file* Parquet column compression codec.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["ParquetOptions", "ParquetIO"]


@dataclass(slots=True)
class ParquetOptions(MediaOptions):
    """Options for Parquet I/O.

    Inherits all :class:`MediaOptions` fields and adds Parquet-specific
    knobs for write-time compression and statistics.

    Parameters
    ----------
    compression:
        Intra-file column compression codec passed to
        :func:`pyarrow.parquet.write_table`.
    compression_level:
        Codec-specific compression level.
    use_dictionary:
        Enable dictionary encoding for eligible columns.
    use_statistics:
        Write column statistics in the Parquet metadata.
    allow_truncated_timestamps:
        When ``True``, coerce timestamps to microsecond precision.
    """


    # ---- write options ----
    compression: str | None = "zstd"
    compression_level: int | None = None
    use_dictionary: bool = True
    use_statistics: bool = True
    allow_truncated_timestamps: bool = True

    @classmethod
    def resolve(cls, *, options: "ParquetOptions | None" = None, **overrides) -> "ParquetOptions":
        """Merge *overrides* into *options* (or a fresh default)."""
        base = options or cls()
        valid = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        unknown = set(overrides) - set(valid)
        if unknown:
            raise TypeError(f"{cls.__name__}.resolve(): unknown option(s): {sorted(unknown)}")
        for k, v in overrides.items():
            setattr(base, k, v)
        return base


@dataclass(slots=True)
class ParquetIO(MediaIO[ParquetOptions]):
    """Parquet I/O backed by :mod:`pyarrow.parquet`."""

    @classmethod
    def check_options(cls, options: Optional[ParquetOptions], *args, **kwargs) -> ParquetOptions:
        """Validate and merge caller-supplied options."""
        return ParquetOptions.check_parameters(options=options, **kwargs)

    def _read_arrow_batches(self, *, options: ParquetOptions) -> Iterator["pyarrow.RecordBatch"]:
        """Yield record batches from the (uncompressed) Parquet buffer."""
        if self.buffer.size <= 0:
            return

        arrow_io = self.buffer.to_arrow_io("r")
        try:
            pf = pq.ParquetFile(arrow_io)
            for batch in pf.iter_batches(
                columns=options.columns,
                use_threads=options.use_threads,
            ):
                yield options.cast.cast_arrow(batch)
        finally:
            arrow_io.close()

    def _collect_arrow_schema(self) -> "pyarrow.Schema":
        """Return the Parquet schema by reading only the file footer."""
        if self.buffer.size <= 0:
            return pa.schema([])

        buf, decompressed = self._decompressed_buffer()
        orig_buffer = self.buffer
        try:
            if decompressed:
                self.buffer = buf
            arrow_io = self.buffer.to_arrow_io("r")
            try:
                return pq.ParquetFile(arrow_io).schema_arrow
            finally:
                arrow_io.close()
        finally:
            if decompressed:
                self.buffer = orig_buffer
                buf.close()

    def _write_arrow_batches(
        self,
        *,
        batches: Iterator["pyarrow.RecordBatch"],
        schema: "pyarrow.Schema",
        options: ParquetOptions,
    ) -> None:
        """Write record batches as Parquet into the (uncompressed) buffer."""
        arrow_io = self.buffer.to_arrow_io("w")
        try:
            writer = pq.ParquetWriter(
                arrow_io,
                schema,
                compression=options.compression,
                compression_level=options.compression_level,
                use_dictionary=options.use_dictionary,
                write_statistics=options.use_statistics,
                coerce_timestamps="us" if options.allow_truncated_timestamps else None,
                use_deprecated_int96_timestamps=not options.allow_truncated_timestamps,
            )
            try:
                for batch in batches:
                    casted = options.cast.cast_arrow(batch)
                    writer.write_batch(casted)
            finally:
                writer.close()
        finally:
            arrow_io.close()
