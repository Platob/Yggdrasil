# yggdrasil/io/buffer/parquet_io.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, Optional

from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow  # for type checkers only


__all__ = ["ParquetOptions", "ParquetIO"]


@dataclass(slots=True)
class ParquetOptions(MediaOptions):
    """
    Options for Parquet IO.

    Notes:
    - We include common read params (columns/use_threads) because your ParquetIO.read_table
      already expects them (they were referenced before but not declared in the options class).
    - We keep write params (compression, etc.) with sane defaults.
    """
    # ---- read options ----
    columns: list[str] | None = None
    use_threads: bool = True

    # ---- write options ----
    compression: str | None = "zstd"
    compression_level: int | None = None
    use_dictionary: bool = True
    use_statistics: bool = True
    allow_truncated_timestamps: bool = True

    # If you want "kwargs" overrides from public signatures to merge into an options instance,
    # keep this helper. It preserves concrete type (Self).
    @classmethod
    def resolve(cls, *, options: Self | None = None, **overrides) -> Self:
        base = options or cls()
        # Only apply known fields (prevents silent typos)
        valid = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        unknown = set(overrides) - set(valid)
        if unknown:
            raise TypeError(f"{cls.__name__}.resolve(): unknown option(s): {sorted(unknown)}")
        for k, v in overrides.items():
            setattr(base, k, v)
        return base


@dataclass(slots=True)
class ParquetIO(MediaIO[ParquetOptions]):
    """
    Concrete IO for Parquet.
    """

    @classmethod
    def check_options(cls, options: Optional[ParquetOptions], *args, **kwargs) -> ParquetOptions:
        return ParquetOptions.check_parameters(
            options=options,
            **kwargs
        )

    def _read_arrow_table(self, *, options: ParquetOptions) -> "pyarrow.Table":
        if self.buffer.size <= 0:
            import pyarrow as pa
            return pa.Table.from_batches([], schema=pa.schema([])) # noqa

        import pyarrow.parquet as pq

        arrow_io = self.buffer.to_arrow_io("r")
        try:
            return pq.read_table(
                arrow_io,
                columns=options.columns,
                use_threads=options.use_threads,
            )
        finally:
            arrow_io.close()

    def _write_arrow_table(self, *, table: "pyarrow.Table", options: ParquetOptions) -> None:
        import pyarrow.parquet as pq

        arrow_io = self.buffer.to_arrow_io("w")
        try:
            pq.write_table(
                table,
                arrow_io,
                compression=options.compression,
                compression_level=options.compression_level,
                use_dictionary=options.use_dictionary,
                write_statistics=options.use_statistics,
                coerce_timestamps="us" if options.allow_truncated_timestamps else None,
                use_deprecated_int96_timestamps=not options.allow_truncated_timestamps,
                # If you actually use this option in your project, wire it to pq.write_table params
                # or to a coercion/normalization step before writing.
            )
        finally:
            arrow_io.close()
