"""In-memory :class:`Tabular` holding a (mutable) polars DataFrame.

The polars sibling of :class:`yggdrasil.arrow.tabular.ArrowTabular` and
:class:`yggdrasil.spark.tabular.SparkDataset`: the frame is the holder's
only state, reads return it natively (zero-copy through the shared Arrow
buffers where polars allows it), writes append / overwrite it in place.

Cross-engine reads (``read_arrow_*``) convert on demand via
:mod:`yggdrasil.polars.cast`; an Arrow / pandas / Spark consumer therefore
sees the same data without the holder ever leaving polars internally.

> PARITY: mirrors ``ArrowTabular`` / ``SparkDataset`` — keep the three
> in-memory holders' surface (``frame`` property, ``is_empty``, native
> read/write overrides) aligned.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import pyarrow as pa

from yggdrasil.data import Schema, StructField
from yggdrasil.data.options import CastOptions
from yggdrasil.enums import Mode
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.lazy_imports import polars_module

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class PolarsTabular(Tabular[CastOptions]):
    """:class:`Tabular` wrapping a single in-memory ``polars.DataFrame``."""

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_media_type(cls) -> Any:
        # In-memory container — claim no wire format so it never wins
        # media-type factory dispatch (mirrors ArrowTabular / SparkDataset).
        return None

    def __init__(
        self,
        frame: Optional[pl.DataFrame] = None,
        schema: Schema | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._frame: Optional[pl.DataFrame] = frame
        if schema is not None:
            self._persist_schema(schema)

    def __repr__(self) -> str:
        if self._frame is None:
            return "PolarsTabular(frame=None)"
        return f"PolarsTabular(rows={self._frame.height}, cols={self._frame.width})"

    # -- Accessors -------------------------------------------------------

    @property
    def frame(self) -> Optional[pl.DataFrame]:
        return self._frame

    @frame.setter
    def frame(self, value: Optional[pl.DataFrame]) -> None:
        self._frame = value

    def is_empty(self) -> bool:
        return self._frame is None or self._frame.height == 0

    def __bool__(self) -> bool:
        return self._frame is not None

    def _collect_schema(self, options: CastOptions) -> Schema:
        cached = self._schema_cache
        if cached is not ...:
            return cached
        if self._frame is None:
            return StructField.empty()
        return StructField.from_polars_schema(self._frame.schema)

    # -- Native polars read / write (no Arrow round-trip) ----------------

    def _read_polars_frame(self, options: CastOptions) -> pl.DataFrame:
        if self._frame is None:
            schema = options.merged
            return polars_module().DataFrame(
                schema=schema.to_polars_schema() if schema is not None else None,
            )
        from yggdrasil.polars.cast import cast_polars_dataframe

        frame = self._frame
        if options.target is not None or options.unique_by or options.time_sample_by:
            frame = cast_polars_dataframe(frame, options)
        if options.row_limit is not None:
            frame = frame.head(options.row_limit)
        return frame

    def _write_polars_frame(self, frame: pl.DataFrame, options: CastOptions) -> None:
        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return
        if action is Mode.OVERWRITE or self._frame is None:
            self._frame = frame
            return
        if action is Mode.APPEND:
            # ``diagonal_relaxed`` unions columns by name and reconciles
            # differing-but-compatible dtypes — matches the "allow missing
            # columns" append the Spark / Arrow holders give.
            self._frame = polars_module().concat(
                [self._frame, frame], how="diagonal_relaxed",
            )
            return
        raise NotImplementedError(
            f"{type(self).__name__}._write_polars_frame handles "
            f"OVERWRITE / APPEND / IGNORE; got {action!r}."
        )

    # -- Arrow read / write (cross-engine bridge) ------------------------

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        if self._frame is None:
            return
        pl = polars_module()
        frame = self._frame
        if options.row_limit is not None:
            frame = frame.head(options.row_limit)
        # ``CompatLevel.oldest()`` keeps flat ``string`` / ``binary`` (not the
        # ``string_view`` shapes ``newest()`` emits) — view buffers raise in
        # the pyarrow parquet writer and several compute kernels (e.g.
        # ``equal(string_view, string)``). Matches the base polars writer.
        table = frame.to_arrow(compat_level=pl.CompatLevel.oldest())
        for batch in table.to_batches(max_chunksize=options.row_size):
            yield options.cast_arrow_batch(batch)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        materialized = list(batches)
        if not materialized:
            return
        from yggdrasil.polars.cast import any_to_polars_dataframe

        frame = any_to_polars_dataframe(pa.Table.from_batches(materialized))
        self._write_polars_frame(frame, options)

    def _union(self, other: Tabular, *, mode: Mode = ...) -> PolarsTabular:
        frame = other.read_polars_frame()
        self._write_polars_frame(frame, self.check_options(None, mode=Mode.APPEND))
        return self

    def _delete(
        self,
        predicate: Any = None,
        *,
        wait: Any = True,
        missing_ok: bool = False,
        delete_staging: bool = True,
        **kwargs: Any,
    ) -> int:
        return self._delete_rewrite(predicate, **kwargs)
