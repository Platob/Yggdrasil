"""In-memory :class:`Tabular` holding a (mutable) pandas DataFrame.

The pandas sibling of :class:`yggdrasil.arrow.tabular.ArrowTabular`,
:class:`yggdrasil.polars.tabular.PolarsTabular` and
:class:`yggdrasil.spark.tabular.SparkDataset`: the frame is the holder's
only state, reads return it natively, writes append / overwrite it in
place. Cross-engine reads convert on demand via
:mod:`yggdrasil.pandas.cast`.

> PARITY: mirrors ``ArrowTabular`` / ``PolarsTabular`` / ``SparkDataset`` —
> keep the in-memory holders' surface aligned.
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
from yggdrasil.lazy_imports import pandas_module

if TYPE_CHECKING:
    import pandas

logger = logging.getLogger(__name__)


class PandasTabular(Tabular[CastOptions]):
    """:class:`Tabular` wrapping a single in-memory ``pandas.DataFrame``."""

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_media_type(cls) -> Any:
        return None  # in-memory container — see ArrowTabular for the rationale

    def __init__(
        self,
        frame: Optional[pandas.DataFrame] = None,
        schema: Schema | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._frame: Optional[pandas.DataFrame] = frame
        if schema is not None:
            self._persist_schema(schema)

    def __repr__(self) -> str:
        if self._frame is None:
            return "PandasTabular(frame=None)"
        rows, cols = self._frame.shape
        return f"PandasTabular(rows={rows}, cols={cols})"

    # -- Accessors -------------------------------------------------------

    @property
    def frame(self) -> Optional[pandas.DataFrame]:
        return self._frame

    @frame.setter
    def frame(self, value: Optional[pandas.DataFrame]) -> None:
        self._frame = value

    def is_empty(self) -> bool:
        return self._frame is None or len(self._frame) == 0

    def __bool__(self) -> bool:
        return self._frame is not None

    def _collect_schema(self, options: CastOptions) -> Schema:
        cached = self._schema_cache
        if cached is not ...:
            return cached
        if self._frame is None:
            return StructField.empty()
        # Round-trip through Arrow so index handling / dtype mapping matches
        # what a later ``read_arrow`` would produce.
        return StructField.from_arrow_schema(
            pa.Schema.from_pandas(self._frame, preserve_index=False),
        )

    # -- Native pandas read / write --------------------------------------

    def _read_pandas_frame(self, options: CastOptions) -> pandas.DataFrame:
        if self._frame is None:
            return pandas_module().DataFrame()
        from yggdrasil.pandas.cast import cast_pandas_dataframe

        frame = self._frame
        if options.target is not None:
            frame = cast_pandas_dataframe(frame, options)
        if options.row_limit is not None:
            frame = frame.head(options.row_limit)
        return frame

    def _write_pandas_frame(self, frame: pandas.DataFrame, options: CastOptions) -> None:
        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return
        if action is Mode.OVERWRITE or self._frame is None:
            self._frame = frame
            return
        if action is Mode.APPEND:
            self._frame = pandas_module().concat(
                [self._frame, frame], ignore_index=True,
            )
            return
        raise NotImplementedError(
            f"{type(self).__name__}._write_pandas_frame handles "
            f"OVERWRITE / APPEND / IGNORE; got {action!r}."
        )

    # -- Arrow read / write (cross-engine bridge) ------------------------

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        if self._frame is None:
            return
        from yggdrasil.pandas.cast import pandas_dataframe_to_arrow_table

        table = pandas_dataframe_to_arrow_table(self._frame)
        if options.row_limit is not None and table.num_rows > options.row_limit:
            table = table.slice(0, options.row_limit)
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
        from yggdrasil.pandas.cast import any_to_pandas_dataframe

        frame = any_to_pandas_dataframe(pa.Table.from_batches(materialized))
        self._write_pandas_frame(frame, options)

    def _union(self, other: Tabular, *, mode: Mode = ...) -> PandasTabular:
        frame = other.read_pandas_frame()
        self._write_pandas_frame(frame, self.check_options(None, mode=Mode.APPEND))
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
