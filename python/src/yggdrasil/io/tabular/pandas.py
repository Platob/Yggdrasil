"""In-memory :class:`Tabular` holding a (mutable) pandas DataFrame.

Mirror of :class:`yggdrasil.io.tabular.polars.PolarsTabular` /
:class:`yggdrasil.io.tabular.spark.SparkTabular` for pandas: the held
frame is the holder's only state. Reads of :meth:`_read_pandas_frame`
return it unchanged (no Arrow round-trip); writes mutate it in place
subject to ``options.mode`` (AUTO / OVERWRITE / TRUNCATE → replace,
APPEND → ``pd.concat`` with ``ignore_index=True``, IGNORE /
ERROR_IF_EXISTS follow the same shape as the other holders).

What we ingest
--------------

:meth:`_coerce_frame` accepts the shapes a real caller actually has
without forcing a manual conversion to pandas:

- :class:`pandas.DataFrame` (passthrough)
- :class:`pyarrow.Table` / :class:`pyarrow.RecordBatch`
- :class:`polars.DataFrame` / :class:`polars.LazyFrame`
- :class:`pyspark.sql.DataFrame` (driver-side via ``toPandas``)
- ``list[dict]`` rows / ``dict[str, list]`` columns
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Optional

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.enums import MimeType, Mode
from yggdrasil.io.tabular import Tabular
from yggdrasil.pickle.serde import ObjectSerde

if TYPE_CHECKING:
    import pandas


__all__ = ["PandasTabular"]


class PandasTabular(Tabular[CastOptions]):
    """:class:`Tabular` whose backing store is a single pandas DataFrame.

    The frame is the holder's only state; reads return it as-is,
    writes replace (OVERWRITE) or concat (APPEND) it. Use this when
    you want a :class:`Tabular` over pandas data you already have on
    the driver and don't want to round-trip through IPC bytes.
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_media_type(cls) -> Optional[MimeType]:
        # In-memory containers don't claim a wire format — same
        # rationale as ArrowTabular / SparkTabular.
        return None

    def __init__(
        self,
        frame: "pandas.DataFrame | Any | None" = None,
    ) -> None:
        super().__init__()
        self._frame: "pandas.DataFrame | None" = (
            self._coerce_frame(frame) if frame is not None else None
        )

    def __repr__(self) -> str:
        if self._frame is None:
            return "PandasTabular(frame=None)"
        rows, cols = self._frame.shape
        return f"PandasTabular(num_rows={rows}, num_columns={cols})"

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def frame(self) -> "pandas.DataFrame | None":
        """Currently-held pandas DataFrame, or ``None`` when empty."""
        return self._frame

    @frame.setter
    def frame(self, value: "pandas.DataFrame | Any | None") -> None:
        self._frame = self._coerce_frame(value) if value is not None else None

    def is_empty(self) -> bool:
        return self._frame is None

    def __bool__(self) -> bool:
        return self._frame is not None

    @property
    def num_rows(self) -> int:
        return 0 if self._frame is None else len(self._frame.index)

    def __len__(self) -> int:
        return self.num_rows

    # ------------------------------------------------------------------
    # Tabular contract — cache & persist
    # ------------------------------------------------------------------

    @property
    def cached(self) -> bool:
        return self._frame is not None

    def unpersist(self) -> None:
        self._frame = None

    def persist(
        self,
        engine: str = "auto",
        *,
        data: Any = None,
    ) -> "Tabular":
        if data is not None:
            self._frame = self._coerce_frame(data)
        return self

    # ------------------------------------------------------------------
    # Pandas read / write — no Arrow round-trip on the pandas path
    # ------------------------------------------------------------------

    def stat(self):
        return self._stats

    def _read_pandas_frame(self, options: CastOptions) -> "pandas.DataFrame":
        from yggdrasil.pandas.lib import pandas as pd

        if self._frame is None:
            schema = options.merged_schema
            if schema is None:
                return pd.DataFrame()
            return schema.to_arrow_schema().empty_table().to_pandas()
        if options.target_field is None:
            return self._frame
        return options.cast_pandas(self._frame)

    def _write_pandas_frame(
        self,
        frame: "pandas.DataFrame",
        options: CastOptions,
    ) -> None:
        from yggdrasil.pandas.lib import pandas as pd

        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return

        if action is Mode.OVERWRITE or self._frame is None:
            self._frame = frame
            return
        if action is Mode.APPEND:
            # ``ignore_index=True`` keeps the result with a default
            # range index — matching the way :meth:`Tabular._write_pandas_frame`
            # treats unindexed frames as the canonical shape.
            self._frame = pd.concat(
                [self._frame, frame], ignore_index=True, sort=False,
            )
            return
        raise NotImplementedError(
            f"{type(self).__name__}._write_pandas_frame handles "
            f"OVERWRITE / APPEND / IGNORE; got {action!r}."
        )

    # ------------------------------------------------------------------
    # Arrow read / write — go through the pandas frame so the held
    # shape stays in sync with what reads see.
    # ------------------------------------------------------------------

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        if self._frame is None:
            return
        is_default_range = self._has_default_index(self._frame)
        table = pa.Table.from_pandas(
            self._frame, preserve_index=not is_default_range,
        )
        for batch in table.to_batches(max_chunksize=options.row_size or None):
            yield options.cast_arrow_tabular(batch)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        materialized = list(batches)
        if not materialized:
            return
        table = pa.Table.from_batches(materialized)
        self._write_pandas_frame(table.to_pandas(), options)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_save_mode(self, mode: Any) -> Mode:
        m = Mode.from_(mode, default=Mode.AUTO)
        if m in (Mode.AUTO, Mode.OVERWRITE, Mode.TRUNCATE):
            return Mode.OVERWRITE
        if m is Mode.IGNORE:
            return Mode.IGNORE if self._frame is not None else Mode.OVERWRITE
        if m is Mode.ERROR_IF_EXISTS:
            if self._frame is not None:
                raise FileExistsError(
                    f"{type(self).__name__} write with Mode.ERROR_IF_EXISTS "
                    "but buffer is non-empty."
                )
            return Mode.OVERWRITE
        if m is Mode.APPEND:
            return Mode.APPEND
        raise ValueError(
            f"{type(self).__name__} does not support Mode.{m.name}; "
            f"valid: AUTO, OVERWRITE, TRUNCATE, APPEND, IGNORE, ERROR_IF_EXISTS."
        )

    @staticmethod
    def _has_default_index(frame: "pandas.DataFrame") -> bool:
        from yggdrasil.pandas.lib import pandas as pd

        return isinstance(frame.index, pd.RangeIndex) and frame.index.name is None

    @classmethod
    def _coerce_frame(cls, value: Any) -> "pandas.DataFrame":
        """Coerce *value* to a :class:`pandas.DataFrame`.

        Mirrors :class:`ArrowTabular._ingest`'s shape-detection logic
        but lands on a pandas frame. We don't have an
        ``any_to_pandas_dataframe`` helper at the moment, so the
        conversion ladder lives here — same module-name sniffing the
        rest of :mod:`yggdrasil.io.tabular` uses.
        """
        from yggdrasil.pandas.lib import pandas as pd

        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, pa.Table):
            return value.to_pandas()
        if isinstance(value, pa.RecordBatch):
            return pa.Table.from_batches([value]).to_pandas()
        if isinstance(value, pa.RecordBatchReader):
            return value.read_all().to_pandas()

        ns = ObjectSerde.full_namespace(value)
        if ns.startswith("polars"):
            from yggdrasil.polars.lib import polars as pl

            if isinstance(value, pl.LazyFrame):
                value = value.collect()
            return value.to_pandas()
        if ns.startswith("pyspark"):
            to_arrow = getattr(value, "toArrow", None)
            if to_arrow is not None:
                return to_arrow().to_pandas()
            return value.toPandas()

        # Pure-Python row-list / column-dict shapes go straight
        # through the pandas constructor.
        return pd.DataFrame(value)
