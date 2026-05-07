"""In-memory :class:`Tabular` holding a (mutable) Polars DataFrame.

Mirror of :class:`yggdrasil.io.tabular.spark.SparkTabular` for Polars:
the held frame is the holder's only state. Reads of
:meth:`_read_polars_frame` / :meth:`_scan_polars_frame` return it
unchanged (no Arrow round-trip); writes mutate it in place subject
to ``options.mode`` (AUTO / OVERWRITE / TRUNCATE → replace, APPEND
→ ``pl.concat`` with ``how="diagonal_relaxed"`` so missing columns
align with nulls, IGNORE / ERROR_IF_EXISTS follow the same shape as
the Spark / Arrow holders).

What we ingest
--------------

:meth:`_coerce_frame` accepts the shapes a real caller actually has
without forcing a manual conversion to Polars:

- :class:`polars.DataFrame` / :class:`polars.LazyFrame` (LazyFrame
  collects on ingest — the holder is in-memory by design)
- :class:`pyarrow.Table` / :class:`pyarrow.RecordBatch`
- :class:`pandas.DataFrame`
- :class:`pyspark.sql.DataFrame` (driver-side)
- ``list[dict]`` rows / ``dict[str, list]`` columns

That keeps the most common conversion glue on this side of the API
instead of every caller writing the same five-line ``isinstance``
ladder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Optional

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.enums import MimeType, Mode
from yggdrasil.io.tabular import Tabular

if TYPE_CHECKING:
    import polars as pl


__all__ = ["PolarsTabular"]


class PolarsTabular(Tabular[CastOptions]):
    """:class:`Tabular` whose backing store is a single Polars DataFrame.

    The frame is the holder's only state; reads return it as-is,
    writes replace (OVERWRITE) or concat (APPEND) it. Use this when
    you want a :class:`Tabular` over Polars data you already have on
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
        frame: "pl.DataFrame | pl.LazyFrame | Any | None" = None,
    ) -> None:
        super().__init__()
        self._frame: "pl.DataFrame | None" = (
            self._coerce_frame(frame) if frame is not None else None
        )

    def __repr__(self) -> str:
        if self._frame is None:
            return "PolarsTabular(frame=None)"
        return (
            f"PolarsTabular(num_rows={self._frame.height}, "
            f"num_columns={self._frame.width})"
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def frame(self) -> "pl.DataFrame | None":
        """Currently-held Polars DataFrame, or ``None`` when empty."""
        return self._frame

    @frame.setter
    def frame(self, value: "pl.DataFrame | pl.LazyFrame | Any | None") -> None:
        self._frame = self._coerce_frame(value) if value is not None else None

    def is_empty(self) -> bool:
        return self._frame is None

    def __bool__(self) -> bool:
        return self._frame is not None

    @property
    def num_rows(self) -> int:
        return 0 if self._frame is None else self._frame.height

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
    # Polars read / write — no Arrow round-trip on the polars path
    # ------------------------------------------------------------------

    def stat(self):
        return self._stats

    def _read_polars_frame(self, options: CastOptions) -> "pl.DataFrame":
        if self._frame is None:
            from yggdrasil.polars.lib import polars as pl

            schema = options.merged_schema
            polars_schema = (
                schema.to_polars_schema() if schema is not None else None
            )
            return pl.DataFrame(schema=polars_schema)
        return options.cast_polars_tabular(self._frame)

    def _scan_polars_frame(self, options: CastOptions) -> "pl.LazyFrame":
        return self._read_polars_frame(options).lazy()

    def _write_polars_frame(
        self,
        frame: "pl.DataFrame | pl.LazyFrame",
        options: CastOptions,
    ) -> None:
        from yggdrasil.polars.lib import polars as pl

        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return

        incoming = frame.collect() if isinstance(frame, pl.LazyFrame) else frame

        if action is Mode.OVERWRITE or self._frame is None:
            self._frame = incoming
            return
        if action is Mode.APPEND:
            # ``diagonal_relaxed`` mirrors Spark's ``unionByName(...,
            # allowMissingColumns=True)``: columns present on either
            # side survive, missing ones become null, dtype mismatches
            # widen via the polars supertype rules.
            self._frame = pl.concat(
                [self._frame, incoming], how="diagonal_relaxed",
            )
            return
        raise NotImplementedError(
            f"{type(self).__name__}._write_polars_frame handles "
            f"OVERWRITE / APPEND / IGNORE; got {action!r}."
        )

    # ------------------------------------------------------------------
    # Arrow read / write — go through the polars frame so we keep the
    # held shape in sync with what reads see.
    # ------------------------------------------------------------------

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        if self._frame is None:
            return
        table = self._frame.to_arrow()
        for batch in table.to_batches(max_chunksize=options.row_size or None):
            yield options.cast_arrow_tabular(batch)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        from yggdrasil.polars.lib import polars as pl

        materialized = list(batches)
        if not materialized:
            # APPEND of nothing is a no-op; OVERWRITE of nothing
            # leaves the existing frame. Match the Spark / Arrow
            # holders on an empty iterator.
            return
        table = pa.Table.from_batches(materialized)
        frame = pl.from_arrow(table)
        if isinstance(frame, pl.Series):
            frame = frame.to_frame()
        self._write_polars_frame(frame, options)

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
    def _coerce_frame(value: Any) -> "pl.DataFrame":
        """Coerce *value* to a :class:`pl.DataFrame`.

        Mirrors :class:`ArrowTabular._ingest`'s shape-detection logic
        but lands on a Polars frame instead of an Arrow batch list.
        Reuses :func:`yggdrasil.polars.cast.any_to_polars_dataframe`
        so polars/pandas/arrow/pyspark/dict shapes go through the
        same conversion path as the rest of the codebase.
        """
        from yggdrasil.polars.cast import any_to_polars_dataframe

        return any_to_polars_dataframe(value)
