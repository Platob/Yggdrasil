"""In-memory :class:`TabularIO` holding a (mutable) Spark DataFrame.

The held DataFrame is mutable: writes replace it (OVERWRITE) or union
to it (APPEND). :meth:`read_spark_frame` returns the held frame
untouched (no driver collect); :meth:`read_arrow_batches` falls back
to ``df.toArrow().to_batches()`` (which DOES collect to the driver —
fine when the frame is small enough, but check before reaching for it
in a hot path).

Use this when you want a :class:`TabularIO` over a Spark frame you
already have on the driver and don't want to round-trip through IPC
bytes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Optional

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.io.buffer.base import TabularIO
from yggdrasil.io.enums import MimeType, Mode

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession


__all__ = ["MemorySparkIO"]


class MemorySparkIO(TabularIO[CastOptions]):
    """:class:`TabularIO` whose backing store is a single Spark DataFrame.

    The frame is the holder's only state; reads of
    :meth:`_read_spark_frame` return it as-is, writes mutate it in
    place. The Spark session is cached off the frame on construction
    (or set explicitly) so an empty buffer can still synthesize an
    empty DataFrame on read.
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_media_type(cls) -> Optional[MimeType]:
        return None  # not registered, see MemoryArrowIO for the rationale

    def __init__(
        self,
        frame: Optional["SparkDataFrame"] = None,
        *,
        spark: Optional["SparkSession"] = None,
    ) -> None:
        super().__init__()
        self._frame: Optional["SparkDataFrame"] = frame
        self._spark: Optional["SparkSession"] = spark
        if frame is not None and self._spark is None:
            # Cache the session off the frame so subsequent
            # empty-frame reads / writes don't have to rediscover it.
            self._spark = getattr(frame, "sparkSession", None)

    def __repr__(self) -> str:
        if self._frame is None:
            return "MemorySparkIO(frame=None)"
        return f"MemorySparkIO(frame={self._frame!r})"

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def frame(self) -> Optional["SparkDataFrame"]:
        """Currently-held Spark DataFrame, or ``None`` when empty."""
        return self._frame

    @frame.setter
    def frame(self, value: Optional["SparkDataFrame"]) -> None:
        self._frame = value
        if value is not None and self._spark is None:
            self._spark = getattr(value, "sparkSession", None)

    @property
    def spark(self) -> Optional["SparkSession"]:
        return self._spark

    def is_empty(self) -> bool:
        return self._frame is None

    def __bool__(self) -> bool:
        return self._frame is not None

    # ------------------------------------------------------------------
    # TabularIO contract — cache & persist
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
    ) -> "TabularIO":
        if data is not None:
            self.frame = self._coerce_frame(data)
        return self

    # ------------------------------------------------------------------
    # Spark read / write — no driver collect on the spark path
    # ------------------------------------------------------------------

    def _read_spark_frame(self, options: CastOptions) -> "SparkDataFrame":
        if self._frame is None:
            spark = self._require_spark()
            schema = options.merged_schema
            spark_schema = (
                schema.to_spark_schema() if schema is not None else None
            )
            return spark.createDataFrame([], schema=spark_schema)
        return options.cast_spark_tabular(self._frame)

    def _write_spark_frame(
        self,
        frame: "SparkDataFrame",
        options: CastOptions,
    ) -> None:
        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return
        if action is Mode.OVERWRITE or self._frame is None:
            self.frame = frame
            return
        if action is Mode.APPEND:
            self.frame = self._frame.unionByName(
                frame, allowMissingColumns=True,
            )
            return
        raise NotImplementedError(
            f"{type(self).__name__}._write_spark_frame handles "
            f"OVERWRITE / APPEND / IGNORE; got {action!r}."
        )

    # ------------------------------------------------------------------
    # Arrow read / write — collects on read, builds Spark on write
    # ------------------------------------------------------------------

    def stat(self):
        return self._stats

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        # Forces a driver-side collect via ``df.toArrow()``. Loud
        # rather than silent — the call site is the one asking for
        # Arrow batches off a Spark holder.
        if self._frame is None:
            return
        arrow_table = self._frame.toArrow()
        for batch in arrow_table.to_batches(max_chunksize=options.row_size):
            yield options.cast_arrow_tabular(batch)

    def _read_records(self, options: CastOptions) -> "Iterator[Any]":
        # Skip the Arrow round-trip — `toLocalIterator()` streams
        # rows from the executors one by one, so the driver memory
        # footprint stays bounded even for frames that wouldn't fit
        # in a single ``df.toArrow()`` collect.
        from yggdrasil.data.record import Record

        if self._frame is None:
            return
        yield from Record.from_spark_frame(self._frame)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        # Build a Spark frame from the incoming Arrow stream, then
        # delegate to ``_write_spark_frame`` so OVERWRITE / APPEND /
        # IGNORE branching applies the same way.
        materialized = list(batches)
        if not materialized:
            # APPEND of nothing is a no-op; OVERWRITE of nothing
            # leaves the existing frame. Match the IPC writer's
            # behavior on an empty iterator.
            return
        from yggdrasil.spark.cast import any_to_spark_dataframe

        table = pa.Table.from_batches(materialized)
        frame = any_to_spark_dataframe(table, options=options)
        self._write_spark_frame(frame, options)

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

    def _require_spark(self) -> "SparkSession":
        if self._spark is None:
            from yggdrasil.environ import PyEnv

            self._spark = PyEnv.spark_session(
                create=True, install_spark=False, import_error=True,
            )
        return self._spark

    def _coerce_frame(self, value: Any) -> "SparkDataFrame":
        from yggdrasil.spark.cast import any_to_spark_dataframe

        return any_to_spark_dataframe(value)
