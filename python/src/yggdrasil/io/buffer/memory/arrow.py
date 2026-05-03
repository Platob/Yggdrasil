"""In-memory :class:`TabularIO` holding Arrow record batches.

No bytes, no codec, no spill. Reads yield the held batches as-is;
writes mutate the held batch list in place subject to ``options.mode``
(AUTO / OVERWRITE / TRUNCATE → replace, APPEND → append, IGNORE →
no-op when non-empty). Use this when you want a :class:`TabularIO`
over Arrow data you already have on the driver and don't want to pay
the IPC serialization round-trip.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterable, Iterator, Optional, Union

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.io.buffer.base import TabularIO
from yggdrasil.io.enums import MimeType, Mode


__all__ = ["MemoryArrowIO"]


# Anything we know how to ingest into the internal batch list.
ArrowSource = Union[
    pa.RecordBatch,
    pa.Table,
    Iterable[Union[pa.RecordBatch, pa.Table]],
    None,
]


class MemoryArrowIO(TabularIO[CastOptions]):
    """:class:`TabularIO` whose backing store is an in-memory batch list.

    The schema is tracked separately so an empty buffer still answers
    :meth:`collect_schema` correctly when one was supplied at
    construction (or carried over from a write that was later
    overwritten).
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_mime_type(cls) -> Optional[MimeType]:
        # In-memory containers don't claim a wire format; returning
        # None keeps them out of the media-type registry so they never
        # win factory dispatch by accident.
        return None

    def __init__(
        self,
        data: ArrowSource = None,
        *,
        schema: Optional[pa.Schema] = None,
        **kwargs: Any,
    ) -> None:
        # ``**kwargs`` forwards :class:`TabularIO`-shared init args
        # (``static_values``, ``media_type``, …) without listing
        # them explicitly here — keeps this constructor focused
        # on the in-memory-specific surface.
        super().__init__(**kwargs)
        self._batches: list[pa.RecordBatch] = []
        self._schema: Optional[pa.Schema] = schema
        if data is not None:
            self._ingest(data)

    def __repr__(self) -> str:
        return (
            f"MemoryArrowIO(num_batches={len(self._batches)}, "
            f"num_rows={sum(b.num_rows for b in self._batches)})"
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def batches(self) -> list[pa.RecordBatch]:
        """Defensive copy of the held batches."""
        return list(self._batches)

    @property
    def schema(self) -> Optional[pa.Schema]:
        """Arrow schema, when known.

        Set by the first ingested batch / table, by an explicit
        constructor argument, or by :meth:`_write_arrow_batches` on
        its first write. ``None`` only when the buffer has never seen
        data and no schema was passed in.
        """
        return self._schema

    @schema.setter
    def schema(self, value: Optional[pa.Schema]) -> None:
        self._schema = value

    def is_empty(self) -> bool:
        return not self._batches

    @property
    def num_rows(self) -> int:
        return sum(b.num_rows for b in self._batches)

    def __len__(self) -> int:
        return self.num_rows

    def __bool__(self) -> bool:
        return bool(self._batches)

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        return iter(self._batches)

    # ------------------------------------------------------------------
    # TabularIO contract — cache & persist
    # ------------------------------------------------------------------

    @property
    def cached(self) -> bool:
        # Always materialised — that's the whole point of this class.
        return True

    def unpersist(self) -> None:
        self._batches.clear()

    def persist(
        self,
        engine: str = "auto",
        *,
        data: Any = None,
    ) -> "TabularIO":
        # ``persist`` on a memory IO either no-ops (already cached) or
        # replaces the internal data. The engine arg is ignored — the
        # holder is always Arrow-backed.
        if data is not None:
            self._batches.clear()
            self._ingest(data)
        return self

    # ------------------------------------------------------------------
    # TabularIO contract — read / write hooks
    # ------------------------------------------------------------------

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        for batch in self._batches:
            yield options.cast_arrow_tabular(batch)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return
        if action is Mode.OVERWRITE:
            self._batches.clear()
        elif action is not Mode.APPEND:
            raise NotImplementedError(
                f"{type(self).__name__}._write_arrow_batches handles "
                f"OVERWRITE / APPEND / IGNORE; got {action!r}."
            )
        for batch in batches:
            self._batches.append(batch)
            if self._schema is None:
                self._schema = batch.schema

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_save_mode(self, mode: Any) -> Mode:
        m = Mode.from_(mode, default=Mode.AUTO)
        if m in (Mode.AUTO, Mode.OVERWRITE, Mode.TRUNCATE):
            return Mode.OVERWRITE
        if m is Mode.IGNORE:
            return Mode.IGNORE if self._batches else Mode.OVERWRITE
        if m is Mode.ERROR_IF_EXISTS:
            if self._batches:
                raise FileExistsError(
                    f"{type(self).__name__} write with Mode.ERROR_IF_EXISTS "
                    f"but buffer is non-empty ({len(self._batches)} batch(es))."
                )
            return Mode.OVERWRITE
        if m is Mode.APPEND:
            return Mode.APPEND
        raise ValueError(
            f"{type(self).__name__} does not support Mode.{m.name}; "
            f"valid: AUTO, OVERWRITE, TRUNCATE, APPEND, IGNORE, ERROR_IF_EXISTS."
        )

    def _ingest(self, source: ArrowSource) -> None:
        if source is None:
            return
        if isinstance(source, pa.RecordBatch):
            self._batches.append(source)
            if self._schema is None:
                self._schema = source.schema
        elif isinstance(source, pa.Table):
            for batch in source.to_batches():
                self._batches.append(batch)
            if self._schema is None:
                self._schema = source.schema
        else:
            for inner in source:
                self._ingest(inner)
