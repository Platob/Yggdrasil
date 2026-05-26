"""In-memory :class:`Tabular` holding a single :class:`pa.Table`.

:class:`ArrowTabular` is the simplest Tabular — it wraps a pyarrow
Table and exposes reads/writes through the Tabular protocol. Writes
replace or append to the held table.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Iterable, Iterator, Optional, Union

import pyarrow as pa
from yggdrasil.data import StructField, Schema

from yggdrasil.data.options import CastOptions
from yggdrasil.io.tabular import O
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.enums import MimeType, Mode

logger = logging.getLogger(__name__)


__all__ = ["ArrowTabular"]


ArrowSource = Union[
    pa.RecordBatch,
    pa.Table,
    pa.RecordBatchReader,
    "Tabular",
    Iterable[Union[pa.RecordBatch, pa.Table]],
    Any,
    None,
]


class ArrowTabular(Tabular[CastOptions]):
    """In-memory Tabular holding a :class:`pa.Table`."""

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_media_type(cls) -> Optional[MimeType]:
        return None

    def __init__(
        self,
        data: ArrowSource = None,
        *more: ArrowSource,
        schema: "Optional[pa.Schema | StructField]" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._table: Optional[pa.Table] = None

        if schema is not None:
            if isinstance(schema, pa.Schema):
                self._persist_schema(StructField.from_arrow(schema))
            else:
                self._persist_schema(schema)

        if data is not None:
            self._ingest(data)
        for src in more:
            if src is not None:
                self._ingest(src)

    def _ingest(self, data: Any) -> None:
        if isinstance(data, pa.Table):
            self._append_table(data)
        elif isinstance(data, pa.RecordBatch):
            self._append_table(pa.Table.from_batches([data]))
        elif isinstance(data, pa.RecordBatchReader):
            self._append_table(data.read_all())
        elif isinstance(data, Tabular):
            self._append_table(data.read_arrow_table())
        elif isinstance(data, list):
            batches = []
            for item in data:
                if isinstance(item, pa.RecordBatch):
                    batches.append(item)
                elif isinstance(item, pa.Table):
                    batches.extend(item.to_batches())
            if batches:
                self._append_table(pa.Table.from_batches(batches))
        elif hasattr(data, "__iter__"):
            batches = []
            for item in data:
                if isinstance(item, pa.RecordBatch):
                    batches.append(item)
                elif isinstance(item, pa.Table):
                    batches.extend(item.to_batches())
            if batches:
                self._append_table(pa.Table.from_batches(batches))

    def _append_table(self, table: pa.Table) -> None:
        if table.num_rows == 0 and self._table is not None:
            return
        if self._table is None:
            self._table = table
        else:
            self._table = pa.concat_tables(
                [self._table, table], promote_options="permissive",
            )

    def __repr__(self) -> str:
        rows = self._table.num_rows if self._table is not None else 0
        cols = self._table.num_columns if self._table is not None else 0
        return f"ArrowTabular(rows={rows}, cols={cols})"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def table(self) -> Optional[pa.Table]:
        return self._table

    @property
    def batches(self) -> list[pa.RecordBatch]:
        if self._table is None:
            return []
        return self._table.to_batches()

    @property
    def schema(self) -> Optional[pa.Schema]:
        if self._table is not None:
            return self._table.schema
        s = self._schema_cache
        if s is not None and s is not ...:
            return s.to_arrow_schema()
        return None

    @property
    def num_rows(self) -> int:
        return self._table.num_rows if self._table is not None else 0

    @property
    def num_columns(self) -> int:
        return self._table.num_columns if self._table is not None else 0

    def __len__(self) -> int:
        return self.num_rows

    def _count(self, options=None) -> int:
        if options is None:
            return self.num_rows
        return sum(
            options.cast_arrow_batch(b).num_rows for b in self.batches
        )

    def __bool__(self) -> bool:
        return self._table is not None and self._table.num_rows > 0

    def is_empty(self) -> bool:
        return self._table is None or self._table.num_rows == 0

    # ------------------------------------------------------------------
    # Tabular implementation
    # ------------------------------------------------------------------

    def _collect_schema(self, options=None) -> "Schema":
        cached = self._schema_cache
        if cached is not None and cached is not ...:
            return cached
        if self._table is not None:
            return Schema.from_arrow(self._table.schema)
        return Schema.empty()

    def _read_arrow_batches(
        self, options: CastOptions = None,
    ) -> Iterator[pa.RecordBatch]:
        if self._table is None:
            return
        for batch in self._table.to_batches():
            if options is not None:
                batch = options.cast_arrow_batch(batch)
            yield batch

    def _read_arrow_table(self, options: CastOptions = None) -> pa.Table:
        if self._table is None:
            s = self.schema
            if s is not None:
                return s.empty_table()
            return pa.table({})
        if options is not None:
            return options.cast_arrow_table(self._table)
        return self._table

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions = None,
    ) -> None:
        mode = options.mode if options else Mode.AUTO
        if mode in (Mode.OVERWRITE, Mode.TRUNCATE):
            self._table = None

        new_batches = [b for b in batches if b.num_rows > 0]
        if not new_batches:
            return

        new_table = pa.Table.from_batches(new_batches)
        self._append_table(new_table)

        if self._table is not None:
            self._persist_schema(Schema.from_arrow(self._table.schema))

    @classmethod
    def options_class(cls):
        return CastOptions

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to_pandas(self, **kwargs):
        if self._table is None:
            import pandas as pd
            return pd.DataFrame()
        return self._table.to_pandas(**kwargs)

    def to_pydict(self) -> dict:
        if self._table is None:
            return {}
        return self._table.to_pydict()

    def to_pylist(self) -> list[dict]:
        if self._table is None:
            return []
        return self._table.to_pylist()
