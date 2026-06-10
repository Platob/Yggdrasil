"""Tabular inspect/preview — schema + exact row count from the footer.

:meth:`inspect` is hit on every file open in the UI. For parquet the
schema and row count live in the footer (O(1)) — no need to read cap+1
rows just to size the file. The service reads only the metadata for the
row count and the Arrow schema for the column list, and reports an
``editable`` flag (parquet/arrow are rewriteable in place).
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel

from yggdrasil.exceptions.api import NotFoundError
from yggdrasil.node.config import Settings
from yggdrasil.node.api.services.fs import FsService

_EDITABLE_SUFFIXES = {".parquet", ".pq", ".arrow", ".feather", ".csv", ".json"}


class ColumnInfo(BaseModel):
    name: str
    dtype: str


class TabularInfo(BaseModel):
    path: str
    row_count: int
    columns: list[ColumnInfo]
    editable: bool
    fmt: str


class TabularService:
    def __init__(self, settings: Settings, fs: FsService) -> None:
        self.settings = settings
        self.fs = fs

    async def inspect(self, rel: str) -> TabularInfo:
        path = self.fs._resolve(rel)
        if not path.is_file():
            raise NotFoundError(f"No such file: {rel!r}.")
        suffix = path.suffix.lower()
        editable = suffix in _EDITABLE_SUFFIXES

        if suffix in (".parquet", ".pq"):
            meta = pq.read_metadata(str(path))
            schema = meta.schema.to_arrow_schema()
            return TabularInfo(
                path=rel,
                row_count=meta.num_rows,
                columns=[ColumnInfo(name=f.name, dtype=str(f.type)) for f in schema],
                editable=editable,
                fmt="parquet",
            )

        # Non-parquet: fall back to a bounded read through the io handlers.
        from yggdrasil.data.options import CastOptions
        from yggdrasil.path import Path as YggPath

        cap = self.settings.tabular_preview_max_rows
        with YggPath.from_(str(path)).open("rb") as bio:
            table = bio.read_arrow_table(options=CastOptions(row_limit=cap + 1))
        return TabularInfo(
            path=rel,
            row_count=table.num_rows,
            columns=[ColumnInfo(name=n, dtype=str(t)) for n, t in zip(table.schema.names, table.schema.types)],
            editable=editable,
            fmt=suffix.lstrip(".") or "unknown",
        )
