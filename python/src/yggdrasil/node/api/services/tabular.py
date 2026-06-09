"""TabularService — schema + exact row count without reading rows.

For parquet, ``inspect`` reads the footer (``pq.read_metadata``): schema and
row count are O(1) there, vs the old path that pulled cap+1 rows just to size
the file. For non-parquet tabular formats it falls back to a bounded read.
``preview`` returns the first N rows as row-dicts for the grid.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pydantic import BaseModel


class InspectColumn(BaseModel):
    name: str
    type: str


class InspectResult(BaseModel):
    path: str
    format: str
    row_count: int
    column_count: int
    columns: list[InspectColumn]
    size_bytes: int
    editable: bool


class PreviewResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool


class TabularService:
    def __init__(self, settings: Any, fs: Any) -> None:
        self.settings = settings
        self.fs = fs
        self.root = Path(settings.node_home)

    def _resolve(self, rel: str) -> Path:
        target = (self.root / rel.lstrip("/")).resolve()
        root = self.root.resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"path {rel!r} escapes node home {root}.")
        return target

    async def inspect(self, path: str) -> InspectResult:
        target = self._resolve(path)
        size = target.stat().st_size
        suffix = target.suffix.lower().lstrip(".")

        if suffix in ("parquet", "pq"):
            md = pq.read_metadata(str(target))
            arrow_schema = md.schema.to_arrow_schema()
            cols = [InspectColumn(name=f.name, type=str(f.type)) for f in arrow_schema]
            return InspectResult(
                path=path, format="parquet", row_count=md.num_rows,
                column_count=len(cols), columns=cols, size_bytes=size,
                editable=md.num_rows <= self.settings.tabular_preview_max_rows,
            )

        # Non-parquet: bounded read to get schema; row count is best-effort.
        from yggdrasil.path import Path as YggPath
        from yggdrasil.data.options import CastOptions

        with YggPath.from_(str(target)).open("rb") as bio:
            tbl = bio.read_arrow_table(options=CastOptions(row_limit=self.settings.tabular_preview_max_rows + 1))
        cols = [InspectColumn(name=f.name, type=str(f.type)) for f in tbl.schema]
        return InspectResult(
            path=path, format=suffix or "unknown", row_count=tbl.num_rows,
            column_count=len(cols), columns=cols, size_bytes=size,
            editable=tbl.num_rows <= self.settings.tabular_preview_max_rows,
        )

    async def preview(self, path: str, limit: int | None = None) -> PreviewResult:
        target = self._resolve(path)
        cap = limit or self.settings.tabular_preview_max_rows
        suffix = target.suffix.lower().lstrip(".")

        if suffix in ("parquet", "pq"):
            import pyarrow as pa
            pf = pq.ParquetFile(str(target))
            total = pf.metadata.num_rows
            batch = next(pf.iter_batches(batch_size=cap), None)
            tbl = pa.Table.from_batches([batch]) if batch is not None else pf.schema_arrow.empty_table()
        else:
            from yggdrasil.path import Path as YggPath
            from yggdrasil.data.options import CastOptions
            with YggPath.from_(str(target)).open("rb") as bio:
                tbl = bio.read_arrow_table(options=CastOptions(row_limit=cap))
            total = tbl.num_rows

        columns = tbl.schema.names
        # Columnar -> row lists once, no per-cell Python loop over the table.
        col_data = [tbl.column(c).to_pylist() for c in columns]
        rows = [list(r) for r in zip(*col_data)] if col_data else []
        return PreviewResult(columns=columns, rows=rows, row_count=len(rows),
                             truncated=total > len(rows))
