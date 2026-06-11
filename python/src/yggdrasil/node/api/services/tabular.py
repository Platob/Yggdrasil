"""Inspect tabular files without a full read.

``inspect`` runs on every file-open in the UI. For parquet the schema and exact
row count live in the footer, so ``pyarrow.parquet.read_metadata`` answers in
O(1) regardless of file size — no data pages are touched. Other formats fall
back to a bounded preview read (at most ``tabular_preview_max_rows`` rows).
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from pydantic import BaseModel

from .fs import FsService


class InspectResult(BaseModel):
    path: str
    row_count: int
    col_count: int
    schema_fields: list[dict]
    editable: bool
    size_bytes: int


def _schema_fields(schema: pa.Schema) -> list[dict]:
    return [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in schema]


class TabularService:
    def __init__(self, settings, fs: FsService) -> None:
        self.settings = settings
        self.fs = fs
        self._root = Path(settings.node_home)
        self._preview_rows = settings.tabular_preview_max_rows

    async def inspect(self, relative: str) -> InspectResult:
        target = self.fs._resolve(relative)
        if not target.is_file():
            raise FileNotFoundError(f"No such file {relative!r} under node_home.")
        size = target.stat().st_size
        suffix = target.suffix.lower()

        if suffix in (".parquet", ".pq"):
            md = pq.read_metadata(str(target))
            schema = md.schema.to_arrow_schema()
            return InspectResult(
                path=relative,
                row_count=md.num_rows,
                col_count=md.num_columns,
                schema_fields=_schema_fields(schema),
                editable=md.num_rows <= self._preview_rows,
                size_bytes=size,
            )

        if suffix in (".arrow", ".arrows", ".ipc", ".feather"):
            # IPC carries the schema in its header and per-batch row counts in
            # the footer/metadata — no need to materialize the columns.
            with pa.memory_map(str(target), "r") as src:
                reader = ipc.open_stream(src) if suffix == ".arrows" else ipc.open_file(src)
                schema = reader.schema
                if hasattr(reader, "num_record_batches"):
                    rows = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
                else:
                    rows = reader.read_all().num_rows
            return InspectResult(
                path=relative,
                row_count=rows,
                col_count=len(schema),
                schema_fields=_schema_fields(schema),
                editable=rows <= self._preview_rows,
                size_bytes=size,
            )

        raise ValueError(
            f"Can't inspect {relative!r}: unsupported extension {suffix!r}. "
            f"Expected one of .parquet/.pq/.arrow/.arrows/.ipc/.feather."
        )
