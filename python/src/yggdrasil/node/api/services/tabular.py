"""Tabular inspection service.

Inspect is hit on every file open in the UI; it needs schema, an exact row
count, and an editable flag. For parquet all of that lives in the footer
(O(1)), so we read the metadata and never touch a single data row. CSV/JSON
are sampled for schema inference.
"""
from __future__ import annotations

from ...config import Settings
from ..schemas.tabular import TabularInspectResult
from .fs import FsService

_PARQUET_SUFFIXES = {".parquet", ".pq"}
_DELIMITED_SUFFIXES = {".csv", ".tsv", ".txt"}
_JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}


class TabularService:
    def __init__(self, settings: Settings, fs: FsService) -> None:
        self.settings = settings
        self.fs = fs

    async def inspect(self, path: str) -> TabularInspectResult:
        target = self.fs._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"no such file: {path!r}")
        suffix = target.suffix.lower()

        if suffix in _PARQUET_SUFFIXES:
            import pyarrow.parquet as pq

            # Footer only — never reads data rows. num_rows + schema are exact.
            meta = pq.read_metadata(str(target))
            schema = meta.schema.to_arrow_schema()
            return TabularInspectResult(
                path=str(path),
                row_count=meta.num_rows,
                col_count=len(schema),
                schema=[{"name": f.name, "type": str(f.type)} for f in schema],
                editable=False,
                format="parquet",
            )

        if suffix in _DELIMITED_SUFFIXES:
            import pyarrow.csv as pacsv

            delimiter = "\t" if suffix == ".tsv" else ","
            table = pacsv.read_csv(
                str(target),
                parse_options=pacsv.ParseOptions(delimiter=delimiter),
                read_options=pacsv.ReadOptions(block_size=1 << 20),
            )
            sample = table.slice(0, 1000)
            return TabularInspectResult(
                path=str(path),
                row_count=table.num_rows,
                col_count=table.num_columns,
                schema=[{"name": f.name, "type": str(f.type)} for f in sample.schema],
                editable=True,
                format="csv" if suffix != ".tsv" else "tsv",
            )

        if suffix in _JSON_SUFFIXES:
            import pyarrow.json as pajson

            table = pajson.read_json(str(target))
            return TabularInspectResult(
                path=str(path),
                row_count=table.num_rows,
                col_count=table.num_columns,
                schema=[{"name": f.name, "type": str(f.type)} for f in table.schema],
                editable=True,
                format="json",
            )

        raise ValueError(
            f"unsupported tabular format {suffix!r} for {path!r}. "
            f"Supported: parquet, csv, tsv, json."
        )
