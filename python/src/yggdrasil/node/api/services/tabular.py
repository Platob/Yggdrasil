"""Tabular inspect / preview / bounded-edit over the LazyTabular IO layer.

Reuses the same ``YggPath.from_(path).open(...).read_arrow_table()`` dispatch
the Excel service relies on, so parquet/csv/json/arrow/xlsx all decode through
one typed path. Reads are bounded by ``tabular_preview_max_rows`` — a file at
or under that count is read whole (and is therefore safe to edit and save back
in place), anything bigger stays a read-only preview.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import itertools
from decimal import Decimal
from functools import partial

import pyarrow as pa
from fastapi.concurrency import run_in_threadpool

from yggdrasil.data.options import CastOptions
from yggdrasil.enums.media_type import MediaType
from yggdrasil.exceptions.api import BadRequestError
from yggdrasil.path import Path as YggPath

from ...config import Settings
from ...exceptions import ForbiddenError, NotFoundError
from ..schemas.tabular import (
    TabularColumn,
    TabularInspect,
    TabularPreview,
    TabularWriteRequest,
    TabularWriteResponse,
)
from .fs import FsService

TABULAR_EXTS = {"csv", "parquet", "pq", "json", "ndjson", "arrow", "feather", "xlsx", "xls"}


class TabularService:
    def __init__(self, settings: Settings, *, fs: FsService) -> None:
        self.settings = settings
        self.fs = fs

    async def inspect(self, path: str) -> TabularInspect:
        return await run_in_threadpool(partial(self._inspect, path))

    async def preview(self, path: str, limit: int, offset: int = 0) -> TabularPreview:
        return await run_in_threadpool(partial(self._preview, path, limit, offset))

    async def write(self, req: TabularWriteRequest) -> TabularWriteResponse:
        return await run_in_threadpool(partial(self._write, req))

    # -- Arrow IPC preview --------------------------------------------------

    async def preview_arrow(self, path: str, limit: int, offset: int = 0) -> bytes:
        """Bounded, paged preview as an Arrow IPC stream — no JSON/Python row
        materialization. ``offset`` reads a row window so the grid can page."""
        return await run_in_threadpool(partial(self._preview_arrow, path, limit, offset))

    def _preview_arrow(self, path: str, limit: int, offset: int = 0) -> bytes:
        from ... import transport
        resolved = self.fs._resolve(path)
        if not resolved.exists() or resolved.is_dir():
            raise NotFoundError(f"File not found: {path!r}")
        limit = max(1, min(limit, self.settings.tabular_preview_max_rows))
        offset = max(0, offset)
        try:
            with YggPath.from_(str(resolved)).open("rb") as bio:
                table = bio.read_arrow_table(options=CastOptions(row_limit=offset + limit))
        except Exception as exc:
            raise BadRequestError(f"Cannot read {path!r} as a table: {exc}")
        if offset:
            table = table.slice(offset, limit)
        return transport.write_arrow_stream_bytes(table)

    # -- Workbook (xlsx) ----------------------------------------------------

    async def workbook_sheets(self, path: str):
        return await run_in_threadpool(partial(self._workbook_sheets, path))

    async def read_sheet_arrow(
        self, path: str, sheet: str | None, *,
        header: bool, skip_rows: int, n_rows: int | None, columns: list[str] | None,
    ) -> bytes:
        return await run_in_threadpool(partial(
            self._read_sheet_arrow, path, sheet, header, skip_rows, n_rows, columns,
        ))

    async def edit_workbook(self, req) -> int:
        return await run_in_threadpool(partial(self._edit_workbook, req))

    def _resolve_xlsx(self, path: str):
        resolved = self.fs._resolve(path)
        if not resolved.exists() or resolved.is_dir():
            raise NotFoundError(f"File not found: {path!r}")
        if resolved.suffix.lstrip(".").lower() not in ("xlsx", "xls"):
            raise BadRequestError(f"Not an Excel workbook: {path!r}")
        return resolved

    def _workbook_sheets(self, path: str):
        resolved = self._resolve_xlsx(path)
        with YggPath.from_(str(resolved)).open("rb") as wb:
            return wb.sheet_infos()

    def _read_sheet_arrow(self, path, sheet, header, skip_rows, n_rows, columns) -> bytes:
        from ... import transport
        resolved = self._resolve_xlsx(path)
        with YggPath.from_(str(resolved)).open("rb") as wb:
            table = wb.read_range(
                sheet, header=header, skip_rows=skip_rows, n_rows=n_rows, columns=columns,
            )
        return transport.write_arrow_stream_bytes(table)

    def _edit_workbook(self, req) -> int:
        resolved = self._resolve_xlsx(req.path)
        with YggPath.from_(str(resolved)).open("rb") as wb:
            if req.values is not None:
                if req.start_row is None or req.start_col is None:
                    raise BadRequestError("range edit needs start_row and start_col")
                return wb.edit_range(
                    req.sheet, req.start_row, req.start_col, req.values, create=req.create,
                )
            if not req.cells:
                raise BadRequestError("provide either `cells` or `values`+start_row/col")
            return wb.apply_edits(req.sheet, req.cells, create=req.create)

    # -- sync workers -------------------------------------------------------

    def _inspect(self, path: str) -> TabularInspect:
        resolved = self.fs._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"File not found: {path!r}")
        if resolved.is_dir():
            raise ForbiddenError(f"Not a file: {path!r}")

        # Honor compression wrappers (``data.csv.gz``, ``trades.parquet.zst``):
        # strip a trailing codec suffix so the *inner* format decides whether
        # the file is tabular — the IO layer decompresses transparently on read.
        suffixes = [s.lower() for s in resolved.name.split(".")[1:]]
        ext = suffixes[-1] if suffixes else ""
        codec_media = MediaType.from_(ext, default=None) if ext else None
        compressed = bool(codec_media and codec_media.codec)
        if compressed and len(suffixes) >= 2:
            ext = suffixes[-2]
        is_tabular = ext in TABULAR_EXTS
        media = MediaType.from_(resolved, default=None)
        cap = self.settings.tabular_preview_max_rows
        columns: list[TabularColumn] = []
        row_count: int | None = None
        schema_hash = ""
        editable = False
        schema_error: str | None = None

        if is_tabular:
            try:
                if ext in ("parquet", "pq") and not compressed:
                    # Parquet carries schema + exact row count in its footer —
                    # an O(1) read. No need to pull cap+1 rows just to size the
                    # file, so even a multi-GB parquet inspects instantly and
                    # still reports its true row_count. (An externally-compressed
                    # parquet falls to the codec-aware read below instead.)
                    import pyarrow.parquet as _pq
                    pf = _pq.ParquetFile(str(resolved))
                    arrow_schema = pf.schema_arrow
                    row_count = pf.metadata.num_rows
                    editable = row_count <= cap
                else:
                    # No cheap metadata (csv/json/…): one bounded read gives both
                    # the schema and whether the whole file fits the editor cap.
                    with YggPath.from_(str(resolved)).open("rb") as bio:
                        table = bio.read_arrow_table(options=CastOptions(row_limit=cap + 1))
                    arrow_schema = table.schema
                    if table.num_rows <= cap:
                        row_count = table.num_rows
                        editable = True
                columns = [TabularColumn(name=f.name, type=str(f.type)) for f in arrow_schema]
                schema_hash = hashlib.sha256(
                    "|".join(f"{f.name}:{f.type}" for f in arrow_schema).encode()
                ).hexdigest()[:16]
            except Exception as exc:  # corrupt / unsupported encoding
                schema_error = str(exc)

        return TabularInspect(
            node_id=self.settings.node_id,
            path=path,
            source_url=resolved.as_uri(),
            media_type=media.mime_type.value if media else "application/octet-stream",
            is_tabular=is_tabular,
            columns=columns,
            column_count=len(columns),
            row_count=row_count,
            size_bytes=resolved.stat().st_size,
            schema_hash=schema_hash,
            editable=editable,
            schema_error=schema_error,
        )

    def _preview(self, path: str, limit: int, offset: int = 0) -> TabularPreview:
        resolved = self.fs._resolve(path)
        if not resolved.exists() or resolved.is_dir():
            raise NotFoundError(f"File not found: {path!r}")
        limit = max(1, min(limit, self.settings.tabular_preview_max_rows))
        offset = max(0, offset)

        try:
            with YggPath.from_(str(resolved)).open("rb") as bio:
                table = bio.read_arrow_table(options=CastOptions(row_limit=offset + limit + 1))
        except Exception as exc:
            raise BadRequestError(f"Cannot read {path!r} as a table: {exc}")

        truncated = table.num_rows > offset + limit
        table = table.slice(offset, limit)

        # Build rows column-wise: one to_pylist per column (vectorized) then a
        # zip-transpose, and skip the per-cell coercion entirely for columns
        # whose Arrow type is already JSON-safe. Faster than to_pylist's
        # row-dicts + a blanket per-cell loop, same output.
        column_values = []
        for field, column in zip(table.schema, table.columns):
            values = column.to_pylist()
            if not _is_json_safe_type(field.type):
                values = [_json_safe(v) for v in values]
            column_values.append(values)
        rows = [list(r) for r in zip(*column_values)] if column_values else []
        return TabularPreview(
            node_id=self.settings.node_id,
            path=path,
            columns=[TabularColumn(name=f.name, type=str(f.type)) for f in table.schema],
            rows=rows,
            row_count=table.num_rows,
            limit=limit,
            truncated=truncated,
        )

    def _write(self, req: TabularWriteRequest) -> TabularWriteResponse:
        cap = self.settings.tabular_preview_max_rows
        if len(req.rows) > cap:
            raise BadRequestError(
                f"Refusing to write {len(req.rows)} rows — the bounded editor "
                f"caps at {cap}. Edit large files through a PyFunc instead."
            )
        if not req.columns:
            raise BadRequestError("No columns to write")

        resolved = self.fs._resolve(req.path)
        # Transpose the row-major edit grid to columns in one pass (zip_longest
        # pads ragged rows with None) rather than re-scanning the rows once per
        # column.
        columns_data = list(itertools.zip_longest(*req.rows, fillvalue=None))
        table = pa.table({
            col: pa.array(columns_data[i]) if i < len(columns_data) else pa.array([None] * len(req.rows))
            for i, col in enumerate(req.columns)
        })
        # Best-effort: cast back to the file's existing column types so a
        # round-trip through the string-cell editor keeps ints as ints etc.
        # The footer schema is enough — no need to read a row for it.
        if resolved.exists():
            try:
                with YggPath.from_(str(resolved)).open("rb") as bio:
                    orig = bio.collect_schema().to_arrow_schema()
                table = table.cast(orig)
            except Exception:
                pass

        target = (
            MediaType.from_(req.fmt or resolved.suffix.lstrip(".") or "parquet", default=None)
            or MediaType.from_("parquet")
        )
        with YggPath.from_(str(resolved)).open("wb", media_type=target) as bio:
            bio.write_arrow_table(table)
        return TabularWriteResponse(
            path=req.path,
            rows=table.num_rows,
            columns=table.num_columns,
            bytes_written=resolved.stat().st_size,
        )


def _is_json_safe_type(t) -> bool:
    """True when an Arrow column's values are already JSON-serializable, so the
    preview can skip per-cell coercion for it."""
    return (
        pa.types.is_integer(t) or pa.types.is_floating(t)
        or pa.types.is_string(t) or pa.types.is_large_string(t)
        or pa.types.is_boolean(t) or pa.types.is_null(t)
    )


def _json_safe(value):
    """Coerce an Arrow cell to something the JSON response can carry."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dt.date, dt.datetime, dt.time, Decimal)):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
