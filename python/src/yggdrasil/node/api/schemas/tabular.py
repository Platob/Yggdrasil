from __future__ import annotations

from typing import Any

from .common import StrictModel


class TabularColumn(StrictModel):
    name: str
    type: str


class TabularInspect(StrictModel):
    node_id: str
    path: str
    source_url: str
    media_type: str
    is_tabular: bool
    columns: list[TabularColumn]
    column_count: int
    # Exact for parquet (read from the footer, free) and for any file that fits
    # under the preview cap; null only when a non-parquet file is too large to
    # count cheaply. Editing is gated on `editable`, not on this.
    row_count: int | None = None
    size_bytes: int = 0
    schema_hash: str = ""
    editable: bool = False
    schema_error: str | None = None


class TabularPreview(StrictModel):
    node_id: str
    path: str
    columns: list[TabularColumn]
    rows: list[list[Any]]
    row_count: int
    limit: int
    # True when the file has more rows than were returned — preview only.
    truncated: bool = False


class TabularWriteRequest(StrictModel):
    path: str
    columns: list[str]
    rows: list[list[Any]]
    fmt: str | None = None


class TabularWriteResponse(StrictModel):
    path: str
    rows: int
    columns: int
    bytes_written: int


# -- Workbook (xlsx) ---------------------------------------------------------

class WorkbookSheet(StrictModel):
    name: str
    rows: int
    cols: int
    visible: bool = True


class WorkbookSheets(StrictModel):
    node_id: str
    path: str
    sheets: list[WorkbookSheet]


class WorkbookEditRequest(StrictModel):
    path: str
    sheet: str
    create: bool = False
    # A batch of (row, col, value) cell edits (1-based)...
    cells: list[tuple[int, int, Any]] | None = None
    # ...or a rectangular block written from (start_row, start_col) (1-based).
    start_row: int | None = None
    start_col: int | None = None
    values: list[list[Any]] | None = None


class WorkbookEditResponse(StrictModel):
    path: str
    sheet: str
    cells_written: int
