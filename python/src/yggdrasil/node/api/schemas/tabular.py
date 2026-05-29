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
    # Exact when the whole file fits under the preview cap; null when it's too
    # large to count cheaply (in which case the editor stays read-only).
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
