"""Tabular file inspection contracts: inspect, preview."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ColumnMeta(BaseModel):
    name: str
    type: str
    nullable: bool = True


class TabularInspect(BaseModel):
    path: str
    format: str
    row_count: int
    columns: list[ColumnMeta]
    editable: bool = False
    size_bytes: int = 0


class TabularPreview(BaseModel):
    path: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool = False
