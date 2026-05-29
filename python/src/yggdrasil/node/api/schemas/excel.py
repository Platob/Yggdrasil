"""Schemas for the Excel-facing node service.

The Excel service is a thin façade over the node's existing capabilities
(filesystem, Python execution) shaped for how Excel / Power Query / an
Office.js add-in want to consume them: a navigation tree, file reads that
come back as typed tables, and a "run Python, get a table" call. Tabular
payloads are returned as Parquet (Power Query reads it natively) or Arrow
(the add-in decodes it in JS); these schemas cover the JSON-shaped
request/metadata around those binary bodies.
"""
from __future__ import annotations

from pydantic import Field

from .common import StrictModel

#: Tabular output encodings the Excel endpoints can emit.
TABLE_FORMATS = ("parquet", "arrow", "json")


class ExcelInfo(StrictModel):
    """Identity + capability card the connector/add-in reads on connect."""
    node_id: str
    node_name: str
    version: str
    table_formats: list[str] = Field(default_factory=lambda: list(TABLE_FORMATS))
    capabilities: list[str] = Field(default_factory=list)


class ExcelQueryRequest(StrictModel):
    """Run a Python snippet and return the named dataframe as a table."""
    code: str
    env: str | None = None              # PyEnv name; None → the node interpreter
    df_name: str = "df"                 # variable in the snippet to return
    packages: list[str] = Field(default_factory=list)
    max_rows: int | None = None
    timeout: float | None = None


class ExcelWriteResponse(StrictModel):
    path: str
    rows: int
    columns: int
    bytes_written: int


class ExcelTreeNode(StrictModel):
    path: str
    name: str
    is_dir: bool
    size: int = 0
    children: list["ExcelTreeNode"] = Field(default_factory=list)


class ExcelTreeResponse(StrictModel):
    node_id: str
    root: str
    tree: list[ExcelTreeNode] = Field(default_factory=list)
