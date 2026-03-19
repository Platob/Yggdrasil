"""Schemas for Databricks SQL execution via the Excel router."""
from __future__ import annotations

from .common import StrictModel
from .python import DataFramePayload


class DatabricksSQLRequest(StrictModel):
    """Execute a SQL query on Databricks and return a DataFrame."""

    statement: str
    host: str | None = None
    token: str | None = None
    catalog_name: str | None = None
    schema_name: str | None = None
    warehouse_id: str | None = None
    warehouse_name: str | None = None
    max_rows: int | None = None
    df_name: str = "df"
    cache_ttl: int | None = None
    force_refresh: bool = False


class DatabricksSQLResponse(StrictModel):
    """Response containing the query result as a DataFrame payload."""

    ok: bool
    data: DataFramePayload
    row_count: int
    truncated: bool
    cache_hit: bool = False

