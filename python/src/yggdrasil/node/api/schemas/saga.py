"""Saga SQL-catalog contracts: catalog / schema / table CRUD, SQL execution,
and the FORECAST asset (a registered, optionally-materialised forecast that
queries like a table).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _SagaModel(BaseModel):
    # ``schema`` is a reserved-ish word and clashes with pydantic internals,
    # so the field is ``schema_`` but the wire/JSON name is ``schema``.
    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# catalog / schema
# ---------------------------------------------------------------------------

class CatalogCreate(BaseModel):
    name: str
    comment: str = ""


class CatalogInfo(BaseModel):
    name: str
    comment: str = ""
    schema_count: int = 0


class CatalogResult(BaseModel):
    catalog: CatalogInfo


class SchemaCreate(BaseModel):
    name: str
    comment: str = ""


class SchemaInfo(_SagaModel):
    name: str
    catalog: str
    comment: str = ""
    table_count: int = 0


class SchemaResult(_SagaModel):
    schema_: SchemaInfo = Field(alias="schema")


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------

class ColumnInfo(BaseModel):
    name: str
    type: str


class TableStatistics(BaseModel):
    row_count: int = 0
    size_bytes: int = 0


class TableInfo(_SagaModel):
    name: str
    catalog: str
    schema_: str = Field(alias="schema")
    source_url: str
    kind: str = "TABLE"  # "TABLE" | "FORECAST"
    columns: list[ColumnInfo] = []
    statistics: TableStatistics = TableStatistics()


class TableCreate(BaseModel):
    name: str
    source_url: str
    infer: bool = True


class TableResult(BaseModel):
    table: TableInfo


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

class SqlRequest(_SagaModel):
    sql: str
    catalog: str | None = None
    schema_: str | None = Field(default=None, alias="schema")
    limit: int | None = None


class SqlResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    node_id: str = ""


# ---------------------------------------------------------------------------
# forecast asset
# ---------------------------------------------------------------------------

class ForecastSpec(BaseModel):
    source: str
    column: str
    x: str
    keys: list[str] = []
    horizon: int = 24
    model: str = "auto"  # "auto" | "ridge" | "gbr" | "xgboost"
    period: int = 24
    materialized: bool = True


class ForecastRegisterRequest(_SagaModel):
    catalog: str
    schema_: str = Field(alias="schema")
    name: str
    spec: ForecastSpec
    materialize: bool = True


class ForecastRegisterResult(BaseModel):
    model_used: str
    rows: int
    materialized: bool
