"""Request models for the Saga catalog + SQL endpoints.

``schema`` is a reserved attribute name on pydantic ``BaseModel``, so the field
is declared as ``schema_`` with an alias of ``schema`` and ``populate_by_name``
on — callers may pass either ``schema=`` (the wire / kwarg shape used by the
benches) or ``schema_=``, and ``.schema_`` reads it back.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CatalogCreate(BaseModel):
    name: str
    comment: str | None = None


class SchemaCreate(BaseModel):
    name: str
    comment: str | None = None


class TableCreate(BaseModel):
    name: str
    source_url: str
    infer: bool = False
    comment: str | None = None


class SqlRequest(BaseModel):
    sql: str
    limit: int | None = None


class ForecastSpec(BaseModel):
    source: str
    column: str
    x: str
    keys: list[str] = Field(default_factory=list)
    horizon: int = 24
    model: Literal["auto", "ridge", "gbr", "xgboost"] = "auto"
    period: int | None = None
    materialized: bool = True


class ForecastRegisterRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    catalog: str
    schema_: str = Field(default="public", alias="schema")
    name: str
    spec: ForecastSpec
    materialize: bool = True
