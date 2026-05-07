"""Catalog navigation — read-only walks over the registered tables.

Mirrors the three-level shape of :class:`TabularEngine`:

- ``GET /catalog`` — every catalog + flat list of qualified names
- ``GET /catalog/{catalog}`` — schemas inside a catalog
- ``GET /catalog/{catalog}/{schema}`` — tables inside a schema
- ``GET /catalog/{catalog}/{schema}/{name}`` — entry metadata
- ``GET /catalog/{catalog}/{schema}/{name}/schema`` — field-level
  schema as JSON

These endpoints intentionally don't move row data — that's
:mod:`yggdrasil.fastapi.routers.data`'s job. Splitting the two
keeps the metadata path cheap (no I/O against the underlying
source unless you actually ask for the schema, which then gets
cached on the entry).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from yggdrasil.io.tabular import TabularEngine

from ..deps import get_engine
from ..exceptions import NotFound
from ..schemas import (
    CatalogListing,
    EngineListing,
    FieldInfo,
    SchemaInfo,
    SchemaListing,
    TableEntry,
)


router = APIRouter(prefix="/catalog", tags=["catalog"])


def _entry_to_table(engine_entry) -> TableEntry:
    return TableEntry(
        catalog=engine_entry.catalog,
        schema=engine_entry.schema,
        name=engine_entry.name,
        qualified_name=engine_entry.qualified_name,
        tabular_class=type(engine_entry.tabular).__name__,
    )


@router.get("", response_model=EngineListing, summary="List catalogs and qualified names")
def list_engine(engine: TabularEngine = Depends(get_engine)) -> EngineListing:
    return EngineListing(
        catalogs=engine.catalogs(),
        qualified_names=engine.qualified_names(),
    )


@router.get(
    "/{catalog}",
    response_model=CatalogListing,
    summary="List schemas inside a catalog",
)
def list_catalog(
    catalog: str, engine: TabularEngine = Depends(get_engine),
) -> CatalogListing:
    schemas = engine.schemas(catalog=catalog)
    if not schemas:
        raise NotFound(
            f"Catalog {catalog!r} is empty or unknown. "
            f"Available catalogs: {engine.catalogs()!r}."
        )
    return CatalogListing(catalog=catalog, schemas=schemas)


@router.get(
    "/{catalog}/{schema}",
    response_model=SchemaListing,
    summary="List tables inside a schema",
)
def list_schema(
    catalog: str,
    schema: str,
    engine: TabularEngine = Depends(get_engine),
) -> SchemaListing:
    tables = engine.tables(catalog=catalog, schema=schema)
    if not tables:
        raise NotFound(
            f"Schema {catalog!r}.{schema!r} is empty or unknown. "
            f"Schemas in {catalog!r}: {engine.schemas(catalog=catalog)!r}."
        )
    return SchemaListing(catalog=catalog, schema=schema, tables=tables)


@router.get(
    "/{catalog}/{schema}/{name}",
    response_model=TableEntry,
    summary="Get a table entry",
)
def get_entry(
    catalog: str,
    schema: str,
    name: str,
    engine: TabularEngine = Depends(get_engine),
) -> TableEntry:
    entry = engine.get(catalog, schema, name)
    if entry is None:
        raise NotFound(
            f"No tabular registered as {catalog!r}.{schema!r}.{name!r}. "
            f"Registered: {engine.qualified_names()!r}."
        )
    return _entry_to_table(entry)


@router.get(
    "/{catalog}/{schema}/{name}/schema",
    response_model=SchemaInfo,
    summary="Inspect a table's field-level schema",
)
def get_schema(
    catalog: str,
    schema: str,
    name: str,
    engine: TabularEngine = Depends(get_engine),
) -> SchemaInfo:
    entry = engine.get(catalog, schema, name)
    if entry is None:
        raise NotFound(
            f"No tabular registered as {catalog!r}.{schema!r}.{name!r}. "
            f"Registered: {engine.qualified_names()!r}."
        )
    table_schema = entry.get_schema()
    fields = [
        FieldInfo(
            name=field.name,
            dtype=field.dtype.to_dict() if hasattr(field.dtype, "to_dict") else str(field.dtype),
            nullable=field.nullable,
            metadata=(
                {k.decode(): v.decode() for k, v in field.metadata.items()}
                if field.metadata else None
            ),
        )
        for field in table_schema.fields
    ]
    return SchemaInfo(
        catalog=catalog, schema=schema, name=name, fields=fields,
    )
