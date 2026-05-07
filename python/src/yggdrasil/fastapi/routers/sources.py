"""Source registration — bind data into ``catalog.schema.name`` slots.

Three registration shapes:

1. **Path** (JSON body) — register a Tabular pointing at a path or
   URL. Anything :meth:`yggdrasil.io.path.Path.from_` accepts works
   (``file://``, ``s3://``, ``dbfs://``, …); the format is sniffed
   from the extension or the optional ``media_type`` field.
2. **Inline rows / columns** (JSON body) — register a small Arrow
   table built from rows or columns embedded in the request.
3. **Binary upload** — POST the raw bytes of a tabular file
   (Arrow IPC stream / file, parquet, csv, json, ndjson, xlsx, …).
   The format comes from the ``Content-Type`` header (or the
   ``?media_type=`` override). Decode goes through
   :meth:`yggdrasil.io.bytes_io.BytesIO.as_media`, so any new
   :class:`Tabular` leaf registered against a mime is upload-ready
   for free, and ``Content-Encoding`` codecs (gzip / zstd / …)
   round-trip via :class:`BytesIO`'s ``_format_view``.

Removal:

- ``DELETE /sources/{catalog}/{schema}/{name}`` drops the entry.
"""

from __future__ import annotations

import pyarrow as pa
from fastapi import APIRouter, Depends, Request

from yggdrasil.data.enums import MediaType
from yggdrasil.io.tabular import ArrowTabular, Tabular, TabularEngine

from ..config import Settings
from ..deps import get_engine, get_settings
from ..exceptions import APIError, NotFound
from ..payloads import decode_payload, extract_wire_mime, read_capped_body
from ..schemas import (
    RegisterInlineRequest,
    RegisterPathRequest,
    RegisterResult,
)


router = APIRouter(prefix="/sources", tags=["sources"])


def _entry_to_result(entry, *, rows: "int | None") -> RegisterResult:
    schema = entry.get_schema()
    return RegisterResult(
        catalog=entry.catalog,
        schema=entry.schema,
        name=entry.name,
        qualified_name=entry.qualified_name,
        rows=rows,
        field_count=len(schema.fields),
    )


def _require_entry(engine: TabularEngine, catalog: str, schema: str, name: str):
    entry = engine.get(catalog, schema, name)
    if entry is None:
        raise NotFound(
            f"No tabular registered as {catalog!r}.{schema!r}.{name!r}. "
            f"Registered: {engine.qualified_names()!r}."
        )
    return entry


# ---------------------------------------------------------------------------
# Read — listing + per-entry metadata
# ---------------------------------------------------------------------------


@router.get(
    "",
    summary="List every registered source",
)
def list_sources(
    catalog: "str | None" = None,
    schema: "str | None" = None,
    engine: TabularEngine = Depends(get_engine),
) -> "list[dict]":
    """Return every registered source as a list of metadata dicts.

    Optional ``?catalog=`` / ``?schema=`` filters scope the result.
    """
    return [
        {
            "catalog": e.catalog,
            "schema": e.schema,
            "name": e.name,
            "qualified_name": e.qualified_name,
            "tabular_class": type(e.tabular).__name__,
        }
        for e in engine.entries(catalog=catalog, schema=schema)
    ]


@router.get(
    "/{catalog}/{schema}/{name}",
    response_model=RegisterResult,
    summary="Get a registered source's metadata",
)
def get_source(
    catalog: str,
    schema: str,
    name: str,
    engine: TabularEngine = Depends(get_engine),
) -> RegisterResult:
    entry = _require_entry(engine, catalog, schema, name)
    return _entry_to_result(entry, rows=None)


@router.post(
    "/{catalog}/{schema}/{name}/path",
    response_model=RegisterResult,
    summary="Register a path-based source (file:// / s3:// / …)",
)
def register_path(
    catalog: str,
    schema: str,
    name: str,
    body: RegisterPathRequest,
    engine: TabularEngine = Depends(get_engine),
) -> RegisterResult:
    from yggdrasil.io.path import Path

    try:
        path = Path.from_(body.path)
    except Exception as exc:  # noqa: BLE001 — re-raise as APIError
        raise APIError(
            f"Could not resolve path {body.path!r}: {exc}. "
            "Pass a local path, a URL string, or any value accepted by "
            "yggdrasil.io.path.Path.from_."
        ) from exc

    media_type = (
        MediaType.from_(body.media_type, default=None) if body.media_type else None
    )
    tabular = Tabular.for_holder(path, media_type=media_type)
    entry = engine.register(catalog, schema, name, tabular)
    return _entry_to_result(entry, rows=None)


@router.post(
    "/{catalog}/{schema}/{name}/inline",
    response_model=RegisterResult,
    summary="Register an in-memory source from JSON rows or columns",
)
def register_inline(
    catalog: str,
    schema: str,
    name: str,
    body: RegisterInlineRequest,
    engine: TabularEngine = Depends(get_engine),
) -> RegisterResult:
    if body.rows is not None and body.columns is not None:
        raise APIError(
            "Pass exactly one of `rows` or `columns`, not both. "
            "If they describe the same data, drop one; "
            "if they describe different data, send two requests."
        )
    if body.rows is None and body.columns is None:
        raise APIError(
            "Inline registration needs either `rows` (list of dicts) or "
            "`columns` (dict of name → list). Got an empty body."
        )

    if body.rows is not None:
        table = pa.Table.from_pylist(body.rows)
    else:
        table = pa.table(body.columns or {})

    entry = engine.register(catalog, schema, name, ArrowTabular(table))
    return _entry_to_result(entry, rows=table.num_rows)


@router.post(
    "/{catalog}/{schema}/{name}/upload",
    response_model=RegisterResult,
    summary="Register a binary tabular upload (Arrow IPC / parquet / csv / …)",
)
async def register_upload(
    catalog: str,
    schema: str,
    name: str,
    request: Request,
    media_type: "str | None" = None,
    engine: TabularEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
) -> RegisterResult:
    payload = await read_capped_body(request, settings.max_request_bytes)
    if not payload:
        raise APIError(
            "Empty upload. Send the tabular bytes in the request body and "
            "set Content-Type to the format mime (or pass ?media_type=)."
        )

    table, rows = decode_payload(
        payload,
        extract_wire_mime(request, override=media_type),
        content_encoding=request.headers.get("content-encoding"),
    )
    entry = engine.register(catalog, schema, name, ArrowTabular(table))
    return _entry_to_result(entry, rows=rows)


@router.delete(
    "/{catalog}/{schema}/{name}",
    summary="Deregister a source",
    status_code=204,
)
def deregister(
    catalog: str,
    schema: str,
    name: str,
    engine: TabularEngine = Depends(get_engine),
) -> None:
    removed = engine.deregister(catalog, schema, name)
    if removed is None:
        raise NotFound(
            f"No tabular registered as {catalog!r}.{schema!r}.{name!r}. "
            f"Registered: {engine.qualified_names()!r}."
        )


# ---------------------------------------------------------------------------
# Bulk delete
# ---------------------------------------------------------------------------


@router.delete(
    "",
    summary="Deregister every source (optionally scoped)",
)
def clear_sources(
    catalog: "str | None" = None,
    schema: "str | None" = None,
    engine: TabularEngine = Depends(get_engine),
) -> "dict[str, int]":
    """Drop every registration, optionally scoped to a catalog / schema.

    Returns ``{"removed": <count>}`` so callers know what was wiped.
    """
    removed = 0
    for entry in engine.entries(catalog=catalog, schema=schema):
        if engine.deregister(entry.catalog, entry.schema, entry.name) is not None:
            removed += 1
    return {"removed": removed}


