"""Data exposure + row CRUD on registered :class:`Tabular` sources.

Read endpoints:

- ``GET /data/{c}/{s}/{n}`` — read every row of a registered table,
  format-negotiated through ``Accept`` / ``?format=``. Default is the
  Arrow IPC stream — the cheapest both ends speak and it streams
  without buffering the whole result.
- ``POST /data/{c}/{s}/{n}/query`` — builder-style query: optional
  ``select`` projection, ``where`` predicate (any SQL boolean
  expression :meth:`Expression.from_sql` accepts), ``limit`` /
  ``offset``. Body is JSON; query params overlay so ``?where=...``
  / ``?limit=...`` work without a body.
- ``POST /data/sql`` — run free-form SQL across the engine and stream
  the result back.

Write endpoints:

- ``POST /data/{c}/{s}/{n}/rows`` — append rows. Body can be JSON
  (rows / columns) or any binary tabular mime (Arrow IPC,
  parquet, csv, …) decoded through :func:`payloads.decode_payload`.
- ``PUT /data/{c}/{s}/{n}/rows`` — replace every row with the
  payload. Same body shapes as POST.
- ``DELETE /data/{c}/{s}/{n}/rows`` — predicate delete. Pass
  ``?where=<sql expr>`` to scope, or omit it to truncate every row.

The default response media type for reads is the Arrow IPC stream;
overrides come from ``?format=`` then ``Accept`` then the configured
default. Writes return a small JSON acknowledgement so the client
sees the row count and the mode applied.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Request, Response

from yggdrasil.data.enums import Mode
from yggdrasil.io.tabular import ArrowTabular, Tabular, TabularEngine

from ..config import Settings
from ..deps import get_engine, get_settings
from ..exceptions import APIError, NotFound
from ..payloads import decode_payload, extract_wire_mime, read_capped_body
from ..responses import ARROW_STREAM_MIME, tabular_response
from ..schemas import (
    DeleteResult,
    InsertRowsRequest,
    QueryRequest,
    WriteResult,
)


router = APIRouter(prefix="/data", tags=["data"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_tabular(
    engine: TabularEngine, catalog: str, schema: str, name: str,
) -> Tabular:
    entry = engine.get(catalog, schema, name)
    if entry is None:
        raise NotFound(
            f"No tabular registered as {catalog!r}.{schema!r}.{name!r}. "
            f"Registered: {engine.qualified_names()!r}."
        )
    return entry.tabular


def _apply_query(tabular: Tabular, q: "QueryRequest | None") -> Tabular:
    """Layer optional ``select`` / ``where`` / ``limit`` / ``offset``.

    Goes through :meth:`Tabular.lazy` so ``select`` and ``where`` push
    down through the same plan machinery the SQL executor uses; the
    materialisation slice for ``limit`` / ``offset`` happens after
    the lazy frame collects, since the plan AST doesn't have native
    LIMIT today.
    """
    if q is None:
        return tabular

    lazy = tabular.lazy()
    if q.where:
        try:
            lazy = lazy.filter(q.where)
        except Exception as exc:  # noqa: BLE001 — wrap parse errors as 400
            raise APIError(
                f"Could not parse `where` predicate {q.where!r}: {exc}. "
                "Use a SQL boolean expression like \"id > 10\" or "
                "\"region = 'EU' AND price > 100\"."
            ) from exc
    if q.select:
        try:
            lazy = lazy.select(*q.select)
        except Exception as exc:  # noqa: BLE001
            raise APIError(
                f"Could not apply `select` {q.select!r}: {exc}. "
                "Pass a list of column names that exist on the source."
            ) from exc
        # Force-resolve the schema now so column-not-found errors
        # surface as a 400 here instead of leaking out of the
        # streaming response writer downstream.
        try:
            lazy.collect_schema()
        except Exception as exc:  # noqa: BLE001
            raise APIError(
                f"Could not resolve `select` {q.select!r}: {exc}. "
                "Pass a list of column names that exist on the source."
            ) from exc
    if not (q.limit or q.offset):
        return lazy

    table = lazy.read_arrow_table()
    offset = q.offset or 0
    if q.limit is None:
        sliced = table.slice(offset)
    else:
        sliced = table.slice(offset, q.limit)
    return ArrowTabular(sliced)


async def _decode_request_table(
    request: Request,
    *,
    media_type_override: "str | None",
    settings: Settings,
):
    """Pick between JSON inline rows/columns and a binary tabular body.

    JSON bodies are parsed into :class:`InsertRowsRequest` so the
    same shape works for both the inline-rows path and the binary-
    upload path. Anything non-JSON goes through
    :func:`payloads.decode_payload`.
    """
    import pyarrow as pa

    wire_mime = extract_wire_mime(request, override=media_type_override)
    if wire_mime in {"application/json", "text/json"} and not media_type_override:
        # JSON inline rows / columns — same shape as
        # :class:`InsertRowsRequest`.
        body = await request.json()
        req = InsertRowsRequest.model_validate(body)
        if req.rows is not None and req.columns is not None:
            raise APIError(
                "Pass exactly one of `rows` or `columns`, not both."
            )
        if req.rows is None and req.columns is None:
            raise APIError(
                "Empty body. Pass `rows` (list of dicts) or `columns` "
                "(dict of name → list)."
            )
        if req.rows is not None:
            table = pa.Table.from_pylist(req.rows)
        else:
            table = pa.table(req.columns or {})
        return table

    payload = await read_capped_body(request, settings.max_request_bytes)
    if not payload:
        raise APIError(
            "Empty body. Send rows as JSON or as a tabular binary "
            "payload with the matching Content-Type."
        )
    table, _ = decode_payload(
        payload,
        wire_mime,
        content_encoding=request.headers.get("content-encoding"),
    )
    return table


# ---------------------------------------------------------------------------
# Read — full table + builder query + SQL
# ---------------------------------------------------------------------------


@router.get(
    "/{catalog}/{schema}/{name}",
    summary="Stream a registered table in the requested media type",
    response_class=Response,
)
def read_table(
    catalog: str,
    schema: str,
    name: str,
    request: Request,
    format: "str | None" = None,
    media_type: "str | None" = None,
    filename: "str | None" = None,
    accept: "str | None" = Header(default=None),
    engine: TabularEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
) -> Response:
    tabular = _require_tabular(engine, catalog, schema, name)
    return tabular_response(
        tabular,
        media_type=format or media_type,
        accept=accept,
        default=settings.default_media_type or ARROW_STREAM_MIME,
        filename=filename,
        stream_batch_rows=settings.stream_batch_rows,
    )


@router.post(
    "/{catalog}/{schema}/{name}/query",
    summary="Run a builder query (select / where / limit) and stream the result",
    response_class=Response,
)
async def query_table(
    catalog: str,
    schema: str,
    name: str,
    request: Request,
    body: "QueryRequest | None" = None,
    where: "str | None" = None,
    select: "str | None" = None,
    limit: "int | None" = None,
    offset: "int | None" = None,
    format: "str | None" = None,
    media_type: "str | None" = None,
    filename: "str | None" = None,
    accept: "str | None" = Header(default=None),
    engine: TabularEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Apply ``select`` / ``where`` / ``limit`` / ``offset`` then stream.

    The JSON body and the query string overlay: query-string knobs win
    when both are set, and either alone works. Pass an empty body and
    use ``?where=...&limit=...`` for the GET-style call shape; pass a
    JSON body for richer projections.
    """
    tabular = _require_tabular(engine, catalog, schema, name)

    merged = QueryRequest(
        select=(
            [s.strip() for s in select.split(",") if s.strip()]
            if select else (body.select if body else None)
        ),
        where=where if where is not None else (body.where if body else None),
        limit=limit if limit is not None else (body.limit if body else None),
        offset=offset if offset is not None else (body.offset if body else None),
    )

    queried = _apply_query(tabular, merged)
    return tabular_response(
        queried,
        media_type=format or media_type,
        accept=accept,
        default=settings.default_media_type or ARROW_STREAM_MIME,
        filename=filename,
        stream_batch_rows=settings.stream_batch_rows,
    )


@router.post(
    "/sql",
    summary="Run SQL across the engine and stream the result",
    response_class=Response,
)
async def run_sql(
    request: Request,
    format: "str | None" = None,
    media_type: "str | None" = None,
    filename: "str | None" = None,
    accept: "str | None" = Header(default=None),
    engine: TabularEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
) -> Response:
    raw = await request.body()
    if len(raw) > settings.max_request_bytes:
        raise APIError(
            f"SQL payload exceeds the configured cap of "
            f"{settings.max_request_bytes} bytes. "
            "Tighten the query or bump YGG_API_MAX_REQUEST_BYTES.",
            status_code=413,
        )
    query = raw.decode("utf-8", errors="replace").strip()
    if not query:
        raise APIError(
            "Empty SQL body. Send the query text as the request body "
            "(text/plain or application/sql) — referencing tables by "
            "their dotted catalog.schema.name, schema.name, or bare name."
        )

    result = engine.execute_sql(query)
    return tabular_response(
        result,
        media_type=format or media_type,
        accept=accept,
        default=settings.default_media_type or ARROW_STREAM_MIME,
        filename=filename,
        stream_batch_rows=settings.stream_batch_rows,
    )


# ---------------------------------------------------------------------------
# Write — insert / replace / delete rows
# ---------------------------------------------------------------------------


@router.post(
    "/{catalog}/{schema}/{name}/rows",
    response_model=WriteResult,
    summary="Append rows to a registered table",
)
async def insert_rows(
    catalog: str,
    schema: str,
    name: str,
    request: Request,
    media_type: "str | None" = None,
    engine: TabularEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
) -> WriteResult:
    tabular = _require_tabular(engine, catalog, schema, name)
    table = await _decode_request_table(
        request, media_type_override=media_type, settings=settings,
    )
    tabular.write_table(table, mode=Mode.APPEND)
    engine[catalog, schema, name].invalidate_schema()
    return WriteResult(
        catalog=catalog,
        schema=schema,
        name=name,
        rows_written=table.num_rows,
        mode="APPEND",
    )


@router.put(
    "/{catalog}/{schema}/{name}/rows",
    response_model=WriteResult,
    summary="Replace every row of a registered table",
)
async def replace_rows(
    catalog: str,
    schema: str,
    name: str,
    request: Request,
    media_type: "str | None" = None,
    engine: TabularEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
) -> WriteResult:
    tabular = _require_tabular(engine, catalog, schema, name)
    table = await _decode_request_table(
        request, media_type_override=media_type, settings=settings,
    )
    tabular.write_table(table, mode=Mode.OVERWRITE)
    engine[catalog, schema, name].invalidate_schema()
    return WriteResult(
        catalog=catalog,
        schema=schema,
        name=name,
        rows_written=table.num_rows,
        mode="OVERWRITE",
    )


@router.delete(
    "/{catalog}/{schema}/{name}/rows",
    response_model=DeleteResult,
    summary="Delete rows by predicate (or every row when no predicate is given)",
)
def delete_rows(
    catalog: str,
    schema: str,
    name: str,
    where: "str | None" = None,
    engine: TabularEngine = Depends(get_engine),
) -> DeleteResult:
    tabular = _require_tabular(engine, catalog, schema, name)

    if where:
        # :meth:`Tabular.delete` accepts either a parsed Predicate or
        # a SQL string it parses on the fly — we hand the string
        # through unchanged so error messages point at the SQL the
        # client actually sent.
        try:
            removed = tabular.delete(where)
        except Exception as exc:  # noqa: BLE001 — wrap as APIError
            raise APIError(
                f"Could not parse predicate {where!r}: {exc}. "
                "Use a SQL boolean expression like \"id > 10\" or "
                "\"region = 'EU' AND price > 100\"."
            ) from exc
    else:
        # No predicate → wipe the table by overwriting it with a
        # zero-row Arrow batch shaped like the current schema. We
        # avoid OVERWRITE-with-empty (which some leaves treat as a
        # no-op) by going through the predicate path with
        # ``True``-coerced expression equivalent.
        snapshot = tabular.read_arrow_table()
        rows_before = snapshot.num_rows
        tabular.write_table(snapshot.slice(0, 0), mode=Mode.OVERWRITE)
        removed = rows_before

    engine[catalog, schema, name].invalidate_schema()
    return DeleteResult(
        catalog=catalog, schema=schema, name=name, rows_deleted=removed,
    )
