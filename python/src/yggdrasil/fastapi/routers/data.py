"""Data exposure — read rows out of registered sources or ad-hoc SQL.

Endpoints:

- ``GET /data/{catalog}/{schema}/{name}`` — read a registered
  :class:`Tabular`'s rows, format-negotiated through the ``Accept``
  header / ``?format=`` query string.
- ``POST /data/sql`` — run a SQL query against the engine and stream
  the result back; the body is the SQL text (any non-empty text/sql
  payload), the response format is negotiated the same way.

The default response is the **Arrow IPC stream** because it's the
cheapest format both ends speak and it streams without buffering
the whole result. Callers can opt in to anything else the
:class:`Tabular` registry knows about (parquet, csv, json,
ndjson, xlsx, …) via either the ``Accept`` header or the
``?format=`` shorthand.

Power Query / Excel callers tend to send unhelpful ``Accept``
headers, so the query string takes precedence. Same for the
``?filename=`` knob, which sets ``Content-Disposition`` so the
browser's download UI gets a sensible name.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Request, Response

from yggdrasil.io.tabular import TabularEngine

from ..config import Settings
from ..deps import get_engine, get_settings
from ..exceptions import APIError, NotFound
from ..responses import ARROW_STREAM_MIME, tabular_response


router = APIRouter(prefix="/data", tags=["data"])


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
    entry = engine.get(catalog, schema, name)
    if entry is None:
        raise NotFound(
            f"No tabular registered as {catalog!r}.{schema!r}.{name!r}. "
            f"Registered: {engine.qualified_names()!r}."
        )

    return tabular_response(
        entry.tabular,
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
