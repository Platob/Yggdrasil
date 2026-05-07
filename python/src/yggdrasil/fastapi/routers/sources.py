"""Source registration — bind data into ``catalog.schema.name`` slots.

Three registration shapes:

1. **Path** (JSON body) — register a Tabular pointing at a path or
   URL. Anything :class:`yggdrasil.io.path.Path.from_` accepts works
   (``file://``, ``s3://``, ``dbfs://``, etc.); the format is
   sniffed from the extension or the optional ``media_type`` field.
2. **Inline rows / columns** (JSON body) — register a small Arrow
   table built from rows or columns embedded in the request. Useful
   for tests, Power Query parameter tables, ad-hoc demos.
3. **Binary upload** — POST the raw bytes of a tabular file
   (Arrow IPC, parquet, csv, …); the format comes from the
   ``Content-Type`` header (or the ``?media_type=`` override) and
   the bytes are wrapped in an :class:`ArrowTabular` after a single
   read so re-reads don't pay the deserialization cost twice.

Removal:

- ``DELETE /sources/{catalog}/{schema}/{name}`` — drops the entry.

The path / inline shapes use ``application/json``; the binary
upload accepts any registered tabular mime. Picking the right
endpoint is mime-driven so a single ``POST`` URL works for every
shape.
"""

from __future__ import annotations

import pyarrow as pa
from fastapi import APIRouter, Depends, Request

from yggdrasil.data.enums import MediaType
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.tabular import ArrowTabular, Tabular, TabularEngine

# Same side-effect import as :mod:`responses` — keep the registry
# warm so an upload arriving on a fresh worker has every leaf
# already wired up.
import yggdrasil.io.primitive  # noqa: F401

from ..deps import get_engine, get_settings
from ..exceptions import APIError, NotFound
from ..responses import ARROW_FILE_MIME, ARROW_STREAM_MIME
from ..schemas import (
    RegisterInlineRequest,
    RegisterPathRequest,
    RegisterResult,
)
from ..config import Settings


router = APIRouter(prefix="/sources", tags=["sources"])


# Mimes we treat as the "JSON registration" path. Anything else
# coming through the binary upload endpoint is treated as a tabular
# payload and decoded via the registry / Arrow IPC fast path.
_JSON_MIMES = frozenset({
    "application/json",
    "text/json",
})


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

    media_type = MediaType.from_(body.media_type, default=None) if body.media_type else None
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
    payload = await _read_capped_body(request, settings.max_request_bytes)
    if not payload:
        raise APIError(
            "Empty upload. Send the tabular bytes in the request body and "
            "set Content-Type to the format mime (or pass ?media_type=)."
        )

    wire_mime = (media_type or request.headers.get("content-type") or "").split(";")[0].strip()
    if not wire_mime:
        raise APIError(
            "Missing media type. Pass ?media_type=<mime> or set Content-Type "
            "(e.g. application/vnd.apache.arrow.stream / "
            "application/vnd.apache.arrow.file / "
            "application/vnd.apache.parquet / text/csv / application/json)."
        )

    table, rows = _decode_upload(payload, wire_mime)
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
# Internal — body reads and format decode
# ---------------------------------------------------------------------------


async def _read_capped_body(request: Request, cap: int) -> bytes:
    """Read the request body, refusing to buffer past *cap* bytes.

    The streaming form runs over ``request.stream()`` so we don't
    rely on ``Content-Length`` (some clients omit it); the cap is
    inclusive of the running total so a payload exactly at the cap
    is accepted but anything past it raises a 413.
    """
    chunks: "list[bytes]" = []
    total = 0
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > cap:
            raise APIError(
                f"Upload exceeds the configured cap of {cap} bytes. "
                "Bump YGG_API_MAX_REQUEST_BYTES or split the payload "
                "and POST in chunks.",
                status_code=413,
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_upload(payload: bytes, wire_mime: str) -> "tuple[pa.Table, int]":
    """Decode *payload* bytes of *wire_mime* into a single :class:`pa.Table`.

    Arrow IPC stream is special-cased — pyarrow's
    :class:`pa.ipc.open_stream` is the cheapest path (zero-copy
    against the buffer, no temp file). Everything else routes
    through the :class:`Tabular` registry: write the bytes into a
    matching :class:`BytesIO`, then read the table back out via
    the leaf's reader. JSON is interpreted as a list-of-dicts.
    """
    head = wire_mime.lower().strip()

    if head == ARROW_STREAM_MIME:
        reader = pa.ipc.open_stream(pa.py_buffer(payload))
        table = reader.read_all()
        return table, table.num_rows

    if head == ARROW_FILE_MIME:
        reader = pa.ipc.open_file(pa.py_buffer(payload))
        table = reader.read_all()
        return table, table.num_rows

    if head in _JSON_MIMES:
        from yggdrasil.pickle import json as ygg_json

        decoded = ygg_json.loads(payload)
        if isinstance(decoded, list):
            table = pa.Table.from_pylist(decoded)
        elif isinstance(decoded, dict):
            table = pa.table(decoded)
        else:
            raise APIError(
                f"JSON upload must be a list of row dicts or a dict of "
                f"column → list; got {type(decoded).__name__}."
            )
        return table, table.num_rows

    leaf_cls = Tabular.class_for_media_type(head, default=None)
    if leaf_cls is None:
        raise APIError(
            f"Unsupported upload media type {wire_mime!r}. Supported: "
            f"{sorted(Tabular.registered_classes())}, "
            f"{ARROW_STREAM_MIME!r}, {ARROW_FILE_MIME!r}, "
            "or application/json.",
            status_code=415,
        )

    target = MediaType.from_(head, default=None)
    sink = BytesIO(data=payload, media_type=target)
    leaf = leaf_cls(holder=sink._holder)
    table = leaf.read_arrow_table()
    return table, table.num_rows
