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

from yggdrasil.data.enums import Codec, MediaType
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.tabular import ArrowTabular, Tabular, TabularEngine

from ..config import Settings
from ..deps import get_engine, get_settings
from ..exceptions import APIError, NotFound
from ..responses import ARROW_FILE_MIME, ARROW_STREAM_MIME
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
    payload = await _read_capped_body(request, settings.max_request_bytes)
    if not payload:
        raise APIError(
            "Empty upload. Send the tabular bytes in the request body and "
            "set Content-Type to the format mime (or pass ?media_type=)."
        )

    wire_mime = (
        media_type
        or (request.headers.get("content-type") or "").split(";", 1)[0].strip()
    )
    if not wire_mime:
        raise APIError(
            "Missing media type. Pass ?media_type=<mime> or set Content-Type "
            "(e.g. application/vnd.apache.arrow.stream / "
            "application/vnd.apache.arrow.file / "
            "application/vnd.apache.parquet / text/csv / application/json)."
        )

    table, rows = _decode_upload(
        payload,
        wire_mime,
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
# Internal — body reads and format decode
# ---------------------------------------------------------------------------


async def _read_capped_body(request: Request, cap: int) -> bytes:
    """Read the request body, refusing to buffer past *cap* bytes."""
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


def _decode_upload(
    payload: bytes,
    wire_mime: str,
    *,
    content_encoding: "str | None" = None,
) -> "tuple[pa.Table, int]":
    """Decode *payload* into a :class:`pa.Table`.

    Resolution:

    1. **Arrow IPC stream** — the wire format isn't a registered
       :class:`Tabular` leaf (the registry only tracks the on-disk
       file format), so this branch goes straight through
       :func:`pa.ipc.open_stream` against a zero-copy
       :class:`pa.py_buffer`. Cheapest path.
    2. **Anything else** — :class:`BytesIO` over the payload, stamped
       with the resolved :class:`MediaType` (and ``Content-Encoding``
       folded into the codec slot). :meth:`BytesIO.as_media`
       dispatches to the registered :class:`Tabular` leaf and reads
       a single :class:`pa.Table` back. Codec on the buffer is
       handled by :class:`BytesIO`'s ``_format_view`` for the leaves
       that consume it on read.
    """
    head = wire_mime.lower().strip()

    if head == ARROW_STREAM_MIME:
        # No registered leaf for the streaming wire format — pyarrow
        # has the cheapest path.
        reader = pa.ipc.open_stream(pa.py_buffer(payload))
        table = reader.read_all()
        return table, table.num_rows

    media = MediaType.from_(head, default=None)
    if media is None:
        raise APIError(
            f"Unsupported upload media type {wire_mime!r}. Supported: "
            f"{sorted(Tabular.registered_classes())}, "
            f"{ARROW_STREAM_MIME!r}, {ARROW_FILE_MIME!r}, "
            "or application/json.",
            status_code=415,
        )

    # Surface ``Content-Encoding`` as the codec on the buffer's
    # MediaType. :class:`BytesIO`'s ``_format_view`` reads it back
    # on the leaf side and decompresses transparently.
    if content_encoding and media.codec is None:
        codec = Codec.from_(content_encoding, default=None)
        if codec is not None:
            media = media.with_codec(codec)

    buf = BytesIO(data=payload, media_type=media)
    try:
        leaf = buf.as_media()
    except KeyError as exc:
        raise APIError(
            f"Unsupported upload media type {wire_mime!r}. Supported: "
            f"{sorted(Tabular.registered_classes())}, "
            f"{ARROW_STREAM_MIME!r}, {ARROW_FILE_MIME!r}.",
            status_code=415,
        ) from exc

    table = leaf.read_arrow_table()
    return table, table.num_rows
