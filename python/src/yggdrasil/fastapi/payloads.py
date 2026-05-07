"""Decode binary HTTP payloads into Arrow tables / Tabulars.

Shared between :mod:`routers.sources` (registration uploads) and
:mod:`routers.data` (row inserts / replaces).

Dispatch uses :class:`yggdrasil.io.bytes_io.BytesIO` +
:meth:`BytesIO.as_media`, so any registered :class:`Tabular` leaf
(parquet / csv / json / ndjson / xlsx / arrow IPC file …) is handled
without per-mime branches in the routers, and ``Content-Encoding``
codecs round-trip via :class:`BytesIO`'s ``_format_view``.

Arrow IPC stream is the only special case — the streaming wire
format isn't a registered :class:`Tabular` leaf (the registry
tracks the on-disk *file* format under ``ARROW_IPC``), so it goes
straight through :func:`pyarrow.ipc.open_stream` against a zero-copy
:class:`pa.py_buffer`.
"""

from __future__ import annotations

import pyarrow as pa
from fastapi import Request

from yggdrasil.data.enums import Codec, MediaType
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.tabular import Tabular

from .exceptions import APIError
from .responses import ARROW_FILE_MIME, ARROW_STREAM_MIME


__all__ = [
    "read_capped_body",
    "decode_payload",
    "extract_wire_mime",
]


async def read_capped_body(request: Request, cap: int) -> bytes:
    """Read the request body, refusing to buffer past *cap* bytes.

    Streamed off ``request.stream()`` so we don't depend on
    ``Content-Length`` (some clients omit it). The cap is inclusive
    of the running total: a payload exactly at the cap is accepted,
    anything past it raises 413.
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


def extract_wire_mime(
    request: Request, *, override: "str | None" = None,
) -> str:
    """Resolve the request's content type. ``override`` wins."""
    if override:
        return override.split(";", 1)[0].strip()
    raw = request.headers.get("content-type") or ""
    return raw.split(";", 1)[0].strip()


def decode_payload(
    payload: bytes,
    wire_mime: str,
    *,
    content_encoding: "str | None" = None,
) -> "tuple[pa.Table, int]":
    """Decode *payload* into ``(arrow_table, row_count)``.

    Resolution:

    1. **Arrow IPC stream** — straight through :func:`pa.ipc.open_stream`.
    2. **Anything else** — :class:`BytesIO` over the payload, stamped
       with the resolved :class:`MediaType` (and ``Content-Encoding``
       folded into the codec slot). :meth:`BytesIO.as_media` resolves
       the registered :class:`Tabular` leaf and reads a single
       :class:`pa.Table` back; codec on the buffer's MediaType is
       handled by the leaf's ``_format_view`` on read.
    """
    head = wire_mime.lower().strip()

    if not head:
        raise APIError(
            "Missing media type. Pass ?media_type=<mime> or set Content-Type "
            "(e.g. application/vnd.apache.arrow.stream / "
            "application/vnd.apache.arrow.file / "
            "application/vnd.apache.parquet / text/csv / application/json)."
        )

    if head == ARROW_STREAM_MIME:
        reader = pa.ipc.open_stream(pa.py_buffer(payload))
        table = reader.read_all()
        return table, table.num_rows

    media = MediaType.from_(head, default=None)
    if media is None:
        raise APIError(
            f"Unsupported upload media type {wire_mime!r}. Supported: "
            f"{sorted(Tabular.registered_classes())}, "
            f"{ARROW_STREAM_MIME!r}, {ARROW_FILE_MIME!r}.",
            status_code=415,
        )

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
