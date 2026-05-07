"""Media-type-aware streaming responses for :class:`Tabular` data.

Two transport flavors:

- Arrow IPC **stream** (default) — :func:`stream_arrow_ipc` writes
  the source's :meth:`pa.RecordBatchReader` straight into the HTTP
  response with :class:`pa.ipc.RecordBatchStreamWriter`. No footer,
  no seek, true streaming. The wire mime is
  ``application/vnd.apache.arrow.stream``.
- Anything else — :func:`tabular_response` resolves the requested
  :class:`MediaType` through :meth:`Tabular.class_for_media_type`,
  writes the source into a fresh :class:`BytesIO` of that format,
  and returns the bytes. That covers parquet / CSV / JSON / NDJSON /
  XLSX / Arrow IPC file out of the box, and any new
  :class:`Tabular` leaf registered against a mime gets exposed for
  free.

Content negotiation
-------------------

:func:`resolve_media_type` looks at, in order:

1. an explicit ``?format=`` / ``?media_type=`` query parameter;
2. the ``Accept`` request header (first acceptable type wins);
3. the configured ``default_media_type``.

The query param wins because it's the most explicit — Power
Query / Excel / curl scripts often can't set ``Accept`` cleanly
and a query string is the obvious override path.
"""

from __future__ import annotations

import io
from typing import Any, Iterator

import pyarrow as pa
from fastapi import Response

from yggdrasil.data.enums import MediaType, MimeTypes
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.tabular import Tabular

# Pull every concrete tabular leaf into the registry so
# :meth:`Tabular.class_for_media_type` can dispatch parquet / csv /
# json / ndjson / xlsx / arrow-ipc the moment the API boots.
import yggdrasil.io.primitive  # noqa: F401  — side-effect import

from .exceptions import APIError


__all__ = [
    "ARROW_STREAM_MIME",
    "ARROW_FILE_MIME",
    "resolve_media_type",
    "tabular_response",
    "stream_arrow_ipc",
]


# Wire mime for the Arrow IPC stream format. Not in
# :class:`MimeTypes` (the registry tracks the file format under
# ``ARROW_IPC``); we keep it as a literal here and translate to the
# file MimeType when we need to dispatch through
# :meth:`Tabular.class_for_media_type`.
ARROW_STREAM_MIME = "application/vnd.apache.arrow.stream"
ARROW_FILE_MIME = MimeTypes.ARROW_IPC.value  # application/vnd.apache.arrow.file


def _is_stream_mime(value: str) -> bool:
    v = value.strip().lower()
    return v in {ARROW_STREAM_MIME, "arrow", "arrow_stream", "ipc_stream", "stream"}


def resolve_media_type(
    *,
    explicit: "str | None",
    accept: "str | None",
    default: str,
) -> "tuple[str, MediaType | None]":
    """Pick the response media type. Returns ``(wire_mime, MediaType|None)``.

    ``MediaType`` is ``None`` when the wire mime is the Arrow IPC
    stream — that one doesn't have a registered :class:`MimeType`
    in :mod:`yggdrasil.data.enums`, and dispatch goes through
    :func:`stream_arrow_ipc` directly.
    """
    candidates = _candidate_chain(explicit=explicit, accept=accept, default=default)
    for raw in candidates:
        if not raw:
            continue
        if _is_stream_mime(raw):
            return ARROW_STREAM_MIME, None
        mt = MediaType.from_(raw, default=None)
        if mt is not None and mt.mime_type is not MimeTypes.OCTET_STREAM:
            return mt.mime_type.value, mt

    # Final fallback — Arrow stream. The most efficient option a
    # generic caller can ask for, and we can always produce it.
    return ARROW_STREAM_MIME, None


def _candidate_chain(
    *, explicit: "str | None", accept: "str | None", default: str,
) -> "list[str]":
    chain: "list[str]" = []
    if explicit:
        chain.append(explicit)
    if accept:
        # Strip accept-params (q=, charset=…) and split on ',' so we
        # walk the client's preference list in order.
        for part in accept.split(","):
            head = part.split(";", 1)[0].strip()
            if head and head != "*/*":
                chain.append(head)
    if default:
        chain.append(default)
    return chain


def tabular_response(
    tabular: Tabular,
    *,
    media_type: "str | MediaType | None" = None,
    accept: "str | None" = None,
    default: str = ARROW_STREAM_MIME,
    filename: "str | None" = None,
    stream_batch_rows: int = 65_536,
) -> "Response":
    """Serialize *tabular* into the requested media type and respond.

    Goes through :func:`stream_arrow_ipc` when the negotiated mime
    is the Arrow IPC stream, otherwise routes through
    :meth:`Tabular.class_for_media_type` to find the leaf for that
    mime and writes the data into a fresh :class:`BytesIO`.
    """
    explicit = media_type if isinstance(media_type, str) else None
    if isinstance(media_type, MediaType):
        wire = media_type.mime_type.value
        target = media_type
    else:
        wire, target = resolve_media_type(
            explicit=explicit, accept=accept, default=default,
        )

    if target is None:  # Arrow IPC stream
        return stream_arrow_ipc(
            tabular,
            filename=filename,
            stream_batch_rows=stream_batch_rows,
        )

    return _materialize_response(
        tabular, target=target, wire_mime=wire, filename=filename,
    )


def stream_arrow_ipc(
    tabular: Tabular,
    *,
    filename: "str | None" = None,
    stream_batch_rows: int = 65_536,
) -> "Response":
    """Stream *tabular* as Arrow IPC stream — the fast path.

    Drives :class:`pa.ipc.RecordBatchStreamWriter` over the source's
    :class:`pa.RecordBatchReader`, yielding each emitted IPC chunk
    as a separate response chunk. No footer, no seek, no full
    materialization — the response starts flowing as soon as the
    first batch is ready.
    """
    from fastapi.responses import StreamingResponse

    options = tabular.options_class()(row_size=stream_batch_rows or None)
    reader = tabular._read_arrow_batch_reader(options)

    def _iter_ipc_bytes() -> Iterator[bytes]:
        sink = _ChunkSink()
        writer = pa.ipc.RecordBatchStreamWriter(sink, reader.schema)
        try:
            for batch in reader:
                writer.write_batch(batch)
                # Drain whatever the writer produced for this batch
                # so we hand bytes to ASGI as we go instead of
                # buffering the entire stream.
                chunk = sink.drain()
                if chunk:
                    yield chunk
        finally:
            writer.close()
        tail = sink.drain()
        if tail:
            yield tail

    headers: "dict[str, str]" = {}
    if filename:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    return StreamingResponse(
        _iter_ipc_bytes(),
        media_type=ARROW_STREAM_MIME,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _ChunkSink(io.RawIOBase):
    """Tiny ``write``-only sink that hands buffered bytes back via :meth:`drain`.

    Arrow's stream writer accepts any object with a ``write`` method
    that takes bytes-like input; we don't need ``seek``/``tell`` for
    the stream format. Buffering one batch at a time keeps memory
    bounded — :meth:`drain` returns the accumulated chunk and clears
    the slot so the next ``write_batch`` starts fresh.
    """

    def __init__(self) -> None:
        super().__init__()
        self._buf: "list[bytes]" = []

    def writable(self) -> bool:  # noqa: D401 — stdlib protocol
        return True

    def write(self, data: Any) -> int:  # noqa: D401
        if isinstance(data, memoryview):
            data = data.tobytes()
        elif not isinstance(data, (bytes, bytearray)):
            data = bytes(data)
        self._buf.append(bytes(data))
        return len(data)

    def drain(self) -> bytes:
        if not self._buf:
            return b""
        out = b"".join(self._buf)
        self._buf.clear()
        return out


def _materialize_response(
    tabular: Tabular,
    *,
    target: MediaType,
    wire_mime: str,
    filename: "str | None",
) -> "Response":
    """Write *tabular* into a :class:`BytesIO` of *target* mime and respond.

    Used for non-stream formats. The :class:`Tabular` registry maps
    a mime to the leaf class that knows how to serialize into it
    (parquet → :class:`ParquetIO`, csv → :class:`CsvIO`, json →
    :class:`JsonIO`, …). We pop the writer's bytes once at the end —
    formats that need a footer (parquet, arrow IPC file, xlsx)
    can't be true-streamed in HTTP without a tee-and-buffer dance,
    and the buffered path here is the honest implementation.
    """
    from fastapi.responses import Response

    leaf_cls = Tabular.class_for_media_type(target, default=None)
    if leaf_cls is None:
        raise APIError(
            (
                f"No serializer registered for media type {target!r}. "
                f"Supported: {sorted(Tabular.registered_classes())}. "
                "Pass ?format=arrow_stream for the streaming default, "
                "or any of the registered tabular mimes."
            ),
            status_code=415,
        )

    sink = BytesIO(media_type=target)
    leaf = leaf_cls(holder=sink._holder)
    leaf.write_table(tabular)
    payload = sink.to_bytes()

    headers: "dict[str, str]" = {}
    if filename:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    headers["Content-Length"] = str(len(payload))

    return Response(content=payload, media_type=wire_mime, headers=headers)
