"""Media-type-aware streaming responses for :class:`Tabular` data.

Two transport flavors:

- **Arrow IPC stream** (default) — :func:`stream_arrow_ipc` writes the
  source's :class:`pa.RecordBatchReader` straight into the HTTP
  response with :class:`pa.ipc.RecordBatchStreamWriter`. No footer,
  no seek, true streaming. The wire mime is
  ``application/vnd.apache.arrow.stream``. The streaming format
  isn't a registered :class:`Tabular` leaf (the registry tracks the
  on-disk *file* format under ``ARROW_IPC``), so this case stays a
  thin pyarrow-direct path.
- **Anything else** — :func:`tabular_response` resolves the
  requested :class:`MediaType` and writes the source through
  :meth:`yggdrasil.io.bytes_io.BytesIO.as_media`. That returns the
  :class:`Tabular` leaf already registered for the mime
  (parquet / csv / json / ndjson / xlsx / arrow IPC file …), so
  every new tabular leaf gets exposed automatically and codec
  handling (``Content-Encoding``) round-trips through
  :class:`BytesIO`'s ``_format_view``.

Content negotiation
-------------------

:func:`resolve_media_type` walks, in order:

1. an explicit ``?format=`` / ``?media_type=`` query parameter,
2. each entry in the ``Accept`` header,
3. the configured default mime.

Each candidate goes through :meth:`MediaType.from_` so aliases,
extension hints and ``mime+codec`` strings are normalised the same
way the rest of the library does it.
"""

from __future__ import annotations

import io
from typing import Iterator

import pyarrow as pa
from fastapi import Response
from fastapi.responses import StreamingResponse

from yggdrasil.data.enums import MediaType, MimeTypes
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.tabular import Tabular

from .exceptions import APIError


__all__ = [
    "ARROW_STREAM_MIME",
    "ARROW_FILE_MIME",
    "resolve_media_type",
    "tabular_response",
    "stream_arrow_ipc",
]


# Wire mime for the Arrow IPC stream format. Not in :class:`MimeTypes`
# (the registry tracks the file format under ``ARROW_IPC``); kept as a
# literal so the negotiation chain can recognise it without forcing a
# new mime registration.
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
    stream — the writer goes through :func:`stream_arrow_ipc`
    directly and there's no registered :class:`MimeType` for it.
    Every other candidate is normalised by :meth:`MediaType.from_`.
    """
    for raw in _candidate_chain(explicit=explicit, accept=accept, default=default):
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
) -> Response:
    """Serialize *tabular* into the requested media type and respond.

    Routes through :func:`stream_arrow_ipc` when the negotiated mime
    is the Arrow IPC stream; otherwise hands the work to
    :meth:`BytesIO.as_media` which dispatches to the registered
    :class:`Tabular` leaf for that mime.
    """
    explicit = media_type if isinstance(media_type, str) else None
    if isinstance(media_type, MediaType):
        wire = media_type.mime_type.value
        target: "MediaType | None" = media_type
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
) -> Response:
    """Stream *tabular* as Arrow IPC stream — the fast path.

    Drives :class:`pa.ipc.RecordBatchStreamWriter` over the source's
    :class:`pa.RecordBatchReader`, draining the writer's bytes into
    the response after each batch. No footer, no seek, no full
    materialisation — the response starts flowing as soon as the
    first batch is ready.
    """
    options = tabular.options_class()(row_size=stream_batch_rows or None)
    reader = tabular._read_arrow_batch_reader(options)

    def _iter_ipc_bytes() -> Iterator[bytes]:
        # stdlib ``io.BytesIO`` is already an ideal sink for this:
        # ``write`` + ``seek`` + ``getvalue`` + ``truncate`` is all
        # the IPC writer needs, and we drain it after every batch
        # so the response streams instead of buffering.
        sink = io.BytesIO()
        writer = pa.ipc.RecordBatchStreamWriter(sink, reader.schema)
        try:
            for batch in reader:
                writer.write_batch(batch)
                chunk = sink.getvalue()
                if chunk:
                    yield chunk
                    sink.seek(0)
                    sink.truncate(0)
        finally:
            writer.close()
        tail = sink.getvalue()
        if tail:
            yield tail

    return StreamingResponse(
        _iter_ipc_bytes(),
        media_type=ARROW_STREAM_MIME,
        headers=_disposition_headers(filename),
    )


# ---------------------------------------------------------------------------
# Internal — non-stream materialisation
# ---------------------------------------------------------------------------


def _materialize_response(
    tabular: Tabular,
    *,
    target: MediaType,
    wire_mime: str,
    filename: "str | None",
) -> Response:
    """Write *tabular* into a :class:`BytesIO` of *target* and respond.

    The serializer is whatever :meth:`BytesIO.as_media` resolves to —
    that walks the same registry :meth:`Tabular.class_for_media_type`
    uses, so a parquet target lands on :class:`ParquetIO`, a csv
    target on :class:`CsvIO`, and so on. Codec handling on the way
    out is :class:`BytesIO`'s job (``Content-Encoding`` round-trips
    via the holder's :class:`MediaType`).
    """
    sink = BytesIO(media_type=target)
    try:
        leaf = sink.as_media()
    except KeyError as exc:
        raise APIError(
            (
                f"No serializer registered for media type {target!r}. "
                f"Supported: {sorted(Tabular.registered_classes())}. "
                "Pass ?format=arrow_stream for the streaming default, "
                "or any of the registered tabular mimes."
            ),
            status_code=415,
        ) from exc

    leaf.write_table(tabular)
    payload = sink.to_bytes()

    headers = _disposition_headers(filename)
    headers["Content-Length"] = str(len(payload))
    if target.codec is not None:
        # Mirror what :meth:`yggdrasil.io.response.Response._to_asgi_payload`
        # does — surface the wrapper codec on the wire so the client
        # decompresses correctly.
        headers["Content-Encoding"] = target.codec.name

    return Response(content=payload, media_type=wire_mime, headers=headers)


def _disposition_headers(filename: "str | None") -> "dict[str, str]":
    if not filename:
        return {}
    return {"Content-Disposition": f'attachment; filename="{filename}"'}
