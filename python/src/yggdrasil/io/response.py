"""HTTP response model with Arrow, Polars, pandas, and ASGI serialisation.

This module defines :class:`Response` — the canonical response dataclass used
throughout the ``yggdrasil.io`` stack — and the two Arrow schemas that describe
how responses are persisted:

* :data:`ARROW_SCHEMA` — response-only columns (status code, headers, body, …).
* :data:`RESPONSE_ARROW_SCHEMA` — full request + response flattened into a
  single schema, used for Delta-table caching in
  :class:`~yggdrasil.io.session.Session`.

Public symbols
--------------
.. autosummary::

   Response
   ARROW_SCHEMA
   RESPONSE_ARROW_SCHEMA

Serialisation methods on :class:`Response`
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Output
   * - :meth:`~Response.to_arrow_batch`
     - :class:`pyarrow.RecordBatch` (raw schema row or parsed body)
   * - :meth:`~Response.to_arrow_table`
     - :class:`pyarrow.Table` (single-row or parsed body)
   * - :meth:`~Response.to_parquet`
     - ``bytes`` — Parquet-encoded Arrow table
   * - :meth:`~Response.to_ipc`
     - ``bytes`` — Arrow IPC stream
   * - :meth:`~Response.to_polars`
     - :class:`polars.DataFrame` or :class:`polars.LazyFrame`
   * - :meth:`~Response.to_pandas`
     - :class:`pandas.DataFrame`
   * - :meth:`~Response.to_starlette`
     - :class:`starlette.responses.Response`
   * - :meth:`~Response.to_fastapi`
     - :class:`fastapi.Response`
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
)

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

import yggdrasil.pickle.json as json_module
from yggdrasil.dataclasses.dataclass import get_from_dict
from .buffer import BytesIO
from .enums import MediaType, Codec, MimeType
from .headers import PromotedHeaders, normalize_headers, DEFAULT_HOSTNAME
from .request import PreparedRequest, REQUEST_ARROW_SCHEMA

if TYPE_CHECKING:
    import io as _io
    import polars as pl
    import pandas as pd
    from starlette.responses import Response as StarletteResponse
    from fastapi import Response as FastAPIResponse

__all__ = [
    "Response",
    "ARROW_SCHEMA",
    "RESPONSE_ARROW_SCHEMA",
]

# ---------------------------------------------------------------------------
# Private header / body helpers
# ---------------------------------------------------------------------------

def _get_header(headers: Mapping[str, str] | None, name: str) -> str | None:
    """Return the value of *name* from *headers*, case-insensitively.

    Parameters
    ----------
    headers:
        Header mapping to search, or ``None``.
    name:
        Header name (any casing).

    Returns
    -------
    str | None
        The header value as a string, or ``None`` if not found.
    """
    if not headers:
        return None

    value = headers.get(name)
    if value is not None:
        return str(value)

    target = name.lower()
    for key, value in headers.items():
        if isinstance(key, str):
            if key == name or key.lower() == target:
                return str(value)
        elif str(key).lower() == target:
            return str(value)

    return None


def _pop_header(headers: MutableMapping[str, str], name: str) -> str | None:
    """Remove and return the value of *name* from *headers*, case-insensitively.

    Parameters
    ----------
    headers:
        Mutable header mapping to mutate.
    name:
        Header name (any casing).

    Returns
    -------
    str | None
        The removed value, or ``None`` if the header was not present.
    """
    value = headers.pop(name, None)
    if value is not None:
        return str(value)

    target = name.lower()
    for key in list(headers.keys()):
        if isinstance(key, str):
            if key == name or key.lower() == target:
                return str(headers.pop(key))
        elif str(key).lower() == target:
            return str(headers.pop(key))

    return None


def _get_charset(headers: Mapping[str, str]) -> str:
    """Extract the ``charset`` parameter from the ``Content-Type`` header.

    Parameters
    ----------
    headers:
        Response (or request) headers.

    Returns
    -------
    str
        The charset string (e.g. ``"utf-8"``).  Defaults to ``"utf-8"``
        when the header is absent or carries no charset parameter.
    """
    content_type = _get_header(headers, "Content-Type")
    if not content_type:
        return "utf-8"

    for part in str(content_type).split(";")[1:]:
        part = part.strip()
        if part.lower().startswith("charset="):
            charset = part.split("=", 1)[1].strip().strip('"')
            return charset or "utf-8"

    return "utf-8"


def _parse_content_type(headers: Mapping[str, str] | None) -> str | None:
    """Return the bare MIME type from ``Content-Type`` (no parameters).

    Parameters
    ----------
    headers:
        Header mapping, or ``None``.

    Returns
    -------
    str | None
        Lower-cased MIME type (e.g. ``"application/json"``), or ``None``.
    """
    value = _get_header(headers, "Content-Type")
    if not value:
        return None
    bare = str(value).split(";", 1)[0].strip().lower()
    return bare or None


def _parse_content_encoding(headers: Mapping[str, str] | None) -> str | None:
    """Return the normalised ``Content-Encoding`` value.

    Multiple encodings are joined with ``","`` and each token is lower-cased.

    Parameters
    ----------
    headers:
        Header mapping, or ``None``.

    Returns
    -------
    str | None
        Comma-joined encoding string, or ``None`` if absent.
    """
    value = _get_header(headers, "Content-Encoding")
    if not value:
        return None
    parts = [p.strip().lower() for p in str(value).split(",") if p.strip()]
    return ",".join(parts) or None


def _parse_content_length(headers: Mapping[str, str] | None) -> int | None:
    """Parse ``Content-Length`` as an integer.

    Parameters
    ----------
    headers:
        Header mapping, or ``None``.

    Returns
    -------
    int | None
        The content length, or ``None`` if the header is absent or unparseable.
    """
    value = _get_header(headers, "Content-Length")
    if value in (None, ""):
        return None
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _is_probably_placeholder_content_type(value: str | None) -> bool:
    """Return ``True`` when *value* is a generic / placeholder MIME type.

    These values carry no useful information and should be replaced by the
    sniffed media type from the body bytes.

    Parameters
    ----------
    value:
        Content-Type string to inspect, or ``None``.

    Returns
    -------
    bool
    """
    if not value:
        return True
    return value.strip().lower() in {
        "",
        "application/octet-stream",
        "binary/octet-stream",
        "unknown/unknown",
    }


def _sniff_media_from_body(
    body: BytesIO,
    *,
    content_type: str | None,
    content_encoding: str | None,
) -> MediaType:
    """Determine the :class:`~yggdrasil.io.enums.MediaType` of *body*.

    Prefers the declared ``Content-Type`` when it is not a placeholder.
    Falls back to magic-byte sniffing via :attr:`BytesIO.media_type`.

    Parameters
    ----------
    body:
        The response body buffer.
    content_type:
        Bare MIME type from the ``Content-Type`` header, or ``None``.
    content_encoding:
        Value from the ``Content-Encoding`` header, or ``None``.

    Returns
    -------
    MediaType
        The resolved media type (MIME + optional codec).
    """
    content_codec = Codec.parse(content_encoding, default=None)

    if content_type and not _is_probably_placeholder_content_type(content_type):
        try:
            return MediaType.parse(content_type, codec=content_codec)
        except Exception:
            pass

    try:
        sniffed = body.media_type
    except Exception:
        sniffed = MediaType(MimeType.OCTET_STREAM, content_codec)

    return MediaType(sniffed.mime_type, codec=content_codec) if content_encoding else sniffed


def _ensure_media_headers(
    headers: MutableMapping[str, str],
    body: BytesIO,
) -> MediaType:
    """Back-fill ``Content-Type``, ``Content-Encoding``, and ``Content-Length``.

    Sniffs the body when the declared type is a placeholder, and writes the
    resolved values back into *headers*.

    Parameters
    ----------
    headers:
        Mutable header mapping to update in-place.
    body:
        The response body buffer.

    Returns
    -------
    MediaType
        The resolved media type applied to *headers*.
    """
    declared_type = _parse_content_type(headers)
    declared_encoding = _parse_content_encoding(headers)

    media = _sniff_media_from_body(
        body,
        content_type=declared_type,
        content_encoding=declared_encoding,
    )

    if _is_probably_placeholder_content_type(declared_type):
        headers["Content-Type"] = media.mime_type.value

    if not declared_encoding and media.codec is not None:
        headers["Content-Encoding"] = media.codec.name

    if _parse_content_length(headers) is None:
        headers["Content-Length"] = str(body.size)

    return media


# ---------------------------------------------------------------------------
# Arrow record parsing helpers
# ---------------------------------------------------------------------------

def _parse_headers(obj: Mapping[str, Any], *, prefix: str) -> MutableMapping[str, str]:
    """Extract and normalise headers from an arbitrary dict representation.

    Tries common field names (``headers``, ``header``, ``hdrs``, …) with
    and without *prefix*.  When no header dict is found, falls back to
    assembling promoted headers from individual fields
    (``host``, ``user_agent``, …).

    Parameters
    ----------
    obj:
        Source mapping (e.g. a JSON-decoded response record).
    prefix:
        Key prefix to try first (e.g. ``"response_"``).

    Returns
    -------
    MutableMapping[str, str]
        A ``{header-name: value}`` dict (may be empty).
    """
    headers = get_from_dict(obj, keys=("headers", "header", "hdrs", "response_headers"), prefix=prefix)
    if headers is MISSING:
        headers = get_from_dict(obj, keys=("headers", "header", "hdrs", "response_headers"), prefix="")

    if isinstance(headers, Mapping):
        parsed = {str(k): str(v) for k, v in headers.items()}
        if parsed:
            return parsed

    dumped_headers = {
        "Host": get_from_dict(obj, keys=("host",), prefix=prefix),
        "User-Agent": get_from_dict(obj, keys=("user_agent",), prefix=prefix),
        "Accept": get_from_dict(obj, keys=("accept",), prefix=prefix),
        "Accept-Encoding": get_from_dict(obj, keys=("accept_encoding",), prefix=prefix),
        "Accept-Language": get_from_dict(obj, keys=("accept_language",), prefix=prefix),
        "Content-Type": get_from_dict(obj, keys=("content_type",), prefix=prefix),
        "Content-Length": get_from_dict(obj, keys=("content_length",), prefix=prefix),
        "Content-Encoding": get_from_dict(obj, keys=("content_encoding",), prefix=prefix),
        "Transfer-Encoding": get_from_dict(obj, keys=("transfer_encoding",), prefix=prefix),
        "Location": get_from_dict(obj, keys=("location",), prefix=prefix),
        "ETag": get_from_dict(obj, keys=("etag",), prefix=prefix),
        "Last-Modified": get_from_dict(obj, keys=("last_modified",), prefix=prefix),
    }

    return {
        header_name: str(value)
        for header_name, value in dumped_headers.items()
        if value is not MISSING and value not in (None, "")
    }


def _parse_tags(obj: Mapping[str, Any], *, prefix: str) -> dict[str, str]:
    """Extract ``tags`` from *obj*, trying *prefix* first, then no prefix.

    Parameters
    ----------
    obj:
        Source mapping.
    prefix:
        Key prefix to try first.

    Returns
    -------
    dict[str, str]
        Tag mapping (may be empty).
    """
    tags = get_from_dict(obj, keys=("tags", "response_tags"), prefix=prefix)
    if tags is MISSING:
        tags = get_from_dict(obj, keys=("tags", "response_tags"), prefix="")
    if not isinstance(tags, Mapping):
        return {}
    return {str(k): str(v) for k, v in tags.items()}


def _parse_buffer(obj: Mapping[str, Any], *, prefix: str) -> BytesIO:
    """Extract the response body from *obj* into a :class:`BytesIO`.

    Tries common field names (``buffer``, ``body``, ``content``, ``data``,
    ``response_body``) with and without *prefix*.

    Parameters
    ----------
    obj:
        Source mapping.
    prefix:
        Key prefix to try first.

    Returns
    -------
    BytesIO
        An empty buffer when the body field is absent or ``None``.
    """
    body = get_from_dict(obj, keys=("buffer", "body", "content", "data", "response_body"), prefix=prefix)
    if body is MISSING:
        body = get_from_dict(obj, keys=("buffer", "body", "content", "data", "response_body"), prefix="")
    if body is MISSING or body is None:
        return BytesIO()
    return BytesIO.parse(obj=body)


def _parse_status_code(obj: Mapping[str, Any], *, prefix: str) -> int:
    """Extract and parse the HTTP status code from *obj*.

    Parameters
    ----------
    obj:
        Source mapping.
    prefix:
        Key prefix to try first.

    Returns
    -------
    int
        The HTTP status code.

    Raises
    ------
    ValueError
        If no status code field is found.
    """
    status = get_from_dict(obj, keys=("status_code", "status", "code"), prefix=prefix)
    if status is MISSING:
        status = get_from_dict(obj, keys=("status_code", "status", "code"), prefix="")
    if status is MISSING or status in (None, ""):
        raise ValueError("Response.parse_dict: missing status_code/status/code")
    return int(status) if isinstance(status, int) else int(float(str(status).strip()))


def _parse_received_at_timestamp(obj: Mapping[str, Any], *, prefix: str) -> int:
    """Extract the received-at timestamp (microseconds since epoch) from *obj*.

    Tries several common field names in order:
    ``received_at_timestamp``, ``received_at``, ``time_us``, ``timestamp``,
    ``time_ns``, ``received_at_epoch``, ``response_received_at_epoch``,
    ``response_received_at``.

    Parameters
    ----------
    obj:
        Source mapping.
    prefix:
        Key prefix to try first.

    Returns
    -------
    int
        Microseconds since the Unix epoch (UTC).  Returns ``0`` when absent.
    """
    _KEYS = (
        "received_at_timestamp", "received_at", "time_us", "timestamp",
        "time_ns", "received_at_epoch", "response_received_at_epoch",
        "response_received_at",
    )
    value = get_from_dict(obj, keys=_KEYS, prefix=prefix)
    if value is MISSING:
        value = get_from_dict(obj, keys=_KEYS, prefix="")
    if value is MISSING or value in (None, ""):
        return 0
    return int(value) if isinstance(value, int) else int(float(str(value).strip()))


def _arrow_ts_col_to_us(col: pa.ChunkedArray | pa.Array, i: int) -> int:
    """Read a single Arrow timestamp scalar as microseconds since epoch.

    Parameters
    ----------
    col:
        An Arrow timestamp column (any unit).
    i:
        Row index.

    Returns
    -------
    int
        The integer timestamp value (native to the column unit), or ``0``
        when the scalar is null.
    """
    scalar = col[i]
    if scalar is None or not scalar.is_valid:
        return 0
    return int(scalar.value) if scalar.value is not None else 0


def _map_to_str_dict(value: Any) -> dict[str, str]:
    """Coerce *value* to a ``dict[str, str]``, dropping ``None`` pairs.

    Parameters
    ----------
    value:
        A mapping, an iterable of ``(key, value)`` pairs, or anything else.
        Non-mapping, non-iterable values return an empty dict.

    Returns
    -------
    dict[str, str]
    """
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if k is not None and v is not None}
    try:
        return {str(k): str(v) for k, v in value if k is not None and v is not None}
    except Exception:
        return {}


def _first_present(cols: Mapping[str, Any], i: int, *names: str) -> Any:
    """Return ``cols[name][i].as_py()`` for the first *name* found in *cols*.

    Parameters
    ----------
    cols:
        ``{column_name: arrow_column}`` mapping.
    i:
        Row index.
    *names:
        Column names to try in order.

    Returns
    -------
    Any
        The Python scalar at row *i*, or ``None`` if none of the names
        exist in *cols*.
    """
    for name in names:
        if name in cols:
            return cols[name][i].as_py()
    return None


# ---------------------------------------------------------------------------
# Hop-by-hop header names (RFC 7230 §6.1)
# ---------------------------------------------------------------------------

_HOP_BY_HOP: frozenset[str] = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
})

# ---------------------------------------------------------------------------
# Arrow schemas
# ---------------------------------------------------------------------------

ARROW_SCHEMA = pa.schema(
    [
        pa.field(
            "response_status_code",
            pa.int32(),
            nullable=False,
            metadata={"comment": "HTTP status code returned by the server"},
        ),
        pa.field("response_host", pa.string(), nullable=True,
                 metadata={"comment": "Host header"}),
        pa.field("response_user_agent", pa.string(), nullable=True,
                 metadata={"comment": "User-Agent header"}),
        pa.field("response_accept", pa.string(), nullable=True,
                 metadata={"comment": "Accept header"}),
        pa.field("response_accept_encoding", pa.string(), nullable=True,
                 metadata={"comment": "Accept-Encoding header"}),
        pa.field("response_accept_language", pa.string(), nullable=True,
                 metadata={"comment": "Accept-Language header"}),
        pa.field("response_content_type", pa.string(), nullable=True,
                 metadata={"comment": "Content-Type header"}),
        pa.field(
            "response_content_length",
            pa.int64(),
            nullable=False,
            metadata={"comment": "Content-Length header parsed as integer"},
        ),
        pa.field("response_content_encoding", pa.string(), nullable=True,
                 metadata={"comment": "Content-Encoding header"}),
        pa.field("response_transfer_encoding", pa.string(), nullable=True,
                 metadata={"comment": "Transfer-Encoding header"}),
        pa.field(
            "response_headers",
            pa.map_(pa.string(), pa.string()),
            nullable=False,
            metadata={
                "comment": "Response headers excluding promoted common headers",
                "keys_sorted": "false",
            },
        ),
        pa.field(
            "response_tags",
            pa.map_(pa.string(), pa.string()),
            nullable=False,
            metadata={"comment": "Arbitrary string tags attached to this response"},
        ),
        pa.field(
            "response_body",
            pa.binary(),
            nullable=True,
            metadata={"comment": "Raw binary payload of the response"},
        ),
        pa.field(
            "response_body_hash",
            pa.int64(),
            nullable=True,
            metadata={
                "comment": "Signed Int64 XXH3 digest of response_body",
                "algorithm": "xxh3_64",
            },
        ),
        pa.field(
            "response_received_at",
            pa.timestamp("us", "UTC"),
            nullable=False,
            metadata={
                "comment": "UTC timestamp when the response was captured",
                "unit": "us",
                "tz": "UTC",
            },
        ),
        pa.field(
            "response_received_at_epoch",
            pa.int64(),
            nullable=False,
            metadata={
                "comment": "Microseconds since Unix epoch (UTC) when the response was captured",
                "unit": "us",
            },
        ),
    ],
    metadata={
        "comment": (
            "Response record (single row), designed for deterministic "
            "logging and replay."
        ),
    },
)

RESPONSE_ARROW_SCHEMA = pa.schema(
    list(REQUEST_ARROW_SCHEMA) + list(ARROW_SCHEMA),
    metadata={
        "comment": (
            "Prepared request and response flattened into a single row "
            "schema for Delta-table caching."
        ),
    },
)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class Response:
    """An HTTP response with its originating request and a smart body buffer.

    :class:`Response` is the central data object in the ``yggdrasil.io``
    stack.  It is produced by :meth:`~yggdrasil.io.session.Session.send`,
    stored in Delta tables by
    :meth:`~yggdrasil.io.session.Session.send_many`, and can be round-tripped
    through Arrow via :meth:`to_arrow_batch` / :meth:`from_arrow`.

    Parameters
    ----------
    request:
        The prepared request that produced this response.
    status_code:
        HTTP status code (e.g. 200, 404).
    headers:
        Mutable response header mapping.  ``Content-Type``,
        ``Content-Encoding``, and ``Content-Length`` are back-filled from the
        body when absent.
    tags:
        Arbitrary string metadata attached at response creation time.
    buffer:
        The response body as a :class:`~yggdrasil.io.buffer.BytesIO`.  Knows
        its own media type and supports lazy decompression.
    received_at_timestamp:
        UTC microseconds since the Unix epoch when the response was captured.

    Notes
    -----
    **Subclassing** — :class:`~yggdrasil.io.http_.response.HTTPResponse`
    extends this class with urllib3-specific construction helpers
    (:meth:`from_urllib3`, :meth:`drain_urllib3`).

    :meth:`from_arrow` auto-dispatches to :class:`HTTPResponse` for HTTP/HTTPS
    URLs when called as ``Response.from_arrow(…)`` (the concrete subclass is
    used when called directly on it).

    **Thread safety** — instances are *not* thread-safe.  Clone with
    :func:`dataclasses.replace` before sharing across threads.

    Examples
    --------
    Round-trip through Arrow::

        batch  = response.to_arrow_batch()         # raw schema row
        table  = response.to_arrow_table()         # pa.Table wrapping the batch
        parquet_bytes = response.to_parquet()      # Parquet bytes
        ipc_bytes     = response.to_ipc()          # Arrow IPC stream bytes

    Parse the body into a typed table::

        df = response.to_polars(parse=True)
        tbl = response.to_arrow_table(parse=True)  # parsed body columns
    """

    request: PreparedRequest
    status_code: int
    headers: MutableMapping[str, str]
    tags: MutableMapping[str, str]
    buffer: BytesIO
    received_at_timestamp: int

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        url = self.request.url.to_string() if self.request else "?"
        return (
            f"{type(self).__name__}("
            f"status_code={self.status_code}, "
            f"url={url!r}, "
            f"content_length={self.buffer.size if self.buffer else 0}"
            f")"
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, obj: Any, *, normalize: bool = True) -> "Response":
        """Coerce *obj* into a :class:`Response`.

        Dispatches to :meth:`parse_str` for strings and :meth:`parse_dict`
        for mappings.  An existing :class:`Response` is returned unchanged.

        Parameters
        ----------
        obj:
            A :class:`Response`, a JSON string, or a ``Mapping``.
        normalize:
            Whether to normalise headers and the URL.

        Returns
        -------
        Response
        """
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, str):
            return cls.parse_str(obj, normalize=normalize)
        if isinstance(obj, Mapping):
            return cls.parse_dict(obj, normalize=normalize)
        return cls.parse_str(str(obj), normalize=normalize)

    @classmethod
    def parse_str(cls, raw: str, *, normalize: bool = True) -> "Response":
        """Parse a JSON string into a :class:`Response`.

        Parameters
        ----------
        raw:
            A JSON object string encoding a response record.
        normalize:
            Whether to normalise headers and the URL.

        Returns
        -------
        Response

        Raises
        ------
        ValueError
            If *raw* is empty, not valid JSON, or not a JSON object.
        """
        s = raw.strip()
        if not s:
            raise ValueError("Response.parse_str: empty string")
        try:
            d = json_module.loads(s)
        except Exception as exc:
            raise ValueError("Response.parse_str: expected JSON object string") from exc
        if not isinstance(d, Mapping):
            raise ValueError("Response.parse_str: JSON must decode to a mapping")
        return cls.parse_dict(d, normalize=normalize)

    @classmethod
    def parse_dict(
        cls,
        obj: Mapping[str, Any],
        *,
        normalize: bool = True,
        prefix: str = "response_",
    ) -> "Response":
        """Build a :class:`Response` from a flat or nested mapping.

        The mapping may be a raw JSON-decoded dict, a flat Arrow-schema row,
        or any structure where request fields live at the top level or under a
        ``"request"`` key.

        Parameters
        ----------
        obj:
            Source mapping.
        normalize:
            Whether to normalise headers (back-fill Content-Type, …) and URLs.
        prefix:
            Prefix used when looking up response-specific fields
            (default ``"response_"``).

        Returns
        -------
        Response

        Raises
        ------
        ValueError
            If *obj* is empty or the status code cannot be parsed.
        """
        if not obj:
            raise ValueError("Response.parse_dict: empty mapping")

        req_obj = get_from_dict(obj, keys=("request",), prefix="")
        request = PreparedRequest.parse(
            obj if req_obj is MISSING or req_obj in (None, "") else req_obj,
            normalize=normalize,
        )
        status_code = _parse_status_code(obj, prefix=prefix)
        headers = _parse_headers(obj, prefix=prefix)
        buffer = _parse_buffer(obj, prefix=prefix)
        received_at_timestamp = _parse_received_at_timestamp(obj, prefix=prefix)
        tags = _parse_tags(obj, prefix=prefix)

        if normalize:
            headers = normalize_headers(headers, body=buffer, is_request=False)

        _ensure_media_headers(headers, buffer)

        return cls(
            request=request,
            status_code=status_code,
            headers=headers,
            buffer=buffer,
            received_at_timestamp=received_at_timestamp,
            tags=tags,
        )

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def update_headers(
        self,
        headers: MutableMapping[str, str],
        normalize: bool = True,
    ) -> "Response":
        """Merge *headers* into :attr:`headers` in-place and return *self*.

        Parameters
        ----------
        headers:
            Headers to merge.  Existing keys are overwritten.
        normalize:
            When ``True``, back-fills ``Content-Type``, ``Content-Encoding``,
            and ``Content-Length`` from the body after merging.

        Returns
        -------
        Response
            *self* (for chaining).
        """
        if not headers:
            return self

        if not self.headers:
            self.headers = dict(headers)
        else:
            for k, v in headers.items():
                self.headers[str(k)] = str(v)

        if normalize:
            _ensure_media_headers(self.headers, self.buffer)

        return self

    def update_tags(
        self,
        tags: MutableMapping[str, str],
    ) -> "Response":
        """Merge *tags* into :attr:`tags` in-place and return *self*.

        Parameters
        ----------
        tags:
            Tags to merge.  Existing keys are overwritten.

        Returns
        -------
        Response
            *self* (for chaining).
        """
        if not tags:
            return self

        if not self.tags:
            self.tags = dict(tags)
        else:
            self.tags.update(tags)

        return self

    # ------------------------------------------------------------------
    # Media type
    # ------------------------------------------------------------------

    @property
    def media_type(self) -> MediaType:
        """Resolved :class:`~yggdrasil.io.enums.MediaType` of the body.

        Back-fills ``Content-Type``, ``Content-Encoding``, and
        ``Content-Length`` into :attr:`headers` on first access.
        """
        if self.headers is None:
            self.headers = {}
        return _ensure_media_headers(self.headers, self.buffer)

    @media_type.setter
    def media_type(self, value: MediaType) -> None:
        """Set the media type (calls :meth:`set_media_type` with ``safe=True``)."""
        self.set_media_type(value, safe=True)

    def set_media_type(
        self,
        value: MediaType,
        *,
        safe: bool = True,
    ) -> "Response":
        """Update the media type on both the buffer and the headers.

        Parameters
        ----------
        value:
            New :class:`~yggdrasil.io.enums.MediaType`.
        safe:
            When ``True``, raises if the media type cannot be parsed.

        Returns
        -------
        Response
            *self* (for chaining).
        """
        if self.headers is None:
            self.headers = {}

        self.request.accept_media_type = value
        self.buffer.set_media_type(value, safe=safe)
        self.headers["Content-Type"] = value.mime_type.value

        if value.codec is not None:
            self.headers["Content-Encoding"] = value.codec.name
        elif "Content-Encoding" in self.headers:
            del self.headers["Content-Encoding"]

        self.headers["Content-Length"] = str(self.buffer.size)
        return self

    # ------------------------------------------------------------------
    # Body accessors
    # ------------------------------------------------------------------

    @property
    def body(self) -> BytesIO:
        """Alias for :attr:`buffer` — the raw response body."""
        return self.buffer

    @property
    def codec(self) -> Optional[Codec]:
        """The body compression codec, or ``None`` if uncompressed."""
        return self.media_type.codec

    @property
    def content(self) -> bytes:
        """Return the response body as raw bytes, decompressing if needed."""
        codec = self.codec
        if codec is not None:
            with self.buffer.decompress(codec=codec, copy=True) as b:
                return b.to_bytes()
        return self.buffer.to_bytes()

    @property
    def text(self) -> str:
        """Return the response body decoded as text.

        The charset is read from the ``Content-Type`` header (defaults to
        ``"utf-8"``).  Decoding errors are replaced with the Unicode
        replacement character.
        """
        return self.content.decode(_get_charset(self.headers), errors="replace")

    def json(
        self,
        orient: Optional[Literal["records", "split", "index", "columns", "values"]] = None,
        *,
        media_type: Optional[MediaType] = None,
    ) -> Any:
        """Deserialise the body as JSON.

        Parameters
        ----------
        orient:
            Pandas-compatible orient hint forwarded to
            :meth:`BytesIO.json_load`.  ``None`` returns the raw decoded
            Python object.
        media_type:
            Override the media type used during deserialisation.

        Returns
        -------
        Any
            The decoded Python object (list, dict, …).
        """
        return self.buffer.json_load(
            orient=orient,
            media_type=media_type or self.media_type,
        )

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def ok(self) -> bool:
        """``True`` when :attr:`status_code` is in the 2xx–3xx range."""
        return 200 <= self.status_code < 400

    def raise_for_status(self) -> None:
        """Raise the response error if :attr:`ok` is ``False``.

        Raises
        ------
        ResponseError
            Built by :func:`~yggdrasil.io.errors.make_for_status`.
        """
        if not self.ok:
            raise self.error()

    def error(self) -> Optional[Exception]:
        """Return the response error, or ``None`` when :attr:`ok` is ``True``.

        Returns
        -------
        Exception | None
        """
        if not self.ok:
            from .errors import make_for_status
            return make_for_status(self)
        return None

    # ------------------------------------------------------------------
    # Anonymisation
    # ------------------------------------------------------------------

    def anonymize(self, mode: Literal["remove", "redact"] = "remove") -> "Response":
        """Return a copy with sensitive header values stripped or redacted.

        Parameters
        ----------
        mode:
            * ``"remove"`` — sensitive values are replaced with ``None``.
            * ``"redact"`` — sensitive values are replaced with
              ``"<REDACTED>"``.

        Returns
        -------
        Response
            A shallow copy via :func:`dataclasses.replace`.
        """
        return replace(
            self,
            request=self.request.anonymize(mode=mode),
            headers=normalize_headers(
                self.headers,
                is_request=False,
                mode=mode,
                body=self.body,
                anonymize=True,
            ),
        )

    # ------------------------------------------------------------------
    # Serialisation — Arrow
    # ------------------------------------------------------------------

    def to_arrow_batch(self, parse: bool = False) -> pa.RecordBatch:
        """Serialise the response to a single-row :class:`pyarrow.RecordBatch`.

        Two modes are supported:

        ``parse=False`` (default)
            Returns a row conforming to :data:`RESPONSE_ARROW_SCHEMA` —
            request + response metadata flattened into columns.  Used for
            Delta-table caching and replay.

        ``parse=True``
            Parses the body (via Polars) and returns a batch whose schema
            reflects the *body* data, not the response metadata.  Useful
            when the body is a structured table (Parquet, Arrow IPC, JSON).

        Parameters
        ----------
        parse:
            When ``True``, parse the body as a structured table.

        Returns
        -------
        pyarrow.RecordBatch

        Raises
        ------
        NotImplementedError
            ``parse=True`` is delegated via Polars; unsupported body formats
            will raise inside the Polars read step.
        """
        if parse:
            from yggdrasil.polars.cast import polars_dataframe_to_arrow_table
            return polars_dataframe_to_arrow_table(
                self.to_polars(parse=True)
            ).to_batches()[0]

        req_rb = self.request.to_arrow_batch(parse=False)
        promoted = PromotedHeaders.extract(self.headers or {}, host=DEFAULT_HOSTNAME)
        tags_v = {str(k): str(v) for k, v in (self.tags or {}).items()}

        if self.buffer is not None:
            body_bytes = self.buffer.to_bytes()
            body_hash = self.buffer.xxh3_int64()
        else:
            body_bytes = None
            body_hash = None

        values = {
            "response_status_code": self.status_code,
            "response_host": promoted.host or DEFAULT_HOSTNAME,
            "response_user_agent": promoted.user_agent,
            "response_accept": promoted.accept,
            "response_accept_encoding": promoted.accept_encoding,
            "response_accept_language": promoted.accept_language,
            "response_content_type": promoted.content_type,
            "response_content_length": promoted.content_length,
            "response_content_encoding": promoted.content_encoding,
            "response_transfer_encoding": promoted.transfer_encoding,
            "response_headers": promoted.remaining,
            "response_tags": tags_v,
            "response_body": body_bytes,
            "response_body_hash": body_hash,
            "response_received_at": self.received_at_timestamp,
            "response_received_at_epoch": self.received_at_timestamp,
        }

        return pa.RecordBatch.from_arrays(
            list(req_rb.columns) + [
                pa.array([values[f.name]], type=f.type)
                for f in ARROW_SCHEMA
            ],
            schema=RESPONSE_ARROW_SCHEMA,
        )

    def to_arrow_table(self, parse: bool = False) -> pa.Table:
        """Serialise the response to a :class:`pyarrow.Table`.

        Wraps :meth:`to_arrow_batch` in a single-row table.

        Parameters
        ----------
        parse:
            Forwarded to :meth:`to_arrow_batch`.  When ``True``, the body is
            parsed and the table schema reflects the body data.

        Returns
        -------
        pyarrow.Table
            A single-row table (``parse=False``) or a parsed body table
            (``parse=True``).
        """
        if parse:
            from yggdrasil.polars.cast import polars_dataframe_to_arrow_table
            return polars_dataframe_to_arrow_table(self.to_polars(parse=True))

        return pa.Table.from_batches([self.to_arrow_batch(parse=False)])

    def to_parquet(
        self,
        parse: bool = False,
        *,
        compression: str = "snappy",
        **kwargs: Any,
    ) -> bytes:
        """Serialise the response to Parquet bytes.

        Parameters
        ----------
        parse:
            When ``True``, encodes the parsed body table; when ``False``
            (default), encodes the raw :data:`RESPONSE_ARROW_SCHEMA` row.
        compression:
            Parquet compression codec (``"snappy"``, ``"zstd"``, ``"lz4"``,
            ``"none"``, …).  Defaults to ``"snappy"``.
        **kwargs:
            Forwarded to :func:`pyarrow.parquet.write_table`.

        Returns
        -------
        bytes
            In-memory Parquet file.
        """
        import io as _io
        table = self.to_arrow_table(parse=parse)
        buf = _io.BytesIO()
        pq.write_table(table, buf, compression=compression, **kwargs)
        return buf.getvalue()

    def to_ipc(self, parse: bool = False) -> bytes:
        """Serialise the response to Arrow IPC stream bytes.

        Parameters
        ----------
        parse:
            When ``True``, encodes the parsed body table; when ``False``
            (default), encodes the raw :data:`RESPONSE_ARROW_SCHEMA` row.

        Returns
        -------
        bytes
            Arrow IPC stream (``schema + record_batch + EOS``).
        """
        import io as _io
        table = self.to_arrow_table(parse=parse)
        buf = _io.BytesIO()
        with pa_ipc.new_stream(buf, table.schema) as writer:
            writer.write_table(table)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Serialisation — Polars / pandas
    # ------------------------------------------------------------------

    def to_polars(
        self,
        parse: bool = True,
        *,
        lazy: bool = False,
    ) -> "pl.DataFrame | pl.LazyFrame":
        """Deserialise the body as a Polars DataFrame.

        Parameters
        ----------
        parse:
            * ``True`` (default) — decode the body using the detected
              ``MediaType`` (Parquet, Arrow IPC, JSON, …).
            * ``False`` — convert the raw :data:`RESPONSE_ARROW_SCHEMA`
              Arrow batch to a Polars frame without body parsing.
        lazy:
            When ``True``, return a :class:`polars.LazyFrame` instead of a
            materialised :class:`polars.DataFrame`.  Only honoured when
            ``parse=True``.

        Returns
        -------
        polars.DataFrame | polars.LazyFrame
        """
        from yggdrasil.polars.lib import polars as _pl

        if parse:
            mt = self.media_type
            mio = self.buffer.media_io(media=mt)
            return mio.read_polars_frame(lazy=lazy)

        return _pl.from_arrow(self.to_arrow_batch(parse=False))

    def to_pandas(self, parse: bool = True) -> "pd.DataFrame":
        """Deserialise the body as a pandas DataFrame.

        Parameters
        ----------
        parse:
            Forwarded to :meth:`to_polars`.

        Returns
        -------
        pandas.DataFrame
        """
        return self.to_polars(parse=parse).to_pandas()

    # ------------------------------------------------------------------
    # Arrow deserialization
    # ------------------------------------------------------------------

    @classmethod
    def from_arrow(
        cls,
        batch: pa.RecordBatch | pa.Table,
        *,
        parse: bool = False,
        normalize: bool = True,
    ) -> Iterator["Response"]:
        """Yield :class:`Response` objects from an Arrow batch or table.

        Each row in *batch* / *table* is reconstructed into a
        :class:`Response`.  When called as ``Response.from_arrow(…)`` the
        method auto-dispatches to
        :class:`~yggdrasil.io.http_.response.HTTPResponse` for rows whose
        request URL uses an HTTP/HTTPS scheme.

        Parameters
        ----------
        batch:
            An Arrow :class:`~pyarrow.RecordBatch` or :class:`~pyarrow.Table`
            conforming to :data:`RESPONSE_ARROW_SCHEMA`.
        parse:
            Reserved for future use (body parsing on deserialisation).
            Currently raises :exc:`NotImplementedError` when ``True``.
        normalize:
            Whether to normalise reconstructed headers and URLs.

        Yields
        ------
        Response
            One instance per row, in row order.

        Raises
        ------
        NotImplementedError
            When ``parse=True``.
        """
        if parse:
            raise NotImplementedError("parse=True is not yet implemented for from_arrow")

        req_cols = [f.name for f in REQUEST_ARROW_SCHEMA]
        resp_cols = [f.name for f in ARROW_SCHEMA]

        def _iter_batches(obj: pa.RecordBatch | pa.Table) -> Iterator[pa.RecordBatch]:
            if isinstance(obj, pa.RecordBatch):
                yield obj
            else:
                yield from obj.to_batches()

        for rb in _iter_batches(batch):
            cols = {
                name: rb.column(name)
                for name in req_cols + resp_cols
                if name in rb.schema.names
            }

            for i in range(rb.num_rows):
                # --- Reconstruct the request ---
                method = _first_present(cols, i, "request_method") or "GET"

                url_str = _first_present(cols, i, "request_url_str")
                if url_str not in (None, ""):
                    req_url_str: str | None = str(url_str)
                    req_url_struct: Any = None
                else:
                    scheme    = _first_present(cols, i, "request_url_scheme")
                    userinfo  = _first_present(cols, i, "request_url_userinfo")
                    host      = _first_present(cols, i, "request_url_host")
                    port      = _first_present(cols, i, "request_url_port")
                    path      = _first_present(cols, i, "request_url_path")
                    query     = _first_present(cols, i, "request_url_query")
                    fragment  = _first_present(cols, i, "request_url_fragment")

                    if any(part not in (None, "", 0)
                           for part in (scheme, userinfo, host, port, path, query, fragment)):
                        req_url_str = None
                        req_url_struct = {
                            "scheme":   scheme or "",
                            "userinfo": userinfo or "",
                            "host":     host or "",
                            "port":     0 if port in (None, "") else int(port),
                            "path":     path or "/",
                            "query":    query or "",
                            "fragment": fragment or "",
                        }
                    else:
                        legacy = _first_present(cols, i, "request_url")
                        req_url_str = None
                        req_url_struct = legacy if isinstance(legacy, Mapping) else ""

                req_headers = _map_to_str_dict(_first_present(cols, i, "request_headers"))
                for hk, col_name in (
                    ("Host",              "request_host"),
                    ("User-Agent",        "request_user_agent"),
                    ("Accept",            "request_accept"),
                    ("Accept-Encoding",   "request_accept_encoding"),
                    ("Accept-Language",   "request_accept_language"),
                    ("Content-Type",      "request_content_type"),
                    ("Content-Length",    "request_content_length"),
                    ("Content-Encoding",  "request_content_encoding"),
                    ("Transfer-Encoding", "request_transfer_encoding"),
                ):
                    hv = _first_present(cols, i, col_name)
                    if hv not in (None, ""):
                        req_headers[hk] = str(hv)

                request = PreparedRequest.parse_dict(
                    {
                        "method":             method,
                        "url_str":            req_url_str,
                        "url":                req_url_struct,
                        "headers":            req_headers,
                        "tags":               _map_to_str_dict(_first_present(cols, i, "request_tags")),
                        "buffer":             _first_present(cols, i, "request_body"),
                        "sent_at_timestamp":  (
                            _arrow_ts_col_to_us(cols["request_sent_at"], i)
                            if "request_sent_at" in cols else 0
                        ),
                    },
                    normalize=normalize,
                )

                # --- Reconstruct response headers ---
                resp_headers = _map_to_str_dict(_first_present(cols, i, "response_headers"))
                for hk, col_name in (
                    ("Host",              "response_host"),
                    ("User-Agent",        "response_user_agent"),
                    ("Accept",            "response_accept"),
                    ("Accept-Encoding",   "response_accept_encoding"),
                    ("Accept-Language",   "response_accept_language"),
                    ("Content-Type",      "response_content_type"),
                    ("Content-Length",    "response_content_length"),
                    ("Content-Encoding",  "response_content_encoding"),
                    ("Transfer-Encoding", "response_transfer_encoding"),
                ):
                    hv = _first_present(cols, i, col_name)
                    if hv not in (None, ""):
                        resp_headers[hk] = str(hv)

                body_bytes = _first_present(cols, i, "response_body")
                buffer = BytesIO(body_bytes) if body_bytes is not None else BytesIO()

                if normalize:
                    resp_headers = normalize_headers(
                        resp_headers, is_request=False, body=buffer
                    )

                _ensure_media_headers(resp_headers, buffer)

                received_at = (
                    _arrow_ts_col_to_us(cols["response_received_at"], i)
                    if "response_received_at" in cols else 0
                )

                kwargs = dict(
                    request=request,
                    status_code=int(_first_present(cols, i, "response_status_code") or 0),
                    headers=resp_headers,
                    buffer=buffer,
                    tags=_map_to_str_dict(_first_present(cols, i, "response_tags")),
                    received_at_timestamp=received_at,
                )

                if cls is Response and request.url.is_http:
                    from .http_.response import HTTPResponse
                    yield HTTPResponse(**kwargs)
                else:
                    yield cls(**kwargs)

    # ------------------------------------------------------------------
    # ASGI helpers
    # ------------------------------------------------------------------

    def _to_asgi_payload(self) -> tuple[bytes, dict[str, str], str]:
        """Build a ``(body, headers, media_type_str)`` triple for ASGI frameworks.

        Hop-by-hop headers are stripped.  ``Content-Length`` is recalculated
        from the serialised body.  ``Content-Encoding`` is added when the
        body is compressed and the header is not already present.

        Returns
        -------
        tuple[bytes, dict[str, str], str]
            ``(raw_body, headers_dict, mime_type_string)``
        """
        body = self.buffer.to_bytes() if self.buffer is not None else b""
        media = self.media_type

        headers = {
            str(k): str(v)
            for k, v in (self.headers or {}).items()
            if str(k).lower() not in _HOP_BY_HOP
        }
        _pop_header(headers, "Content-Type")
        headers["Content-Length"] = str(len(body))

        if media.codec is not None and _parse_content_encoding(headers) is None:
            headers["Content-Encoding"] = media.codec.name

        return body, headers, media.mime_type.value

    def to_starlette(self) -> "StarletteResponse":
        """Convert to a :class:`starlette.responses.Response`.

        Hop-by-hop headers are stripped and ``Content-Length`` is recalculated.

        Returns
        -------
        starlette.responses.Response
        """
        from starlette.responses import Response as _StarletteResponse
        body, headers, media_type = self._to_asgi_payload()
        return _StarletteResponse(
            content=body,
            status_code=self.status_code,
            headers=headers,
            media_type=media_type,
        )

    def to_fastapi(self) -> "FastAPIResponse":
        """Convert to a :class:`fastapi.Response`.

        Falls back to :meth:`to_starlette` when FastAPI is not installed.

        Returns
        -------
        fastapi.Response | starlette.responses.Response
        """
        try:
            from fastapi import Response as _FastAPIResponse
        except ImportError:
            return self.to_starlette()  # type: ignore[return-value]
        body, headers, media_type = self._to_asgi_payload()
        return _FastAPIResponse(
            content=body,
            status_code=self.status_code,
            headers=headers,
            media_type=media_type,
        )

    # ------------------------------------------------------------------
    # Functional helper
    # ------------------------------------------------------------------

    def apply(self, func: Callable[["Response"], "Response"]) -> "Response":
        """Apply *func* to *self* and return the result.

        Useful for chaining transformations::

            resp = session.get("/data").apply(my_transform)

        Parameters
        ----------
        func:
            A callable that accepts and returns a :class:`Response`.

        Returns
        -------
        Response
        """
        return func(self)

