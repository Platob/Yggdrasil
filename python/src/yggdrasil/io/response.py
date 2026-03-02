from __future__ import annotations

import copy
import json
from dataclasses import dataclass, is_dataclass, replace, MISSING
from typing import Mapping, Any, Iterable, Literal, Sequence, Iterator, TYPE_CHECKING, MutableMapping
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import pyarrow as pa

from yggdrasil.io import MediaType
from yggdrasil.io.headers import anonymize_headers
from .buffer import BytesIO
from .request import PreparedRequest, REQUEST_ARROW_SCHEMA
from ..dataclasses.dataclass import get_from_dict

if TYPE_CHECKING:
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
# Charset / content-type helpers
# ---------------------------------------------------------------------------

def _get_charset(headers: Mapping[str, str]) -> str:
    ct = ""
    for k, v in headers.items():
        if k.lower() == "content-type":
            ct = v
            break
    if not ct:
        return "utf-8"
    parts = [p.strip() for p in ct.split(";")]
    for p in parts[1:]:
        if p.lower().startswith("charset="):
            return p.split("=", 1)[1].strip() or "utf-8"
    return "utf-8"


def _detect_content_type(headers: Mapping[str, str]) -> str | None:
    """Return the bare content-type token (no params) from response headers."""
    for k, v in headers.items():
        if k.lower() == "content-type":
            return v.split(";")[0].strip().lower()
    return None


# ---------------------------------------------------------------------------
# Arrow schemas
# ---------------------------------------------------------------------------

ARROW_SCHEMA = pa.schema(
    [
        pa.field(
            name="response_status_code",
            type=pa.int32(),
            nullable=False,
            metadata={"comment": "HTTP status code returned by the server"},
        ),
        pa.field(
            "response_headers",
            pa.map_(pa.string(), pa.string()),
            nullable=True,
            metadata={"comment": "Raw HTTP response headers as ordered key/value pairs"},
        ),
        pa.field(
            "response_tags",
            pa.map_(pa.string(), pa.string()),
            nullable=True,
            metadata={"comment": "Arbitrary string tags attached to this response"},
        ),
        pa.field(
            name="response_body",
            type=pa.binary(),
            nullable=True,
            metadata={"comment": "Raw binary payload of the response (bytes)"},
        ),
        pa.field(
            name="response_body_hash",
            type=pa.binary(),
            nullable=True,
            metadata={
                "comment": "256-bit BLAKE3 digest of response_body (32 bytes)",
                "algorithm": "blake3",
            },
        ),
        pa.field(
            name="response_received_at",
            type=pa.timestamp("us", "UTC"),
            nullable=False,
            metadata={
                "comment": "UTC timestamp when the response was captured",
                "unit": "us",
                "tz": "UTC",
            },
        ),
        pa.field(
            name="response_received_at_epoch",
            type=pa.int64(),
            nullable=False,
            metadata={
                "comment": "UTC epoch timestamp when the response was captured",
                "unit": "us",
                "tz": "UTC",
            },
        ),
    ],
    metadata={
        "comment": "HTTP response record (single row), designed for deterministic logging + replay.",
    },
)

RESPONSE_ARROW_SCHEMA = pa.schema(
    list(REQUEST_ARROW_SCHEMA) + list(ARROW_SCHEMA),
    metadata={
        "comment": "HTTP prepared request and response flattened into columns for single-row logging batches.",
    },
)


# ---------------------------------------------------------------------------
# Internal URL / header helpers
# ---------------------------------------------------------------------------

def _redact_headers(
    headers: Mapping[str, str] | None,
    *,
    sensitive_keys: Iterable[str],
    replacement: str,
) -> Mapping[str, str] | None:
    if not headers:
        return headers
    sens = {k.lower() for k in sensitive_keys}
    out: dict[str, str] = {
        k: (replacement if k.lower() in sens else v)
        for k, v in headers.items()
    }
    try:
        return type(headers)(out)  # type: ignore[call-arg]
    except Exception:
        return out


def _sanitize_url(
    url: str,
    *,
    sensitive_query_keys: Iterable[str],
    replacement: str,
) -> str:
    if not url:
        return url
    sens = {k.lower() for k in sensitive_query_keys}
    parts = urlsplit(url)
    netloc = parts.netloc
    if "@" in netloc:
        netloc = netloc.split("@", 1)[1]
    if parts.query:
        q = parse_qsl(parts.query, keep_blank_values=True)
        q2 = [(k, replacement if k.lower() in sens else v) for (k, v) in q]
        query = urlencode(q2, doseq=True)
    else:
        query = parts.query
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def _copy_with_updates(obj: Any, **updates: Any) -> Any:
    if not updates:
        return obj
    if is_dataclass(obj):
        return replace(obj, **updates)
    new_obj = copy.copy(obj)
    for k, v in updates.items():
        try:
            setattr(new_obj, k, v)
        except Exception:
            pass
    return new_obj


def _to_url_string(url_obj: Any) -> str | None:
    if url_obj is None:
        return None
    if hasattr(url_obj, "to_string"):
        try:
            return url_obj.to_string()
        except Exception:
            pass
    try:
        return str(url_obj)
    except Exception:
        return None


def _rebuild_url_like(original_url_obj: Any, url_string: str) -> Any:
    if original_url_obj is None:
        return url_string
    url_cls = type(original_url_obj)
    for ctor in ("from_string", "parse", "from_url", "from_str"):
        if hasattr(url_cls, ctor):
            try:
                return getattr(url_cls, ctor)(url_string)
            except Exception:
                pass
    if isinstance(original_url_obj, str):
        return url_string
    return url_string


# ---------------------------------------------------------------------------
# Hop-by-hop headers (shared by to_starlette / to_fastapi)
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
# Response
# ---------------------------------------------------------------------------

@dataclass
class Response:
    request: PreparedRequest

    # Core HTTP bits
    status_code: int
    headers: MutableMapping[str, str]
    buffer: BytesIO

    received_at_timestamp: int   # µs since epoch (time.time_ns() // 1000)
    tags: Mapping[str, str]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, obj: Any, *, normalize: bool = True) -> "Response":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, str):
            return cls.parse_str(obj, normalize=normalize)
        if isinstance(obj, Mapping):
            return cls.parse_dict(obj, normalize=normalize)
        return cls.parse_str(str(obj), normalize=normalize)

    @classmethod
    def parse_str(cls, raw: str, *, normalize: bool = True) -> "Response":
        s = raw.strip()
        if not s:
            raise ValueError("Response.parse_str: empty string")
        try:
            d = json.loads(s)
        except Exception as e:
            raise ValueError("Response.parse_str: expected JSON object string") from e
        if not isinstance(d, Mapping):
            raise ValueError("Response.parse_str: JSON must decode to an object")
        return cls.parse_dict(d, normalize=normalize)

    @classmethod
    def parse_dict(cls, obj: Mapping[str, Any], *, normalize: bool = True, prefix: str = "response_") -> "Response":
        if not obj:
            raise ValueError("Response.parse_dict: empty mapping")

        # request
        req_obj = get_from_dict(obj, keys=("request",), prefix="")
        if req_obj is MISSING or req_obj in (None, ""):
            req_obj = obj
        request = PreparedRequest.parse(req_obj, normalize=normalize)

        # status (prefixed first, then unprefixed)
        status = get_from_dict(obj, keys=("status_code", "status", "code"), prefix=prefix)
        if status is MISSING:
            status = get_from_dict(obj, keys=("status_code", "status", "code"), prefix="")
        if status is MISSING or status in (None, ""):
            raise ValueError("Response.parse_dict: missing status_code/status/code")
        status_code = int(status) if isinstance(status, int) else int(float(str(status).strip()))

        # headers
        headers_obj = get_from_dict(obj, keys=("headers", "header", "hdrs", "response_headers"), prefix=prefix)
        if headers_obj is MISSING:
            headers_obj = get_from_dict(obj, keys=("headers", "header", "hdrs", "response_headers"), prefix="")
        headers_obj = headers_obj if isinstance(headers_obj, Mapping) else {}
        headers = {str(k): str(v) for k, v in headers_obj.items()}

        # body
        body_obj = get_from_dict(obj, keys=("buffer", "body", "content", "data", "response_body"), prefix=prefix)
        if body_obj is MISSING:
            body_obj = get_from_dict(obj, keys=("buffer", "body", "content", "data", "response_body"), prefix="")
        buffer = BytesIO.parse(obj=body_obj) if body_obj is not MISSING and body_obj is not None else BytesIO()

        # received_at_timestamp
        ts_obj = get_from_dict(
            obj,
            keys=("received_at_timestamp", "received_at", "time_us", "timestamp", "time_ns", "received_at_epoch",
                  "response_received_at_epoch"),
            prefix=prefix,
        )
        if ts_obj is MISSING:
            ts_obj = get_from_dict(
                obj,
                keys=("received_at_timestamp", "received_at", "time_us", "timestamp", "time_ns", "received_at_epoch",
                      "response_received_at_epoch"),
                prefix="",
            )
        received_at_ts = 0
        if ts_obj is not MISSING and ts_obj not in (None, ""):
            received_at_ts = int(ts_obj) if isinstance(ts_obj, int) else int(float(str(ts_obj).strip()))

        # tags
        tags_obj = get_from_dict(obj, keys=("tags", "response_tags"), prefix=prefix)
        if tags_obj is MISSING:
            tags_obj = get_from_dict(obj, keys=("tags", "response_tags"), prefix="")
        tags_obj = tags_obj if isinstance(tags_obj, Mapping) else {}
        tags = {str(k): str(v) for k, v in tags_obj.items()}

        return cls(
            request=request,
            status_code=status_code,
            headers=headers,
            buffer=buffer,
            received_at_timestamp=received_at_ts,
            tags=tags,
        )

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------
    @property
    def media_type(self):
        hdr = self.headers.get("Content-Type")

        if hdr:
            return MediaType.parse_str(hdr)

        mt = self.buffer.media_type

        if not self.headers:
            self.headers = {}

        self.headers["Content-Type"] = mt.full_mime_type().value

        return mt

    @property
    def codec(self):
        return self.media_type.codec

    @property
    def content(self) -> bytes:
        return self.buffer.to_bytes()

    @property
    def text(self) -> str:
        return self.content.decode(_get_charset(self.headers), errors="replace")

    def json(self) -> Any:
        return json.loads(self.content)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400

    # ------------------------------------------------------------------
    # raise_for_status — delegates to exceptions module
    # ------------------------------------------------------------------

    def raise_for_status(self, *, max_body: int = 2048) -> None:
        """
        Raise the most specific HTTPStatusError subclass for 4xx/5xx.
        No-op for 1xx/2xx/3xx.

        Raises
        ------
        exceptions.BadRequest          (400)
        exceptions.UnauthorizedError   (401)
        exceptions.ForbiddenError      (403)
        exceptions.NotFoundError       (404)
        exceptions.MethodNotAllowed    (405)
        exceptions.ConflictError       (409)
        exceptions.GoneError           (410)
        exceptions.UnprocessableEntity (422)
        exceptions.TooManyRequests     (429)
        exceptions.InternalServerError (500)
        exceptions.BadGatewayError     (502)
        exceptions.ServiceUnavailable  (503)
        exceptions.GatewayTimeout      (504)
        exceptions.ClientError         (other 4xx)
        exceptions.ServerError         (other 5xx)
        """
        if not self.ok:
            raise self.error()

    def error(self):
        if not self.ok:
            from .errors import make_for_status
            return make_for_status(self)
        return

    # ------------------------------------------------------------------
    # Anonymisation
    # ------------------------------------------------------------------

    def anonymize(self, mode: Literal["remove", "redact", "hash"] = "remove") -> "Response":
        return replace(
            self,
            request=self.request.anonymize(mode=mode),
            headers=anonymize_headers(self.headers, mode=mode),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_polars(
        self,
        parse: bool = True,
        *,
        lazy: bool = False,
    ) -> "pl.DataFrame | pl.LazyFrame":
        from yggdrasil.polars.lib import polars as pl

        if parse:
            mio = self.buffer.media_io(media=self.media_type)
            return mio.read_polars_frame(lazy=lazy)

        return pl.from_arrow(self.to_arrow_batch(parse=False))

    def to_pandas(self, parse: bool = True) -> "pd.DataFrame":
        from yggdrasil.pandas.lib import pandas  # type: ignore
        return self.to_polars(parse=parse).to_pandas()

    def to_arrow_batch(self, parse: bool = False) -> pa.RecordBatch:
        if parse:
            from yggdrasil.polars.cast import polars_dataframe_to_arrow_table
            return polars_dataframe_to_arrow_table(
                self.to_polars(parse=parse)
            ).to_batches()[0]

        req_rb = self.request.to_arrow_batch(parse=False)

        headers_v = {str(k): str(v) for k, v in (self.headers.items() if self.headers else ())}
        tags_v = {str(k): str(v) for k, v in (self.tags.items() if self.tags else ())}

        if self.buffer:
            body_bytes = self.buffer.to_bytes()
            body_blake3_32 = self.buffer.blake3().digest()
        else:
            body_bytes = body_blake3_32 = None

        resp_arrays = [
            pa.array([self.status_code],        type=ARROW_SCHEMA.field("response_status_code").type),
            pa.array([headers_v],               type=ARROW_SCHEMA.field("response_headers").type),
            pa.array([tags_v],                  type=ARROW_SCHEMA.field("response_tags").type),
            pa.array([body_bytes],              type=ARROW_SCHEMA.field("response_body").type),
            pa.array([body_blake3_32],          type=ARROW_SCHEMA.field("response_body_hash").type),
            pa.array([self.received_at_timestamp], type=ARROW_SCHEMA.field("response_received_at").type),
            pa.array([self.received_at_timestamp], type=ARROW_SCHEMA.field("response_received_at_epoch").type),
        ]

        return pa.RecordBatch.from_arrays(
            list(req_rb.columns) + resp_arrays,
            schema=RESPONSE_ARROW_SCHEMA,
        )  # type: ignore

    @classmethod
    def from_arrow(
        cls,
        batch: pa.RecordBatch | pa.Table,
        *,
        parse: bool = False,
        normalize: bool = True,
    ) -> Iterator["Response"]:
        if parse:
            raise NotImplementedError("parse=True not implemented (yet)")

        def _headers_from_map(x: Any) -> Mapping[str, str]:
            if not x:
                return {}
            if isinstance(x, dict):
                return {str(k): str(v) for k, v in x.items() if k is not None and v is not None}
            try:
                return {str(k): str(v) for k, v in x if k is not None and v is not None}
            except Exception:
                return {}

        def _iter_batches(obj: pa.RecordBatch | pa.Table) -> Iterator[pa.RecordBatch]:
            if isinstance(obj, pa.RecordBatch):
                yield obj
                return
            for rb in obj.to_batches():
                yield rb

        def _first_present(cols: Mapping[str, Any], i: int, *names: str) -> Any:
            for n in names:
                if n in cols:
                    return cols[n][i].as_py()
            return None

        req_cols: Sequence[str] = [f.name for f in REQUEST_ARROW_SCHEMA]
        resp_cols: Sequence[str] = [f.name for f in ARROW_SCHEMA]

        for rb in _iter_batches(batch):
            cols = {
                name: rb.column(name)
                for name in list(req_cols) + list(resp_cols)
                if name in rb.schema.names
            }

            for i in range(rb.num_rows):
                # -----------------------------
                # Request (new flattened schema)
                # -----------------------------
                method = _first_present(cols, i, "request_method") or "GET"

                # Prefer full deterministic string first
                url_str = _first_present(cols, i, "request_url_str")
                if url_str not in (None, ""):
                    url_str_out: str | None = str(url_str)
                    url_out: Any | None = None
                else:
                    # Rebuild from exploded columns if present
                    scheme = _first_present(cols, i, "request_url_scheme")
                    userinfo = _first_present(cols, i, "request_url_userinfo")
                    host = _first_present(cols, i, "request_url_host")
                    port = _first_present(cols, i, "request_url_port")
                    path = _first_present(cols, i, "request_url_path")
                    query = _first_present(cols, i, "request_url_query")
                    fragment = _first_present(cols, i, "request_url_fragment")

                    has_exploded = any(
                        x not in (None, "", 0)
                        for x in (scheme, userinfo, host, port, path, query, fragment)
                    )

                    if has_exploded:
                        url_str_out = None
                        url_out = {
                            "scheme": scheme or "",
                            "userinfo": userinfo or "",
                            "host": host or "",
                            "port": 0 if port in (None, "") else int(port),
                            "path": path or "",
                            "query": query or "",
                            "fragment": fragment or "",
                        }
                    else:
                        # Legacy support: older schema had request_url struct
                        legacy_struct = _first_present(cols, i, "request_url")
                        url_str_out = None
                        url_out = legacy_struct if isinstance(legacy_struct, Mapping) else ""

                req_d: dict[str, Any] = {
                    "method": method,
                    # PreparedRequest.parse_dict prefers url_str if present, else url struct/dict
                    "url_str": url_str_out,
                    "url": url_out,
                    "headers": _headers_from_map(_first_present(cols, i, "request_headers") or {}),
                    "tags": _headers_from_map(_first_present(cols, i, "request_tags") or {}),
                    "buffer": _first_present(cols, i, "request_body"),
                    "sent_at_timestamp": (
                        _arrow_ts_col_to_us(cols["request_sent_at"], i)
                        if "request_sent_at" in cols else 0
                    ),
                }
                request = PreparedRequest.parse_dict(req_d, normalize=normalize)

                # -----------------------------
                # Response
                # -----------------------------
                status_code = int(_first_present(cols, i, "response_status_code") or 0)
                headers = _headers_from_map(_first_present(cols, i, "response_headers") or {})
                tags = _headers_from_map(_first_present(cols, i, "response_tags") or {})
                body_bytes = _first_present(cols, i, "response_body")
                buffer = BytesIO.parse(obj=body_bytes) if body_bytes is not None else BytesIO()
                received_at = (
                    _arrow_ts_col_to_us(cols["response_received_at"], i)
                    if "response_received_at" in cols else 0
                )

                yield cls(
                    request=request,
                    status_code=status_code,
                    headers=headers,
                    buffer=buffer,
                    tags=tags,
                    received_at_timestamp=received_at,
                )

    # ------------------------------------------------------------------
    # ASGI — Starlette / FastAPI
    # ------------------------------------------------------------------

    def to_starlette(self) -> "StarletteResponse":
        """
        Convert to a ``starlette.responses.Response`` for direct return
        from a Starlette or FastAPI route handler.

        - Hop-by-hop headers are stripped.
        - ``Content-Length`` is recomputed from the decompressed buffer.
        - ``media_type`` is parsed from ``Content-Type`` so Starlette
          sets the header correctly without duplication.
        """
        from starlette.responses import Response as _SResponse

        body = self.buffer.to_bytes() if self.buffer else b""

        headers: dict[str, str] = {
            k: v
            for k, v in (self.headers or {}).items()
            if k.lower() not in _HOP_BY_HOP
        }
        headers["content-length"] = str(len(body))

        media_type = _detect_content_type(self.headers) or "application/octet-stream"

        return _SResponse(
            content=body,
            status_code=self.status_code,
            headers=headers,
            media_type=media_type,
        )

    def to_fastapi(self) -> "FastAPIResponse":
        """
        Convert to a ``fastapi.Response`` for direct return from a
        FastAPI route handler.

        FastAPI's ``Response`` is a subclass of Starlette's, but
        returning the FastAPI type keeps FastAPI's dependency injection,
        background task hooks, and OpenAPI response modelling intact.

        Falls back to ``to_starlette()`` transparently when FastAPI is
        not installed.
        """
        try:
            from fastapi import Response as _FResponse
        except ImportError:
            return self.to_starlette()  # type: ignore[return-value]

        body = self.buffer.to_bytes() if self.buffer else b""

        headers: dict[str, str] = {
            k: v
            for k, v in (self.headers or {}).items()
            if k.lower() not in _HOP_BY_HOP
        }
        headers["content-length"] = str(len(body))

        media_type = _detect_content_type(self.headers) or "application/octet-stream"

        return _FResponse(
            content=body,
            status_code=self.status_code,
            headers=headers,
            media_type=media_type,
        )


# ---------------------------------------------------------------------------
# DataFrame helpers (module-level → testable independently)
# ---------------------------------------------------------------------------

def _polars_from_json(
    body: BytesIO,
    meta: dict[str, Any],
    headers: Mapping[str, str],
    pl: Any,
) -> Any:
    charset = _get_charset(headers)
    payload = json.loads(body.to_bytes().decode(charset, errors="replace"))

    if isinstance(payload, list):
        if not payload:
            df = pl.DataFrame()
        elif all(isinstance(row, dict) for row in payload):
            df = pl.from_dicts(payload)
        else:
            df = pl.DataFrame({"value": payload})
    elif isinstance(payload, dict):
        row = {
            k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
            for k, v in payload.items()
        }
        df = pl.from_dicts([row])
    else:
        df = pl.from_dicts([{"body_json": json.dumps(payload)}])

    return _attach_meta(df, meta, pl)


def _polars_fallback(
    response: "Response",
    meta: dict[str, Any],
    body: Any,
    pl: Any,
) -> Any:
    raw = body.to_bytes() if hasattr(body, "to_bytes") else (body or b"")
    row = dict(meta)
    row["response_body"] = raw if raw else None
    row["response_body_size"] = len(raw)
    return pl.from_dicts([row])


def _attach_meta(df: Any, meta: dict[str, Any], pl: Any) -> Any:
    import polars as _pl
    existing = set(df.columns)
    for col_name, val in meta.items():
        if col_name not in existing:
            df = df.with_columns(_pl.lit(val).alias(col_name))
    return df


def _arrow_ts_col_to_us(col: pa.ChunkedArray | pa.Array, i: int) -> int:
    scalar = col[i]
    if scalar is None or not scalar.is_valid:
        return 0
    return scalar.value if scalar.value is not None else 0