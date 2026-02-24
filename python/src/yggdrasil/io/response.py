from __future__ import annotations

import copy
import json
from dataclasses import dataclass, is_dataclass, replace
from typing import Mapping, Any, Iterable, Literal, Sequence, Iterator
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import pyarrow as pa

from yggdrasil.io.headers import anonymize_headers
from .buffer import BytesIO
from .request import PreparedRequest, REQUEST_ARROW_SCHEMA

__all__ = ["Response", "HTTPError"]


class HTTPError(RuntimeError):
    """Raised when an HTTP response indicates an error status."""

    def __init__(self, message: str, *, response: "Response"):
        super().__init__(message)
        self.response = response


def _reason_phrase(status_code: int) -> str:
    # Avoid pulling in requests/http; keep it lightweight.
    # This covers the common ones; unknown => empty.
    try:
        from http import HTTPStatus
        return HTTPStatus(status_code).phrase
    except Exception:
        return ""


def _get_charset(headers: Mapping[str, str]) -> str:
    # Try to parse Content-Type: ...; charset=utf-8
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


# -----------------------------
# Response schema (fixed + fully described)
# -----------------------------
ARROW_SCHEMA = pa.schema(
    [
        pa.field(
            name="response_status_code",
            type=pa.int32(),
            nullable=False,
            metadata={
                "comment": "HTTP status code returned by the server",
            },
        ),
        pa.field(
            "response_headers",
            pa.list_(pa.field(
                name="entries",
                type=pa.struct([
                    pa.field("key", pa.string()),
                    pa.field("value", pa.string())
                ])
            )),
            nullable=True,
            metadata={
                "comment": "Raw HTTP response headers as ordered key/value pairs",
                "keys_sorted": "true"
            },
        ),
        pa.field(
            name="response_body",
            type=pa.binary(),
            nullable=True,
            metadata={
                "comment": "Raw binary payload of the response (bytes)",
            },
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
    ],
    metadata={
        "comment": "HTTP response record (single row), designed for deterministic logging + replay.",
    },
)

# -----------------------------
# Full schema (request + response)
# -----------------------------
FULL_ARROW_SCHEMA = pa.schema(
    list(REQUEST_ARROW_SCHEMA) + list(ARROW_SCHEMA),
    metadata={
        "comment": "HTTP prepared request and response flattened into columns for single-row logging batches.",
    },
)


def _redact_headers(
    headers: Mapping[str, str] | None,
    *,
    sensitive_keys: Iterable[str],
    replacement: str,
) -> Mapping[str, str] | None:
    if not headers:
        return headers

    sens = {k.lower() for k in sensitive_keys}

    out: dict[str, str] = {}
    for k, v in headers.items():
        out[k] = replacement if k.lower() in sens else v

    # Preserve mapping type if possible (e.g., OrderedDict)
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

    # Strip userinfo (user:pass@host)
    netloc = parts.netloc
    if "@" in netloc:
        netloc = netloc.split("@", 1)[1]

    # Redact sensitive query param values (keep keys)
    if parts.query:
        q = parse_qsl(parts.query, keep_blank_values=True)
        q2 = [(k, replacement if k.lower() in sens else v) for (k, v) in q]
        query = urlencode(q2, doseq=True)
    else:
        query = parts.query

    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def _copy_with_updates(obj: Any, **updates: Any) -> Any:
    """
    Best-effort copy that supports dataclasses + normal objects.

    - dataclass: dataclasses.replace
    - non-dataclass: copy.copy + setattr
    """
    if not updates:
        return obj

    if is_dataclass(obj):
        return replace(obj, **updates)

    new_obj = copy.copy(obj)
    for k, v in updates.items():
        try:
            setattr(new_obj, k, v)
        except Exception:
            # Don't block anonymization if an attribute is read-only.
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
    """
    Best-effort attempt to reconstruct the same URL object type.
    Falls back to the sanitized string.
    """
    if original_url_obj is None:
        return url_string

    url_cls = type(original_url_obj)

    # Common patterns for URL wrapper types
    for ctor in ("from_string", "parse", "from_url", "from_str"):
        if hasattr(url_cls, ctor):
            try:
                return getattr(url_cls, ctor)(url_string)
            except Exception:
                pass

    # If original was already a string, keep it a string
    if isinstance(original_url_obj, str):
        return url_string

    # Can't rebuild; return string (caller may or may not accept it)
    return url_string


# ------------------- Response -------------------

@dataclass
class Response:
    request: PreparedRequest

    # Core HTTP bits
    status_code: int
    headers: Mapping[str, str]
    buffer: BytesIO

    received_at_timestamp: int  # time.time_ns() // 1000

    @classmethod
    def parse_any(
        cls,
        obj: Any,
        *,
        normalize: bool = True,
    ) -> "Response":
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, str):
            return cls.parse_str(obj, normalize=normalize)

        if isinstance(obj, Mapping):
            return cls.parse_dict(obj, normalize=normalize)

        # last-resort: stringify (useful for wrappers/log objects)
        return cls.parse_str(str(obj), normalize=normalize)

    @classmethod
    def parse_str(
        cls,
        raw: str,
        *,
        normalize: bool = True,
    ) -> "Response":
        """
        Accepts JSON string representing a Response-ish dict.
        """
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
    def parse_dict(
        cls,
        d: Mapping[str, Any],
        *,
        normalize: bool = True,
    ) -> "Response":
        """
        Parses a Response from a mapping.

        Supported shapes (field aliases included):
          - request:  "request" | ("method","url","headers","body" in same dict) fallback
          - status:   "status_code" | "status" | "code"
          - headers:  "headers" | "header" | "hdrs"
          - body:     "buffer" | "body" | "content" | "data"
          - received: "received_at_timestamp" | "received_at" | "timestamp" | "time_us" | "time_ns"
        """
        if not d:
            raise ValueError("Response.parse_dict: empty mapping")

        # ---- request ----
        req_obj = d.get("request")
        if req_obj is None:
            # some logs flatten request fields at top-level
            # (PreparedRequest.parse_dict already supports a bunch of aliases)
            req_obj = d
        request = PreparedRequest.parse_any(req_obj, normalize=normalize)

        # ---- status code ----
        status = (
            d.get("status_code")
            if "status_code" in d
            else d.get("status")
            if "status" in d
            else d.get("code")
        )
        if status is None or status == "":
            raise ValueError("Response.parse_dict: missing status_code/status/code")

        if isinstance(status, int):
            status_code = int(status)
        else:
            s = str(status).strip()
            status_code = int(s) if s.isdigit() else int(float(s))

        # ---- headers ----
        headers_obj = d.get("headers") or d.get("header") or d.get("hdrs") or {}
        if headers_obj is None:
            headers_obj = {}
        if not isinstance(headers_obj, Mapping):
            raise ValueError("Response.parse_dict: headers must be a mapping")
        headers: dict[str, str] = {str(k): str(v) for k, v in headers_obj.items()}

        # ---- body/buffer ----
        body_obj = d.get("buffer")
        if body_obj is None:
            body_obj = d.get("body")
        if body_obj is None:
            body_obj = d.get("content")
        if body_obj is None:
            body_obj = d.get("data")

        if body_obj is None:
            buffer = BytesIO()  # Response.buffer is non-optional
        else:
            buffer = BytesIO.parse_any(obj=body_obj)

        # ---- received_at timestamp (us) ----
        ts_obj = (
            d.get("received_at_timestamp")
            if "received_at_timestamp" in d
            else d.get("received_at")
            if "received_at" in d
            else d.get("time_us")
            if "time_us" in d
            else d.get("timestamp")
            if "timestamp" in d
            else d.get("time_ns")
        )

        received_at_ts = 0
        if ts_obj is not None and ts_obj != "":
            if isinstance(ts_obj, int):
                received_at_ts = ts_obj
            else:
                s = str(ts_obj).strip()
                received_at_ts = int(s) if s.isdigit() else 0

        # If provided as ns, convert to us (Response expects us in this file: time.time_ns() // 1000)
        if "time_ns" in d and isinstance(ts_obj, int):
            received_at_ts = ts_obj

        return cls(
            request=request,
            status_code=status_code,
            headers=headers,
            buffer=buffer,
            received_at_timestamp=received_at_ts,
        )

    @property
    def content(self) -> bytes:
        return self.buffer.to_bytes()

    @property
    def text(self) -> str:
        charset = _get_charset(self.headers)
        return self.content.decode(charset, errors="replace")

    def json(self) -> Any:
        return json.loads(self.content)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400

    def raise_for_status(self, *, max_body: int = 2048) -> None:
        """
        Raise HTTPError if status_code is 4xx/5xx.

        max_body: include up to N bytes of response body in the error message (decoded).
        """
        if 200 <= self.status_code < 400:
            return

        method = self.request.method
        url = self.request.url.to_string()

        reason = _reason_phrase(self.status_code)
        base = f"{self.status_code}{(' ' + reason) if reason else ''} for {method} {url}"

        body_snip = self.content[:max_body]
        msg = base

        if body_snip:
            charset = _get_charset(self.headers)
            text_snip = body_snip.decode(charset, errors="replace").strip()
            if text_snip:
                # Keep it readable; don't dump megabytes
                suffix = "…" if len(self.content) > max_body else ""
                msg = f"{base}\nResponse body ({min(len(self.content), max_body)} bytes){suffix}:\n{text_snip}"

        raise HTTPError(msg, response=self)

    def anonymize(
        self,
        mode: Literal["remove", "redact", "hash"] = "remove",
    ) -> "Response":
        """
        Clean/boring + composable:
        - headers redaction happens here
        - URL redaction happens in URL.anonymize()
        """
        return replace(
            self,
            request=self.request.anonymize(mode=mode),
            headers=anonymize_headers(self.headers, mode=mode),
        )

    def to_arrow_batch(
        self,
        parse: bool = False,
    ) -> pa.RecordBatch:
        """
        Single-row RecordBatch representing this response, with the request exploded
        into columns via request.to_arrow_batch(column_prefix=request_prefix).

        Conventions:
        - response_headers: map<string,string> (null if empty)
        - response_body: raw bytes (null if empty)
        - response_body_blake3: 32-byte digest (null if empty)
        - response_body_hash64: int64 xxh3_64 digest (null if empty)
        """
        if parse:
            raise NotImplementedError

        # 1) Explode request into columns (single-row)
        req_rb = self.request.to_arrow_batch(parse=False)

        # response_headers as map<string,string>
        headers_v = None
        if self.headers:
            headers_v = {
                str(k): str(v)
                for (k, v) in sorted(
                    dict(self.headers).items(),
                    key=lambda kv: (str(kv[0]).lower(), str(kv[0]), str(kv[1])),
                )
                if k and v
            }

        if self.buffer:
            body_bytes = self.buffer.to_bytes()
            body_blake3_32 = self.buffer.blake3().digest()
        else:
            body_bytes, body_blake3_32 = None, None

        resp_arrays = [
            pa.array([self.status_code], type=ARROW_SCHEMA.field("response_status_code").type),
            pa.array([headers_v], type=ARROW_SCHEMA.field("response_headers").type),
            pa.array([body_bytes], type=ARROW_SCHEMA.field("response_body").type),
            pa.array([body_blake3_32], type=ARROW_SCHEMA.field("response_body_hash").type),
            pa.array([self.received_at_timestamp], type=ARROW_SCHEMA.field("response_received_at").type),
        ]

        # 4) Combine request columns + response columns into one RecordBatch
        full_arrays = list(req_rb.columns) + resp_arrays

        return pa.RecordBatch.from_arrays(full_arrays, schema=FULL_ARROW_SCHEMA)  # type: ignore

    @classmethod
    def from_arrow_batch(
        cls,
        batch: pa.RecordBatch | pa.Table,
        *,
        parse: bool = False,
        normalize: bool = True,
    ) -> Iterator["Response"]:
        """
        Zero-copy-ish streaming decode: yields Response per row.

        Accepts:
          - pa.RecordBatch (single chunk)
          - pa.Table (possibly chunked) -> iterates batches

        Notes:
          - request fields are decoded via PreparedRequest.parse_dict
          - response_headers stored as list<struct<key,value>> (order preserved)
          - response_body stored as binary -> DynamicBuffer
        """
        if parse:
            raise NotImplementedError("parse=True not implemented (yet)")

        def _headers_from_map(x: Any) -> Mapping[str, str]:
            if not x:
                return {}
            if isinstance(x, dict):
                return {str(k): str(v) for k, v in x.items() if k is not None and v is not None}
            # fallback: sometimes list of (k,v)
            try:
                return {str(k): str(v) for k, v in x if k is not None and v is not None}
            except Exception:
                return {}

        def _iter_batches(obj: pa.RecordBatch | pa.Table) -> Iterator[pa.RecordBatch]:
            if isinstance(obj, pa.RecordBatch):
                yield obj
                return
            # Table: preserve chunking, avoid materializing whole table
            for rb in obj.to_batches():
                yield rb

        # columns we expect (names in FULL_ARROW_SCHEMA)
        req_cols: Sequence[str] = [f.name for f in REQUEST_ARROW_SCHEMA]
        # response cols (names in ARROW_SCHEMA above)
        resp_cols: Sequence[str] = [f.name for f in ARROW_SCHEMA]

        for rb in _iter_batches(batch):
            # Fast path: column-wise access then row index
            cols = {name: rb.column(name) for name in list(req_cols) + list(resp_cols) if name in rb.schema.names}
            n = rb.num_rows

            for i in range(n):
                # ---- rebuild request dict ----
                # request_headers is list<struct<key,value>>
                req_d: dict[str, Any] = {
                    "method": cols["request_method"][i].as_py() if "request_method" in cols else "GET",
                    "url": cols["request_url"][i].as_py() if "request_url" in cols else ""
                }

                # required fields

                # headers (arrow list<struct>)
                if "request_headers" in cols:
                    req_hdrs = cols["request_headers"][i].as_py()
                    req_d["headers"] = _headers_from_map(req_hdrs)
                else:
                    req_d["headers"] = {}

                # body (binary)
                if "request_body" in cols:
                    b = cols["request_body"][i].as_py()
                    req_d["buffer"] = b  # DynamicBuffer.parse_any can take bytes
                else:
                    req_d["buffer"] = None

                # sent_at
                if "request_sent_at" in cols:
                    # timestamp(us) -> python datetime or int depending on as_py;
                    # we want int microseconds in this file, so handle both.
                    ts = cols["request_sent_at"][i].as_py()
                    if ts is None:
                        req_d["sent_at_timestamp"] = 0
                    elif hasattr(ts, "timestamp"):
                        # datetime -> seconds float; convert to us
                        req_d["sent_at_timestamp"] = int(ts.timestamp() * 1_000_000)
                    else:
                        # sometimes Arrow returns int-like
                        req_d["sent_at_timestamp"] = int(ts)
                else:
                    req_d["sent_at_timestamp"] = 0

                request = PreparedRequest.parse_dict(req_d, normalize=normalize)

                # ---- response pieces ----
                status_code = int(cols["response_status_code"][i].as_py()) if "response_status_code" in cols else 0

                if "response_headers" in cols:
                    resp_hdrs = cols["response_headers"][i].as_py()
                    headers = _headers_from_map(resp_hdrs)
                else:
                    headers = {}

                body_bytes = cols["response_body"][i].as_py() if "response_body" in cols else None
                buffer = BytesIO.parse_any(obj=body_bytes) if body_bytes is not None else BytesIO()

                if "response_received_at" in cols:
                    rts = cols["response_received_at"][i].as_py()
                    if rts is None:
                        received_at = 0
                    elif hasattr(rts, "timestamp"):
                        received_at = int(rts.timestamp() * 1_000_000)
                    else:
                        received_at = int(rts)
                else:
                    received_at = 0

                yield cls(
                    request=request,
                    status_code=status_code,
                    headers=headers,
                    buffer=buffer,
                    received_at_timestamp=received_at,
                )