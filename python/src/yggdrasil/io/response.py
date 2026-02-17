from __future__ import annotations

import copy
import json
from dataclasses import dataclass, is_dataclass, replace
from typing import Mapping, Any, Iterable
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import pyarrow as pa
from yggdrasil.io.headers import anonymize_headers

from .dynamic_buffer import DynamicBuffer
from .request import PreparedRequest

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


# ------------------- anonymization helpers -------------------

_DEFAULT_SENSITIVE_HEADER_KEYS = {
    # auth
    "authorization",
    "proxy-authorization",
    # cookies
    "cookie",
    "set-cookie",
    # common api key headers
    "x-api-key",
    "api-key",
    "apikey",
    "x-auth-token",
    "x-csrf-token",
    "x-xsrf-token",
    # cloud/vendor-ish
    "x-amz-security-token",
    "x-amz-access-token",
}

_DEFAULT_SENSITIVE_QUERY_KEYS = {
    "token",
    "access_token",
    "refresh_token",
    "id_token",
    "api_key",
    "apikey",
    "key",
    "signature",
    "sig",
    "password",
    "passwd",
    "secret",
    "client_secret",
    "session",
    "sid",
}


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
    buffer: DynamicBuffer

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
            buffer = DynamicBuffer()  # Response.buffer is non-optional
        else:
            buffer = DynamicBuffer.parse_any(obj=body_obj)

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
        mode: str = "redact"
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
        *,
        request_prefix: str = "request_",
        column_prefix: str = "response_",
    ) -> pa.RecordBatch:
        """
        Single-row RecordBatch representing this response, with the request exploded
        into columns via request.to_arrow_batch(column_prefix=request_prefix).
        """
        if parse:
            raise NotImplementedError

        # 1) Explode request into columns (single-row)
        req_rb = self.request.to_arrow_batch(parse=False, column_prefix=request_prefix)

        # 2) Response schema with Field Metadata
        resp_schema = pa.schema(
            [
                pa.field(
                    name=f"{column_prefix}status_code",
                    type=pa.int32(),
                    nullable=False,
                    metadata={"comment": "HTTP status code returned by the server"},
                ),
                pa.field(
                    name=f"{column_prefix}headers",
                    type=pa.map_(
                        pa.field("key", pa.string(), nullable=False),
                        pa.field("value", pa.string(), nullable=False),
                    ),
                    nullable=True,
                    metadata={"comment": "Raw HTTP response headers"},
                ),
                pa.field(
                    name=f"{column_prefix}body",
                    type=pa.binary(),
                    nullable=True,
                    metadata={"comment": "Raw binary payload of the response"},
                ),
                pa.field(
                    name=f"{column_prefix}body_hash64",
                    type=pa.int64(),
                    nullable=True,
                    metadata={"algorithm": "xxh3_64", "comment": "64-bit hash of the body"},
                ),
                pa.field(
                    name=f"{column_prefix}received_at",
                    type=pa.timestamp("us", "UTC"),
                    nullable=False,
                    metadata={"comment": "UTC timestamp when the response was captured"},
                ),
            ]
        )

        # 3) Values -> Arrow arrays
        headers_v = None if not self.headers else dict(self.headers)
        body_bytes = self.buffer.to_bytes()

        # Minor optimization: use the already converted body_bytes
        body_h64 = self.buffer.xxh3_64().intdigest() if body_bytes is not None else None

        resp_arrays = [
            pa.array([self.status_code], type=resp_schema.field(0).type),
            pa.array([headers_v], type=resp_schema.field(1).type),
            pa.array([body_bytes], type=resp_schema.field(2).type),
            pa.array([body_h64], type=resp_schema.field(3).type),
            pa.array([self.received_at_timestamp], type=resp_schema.field(4).type),
        ]

        # 4) Combine request columns + response columns into one RecordBatch
        full_schema = pa.schema(list(req_rb.schema) + list(resp_schema))
        full_arrays = list(req_rb.columns) + resp_arrays

        return pa.RecordBatch.from_arrays(full_arrays, schema=full_schema) # type: ignore
