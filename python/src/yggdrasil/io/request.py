from __future__ import annotations

import json as json_module
from dataclasses import dataclass, replace
from typing import Mapping, Any, Optional, MutableMapping, Literal

import pyarrow as pa

from yggdrasil.io.headers import anonymize_headers
from .buffer import BytesIO
from .url import URL

__all__ = ["PreparedRequest", "REQUEST_ARROW_SCHEMA"]


REQUEST_ARROW_SCHEMA = pa.schema(
    [
        pa.field(
            "request_method",
            pa.string(),
            nullable=False,
            metadata={"comment": "HTTP verb (GET, POST, etc.)"},
        ),
        pa.field(
            "request_url",
            pa.string(),
            nullable=False,
            metadata={"comment": "Full request URL as string"},
        ),

        # ---- URL components (best-effort parsed) ----
        pa.field(
            "request_url_scheme",
            pa.string(),
            nullable=False,
            metadata={"comment": "URL scheme (e.g., http, https)"},
        ),
        pa.field(
            "request_url_userinfo",
            pa.string(),
            nullable=True,
            metadata={"comment": "Userinfo from URL authority (e.g., user:pass). Avoid persisting secrets."},
        ),
        pa.field(
            "request_url_host",
            pa.string(),
            nullable=False,
            metadata={"comment": "Host (domain or IP)"},
        ),
        pa.field(
            "request_url_port",
            pa.int32(),
            nullable=True,
            metadata={"comment": "Port number if explicitly specified"},
        ),
        pa.field(
            "request_url_path",
            pa.string(),
            nullable=True,
            metadata={"comment": "Path component of the URL"},
        ),
        pa.field(
            "request_url_query",
            pa.string(),
            nullable=True,
            metadata={"comment": "Raw query string (without leading '?')"},
        ),
        pa.field(
            "request_url_fragment",
            pa.string(),
            nullable=True,
            metadata={"comment": "Fragment identifier (without leading '#')"},
        ),

        # ---- headers/body ----
        pa.field(
            "request_headers",
            pa.map_(pa.string(), pa.string()),
            nullable=False,
            metadata={
                "comment": "HTTP request headers as map<string,string> (keys sorted; duplicates collapsed)",
                "keys_sorted": "true",
            },
        ),
        pa.field(
            "request_tags",
            pa.map_(pa.string(), pa.string()),
            nullable=False,
            metadata={
                "comment": "Raw HTTP request tags as ordered key/value pairs",
            },
        ),
        pa.field(
            "request_body",
            pa.binary(),
            nullable=True,
            metadata={"comment": "Raw request body bytes"},
        ),

        # ---- hashes ----
        pa.field(
            "request_body_hash",
            pa.binary(32),
            nullable=True,
            metadata={"comment": "BLAKE3-256 digest of request_body (32 bytes)", "algorithm": "blake3", "byte_width": "32"},
        ),

        # ---- timing ----
        pa.field(
            "request_sent_at",
            pa.timestamp("us", "UTC"),
            nullable=False,
            metadata={"comment": "UTC timestamp when request was dispatched", "unit": "us", "tz": "UTC"},
        ),
    ],
    metadata={
        "comment": "HTTP prepared request flattened into deterministic columns for logging/replay.",
    },
)


@dataclass
class PreparedRequest:
    method: str
    url: URL
    headers: Mapping[str, str]
    buffer: Optional[BytesIO]
    tags: Optional[Mapping[str, str]]
    sent_at_timestamp: int = 0  # time.time_ns() // 1000

    @classmethod
    def prepare(
        cls,
        method: str,
        url: URL | str,
        headers: Optional[MutableMapping[str, str]] = None,
        body: Optional[Any] = None,
        tags: Optional[Mapping[str, str]] = None,
        *,
        json: Optional[Any] = None,
        normalize: bool = True,
    ) -> "PreparedRequest":
        url = URL.parse_any(url, normalize=normalize)

        if body is not None:
            body = BytesIO.parse_any(obj=body)
        elif json is not None:
            body = BytesIO()
            json_module.dump(json, body)
            body.seek(0)

            if not headers:
                headers = {"Content-Type": "application/json"}
            else:
                headers["Content-Type"] = "application/json"

        if not headers:
            headers = {}

        if body is not None:
            headers["Content-Length"] = body.size

        return cls(
            method=method, url=url, headers=headers, buffer=body,
            tags=tags,
            sent_at_timestamp=0
        )

    @classmethod
    def parse_any(
        cls,
        obj: Any,
        *,
        normalize: bool = True,
    ) -> "PreparedRequest":
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, str):
            return cls.parse_str(obj, normalize=normalize)

        if isinstance(obj, Mapping):
            return cls.parse_dict(obj, normalize=normalize)

        # last-resort: stringify (useful for weird wrappers)
        return cls.parse_str(str(obj), normalize=normalize)

    @classmethod
    def parse_str(
        cls,
        raw: str,
        *,
        normalize: bool = True,
    ) -> "PreparedRequest":
        """
        Accepts JSON string representing a PreparedRequest-ish dict.
        """
        s = raw.strip()
        if not s:
            raise ValueError("PreparedRequest.parse_str: empty string")

        try:
            d = json_module.loads(s)
        except Exception as e:
            raise ValueError("PreparedRequest.parse_str: expected JSON object string") from e

        if not isinstance(d, Mapping):
            raise ValueError("PreparedRequest.parse_str: JSON must decode to an object")

        return cls.parse_dict(d, normalize=normalize)

    @classmethod
    def parse_dict(
        cls,
        d: Mapping[str, Any],
        *,
        normalize: bool = True,
    ) -> "PreparedRequest":
        """
        Parses a PreparedRequest from a mapping.

        Supported shapes (field aliases included):
          - method: "method" | "http_method" | "verb"
          - url:    "url" | "uri" | "href" | ("url_*" exploded fields)
          - headers:"headers" | "header" | "hdrs"
          - body:   "buffer" | "body" | "data"
          - sent_at:"sent_at_timestamp" | "sent_at" | "timestamp" | "time_ns"
        """
        if not d:
            raise ValueError("PreparedRequest.parse_dict: empty mapping")

        # method
        method = (
            d.get("method")
            or d.get("http_method")
            or d.get("verb")
            or "GET"
        )
        method_s = str(method)

        # url (accept URL object, string, dict, or exploded url_* fields)
        url_obj: Any = d.get("url") or d.get("uri") or d.get("href")
        if url_obj is None:
            # exploded fields as used by to_arrow_batch
            has_exploded = any(
                k in d
                for k in (
                    "url_scheme",
                    "url_userinfo",
                    "url_host",
                    "url_port",
                    "url_path",
                    "url_query",
                    "url_fragment",
                )
            )
            if has_exploded:
                q = d.get("url_query")
                query = ""
                if isinstance(q, Mapping):
                    # our arrow encoding is key -> "v1|v2|..."
                    # keep deterministic: sort keys, keep given value order
                    parts: list[tuple[str, str]] = []
                    for k in sorted(q.keys(), key=lambda x: str(x)):
                        v = q[k]
                        if v is None:
                            continue
                        vs = str(v).split("|")
                        for one in vs:
                            parts.append((str(k), one))
                    from urllib.parse import urlencode
                    query = urlencode(parts, doseq=True)

                url_obj = {
                    "scheme": d.get("url_scheme") or "",
                    "userinfo": d.get("url_userinfo") or "",
                    "host": d.get("url_host") or "",
                    "port": d.get("url_port") or 0,
                    "path": d.get("url_path") or "",
                    "query": query,
                    "fragment": d.get("url_fragment") or "",
                }
            else:
                raise ValueError("PreparedRequest.parse_dict: missing url")

        url = URL.parse_any(url_obj, normalize=normalize)

        # headers
        headers_obj = d.get("headers") or d.get("header") or d.get("hdrs") or {}
        if headers_obj is None:
            headers_obj = {}
        if not isinstance(headers_obj, Mapping):
            raise ValueError("PreparedRequest.parse_dict: headers must be a mapping")
        headers: dict[str, str] = {str(k): str(v) for k, v in headers_obj.items()}

        # body/buffer
        body_obj = d.get("buffer")
        if body_obj is None:
            body_obj = d.get("body")
        if body_obj is None:
            body_obj = d.get("data")

        buffer: Optional[BytesIO] = None
        if body_obj is not None:
            buffer = BytesIO.parse_any(obj=body_obj)

        # sent_at timestamp (ns)
        sent_at = (
            d.get("sent_at_timestamp")
            if "sent_at_timestamp" in d
            else d.get("time_ns")
            if "time_ns" in d
            else d.get("timestamp")
            if "timestamp" in d
            else d.get("sent_at")
        )
        sent_at_ts = 0
        if sent_at is not None and sent_at != "":
            if isinstance(sent_at, int):
                sent_at_ts = sent_at
            else:
                s = str(sent_at)
                sent_at_ts = int(s) if s.isdigit() else 0

        tags = d.get("tags", None) or None

        return cls(
            method=method_s,
            url=url,
            headers=headers,
            buffer=buffer,
            tags=tags,
            sent_at_timestamp=sent_at_ts,
        )

    @property
    def body(self):
        return self.buffer

    def anonymize(
        self,
        mode: Literal["remove", "redact", "hash"] = "remove",
    ) -> "PreparedRequest":
        """
        Clean/boring + composable:
        - headers redaction happens here
        - URL redaction happens in URL.anonymize()
        """
        return replace(
            self,
            headers=anonymize_headers(self.headers, mode=mode),
            url=self.url.anonymize(mode=mode),
        )

    def to_arrow_batch(
        self,
        parse: bool = False,
    ) -> pa.RecordBatch:
        if parse:
            raise NotImplementedError

        u = self.url
        url_s = u.to_string()

        scheme_v = u.scheme
        userinfo_v = u.userinfo
        host_v = u.host
        port_v = u.port
        path_v = u.path
        fragment_v = u.fragment
        q_v = u.query

        headers_v = {
            str(k): str(v)
            for k, v in (self.headers.items() if self.headers else ())
        }
        tags_v = {
            str(k): str(v)
            for k, v in (self.tags.items() if self.tags else ())
        }

        if self.buffer:
            body_bytes = self.buffer.to_bytes()
            body_blake3_32 = self.buffer.blake3().digest()
        else:
            body_bytes, body_blake3_32 = None, None

        arrays = [
            pa.array([self.method], type=REQUEST_ARROW_SCHEMA.field("request_method").type),
            pa.array([url_s], type=REQUEST_ARROW_SCHEMA.field("request_url").type),
            pa.array([scheme_v], type=REQUEST_ARROW_SCHEMA.field("request_url_scheme").type),
            pa.array([userinfo_v], type=REQUEST_ARROW_SCHEMA.field("request_url_userinfo").type),
            pa.array([host_v], type=REQUEST_ARROW_SCHEMA.field("request_url_host").type),
            pa.array([port_v], type=REQUEST_ARROW_SCHEMA.field("request_url_port").type),
            pa.array([path_v], type=REQUEST_ARROW_SCHEMA.field("request_url_path").type),
            pa.array([q_v], type=REQUEST_ARROW_SCHEMA.field("request_url_query").type),
            pa.array([fragment_v], type=REQUEST_ARROW_SCHEMA.field("request_url_fragment").type),
            pa.array([headers_v], type=REQUEST_ARROW_SCHEMA.field("request_headers").type),
            pa.array([tags_v], type=REQUEST_ARROW_SCHEMA.field("request_tags").type),
            pa.array([body_bytes], type=REQUEST_ARROW_SCHEMA.field("request_body").type),
            pa.array([body_blake3_32], type=REQUEST_ARROW_SCHEMA.field("request_body_hash").type),
            pa.array([self.sent_at_timestamp], type=REQUEST_ARROW_SCHEMA.field("request_sent_at").type),
        ]

        return pa.RecordBatch.from_arrays(arrays, schema=REQUEST_ARROW_SCHEMA)  # type: ignore
