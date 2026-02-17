from __future__ import annotations

import json as json_module
from dataclasses import dataclass, replace
from typing import Mapping, Any, Optional, MutableMapping

import pyarrow as pa
from yggdrasil.io.headers import anonymize_headers

from .dynamic_buffer import DynamicBuffer
from .url import URL

__all__ = ["PreparedRequest"]


@dataclass
class PreparedRequest:
    method: str
    url: URL
    headers: Mapping[str, str]
    buffer: Optional[DynamicBuffer]
    sent_at_timestamp: int = 0  # time.time_ns() // 1000

    @classmethod
    def prepare(
        cls,
        method: str,
        url: URL | str,
        headers: Optional[MutableMapping[str, str]] = None,
        body: Optional[Any] = None,
        *,
        json: Optional[Any] = None,
        normalize: bool = True,
    ) -> "PreparedRequest":
        url = URL.parse_any(url, normalize=normalize)

        if body is not None:
            body = DynamicBuffer.parse_any(obj=body)
        elif json is not None:
            body = DynamicBuffer()
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

        return cls(method=method, url=url, headers=headers, buffer=body, sent_at_timestamp=0)

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

        buffer: Optional[DynamicBuffer] = None
        if body_obj is not None:
            buffer = DynamicBuffer.parse_any(obj=body_obj)

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

        return cls(
            method=method_s,
            url=url,
            headers=headers,
            buffer=buffer,
            sent_at_timestamp=sent_at_ts,
        )

    @property
    def body(self):
        return self.buffer

    def anonymize(
        self,
        mode: str = "redact"
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
        *,
        column_prefix: str = "",
    ) -> pa.RecordBatch:
        if parse:
            raise NotImplementedError

        schema = pa.schema(
            [
                pa.field(f"{column_prefix}method", pa.string(), nullable=False, metadata={"comment": "The HTTP verb (GET, POST, etc.)"}),
                pa.field(f"{column_prefix}url", pa.string(), nullable=False, metadata={"comment": "The full request URL string"}),
                pa.field(f"{column_prefix}url_scheme", pa.string(), nullable=True, metadata={"comment": "URL protocol (e.g., http, https)"}),
                pa.field(f"{column_prefix}url_userinfo", pa.string(), nullable=True, metadata={"comment": "Authentication information in the URL"}),
                pa.field(f"{column_prefix}url_host", pa.string(), nullable=True, metadata={"comment": "Domain name or IP address of the server"}),
                pa.field(f"{column_prefix}url_port", pa.int32(), nullable=True, metadata={"comment": "TCP port number"}),
                pa.field(f"{column_prefix}url_path", pa.string(), nullable=True, metadata={"comment": "Hierarchical path to the resource"}),
                pa.field(
                    f"{column_prefix}url_query",
                    pa.map_(pa.field("key", pa.string(), nullable=False), pa.field("value", pa.string(), nullable=False)),
                    nullable=True,
                    metadata={"comment": "Parsed query string parameters as key-value pairs"},
                ),
                pa.field(f"{column_prefix}url_fragment", pa.string(), nullable=True, metadata={"comment": "The internal anchor or fragment identifier"}),
                pa.field(f"{column_prefix}body", pa.binary(), nullable=True, metadata={"comment": "Raw binary payload of the request"}),
                pa.field(f"{column_prefix}body_hash64", pa.int64(), nullable=True, metadata={"comment": "64-bit hash (xxh3) of the request body", "algorithm": "xxh3_64"}),
                pa.field(f"{column_prefix}sent_at", pa.timestamp("us", "UTC"), nullable=False, metadata={"comment": "UTC timestamp of when the request was dispatched"}),
            ]
        )

        u = self.url
        url_s = u.to_string()

        scheme_v = u.scheme or None
        userinfo_v = u.userinfo or None
        host_v = u.host or None
        port_v = None if (u.port is None or u.port == 0) else int(u.port)
        path_v = u.path or None
        fragment_v = u.fragment or None

        q = u.query_dict
        q_v = None if not q else {k: "|".join(vs) for k, vs in q.items()}

        body_bytes, body_h64 = (None, None) if self.buffer is None else (self.buffer.to_bytes(), self.buffer.xxh3_64().intdigest())

        arrays = [
            pa.array([self.method], type=pa.string()),
            pa.array([url_s], type=pa.string()),
            pa.array([scheme_v], type=pa.string()),
            pa.array([userinfo_v], type=pa.string()),
            pa.array([host_v], type=pa.string()),
            pa.array([port_v], type=pa.int32()),
            pa.array([path_v], type=pa.string()),
            pa.array([q_v], type=schema.field(f"{column_prefix}url_query").type),
            pa.array([fragment_v], type=pa.string()),
            pa.array([body_bytes], type=pa.binary()),
            pa.array([body_h64], type=pa.int64()),
            pa.array([self.sent_at_timestamp], type=pa.timestamp("us", "UTC")),
        ]

        return pa.RecordBatch.from_arrays(arrays, schema=schema)  # type: ignore
