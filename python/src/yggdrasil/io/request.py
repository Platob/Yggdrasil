from __future__ import annotations

import json
import json as json_module
import time
from dataclasses import dataclass, replace, MISSING
from typing import Mapping, Any, Optional, MutableMapping, Literal, Callable

from yggdrasil.arrow.lib import pyarrow as pa
from yggdrasil.dataclasses.dataclass import get_from_dict
from yggdrasil.io.enums.mime_type import MimeType
from yggdrasil.io.headers import anonymize_headers
from .buffer import BytesIO
from .url import URL

__all__ = ["PreparedRequest", "REQUEST_ARROW_SCHEMA"]


# ----------------------------
# Arrow schema
# ----------------------------

REQUEST_URL_STRUCT = pa.struct(
    [
        pa.field("scheme", pa.string(), nullable=True, metadata={"comment": "URL scheme (e.g., http, https)"}),
        pa.field(
            "userinfo",
            pa.string(),
            nullable=True,
            metadata={"comment": "Userinfo from URL authority (e.g., user:pass). Avoid persisting secrets."},
        ),
        pa.field("host", pa.string(), nullable=True, metadata={"comment": "Host (domain or IP)"}),
        pa.field("port", pa.int32(), nullable=True, metadata={"comment": "Port number if explicitly specified"}),
        pa.field("path", pa.string(), nullable=True, metadata={"comment": "Path component of the URL"}),
        pa.field("query", pa.string(), nullable=True, metadata={"comment": "Raw query string (without leading '?')"}),
        pa.field("fragment", pa.string(), nullable=True, metadata={"comment": "Fragment identifier (without leading '#')"}),
    ]
)

REQUEST_ARROW_SCHEMA = pa.schema(
    [
        pa.field(
            "request_method",
            pa.string(),
            nullable=False,
            metadata={"comment": "HTTP verb (GET, POST, etc.)"},
        ),

        # ✅ full URL string (non-nullable)
        pa.field(
            "request_url_str",
            pa.string(),
            nullable=False,
            metadata={"comment": "Full request URL as string (deterministic)"},
        ),

        # ✅ URL components as a struct (non-nullable)
        pa.field(
            "request_url",
            REQUEST_URL_STRUCT,
            nullable=False,
            metadata={"comment": "URL parsed into a struct (best-effort parsed)"},
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
            metadata={
                "comment": "BLAKE3-256 digest of request_body (32 bytes)",
                "algorithm": "blake3",
                "byte_width": "32",
            },
        ),

        # ---- timing ----
        pa.field(
            "request_sent_at",
            pa.timestamp("us", "UTC"),
            nullable=False,
            metadata={"comment": "UTC timestamp when request was dispatched", "unit": "us", "tz": "UTC"},
        ),
        pa.field(
            "request_sent_at_epoch",
            pa.int64(),
            nullable=False,
            metadata={"comment": "UTC epoch timestamp when request was dispatched", "unit": "us", "tz": "UTC"},
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
    tags: Optional[Mapping[str, str]]
    buffer: Optional[BytesIO]
    sent_at_timestamp: int = 0  # time.time_ns() // 1000

    before_send: Optional[Callable[["PreparedRequest"], "PreparedRequest"]] = None

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        normalize: bool = True,
        prefix: str = "request_"
    ):
        if isinstance(obj, (str, bytes)):
            obj = json.loads(obj)

        if isinstance(obj, dict):
            return cls.parse_dict(obj, normalize=normalize, prefix=prefix)

        raise ValueError(
            f"Cannot make {cls} from {type(obj)}"
        )

    @classmethod
    def parse_dict(
        cls,
        obj: Mapping[str, Any],
        *,
        normalize: bool = True,
        prefix: str = "request_"
    ):
        # method (aliases + request_ prefix)
        method = get_from_dict(obj, keys=("method", "http_method", "verb"), prefix=prefix)
        method = "GET" if method is MISSING or method in (None, "") else str(method)

        # url: prefer new fields first, then legacy ones, then structured url
        url_str = get_from_dict(obj, keys=("url_str", "url", "href", "uri"), prefix=prefix)
        url_struct = get_from_dict(obj, keys=("url",), prefix=prefix)  # could be struct dict from Arrow
        if url_str is not MISSING and url_str not in (None, ""):
            url = url_str
        elif isinstance(url_struct, Mapping):
            # Accept our Arrow struct-like shape: {"scheme":..., "host":..., ...}
            url = {
                "scheme": url_struct.get("scheme") or "",
                "userinfo": url_struct.get("userinfo") or "",
                "host": url_struct.get("host") or "",
                "port": url_struct.get("port") or 0,
                "path": url_struct.get("path") or "",
                "query": url_struct.get("query") or "",
                "fragment": url_struct.get("fragment") or "",
            }
        else:
            # last resort: support older flattened fields if present
            has_exploded = any(
                (k in obj) or (("request_" + k) in obj)
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
            if not has_exploded:
                raise ValueError("PreparedRequest.parse_dict: missing url/url_str/request_url_str/request_url")

            url = {
                "scheme": get_from_dict(obj, ("url_scheme",), prefix=prefix) or "",
                "userinfo": get_from_dict(obj, ("url_userinfo",), prefix=prefix) or "",
                "host": get_from_dict(obj, ("url_host",), prefix=prefix) or "",
                "port": get_from_dict(obj, ("url_port",), prefix=prefix) or 0,
                "path": get_from_dict(obj, ("url_path",), prefix=prefix) or "",
                "query": get_from_dict(obj, ("url_query",), prefix=prefix) or "",
                "fragment": get_from_dict(obj, ("url_fragment",), prefix=prefix) or "",
            }

        # headers/tags (aliases + request_ prefix)
        headers = get_from_dict(obj, keys=("headers", "header", "hdrs", "request_headers"), prefix=prefix)
        headers = headers if isinstance(headers, Mapping) else {}
        headers = {str(k): str(v) for k, v in headers.items()}

        tags = get_from_dict(obj, keys=("tags", "request_tags"), prefix=prefix)
        tags = tags if isinstance(tags, Mapping) else {}
        tags = {str(k): str(v) for k, v in tags.items()}

        # body/buffer (aliases + request_ prefix)
        buffer = get_from_dict(obj, keys=("buffer", "body", "content", "data"), prefix=prefix)
        if buffer is MISSING:
            buffer = None
        if buffer is not None:
            buffer = BytesIO.parse(buffer)

        # sent_at timestamp (epoch micros) - accept several names
        sent_at_timestamp = get_from_dict(
            obj,
            keys=(
            "sent_at_timestamp", "sent_at_timestamp_epoch", "sent_at", "request_sent_at_epoch", "request_sent_at"),
            prefix=prefix,
        )
        if sent_at_timestamp is MISSING or sent_at_timestamp in (None, ""):
            sent_at_timestamp = 0
        else:
            sent_at_timestamp = int(sent_at_timestamp)

        return cls(
            method=method,
            url=URL.parse(url, normalize=normalize),
            headers=headers,
            tags=tags,
            buffer=buffer,
            sent_at_timestamp=sent_at_timestamp,
        )

    def copy(
        self,
        *,
        method: Optional[str] = None,
        url: URL | str | None = None,
        headers: Optional[Mapping[str, str]] = None,
        buffer: Optional[BytesIO] = ...,
        tags: Optional[Mapping[str, str]] = None,
        sent_at_timestamp: Optional[int] = None,
        before_send: Optional[Callable[["PreparedRequest"], "PreparedRequest"]] = ...,
        normalize: bool = True,
        copy_buffer: bool = False,
    ) -> "PreparedRequest":
        new_url = self.url if url is None else URL.parse(url, normalize=normalize)

        new_headers = dict(self.headers) if self.headers else {}
        if headers is not None:
            new_headers = {str(k): str(v) for k, v in headers.items()}

        if buffer is ...:
            new_buf = self.buffer
            if copy_buffer and new_buf is not None:
                new_buf = BytesIO.parse(new_buf.to_bytes())
        else:
            new_buf = buffer

        new_before_send = self.before_send if before_send is ... else before_send

        return self.__class__(
            method=self.method if method is None else str(method),
            url=new_url,
            headers=new_headers,
            buffer=new_buf,
            tags=self.tags if tags is None else tags,
            sent_at_timestamp=self.sent_at_timestamp if sent_at_timestamp is None else int(sent_at_timestamp),
            before_send=new_before_send,
        )

    @classmethod
    def prepare(
        cls,
        method: str,
        url: URL | str,
        headers: Optional[MutableMapping[str, str]] = None,
        body: Optional[Any] = None,
        tags: Optional[Mapping[str, str]] = None,
        before_send: Optional[Callable[["PreparedRequest"], "PreparedRequest"]] = None,
        *,
        json: Optional[Any] = None,
        normalize: bool = True,
    ) -> "PreparedRequest":
        url = URL.parse(url, normalize=normalize)

        if body is not None:
            body = BytesIO.parse(body)
        elif json is not None:
            body = BytesIO()
            json_module.dump(json, body)
            body.seek(0)

            if not headers:
                headers = {"Content-Type": MimeType.JSON.value}
            else:
                headers["Content-Type"] = MimeType.JSON.value

        if not headers:
            headers = {}

        if body is not None:
            headers["Content-Length"] = body.size

        return cls(
            method=method,
            url=url,
            headers=headers,
            buffer=body,
            tags=tags,
            sent_at_timestamp=0,
            before_send=before_send,
        )

    def prepare_to_send(self, add_statistics: bool):
        if self.before_send is not None:
            instance = self.before_send(self)
        else:
            instance = self

        instance.sent_at_timestamp = time.time_ns() // 1000 if add_statistics else 0
        return instance

    # ... parse_any / parse_str / parse_dict unchanged ...

    @property
    def body(self):
        return self.buffer

    def anonymize(self, mode: Literal["remove", "redact", "hash"] = "remove") -> "PreparedRequest":
        return replace(
            self,
            headers=anonymize_headers(self.headers, mode=mode),
            url=self.url.anonymize(mode=mode),
        )

    def to_arrow_batch(self, parse: bool = False) -> pa.RecordBatch:
        if parse:
            raise NotImplementedError

        u = self.url
        url_s = u.to_string()

        url_struct_value = {
            "scheme": u.scheme,
            "userinfo": u.userinfo,
            "host": u.host,
            "port": u.port,
            "path": u.path,
            "query": u.query,
            "fragment": u.fragment,
        }

        headers_v = {str(k): str(v) for k, v in (self.headers.items() if self.headers else ())}
        tags_v = {str(k): str(v) for k, v in (self.tags.items() if self.tags else ())}

        if self.buffer:
            body_bytes = self.buffer.to_bytes()
            body_blake3_32 = self.buffer.blake3().digest()
        else:
            body_bytes, body_blake3_32 = None, None

        arrays = [
            pa.array([self.method], type=REQUEST_ARROW_SCHEMA.field("request_method").type),

            # ✅ full URL string non-nullable
            pa.array([url_s], type=REQUEST_ARROW_SCHEMA.field("request_url_str").type),

            # ✅ struct URL non-nullable (fields inside may be null)
            pa.array([url_struct_value], type=REQUEST_ARROW_SCHEMA.field("request_url").type),

            pa.array([headers_v], type=REQUEST_ARROW_SCHEMA.field("request_headers").type),
            pa.array([tags_v], type=REQUEST_ARROW_SCHEMA.field("request_tags").type),
            pa.array([body_bytes], type=REQUEST_ARROW_SCHEMA.field("request_body").type),
            pa.array([body_blake3_32], type=REQUEST_ARROW_SCHEMA.field("request_body_hash").type),

            # NOTE: current code stores epoch micros into both; keeping behavior as-is
            pa.array([self.sent_at_timestamp], type=REQUEST_ARROW_SCHEMA.field("request_sent_at").type),
            pa.array([self.sent_at_timestamp], type=REQUEST_ARROW_SCHEMA.field("request_sent_at_epoch").type),
        ]

        return pa.RecordBatch.from_arrays(arrays, schema=REQUEST_ARROW_SCHEMA)  # type: ignore