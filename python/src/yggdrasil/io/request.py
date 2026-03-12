# yggdrasil.io.request
from __future__ import annotations

import json as json_module
from dataclasses import MISSING, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, MutableMapping, Optional

from yggdrasil.arrow.lib import pyarrow as pa
from yggdrasil.data.cast.registry import identity
from yggdrasil.dataclasses.dataclass import get_from_dict
from yggdrasil.io import MediaType
from .buffer import BytesIO
from .enums import GZIP, Codec, MimeType
from .headers import PromotedHeaders, normalize_headers
from .url import URL

if TYPE_CHECKING:
    from .response import Response


__all__ = ["PreparedRequest", "REQUEST_ARROW_SCHEMA"]


REQUEST_ARROW_SCHEMA = pa.schema(
    [
        pa.field(
            "request_method",
            pa.string(),
            nullable=False,
            metadata={"comment": "Method (GET, POST, etc.)"},
        ),
        pa.field(
            "request_url_str",
            pa.string(),
            nullable=False,
            metadata={"comment": "Full request URL as string (deterministic)"},
        ),
        pa.field("request_url_scheme", pa.string(), nullable=True, metadata={"comment": "URL scheme (e.g., http, https)"}),
        pa.field("request_url_userinfo", pa.string(), nullable=True, metadata={"comment": "Userinfo from URL authority (e.g., user:pass). Avoid persisting secrets."}),
        pa.field("request_url_host", pa.string(), nullable=True, metadata={"comment": "Host (domain or IP)"}),
        pa.field("request_url_port", pa.int32(), nullable=True, metadata={"comment": "Port number if explicitly specified"}),
        pa.field("request_url_path", pa.string(), nullable=True, metadata={"comment": "Path component of the URL"}),
        pa.field("request_url_query", pa.string(), nullable=True, metadata={"comment": "Raw query string (without leading '?')"}),
        pa.field("request_url_fragment", pa.string(), nullable=True, metadata={"comment": "Fragment identifier (without leading '#')"}),

        pa.field("request_host", pa.string(), nullable=True, metadata={"comment": "Host header"}),
        pa.field("request_user_agent", pa.string(), nullable=True, metadata={"comment": "User-Agent header"}),
        pa.field("request_accept", pa.string(), nullable=True, metadata={"comment": "Accept header"}),
        pa.field("request_accept_encoding", pa.string(), nullable=True, metadata={"comment": "Accept-Encoding header"}),
        pa.field("request_accept_language", pa.string(), nullable=True, metadata={"comment": "Accept-Language header"}),
        pa.field("request_content_type", pa.string(), nullable=True, metadata={"comment": "Content-Type header"}),
        pa.field("request_content_length", pa.int64(), nullable=False, metadata={"comment": "Content-Length header parsed as integer when possible"}),
        pa.field("request_content_encoding", pa.string(), nullable=True, metadata={"comment": "Content-Encoding header"}),
        pa.field("request_transfer_encoding", pa.string(), nullable=True, metadata={"comment": "Transfer-Encoding header"}),

        pa.field(
            "request_headers",
            pa.map_(pa.string(), pa.string()),
            nullable=False,
            metadata={
                "comment": "Request headers excluding promoted common headers",
                "keys_sorted": "false",
            },
        ),
        pa.field(
            "request_tags",
            pa.map_(pa.string(), pa.string()),
            nullable=False,
            metadata={"comment": "Request tags merged with URL query params; explicit tags win on conflict"},
        ),
        pa.field(
            "request_body",
            pa.binary(),
            nullable=True,
            metadata={"comment": "Raw request body bytes"},
        ),
        pa.field(
            "request_body_hash",
            pa.int64(),
            nullable=True,
            metadata={
                "comment": "Signed Int64 XXH3 digest of response_body",
                "algorithm": "xxh3_64",
            },
        ),
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
    metadata={"comment": "Prepared request flattened into deterministic columns for logging/replay."},
)


@dataclass
class PreparedRequest:
    method: str
    url: URL
    headers: MutableMapping[str, str]
    tags: Optional[MutableMapping[str, str]]
    buffer: Optional[BytesIO]
    sent_at_timestamp: int = field(default=0, hash=False, compare=False)

    before_send: Callable[["PreparedRequest"], "PreparedRequest"] = field(
        default=identity,
        hash=False,
        compare=False,
    )
    prepare_response: Callable[["Response"], "Response"] = field(
        default=identity,
        hash=False,
        compare=False,
    )

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        normalize: bool = True,
        prefix: str = "request_",
    ) -> "PreparedRequest":
        if isinstance(obj, (str, bytes)):
            obj = json_module.loads(obj)

        if isinstance(obj, Mapping):
            return cls.parse_dict(obj, normalize=normalize, prefix=prefix)

        raise ValueError(f"Cannot make {cls.__name__} from {type(obj)}")

    @classmethod
    def parse_dict(
        cls,
        obj: Mapping[str, Any],
        *,
        normalize: bool = True,
        prefix: str = "request_",
    ) -> "PreparedRequest":
        method = cls._parse_method(obj, prefix=prefix)
        url = cls._parse_url(obj, normalize=normalize, prefix=prefix)
        headers = cls._parse_headers(obj, prefix=prefix)
        tags = cls._parse_tags(obj, prefix=prefix)
        buffer = cls._parse_buffer(obj, prefix=prefix)
        sent_at_timestamp = cls._parse_sent_at_timestamp(obj, prefix=prefix)

        return cls(
            method=method,
            url=url,
            headers=headers,
            tags=tags,
            buffer=buffer,
            sent_at_timestamp=sent_at_timestamp,
        )

    @staticmethod
    def _parse_method(obj: Mapping[str, Any], *, prefix: str) -> str:
        method = get_from_dict(obj, keys=("method",), prefix=prefix)
        return "GET" if method is MISSING or method in (None, "") else str(method)

    @classmethod
    def _parse_url(
        cls,
        obj: Mapping[str, Any],
        *,
        normalize: bool,
        prefix: str,
    ) -> URL:
        url_str = get_from_dict(obj, keys=("url_str", "url", "href", "uri", "request_url_str"), prefix=prefix)
        url_struct = get_from_dict(obj, keys=("url", "request_url"), prefix=prefix)

        if url_str is not MISSING and url_str not in (None, ""):
            return URL.parse(url_str, normalize=normalize)

        if isinstance(url_struct, Mapping):
            return URL.parse(
                {
                    "scheme": url_struct.get("scheme") or "",
                    "userinfo": url_struct.get("userinfo") or "",
                    "host": url_struct.get("host") or "",
                    "port": url_struct.get("port") or 0,
                    "path": url_struct.get("path") or "",
                    "query": url_struct.get("query") or "",
                    "fragment": url_struct.get("fragment") or "",
                },
                normalize=normalize,
            )

        scheme = get_from_dict(obj, ("url_scheme",), prefix=prefix)
        userinfo = get_from_dict(obj, ("url_userinfo",), prefix=prefix)
        host = get_from_dict(obj, ("url_host",), prefix=prefix)
        port = get_from_dict(obj, ("url_port",), prefix=prefix)
        path = get_from_dict(obj, ("url_path",), prefix=prefix)
        query = get_from_dict(obj, ("url_query",), prefix=prefix)
        fragment = get_from_dict(obj, ("url_fragment",), prefix=prefix)

        parts = (scheme, userinfo, host, port, path, query, fragment)
        if not any(part is not MISSING for part in parts):
            raise ValueError(
                "PreparedRequest.parse_dict: missing url/url_str/request_url_str or exploded url fields"
            )

        return URL.parse(
            {
                "scheme": "" if scheme in (MISSING, None) else str(scheme),
                "userinfo": "" if userinfo in (MISSING, None) else str(userinfo),
                "host": "" if host in (MISSING, None) else str(host),
                "port": 0 if port is MISSING or port in (None, "") else int(port),
                "path": "" if path in (MISSING, None) else str(path),
                "query": "" if query in (MISSING, None) else str(query),
                "fragment": "" if fragment in (MISSING, None) else str(fragment),
            },
            normalize=normalize,
        )

    @staticmethod
    def _parse_headers(obj: Mapping[str, Any], *, prefix: str) -> MutableMapping[str, str]:
        headers = get_from_dict(obj, keys=("headers",), prefix="request_")

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

        out: MutableMapping[str, str] = {}
        for header_name, value in dumped_headers.items():
            if value is MISSING or value in (None, ""):
                continue
            out[header_name] = str(value)

        return out

    @staticmethod
    def _parse_tags(obj: Mapping[str, Any], *, prefix: str) -> dict[str, str]:
        tags = get_from_dict(obj, keys=("tags", "request_tags"), prefix=prefix)
        if not isinstance(tags, Mapping):
            return {}
        return {str(k): str(v) for k, v in tags.items()}

    @staticmethod
    def _parse_buffer(obj: Mapping[str, Any], *, prefix: str) -> Optional[BytesIO]:
        buffer = get_from_dict(obj, keys=("buffer", "body", "content", "data"), prefix=prefix)
        if buffer is MISSING or buffer is None:
            return None
        return BytesIO.parse(buffer)

    @staticmethod
    def _parse_sent_at_timestamp(obj: Mapping[str, Any], *, prefix: str) -> int:
        value = get_from_dict(
            obj,
            keys=(
                "sent_at_timestamp",
                "sent_at_timestamp_epoch",
                "sent_at",
                "request_sent_at_epoch",
                "request_sent_at",
            ),
            prefix=prefix,
        )
        if value is MISSING or value in (None, ""):
            return 0
        return int(value)

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
        prepare_response: Optional[Callable[["Response"], "Response"]] = ...,
        normalize: bool = True,
        copy_buffer: bool = False,
    ) -> "PreparedRequest":
        new_url = self.url if url is None else URL.parse(url, normalize=normalize)

        if headers is None:
            new_headers = dict(self.headers) if self.headers else {}
        else:
            new_headers = {str(k): str(v) for k, v in headers.items()}

        if buffer is ...:
            new_buffer = self.buffer
            if copy_buffer and new_buffer is not None:
                new_buffer = BytesIO.parse(new_buffer.to_bytes())
        else:
            new_buffer = buffer

        new_before_send = self.before_send if before_send is ... else before_send
        new_prepare_response = self.prepare_response if prepare_response is ... else prepare_response

        return self.__class__(
            method=self.method if method is None else str(method),
            url=new_url,
            headers=new_headers,
            buffer=new_buffer,
            tags=self.tags if tags is None else tags,
            sent_at_timestamp=self.sent_at_timestamp if sent_at_timestamp is None else int(sent_at_timestamp),
            before_send=new_before_send,
            prepare_response=new_prepare_response,
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
        after_received: Optional[Callable[["Response"], "Response"]] = None,
        *,
        json: Optional[Any] = None,
        normalize: bool = True,
        compress_threshold: Optional[int] = 4 * 1024 * 1024,
        compress_codec: Optional[Codec] = GZIP,
    ) -> "PreparedRequest":
        parsed_url = URL.parse(url, normalize=normalize)
        out_headers: MutableMapping[str, str] = {str(k): str(v) for k, v in (headers or {}).items()}

        request_body: Optional[BytesIO] = None
        if body is not None:
            request_body = BytesIO(body, copy=False)
        elif json is not None:
            request_body = BytesIO(json_module.dumps(json).encode("utf-8"), copy=False)
            out_headers["Content-Type"] = MimeType.JSON.value

            if compress_threshold and request_body.size > compress_threshold:
                request_body = request_body.compress(codec=compress_codec)
                out_headers["Content-Encoding"] = compress_codec.name

        if request_body is not None:
            out_headers["Content-Length"] = str(request_body.size)

        return cls(
            method=str(method),
            url=parsed_url,
            headers=normalize_headers(out_headers, is_request=True, body=request_body) if normalize else out_headers,
            buffer=request_body,
            tags=tags,
            sent_at_timestamp=0,
            before_send=before_send or identity,
            prepare_response=after_received or identity,
        )

    def prepare_to_send(
        self,
        sent_at_timestamp: int,
        headers: Optional[Mapping[str, str]],
    ) -> "PreparedRequest":
        instance = self.before_send(self) if self.before_send else self

        if instance.headers is None:
            instance.headers = {}

        if headers:
            instance.headers.update(headers)

        instance.sent_at_timestamp = sent_at_timestamp

        return instance

    @property
    def body(self) -> Optional[BytesIO]:
        return self.buffer

    @property
    def content_length(self) -> int:
        if self.buffer is not None:
            return self.buffer.size
        return 0

    @property
    def authorization(self) -> Optional[str]:
        return self.headers.get("Authorization") if self.headers else None

    @property
    def accept_media_type(self):
        if not self.headers:
            return MediaType(MimeType.OCTET_STREAM, None)

        accept = MimeType.parse(self.headers.get("Accept"), default=MimeType.OCTET_STREAM)
        codec = Codec.parse(self.headers.get("Accept-Encoding"), default=None)

        return MediaType(accept, codec)

    @accept_media_type.setter
    def accept_media_type(self, value: MediaType):
        if self.headers is None:
            self.headers = {}
        self.headers["Accept"] = value.mime_type.value
        if value.codec:
            self.headers["Accept-Encoding"] = value.codec.name
        else:
            self.headers.pop("Accept-Encoding", None)

    @authorization.setter
    def authorization(self, value: Optional[str]):
        if self.headers is None:
            self.headers = {}
        if value is not None:
            self.headers["Authorization"] = value
        else:
            self.headers.pop("Authorization", None)

    @property
    def x_api_key(self) -> Optional[str]:
        return self.headers.get("X-API-Key") if self.headers else None

    @x_api_key.setter
    def x_api_key(self, value: Optional[str]):
        if self.headers is None:
            self.headers = {}
        if value is not None:
            self.headers["X-API-Key"] = value
        else:
            self.headers.pop("X-API-Key", None)

    def update_headers(
        self,
        headers: Mapping[str, str],
        *,
        normalize: bool = True,
    ) -> "PreparedRequest":
        if not headers:
            return self

        if normalize:
            headers = normalize_headers(
                headers,
                is_request=True, anonymize=False, add_missing=False
            )

        if not self.headers:
            self.headers = headers
            return self

        self.headers.update(headers)

        return self

    def update_tags(
        self,
        tags: MutableMapping[str, str],
    ) -> "PreparedRequest":
        if not tags:
            return self

        if not self.tags:
            self.tags = tags
            return self

        self.tags.update(tags)

        return self

    def anonymize(self, mode: Literal["remove", "redact"] = "remove") -> "PreparedRequest":
        return replace(
            self,
            headers=normalize_headers(
                self.headers,
                is_request=True, mode=mode, body=self.body, anonymize=True
            ),
            url=self.url.anonymize(mode=mode),
        )

    def to_arrow_batch(self, parse: bool = False) -> pa.RecordBatch:
        if parse:
            raise NotImplementedError

        u = self.url
        url_s = u.to_string()

        promoted = PromotedHeaders.extract(self.headers or {})

        tags_v = dict(u.query_items())
        if self.tags:
            tags_v.update({str(k): str(v) for k, v in self.tags.items()})

        if self.buffer is not None:
            body_bytes = self.buffer.to_bytes()
            body_hash = self.buffer.xxh3_int64()
        else:
            body_bytes = None
            body_hash = None

        values = {
            "request_method": self.method,
            "request_url_str": url_s,
            "request_url_scheme": u.scheme,
            "request_url_userinfo": u.userinfo,
            "request_url_host": u.host,
            "request_url_port": u.port,
            "request_url_path": u.path,
            "request_url_query": u.query,
            "request_url_fragment": u.fragment,

            "request_host": promoted.host,
            "request_user_agent": promoted.user_agent,
            "request_accept": promoted.accept,
            "request_accept_encoding": promoted.accept_encoding,
            "request_accept_language": promoted.accept_language,
            "request_content_type": promoted.content_type,
            "request_content_length": promoted.content_length or 0,
            "request_content_encoding": promoted.content_encoding,
            "request_transfer_encoding": promoted.transfer_encoding,

            "request_headers": promoted.remaining,
            "request_tags": tags_v,
            "request_body": body_bytes,
            "request_body_hash": body_hash,

            "request_sent_at": self.sent_at_timestamp,
            "request_sent_at_epoch": self.sent_at_timestamp,
        }

        arrays = [
            pa.array([values[f.name]], type=f.type)
            for f in REQUEST_ARROW_SCHEMA
        ]

        return pa.RecordBatch.from_arrays(arrays, schema=REQUEST_ARROW_SCHEMA)  # type: ignore[arg-type]

    def apply(
        self,
        func: Callable[["PreparedRequest"], "PreparedRequest"],
    ):
        return func(self)