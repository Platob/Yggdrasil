# yggdrasil.io.request
from __future__ import annotations

import base64
import datetime as dt
import json as json_module
from dataclasses import MISSING, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Literal, Mapping, MutableMapping, Optional

from yggdrasil.arrow.lib import pyarrow as pa
from yggdrasil.data import any_to_datetime
from yggdrasil.dataclasses.dataclass import get_from_dict
from yggdrasil.io import MediaType, MimeTypes

from .buffer import BytesIO
from .enums import GZIP, Codec, MimeType
from .headers import DEFAULT_HOSTNAME, PromotedHeaders, normalize_headers
from .url import URL

if TYPE_CHECKING:
    from .response import Response


__all__ = ["PreparedRequest", "REQUEST_ARROW_SCHEMA"]


def _json_meta(value: Mapping[str, Any]) -> str:
    return json_module.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _build_metadata(
    *,
    comment: str,
    json_tags: Optional[Mapping[str, Any]] = None,
    **extra: Any,
) -> dict[str, str]:
    """
    Build Arrow-friendly metadata using string values only.

    Notes
    -----
    Arrow field/schema metadata is key/value bytes under the hood.
    Persisting JSON as a compact string is the most robust way to keep
    structured classification metadata across Arrow -> Parquet -> Delta hops.
    """
    out: dict[str, str] = {"comment": str(comment)}
    if json_tags:
        out["json_tags"] = _json_meta(json_tags)
    for k, v in extra.items():
        out[str(k)] = str(v)
    return out


def _field(
    name: str,
    dtype: pa.DataType,
    *,
    nullable: bool,
    comment: str,
    json_tags: Optional[Mapping[str, Any]] = None,
    **extra_metadata: Any,
) -> pa.Field:
    return pa.field(
        name,
        dtype,
        nullable=nullable,
        metadata=_build_metadata(
            comment=comment,
            json_tags=json_tags,
            **extra_metadata,
        ),
    )


_REQUEST_SCHEMA_JSON_TAGS: dict[str, str] = {
    "domain": "http",
    "entity": "request",
    "layer": "bronze",
}

REQUEST_ARROW_SCHEMA = pa.schema(
    [
        _field(
            "request_method",
            pa.string(),
            nullable=False,
            comment="HTTP method (GET, POST, etc.)",
            json_tags={
                "entity": "request",
                "group": "routing",
            },
            partition_by="true",
        ),
        _field(
            "request_url_str",
            pa.string(),
            nullable=False,
            comment="Full request URL as deterministic string",
            json_tags={
                "entity": "request",
                "group": "url",
            },
        ),
        _field(
            "request_url_scheme",
            pa.string(),
            nullable=False,
            comment="URL scheme (for example http or https)",
            json_tags={
                "entity": "request",
                "group": "url",
            },
            partition_by="true",
        ),
        _field(
            "request_url_userinfo",
            pa.string(),
            nullable=True,
            comment="Userinfo from URL authority (for example user:pass)",
            json_tags={
                "entity": "request",
                "group": "url",
            },
        ),
        _field(
            "request_url_host",
            pa.string(),
            nullable=False,
            comment="Host name, domain, or IP address from the request URL",
            json_tags={
                "entity": "request",
                "group": "url",
            },
            partition_by="true",
        ),
        _field(
            "request_url_port",
            pa.int32(),
            nullable=True,
            comment="Port number if explicitly specified in the URL",
            json_tags={
                "entity": "request",
                "group": "url",
            },
        ),
        _field(
            "request_url_path",
            pa.string(),
            nullable=False,
            comment="Path component of the request URL",
            json_tags={
                "entity": "request",
                "group": "url",
            },
            partition_by="true",
        ),
        _field(
            "request_url_query",
            pa.string(),
            nullable=True,
            comment="Raw query string without leading question mark",
            json_tags={
                "entity": "request",
                "group": "url",
            },
        ),
        _field(
            "request_url_fragment",
            pa.string(),
            nullable=True,
            comment="Fragment identifier without leading hash",
            json_tags={
                "entity": "request",
                "group": "url",
            },
        ),
        _field(
            "request_host",
            pa.string(),
            nullable=True,
            comment="HTTP Host header",
            json_tags={
                "entity": "request",
                "group": "headers_promoted",
            },
        ),
        _field(
            "request_user_agent",
            pa.string(),
            nullable=True,
            comment="HTTP User-Agent header",
            json_tags={
                "entity": "request",
                "group": "headers_promoted",
            },
        ),
        _field(
            "request_accept",
            pa.string(),
            nullable=True,
            comment="HTTP Accept header",
            json_tags={
                "entity": "request",
                "group": "headers_promoted",
            },
        ),
        _field(
            "request_accept_encoding",
            pa.string(),
            nullable=True,
            comment="HTTP Accept-Encoding header",
            json_tags={
                "entity": "request",
                "group": "headers_promoted",
            },
        ),
        _field(
            "request_accept_language",
            pa.string(),
            nullable=True,
            comment="HTTP Accept-Language header",
            json_tags={
                "entity": "request",
                "group": "headers_promoted",
            },
        ),
        _field(
            "request_content_type",
            pa.string(),
            nullable=True,
            comment="HTTP Content-Type header",
            json_tags={
                "entity": "request",
                "group": "headers_promoted",
            },
        ),
        _field(
            "request_content_length",
            pa.int64(),
            nullable=False,
            comment="HTTP Content-Length header parsed as integer when possible",
            json_tags={
                "entity": "request",
                "group": "headers_promoted",
            },
        ),
        _field(
            "request_content_encoding",
            pa.string(),
            nullable=True,
            comment="HTTP Content-Encoding header",
            json_tags={
                "entity": "request",
                "group": "headers_promoted",
            },
        ),
        _field(
            "request_transfer_encoding",
            pa.string(),
            nullable=True,
            comment="HTTP Transfer-Encoding header",
            json_tags={
                "entity": "request",
                "group": "headers_promoted",
            },
        ),
        _field(
            "request_headers",
            pa.map_(pa.string(), pa.string()),
            nullable=False,
            comment="Request headers excluding promoted common headers",
            json_tags={
                "entity": "request",
                "group": "headers",
            },
            keys_sorted="false",
        ),
        _field(
            "request_tags",
            pa.map_(pa.string(), pa.string()),
            nullable=False,
            comment="Request tags merged with URL query params; explicit tags win on conflict",
            json_tags={
                "entity": "request",
                "group": "tags",
            },
        ),
        _field(
            "request_body",
            pa.binary(),
            nullable=True,
            comment="Raw request body bytes",
            json_tags={
                "entity": "request",
                "group": "payload",
            },
        ),
        _field(
            "request_body_hash",
            pa.int64(),
            nullable=True,
            comment="Signed Int64 XXH3 digest of request_body",
            json_tags={
                "entity": "request",
                "group": "payload",
                "algorithm": "xxh3_64",
            },
            algorithm="xxh3_64",
        ),
        _field(
            "request_sent_at",
            pa.timestamp("us", "UTC"),
            nullable=False,
            comment="UTC timestamp when request was dispatched",
            json_tags={
                "entity": "request",
                "group": "timing",
                "timezone": "UTC",
            },
            unit="us",
            tz="UTC",
        ),
    ],
    metadata=_build_metadata(
        comment="Prepared request flattened into deterministic columns for logging and replay.",
        json_tags=_REQUEST_SCHEMA_JSON_TAGS,
        time_column="request_sent_at",
    ),
)

_REQUEST_FIELD_NAMES: frozenset[str] = frozenset(REQUEST_ARROW_SCHEMA.names)
_PROMOTED_REQUEST_HEADER_FIELDS: tuple[tuple[str, str], ...] = (
    ("Host", "request_host"),
    ("User-Agent", "request_user_agent"),
    ("Accept", "request_accept"),
    ("Accept-Encoding", "request_accept_encoding"),
    ("Accept-Language", "request_accept_language"),
    ("Content-Type", "request_content_type"),
    ("Content-Length", "request_content_length"),
    ("Content-Encoding", "request_content_encoding"),
    ("Transfer-Encoding", "request_transfer_encoding"),
)


def _string_dict(arg: Optional[Mapping[Any, Any]]) -> dict[str, str]:
    if not arg:
        return {}
    return {str(k): str(v) for k, v in arg.items()}


def _map_as_str_dict(value: Any) -> dict[str, str]:
    if not value:
        return {}
    if isinstance(value, Mapping):
        return {str(k): str(v) for k, v in value.items()}
    try:
        return {str(k): str(v) for k, v in value if k is not None and v is not None}
    except Exception:
        return {}


def _epoch_us_to_utc_datetime(value: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(value / 1_000_000, tz=dt.timezone.utc)


@dataclass
class PreparedRequest:
    method: str
    url: URL
    headers: MutableMapping[str, str]
    tags: MutableMapping[str, str]
    buffer: Optional[BytesIO]
    sent_at: dt.datetime

    before_send: Optional[Callable[["PreparedRequest"], "PreparedRequest"]] = field(
        default=None,
        init=False,
        hash=False,
        compare=False,
    )
    prepare_response: Optional[Callable[["Response"], "Response"]] = field(
        default=None,
        init=False,
        hash=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.method = self.method or "GET"
        self.url = URL.parse(self.url)
        self.headers = _string_dict(self.headers)
        self.tags = _string_dict(self.tags)
        self.sent_at = any_to_datetime(self.sent_at) if self.sent_at else dt.datetime.fromtimestamp(
            0, tz=dt.timezone.utc
        )

        if self.buffer is not None and not isinstance(self.buffer, BytesIO):
            self.buffer = BytesIO.parse(self.buffer)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self.method} {self.url.to_string()!r}>"

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        normalize: bool = True,
        prefix: str = "request_",
    ) -> "PreparedRequest":
        if isinstance(obj, str):
            obj = {
                "url": URL.parse_str(obj, normalize=normalize)
            }

        if isinstance(obj, Mapping):
            return cls.parse_mapping(obj, normalize=normalize, prefix=prefix)

        raise ValueError(f"Cannot make {cls.__name__} from {type(obj)}")

    @classmethod
    def parse_mapping(
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
        sent_at = cls._parse_sent_at_timestamp(obj, prefix=prefix)

        if cls is PreparedRequest:
            if url.is_http:
                from .http_ import HTTPRequest

                return HTTPRequest(
                    method=method,
                    url=url,
                    headers=headers,
                    tags=tags,
                    buffer=buffer,
                    sent_at=sent_at
                )

        return cls(
            method=method,
            url=url,
            headers=headers,
            tags=tags,
            buffer=buffer,
            sent_at=sent_at
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
        headers = get_from_dict(obj, keys=("headers",), prefix=prefix)
        if isinstance(headers, Mapping):
            parsed = _string_dict(headers)
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

        out: dict[str, str] = {}
        for header_name, value in dumped_headers.items():
            if value is MISSING or value in (None, ""):
                continue
            out[header_name] = str(value)

        return out

    @staticmethod
    def _parse_tags(obj: Mapping[str, Any], *, prefix: str) -> dict[str, str]:
        tags = get_from_dict(obj, keys=("tags", "request_tags"), prefix=prefix)
        return _string_dict(tags if isinstance(tags, Mapping) else None)

    @staticmethod
    def _parse_buffer(obj: Mapping[str, Any], *, prefix: str) -> Optional[BytesIO]:
        buffer = get_from_dict(obj, keys=("buffer", "body", "content", "data"), prefix=prefix)
        if buffer is MISSING or buffer is None:
            return None
        return BytesIO.parse(buffer)

    @staticmethod
    def _parse_sent_at_timestamp(obj: Mapping[str, Any], *, prefix: str) -> dt.datetime:
        value = get_from_dict(
            obj,
            keys=(
                "sent_at_timestamp",
                "sent_at",
                "request_sent_at",
            ),
            prefix=prefix,
        )
        return any_to_datetime(value) if value not in (None, "", MISSING) else dt.datetime.fromtimestamp(
            0, tz=dt.timezone.utc
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
        out_headers: dict[str, str] = _string_dict(headers)

        request_body: Optional[BytesIO] = None
        if body is not None:
            request_body = BytesIO(body, copy=False)
        elif json is not None:
            request_body = BytesIO(json_module.dumps(json).encode("utf-8"), copy=False)
            out_headers["Content-Type"] = MimeTypes.JSON.value

            if compress_threshold and request_body.size > compress_threshold:
                request_body = request_body.compress(codec=compress_codec)
                if compress_codec is not None:
                    out_headers["Content-Encoding"] = compress_codec.name

        if request_body is not None:
            out_headers["Content-Length"] = str(request_body.size)

        out_class = cls

        if cls is PreparedRequest:
            if parsed_url.is_http:
                from .http_ import HTTPRequest
                out_class = HTTPRequest

        built = out_class(
            method=str(method),
            url=parsed_url,
            headers=normalize_headers(out_headers, is_request=True, body=request_body) if normalize else out_headers,
            tags=_string_dict(tags),
            buffer=request_body,
            sent_at=0,
        )
        built.before_send = before_send
        built.prepare_response = after_received
        return built

    def copy(
        self,
        *,
        method: Optional[str] = None,
        url: URL | str | None = None,
        headers: Optional[Mapping[str, str]] = None,
        buffer: Optional[BytesIO] = ...,
        tags: Optional[Mapping[str, str]] = None,
        sent_at: Optional[int] = None,
        before_send: Optional[Callable[["PreparedRequest"], "PreparedRequest"]] = ...,
        prepare_response: Optional[Callable[["Response"], "Response"]] = ...,
        normalize: bool = True,
        copy_buffer: bool = False,
    ) -> "PreparedRequest":
        new_url = self.url if url is None else URL.parse(url, normalize=normalize)
        new_headers = dict(self.headers) if headers is None else _string_dict(headers)

        if buffer is ...:
            new_buffer = self.buffer
            if copy_buffer and new_buffer is not None:
                new_buffer = BytesIO.parse(new_buffer.to_bytes())
        else:
            new_buffer = buffer

        new_tags = dict(self.tags) if tags is None else _string_dict(tags)

        built = self.__class__(
            method=self.method if method is None else str(method),
            url=new_url,
            headers=new_headers,
            tags=new_tags,
            buffer=new_buffer,
            sent_at=self.sent_at if sent_at is None else any_to_datetime(sent_at),
        )

        built.before_send = self.before_send if before_send is ... else before_send
        built.prepare_response = self.prepare_response if prepare_response is ... else prepare_response
        return built

    def prepare_to_send(
        self,
        sent_at: dt.datetime | dt.date | str | int | None,
        headers: Optional[Mapping[str, str]],
    ) -> "PreparedRequest":
        instance = self.before_send(self) if self.before_send else self

        if instance.headers is None:
            instance.headers = {}

        if headers:
            instance.headers.update(_string_dict(headers))

        instance.sent_at = dt.datetime.now(dt.timezone.utc) if sent_at is None else any_to_datetime(sent_at)

        return instance

    @property
    def body(self) -> Optional[BytesIO]:
        return self.buffer

    @property
    def content_length(self) -> int:
        return self.buffer.size if self.buffer is not None else 0

    @property
    def authorization(self) -> Optional[str]:
        return self.headers.get("Authorization") if self.headers else None

    @authorization.setter
    def authorization(self, value: Optional[str]):
        if self.headers is None:
            self.headers = {}
        if value is None:
            self.headers.pop("Authorization", None)
        else:
            self.headers["Authorization"] = str(value)

    @property
    def x_api_key(self) -> Optional[str]:
        return self.headers.get("X-API-Key") if self.headers else None

    @x_api_key.setter
    def x_api_key(self, value: Optional[str]):
        if self.headers is None:
            self.headers = {}
        if value is None:
            self.headers.pop("X-API-Key", None)
        else:
            self.headers["X-API-Key"] = str(value)

    @property
    def accept_media_type(self) -> MediaType:
        if not self.headers:
            return MediaType(MimeTypes.OCTET_STREAM, None)

        accept = MimeType.parse(self.headers.get("Accept"), default=MimeTypes.OCTET_STREAM)
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

    @property
    def sent_at_timestamp(self) -> int:
        return int(self.sent_at.timestamp() * 1_000_000)

    @property
    def arrow_values(self) -> dict[str, Any]:
        u = self.url
        promoted = PromotedHeaders.extract(self.headers or {})

        tags_v = dict(u.query_items())
        if self.tags:
            tags_v.update(_string_dict(self.tags))

        if self.buffer is not None:
            body_bytes = self.buffer.to_bytes()
            body_hash = self.buffer.xxh3_int64()
        else:
            body_bytes = None
            body_hash = None

        return {
            "request_method": self.method,
            "request_url_str": u.to_string(),
            "request_url_scheme": u.scheme,
            "request_url_userinfo": u.userinfo,
            "request_url_host": u.host,
            "request_url_port": u.port,
            "request_url_path": u.path,
            "request_url_query": u.query,
            "request_url_fragment": u.fragment,
            "request_host": promoted.host or DEFAULT_HOSTNAME,
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
            "request_sent_at": self.sent_at
        }

    def match_value(self, key: str) -> Any:
        values = self.arrow_values
        if key in values:
            return values[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise ValueError(
            f"Unsupported request match key: {key!r}. "
            f"Must be within: {REQUEST_ARROW_SCHEMA.names!r}"
        )

    def match_values(
        self,
        keys: Iterable[str],
    ) -> dict[str, Any]:
        return {str(key): self.match_value(str(key)) for key in keys}

    def match_tuple(
        self,
        keys: Iterable[str],
    ) -> tuple[Any, ...]:
        key_list = [str(key) for key in keys]
        values = self.match_values(key_list)
        return tuple(values[key] for key in key_list)

    def xxh3_64(
        self,
        hash_fields: Optional[Iterable[str]] = None,
    ):
        if not hash_fields:
            hash_fields = ["method", "url", "headers", "buffer"]

        buff = BytesIO()
        for hash_field in sorted(hash_fields):
            v = getattr(self, hash_field, None)

            if isinstance(v, str):
                buff.write(v.encode("utf-8"))
            elif isinstance(v, URL):
                buff.write(v.to_string().encode("utf-8"))
            elif isinstance(v, Mapping):
                for k, val in sorted(v.items()):
                    buff.write(str(k).encode("utf-8"))
                    buff.write(str(val).encode("utf-8"))
            elif isinstance(v, BytesIO):
                buff.write(v.xxh3_64().digest())
            elif v is None:
                buff.write(b"0")
            else:
                raise TypeError(f"Cannot hash field {hash_field} of type {type(v)}")

        return buff.xxh3_64()

    def xxh3_b64(
        self,
        url_safe: bool = True,
    ) -> str:
        h = self.xxh3_64().digest()
        return (
            base64.urlsafe_b64encode(h).decode("ascii")
            if url_safe
            else base64.b64encode(h).decode("ascii")
        )

    def update_headers(
        self,
        headers: MutableMapping[str, str],
        *,
        normalize: bool = True,
    ) -> "PreparedRequest":
        if not headers:
            return self

        next_headers: Mapping[str, str] = headers
        if normalize:
            next_headers = normalize_headers(
                headers,
                is_request=True,
                anonymize=False,
                add_missing=False,
            )

        if not self.headers:
            self.headers = _string_dict(next_headers)
        else:
            self.headers.update(_string_dict(next_headers))

        return self

    def update_tags(
        self,
        tags: MutableMapping[str, str],
    ) -> "PreparedRequest":
        if not tags:
            return self

        if not self.tags:
            self.tags = _string_dict(tags)
        else:
            self.tags.update(_string_dict(tags))

        return self

    def anonymize(self, mode: Literal["remove", "redact"] = "remove") -> "PreparedRequest":
        if not mode:
            return self

        return replace(
            self,
            headers=normalize_headers(
                self.headers,
                is_request=True,
                mode=mode,
                body=self.body,
                anonymize=True,
            ),
            url=self.url.anonymize(mode=mode),
        )

    def to_arrow_batch(self, parse: bool = False) -> pa.RecordBatch:
        if parse:
            raise NotImplementedError

        values = self.arrow_values
        arrays = [
            pa.array([values[f.name]], type=f.type)
            for f in REQUEST_ARROW_SCHEMA
        ]
        return pa.RecordBatch.from_arrays(arrays, schema=REQUEST_ARROW_SCHEMA)  # type: ignore[arg-type]

    def to_arrow_table(self, parse: bool = False) -> pa.Table:
        return pa.Table.from_batches([self.to_arrow_batch(parse=parse)])

    @classmethod
    def from_arrow(
        cls,
        batch: pa.RecordBatch | pa.Table,
        *,
        normalize: bool = True,
    ) -> Iterator["PreparedRequest"]:
        def _iter_batches(obj: pa.RecordBatch | pa.Table) -> Iterator[pa.RecordBatch]:
            if isinstance(obj, pa.RecordBatch):
                yield obj
            else:
                yield from obj.to_batches()

        req_cols = [f.name for f in REQUEST_ARROW_SCHEMA]

        for rb in _iter_batches(batch):
            cols = {
                name: rb.column(name)
                for name in req_cols
                if name in rb.schema.names
            }
            for i in range(rb.num_rows):
                yield cls._from_arrow_cols(cols, i, normalize=normalize)

    @classmethod
    def _from_arrow_cols(
        cls,
        cols: dict[str, Any],
        i: int,
        *,
        normalize: bool = True,
    ) -> "PreparedRequest":
        def _get(name: str) -> Any:
            if name in cols:
                return cols[name][i].as_py()
            return None

        url_str = _get("request_url_str")
        if url_str not in (None, ""):
            url_val = str(url_str)
            url_struct = None
        else:
            scheme = _get("request_url_scheme")
            userinfo = _get("request_url_userinfo")
            host = _get("request_url_host")
            port = _get("request_url_port")
            path = _get("request_url_path")
            query = _get("request_url_query")
            fragment = _get("request_url_fragment")

            if any(part not in (None, "", 0) for part in (scheme, userinfo, host, port, path, query, fragment)):
                url_val = None
                url_struct = {
                    "scheme": scheme or "",
                    "userinfo": userinfo or "",
                    "host": host or "",
                    "port": 0 if port in (None, "") else int(port),
                    "path": path or "/",
                    "query": query or "",
                    "fragment": fragment or "",
                }
            else:
                url_val = ""
                url_struct = None

        headers = _map_as_str_dict(_get("request_headers"))
        for header_name, field_name in _PROMOTED_REQUEST_HEADER_FIELDS:
            value = _get(field_name)
            if value not in (None, ""):
                headers[header_name] = str(value)

        sent_at_value = _get("request_sent_at")
        sent_at_timestamp = any_to_datetime(sent_at_value)

        return cls.parse_mapping(
            {
                "method": _get("request_method") or "GET",
                "url_str": url_val,
                "url": url_struct,
                "headers": headers,
                "tags": _map_as_str_dict(_get("request_tags")),
                "buffer": _get("request_body"),
                "sent_at_timestamp": sent_at_timestamp,
            },
            normalize=normalize,
            prefix="",
        )

    def apply(
        self,
        func: Callable[["PreparedRequest"], "PreparedRequest"],
    ):
        return func(self)