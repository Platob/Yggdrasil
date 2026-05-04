# yggdrasil.io.request
from __future__ import annotations

import base64
import datetime as dt
import json as json_module
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Literal, Mapping, MutableMapping, Optional

import pyarrow as pa

from yggdrasil.data.cast import any_to_datetime
from yggdrasil.data.data_field import field as schema_field
from yggdrasil.data.schema import schema
from yggdrasil.dataclasses.dataclass import get_from_dict
from yggdrasil.io.enums import MediaType, MimeTypes
from .buffer import BytesIO
from .enums import GZIP, Codec, MimeType
from .headers import normalize_headers
from .url import URL

if TYPE_CHECKING:
    from .response import Response
    from .send_config import CacheConfig, SendConfig
    from .session import Session


__all__ = [
    "PreparedRequest",
    "REQUEST_SCHEMA",
    "REQUEST_ARROW_SCHEMA",
    "REQUEST_URL_STRUCT",
]


_REQUEST_SCHEMA_JSON_TAGS: dict[str, str] = {
    "domain": "http",
    "entity": "request",
    "layer": "bronze",
}


# Nested URL struct — parsed components for joins / replay. The full
# string isn't kept here; ``private_url_hash`` covers exact identity
# and ``URL.from_(struct)`` reassembles the URL from its parts.
REQUEST_URL_STRUCT = pa.struct([
    pa.field("scheme",   pa.string(), nullable=False),
    pa.field("userinfo", pa.string(), nullable=True),
    pa.field("host",     pa.string(), nullable=False),
    pa.field("port",     pa.int32(),  nullable=True),
    pa.field("path",     pa.string(), nullable=False),
    pa.field("query",    pa.string(), nullable=True),
    pa.field("fragment", pa.string(), nullable=True),
])


REQUEST_SCHEMA = schema(
    fields=[],
    metadata={
        "comment": "Prepared request flattened into deterministic columns for logging and replay.",
        "time_column": "sent_at",
        # Schema-level identity / partitioning hints — ``autotag`` at
        # the bottom of this block propagates them to the matching
        # children. ``public_hash`` is the primary key (stable across
        # ``anonymize='remove'``); ``partition_key`` is the
        # day-bucket partition column derived from ``sent_at``.
        "primary_key": ["public_hash"],
        "partition_by": ["partition_key"],
    },
    tags=_REQUEST_SCHEMA_JSON_TAGS,
)

REQUEST_SCHEMA["private_hash"] = schema_field(
    "private_hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest over (method, url, headers, body) including sensitive bits "
                   "(URL userinfo, Authorization / API-key headers). Local identity only — "
                   "use ``public_hash`` for cross-system joins / cache lookups.",
        "algorithm": "xxh3_64",
    },
).autotag()

REQUEST_SCHEMA["public_hash"] = schema_field(
    "public_hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest over the anonymize='remove' projection of "
                   "(method, url, headers, body). Stable across cache anonymization, so this "
                   "is the right key for dedup / cross-system identity.",
        "algorithm": "xxh3_64",
    },
).autotag()

REQUEST_SCHEMA["method"] = schema_field(
    "method",
    pa.string(),
    nullable=False,
    metadata={"comment": "HTTP method (GET, POST, etc.)"},
).autotag()

REQUEST_SCHEMA["url"] = schema_field(
    "url",
    REQUEST_URL_STRUCT,
    nullable=False,
    metadata={"comment": "Parsed URL components with full string for replay"},
).autotag()

REQUEST_SCHEMA["private_url_hash"] = schema_field(
    "private_url_hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest of the full URL string (scheme, userinfo, host, port, "
                   "path, query, fragment) — matches the URL exactly as captured.",
        "algorithm": "xxh3_64",
    },
).autotag()

REQUEST_SCHEMA["public_url_hash"] = schema_field(
    "public_url_hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest of ``url.anonymize('remove').to_string()`` — drops "
                   "userinfo and sensitive query params so this hash is stable across "
                   "anonymization. The default cache match key.",
        "algorithm": "xxh3_64",
    },
).autotag()

REQUEST_SCHEMA["headers"] = schema_field(
    "headers",
    pa.map_(pa.string(), pa.string()),
    nullable=False,
    metadata={"comment": "All request headers as a name→value map"},
).autotag()

REQUEST_SCHEMA["tags"] = schema_field(
    "tags",
    pa.map_(pa.string(), pa.string()),
    nullable=False,
    metadata={"comment": "Request tags merged with URL query params; explicit tags win on conflict"},
).autotag()

REQUEST_SCHEMA["body"] = schema_field(
    "body",
    pa.large_binary(),
    nullable=True,
    metadata={"comment": "Raw request body bytes"},
).autotag()

REQUEST_SCHEMA["body_size"] = schema_field(
    "body_size",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "Length of body in bytes; 0 when body is absent",
        "unit": "bytes",
    },
).autotag()

REQUEST_SCHEMA["body_hash"] = schema_field(
    "body_hash",
    pa.int64(),
    nullable=True,
    metadata={
        "comment": "xxh3_64 digest of body bytes; null when body is absent",
        "algorithm": "xxh3_64",
    },
).autotag()

REQUEST_SCHEMA["sent_at"] = schema_field(
    "sent_at",
    pa.timestamp("us", "UTC"),
    nullable=False,
    metadata={"comment": "UTC timestamp when request was dispatched"},
).autotag()

REQUEST_SCHEMA["partition_key"] = schema_field(
    "partition_key",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest of ``f'{url.host}{url.path}'`` — endpoint-bucket "
                   "partition column, so all requests against the same host+path land "
                   "in the same leaf and a per-endpoint cache lookup scans one partition.",
        "algorithm": "xxh3_64",
    },
).autotag()

REQUEST_SCHEMA["_pkl"] = schema_field(
    "_pkl",
    pa.large_binary(),
    nullable=True,
    metadata={
        "comment": "Placeholder for a full ``PreparedRequest`` pickle blob — populated "
                   "by the pickle serializer for lossless round-trips, left null on the "
                   "deterministic-columns-only path.",
    },
).autotag()

# Propagate schema-level ``primary_key`` / ``partition_by`` down to
# the matching children (consumes those metadata keys in place).
REQUEST_SCHEMA = REQUEST_SCHEMA.autotag()

REQUEST_ARROW_SCHEMA: pa.Schema = REQUEST_SCHEMA.to_arrow_schema()


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


def _xxh3_int64_str(text: str) -> int:
    """Signed-int64 xxh3_64 digest of *text*.

    Returns 0 on missing optional ``xxhash`` so the schema's nullable
    semantics still hold without forcing the dependency.
    """
    if not text:
        return 0
    try:
        import xxhash
    except ImportError:
        return 0
    u = xxhash.xxh3_64(text.encode("utf-8")).intdigest()
    return u if u < 2**63 else u - 2**64


def _build_url_struct(url: URL) -> dict[str, Any]:
    """Build the URL struct value (matches REQUEST_URL_STRUCT)."""
    return {
        "scheme":   url.scheme or "",
        "userinfo": url.userinfo,
        "host":     url.host or "",
        "port":     url.port if url.port not in (None, 0) else None,
        "path":     url.path or "",
        "query":    url.query,
        "fragment": url.fragment,
    }


class PreparedRequest:
    """Immutable-ish request descriptor — fields are normalized in __init__.

    ``_session`` is a transient back-reference used by
    :meth:`attach_session`; it's deliberately excluded from
    :meth:`__getstate__` so a request stays portable when pickled to
    Spark executors and re-binds via ``attach_session`` once it lands
    on the worker. Per-request request/response hooks live on the
    owning :class:`Session` (``_prepare_request`` / ``_prepare_response``).
    """

    def __init__(
        self,
        method: str,
        url: URL,
        headers: MutableMapping[str, str],
        tags: MutableMapping[str, str],
        buffer: Optional[BytesIO],
        sent_at: Optional[dt.datetime],
        local_cache_config: Optional["CacheConfig"] = None,
        remote_cache_config: Optional["CacheConfig"] = None,
    ) -> None:
        self.method = method or "GET"
        self.url = URL.from_(url)
        self.headers = _string_dict(headers)
        self.tags = _string_dict(tags)
        self.sent_at = (
            any_to_datetime(sent_at) if sent_at
            else dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)
        )
        self.buffer = (
            BytesIO.from_(buffer)
            if buffer is not None and not isinstance(buffer, BytesIO)
            else buffer
        )
        self.local_cache_config = local_cache_config
        self.remote_cache_config = remote_cache_config
        self._session: "Session | None" = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self.method} {self.url.to_string()!r}>"

    def __str__(self) -> str:
        return self.url.to_string()

    def __getstate__(self) -> dict[str, Any]:
        # Drop the transient session back-reference so requests stay
        # cleanly picklable for cross-executor transport. Re-bind on the
        # worker via :meth:`attach_session`.
        state = self.__dict__.copy()
        state.pop("_session", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._session = None

    # ------------------------------------------------------------------
    # Session attachment — used by Spark broadcast on the worker side
    # ------------------------------------------------------------------

    def attach_session(self, session: "Session") -> "PreparedRequest":
        self._session = session
        return self

    def detach_session(self) -> "PreparedRequest":
        self._session = None
        return self

    @property
    def session(self) -> "Session | None":
        return self._session

    # ------------------------------------------------------------------
    # Sending — delegates to the attached session
    # ------------------------------------------------------------------

    def send(
        self,
        config: "SendConfig | Mapping[str, Any] | None" = None,
        **kwargs: Any,
    ) -> "Response":
        return self._send(config, **kwargs)

    def _send(
        self,
        config: "SendConfig | Mapping[str, Any] | None" = None,
        **kwargs: Any,
    ) -> "Response":
        if self._session is None:
            raise RuntimeError(
                f"{type(self).__name__}.send requires an attached session — "
                "call request.attach_session(session) first, or use "
                "session.send(request) directly."
            )
        return self._session.send(self, config, **kwargs)

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        normalize: bool = True,
    ) -> "PreparedRequest":
        if isinstance(obj, str):
            obj = {
                "url": URL.from_str(obj, normalize=normalize)
            }

        if isinstance(obj, Mapping):
            return cls.from_mapping(obj, normalize=normalize)

        raise ValueError(f"Cannot make {cls.__name__} from {type(obj)}")

    @classmethod
    def from_mapping(
        cls,
        obj: Mapping[str, Any],
        *,
        normalize: bool = True,
    ) -> "PreparedRequest":
        method = cls._parse_method(obj)
        url = cls._parse_url(obj, normalize=normalize)
        headers = cls._parse_headers(obj)
        tags = cls._parse_tags(obj)
        buffer = cls._parse_buffer(obj)
        sent_at = cls._parse_sent_at_timestamp(obj)

        if cls is PreparedRequest:
            if url.is_http:
                from .http_ import HTTPRequest

                return HTTPRequest(
                    method=method,
                    url=url,
                    headers=headers,
                    tags=tags,
                    buffer=buffer,
                    sent_at=sent_at,
                )

        return cls(
            method=method,
            url=url,
            headers=headers,
            tags=tags,
            buffer=buffer,
            sent_at=sent_at,
        )

    @staticmethod
    def _parse_method(obj: Mapping[str, Any]) -> str:
        method = get_from_dict(obj, keys=("method",), prefix=None)
        return "GET" if method is MISSING or method in (None, "") else str(method)

    @classmethod
    def _parse_url(
        cls,
        obj: Mapping[str, Any],
        *,
        normalize: bool,
    ) -> URL:
        # Accept either a flat string ("url"/"href"/"uri"), a struct
        # dict ({"scheme": ..., "host": ..., "path": ...}), or a
        # pre-built URL instance.
        url_value = get_from_dict(obj, keys=("url", "href", "uri"), prefix=None)

        if isinstance(url_value, URL):
            return URL.from_(url_value, normalize=normalize)

        if isinstance(url_value, str) and url_value:
            return URL.from_(url_value, normalize=normalize)

        if isinstance(url_value, Mapping):
            return URL.from_(
                {
                    "scheme":   url_value.get("scheme") or "",
                    "userinfo": url_value.get("userinfo") or "",
                    "host":     url_value.get("host") or "",
                    "port":     url_value.get("port") or 0,
                    "path":     url_value.get("path") or "",
                    "query":    url_value.get("query") or "",
                    "fragment": url_value.get("fragment") or "",
                },
                normalize=normalize,
            )

        raise ValueError(
            "PreparedRequest.from_mapping: missing url — pass a URL, a string, "
            "or a {scheme/host/...} struct."
        )

    @staticmethod
    def _parse_headers(obj: Mapping[str, Any]) -> MutableMapping[str, str]:
        headers = get_from_dict(obj, keys=("headers",), prefix=None)
        return _map_as_str_dict(headers)

    @staticmethod
    def _parse_tags(obj: Mapping[str, Any]) -> dict[str, str]:
        tags = get_from_dict(obj, keys=("tags",), prefix=None)
        return _string_dict(tags if isinstance(tags, Mapping) else None)

    @staticmethod
    def _parse_buffer(obj: Mapping[str, Any]) -> Optional[BytesIO]:
        buffer = get_from_dict(obj, keys=("buffer", "body", "content", "data"), prefix=None)
        if buffer is MISSING or buffer is None:
            return None
        return BytesIO.from_(buffer)

    @staticmethod
    def _parse_sent_at_timestamp(obj: Mapping[str, Any]) -> dt.datetime:
        value = get_from_dict(obj, keys=("sent_at_timestamp", "sent_at"), prefix=None)
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
        local_cache_config: Optional[CacheConfig] = None,
        remote_cache_config: Optional[CacheConfig] = None,
        *,
        json: Optional[Any] = None,
        normalize: bool = True,
        compress_threshold: Optional[int] = 4 * 1024 * 1024,
        compress_codec: Optional[Codec] = GZIP,
    ) -> "PreparedRequest":
        parsed_url = URL.from_(url, normalize=normalize)
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

        return out_class(
            method=str(method),
            url=parsed_url,
            headers=normalize_headers(out_headers, is_request=True, body=request_body) if normalize else out_headers,
            tags=_string_dict(tags),
            buffer=request_body,
            sent_at=0,
            local_cache_config=local_cache_config,
            remote_cache_config=remote_cache_config,
        )

    def copy(
        self,
        *,
        method: str | None = None,
        url: URL | str | None = None,
        headers: Optional[Mapping[str, str]] = None,
        buffer: Optional[BytesIO] = ...,
        tags: Optional[Mapping[str, str]] = None,
        sent_at: int | None = None,
        local_cache_config: Optional["CacheConfig"] = ...,
        remote_cache_config: Optional["CacheConfig"] = ...,
        normalize: bool = True,
        copy_buffer: bool = False,
    ) -> "PreparedRequest":
        new_url = self.url if url is None else URL.from_(url, normalize=normalize)
        new_headers = dict(self.headers) if headers is None else _string_dict(headers)

        if buffer is ...:
            new_buffer = self.buffer
            if copy_buffer and new_buffer is not None:
                new_buffer = BytesIO.from_(new_buffer.to_bytes())
        else:
            new_buffer = buffer

        new_tags = dict(self.tags) if tags is None else _string_dict(tags)

        return self.__class__(
            method=self.method if method is None else str(method),
            url=new_url,
            headers=new_headers,
            tags=new_tags,
            buffer=new_buffer,
            sent_at=self.sent_at if sent_at is None else any_to_datetime(sent_at),
            local_cache_config=self.local_cache_config if local_cache_config is ... else local_cache_config,
            remote_cache_config=self.remote_cache_config if remote_cache_config is ... else remote_cache_config,
        )

    def prepare_to_send(
        self,
        sent_at: dt.datetime | dt.date | str | int | None = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> "PreparedRequest":
        if self.headers is None:
            self.headers = {}

        if headers:
            self.headers.update(_string_dict(headers))

        self.sent_at = dt.datetime.now(dt.timezone.utc) if sent_at is None else any_to_datetime(sent_at)

        return self

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

        accept = MimeType.from_(self.headers.get("Accept"), default=MimeTypes.OCTET_STREAM)
        codec = Codec.from_(self.headers.get("Accept-Encoding"), default=None)
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
    def partition_key(self) -> int:
        """xxh3_64 of ``f'{url.host}{url.path}'`` — endpoint partition key."""
        return _xxh3_int64_str(f"{self.url.host or ''}{self.url.path or ''}")

    @property
    def body_size(self) -> int:
        return self.buffer.size if self.buffer is not None else 0

    @property
    def body_hash(self) -> Optional[int]:
        if self.buffer is None:
            return None
        try:
            return self.buffer.xxh3_int64()
        except ImportError:
            return None

    @property
    def private_url_hash(self) -> int:
        """xxh3_64 of the URL exactly as captured (userinfo + full query)."""
        return _xxh3_int64_str(self.url.to_string())

    @property
    def public_url_hash(self) -> int:
        """xxh3_64 of the URL after ``anonymize('remove')`` — userinfo and
        sensitive query params dropped, so this hash is stable across the
        cache's anonymize step."""
        return _xxh3_int64_str(self.url.anonymize(mode="remove").to_string())

    @property
    def private_hash(self) -> int:
        """xxh3_64 over (method, url, headers, body) including sensitive
        bits — the local-fidelity identity for replay."""
        return self._compute_identity_hash(anonymize=False)

    @property
    def public_hash(self) -> int:
        """xxh3_64 over the anonymize='remove' projection of
        (method, url, headers, body) — the cross-system identity that
        survives cache anonymization."""
        return self._compute_identity_hash(anonymize=True)

    def _compute_identity_hash(self, *, anonymize: bool = False) -> int:
        try:
            import xxhash
        except ImportError:
            return 0

        if anonymize:
            url_str = self.url.anonymize(mode="remove").to_string()
            headers = normalize_headers(
                self.headers or {},
                is_request=True,
                add_missing=False,
                anonymize=True,
                mode="remove",
            )
        else:
            url_str = self.url.to_string()
            headers = self.headers or {}

        h = xxhash.xxh3_64()
        h.update(self.method.encode("utf-8"))
        h.update(b"\x00")
        h.update(url_str.encode("utf-8"))
        h.update(b"\x00")
        for k in sorted(headers):
            h.update(str(k).encode("utf-8"))
            h.update(b"=")
            h.update(str(headers[k]).encode("utf-8"))
            h.update(b"\x00")
        if self.buffer is not None:
            h.update(self.buffer.xxh3_64().digest())
        u = h.intdigest()
        return u if u < 2**63 else u - 2**64

    @property
    def arrow_values(self) -> dict[str, Any]:
        tags_v = dict(self.url.query_items())
        if self.tags:
            tags_v.update(_string_dict(self.tags))

        return {
            "private_hash":     self.private_hash,
            "public_hash":      self.public_hash,
            "method":           self.method,
            "url":              _build_url_struct(self.url),
            "private_url_hash": self.private_url_hash,
            "public_url_hash":  self.public_url_hash,
            "headers":          _string_dict(self.headers),
            "tags":             tags_v,
            "body":             self.buffer.to_bytes() if self.buffer is not None else None,
            "body_size":        self.body_size,
            "body_hash":        self.body_hash,
            "sent_at":          self.sent_at,
            "partition_key":    self.partition_key,
            # ``_pkl`` is a placeholder column populated externally by
            # the pickle serializer; the deterministic projection path
            # leaves it null so writers don't pay for a pickle dump
            # when the structured columns are enough.
            "_pkl":             None,
        }

    def match_value(self, key: str) -> Any:
        # Support dotted paths (``url.path``, ``headers.content_type``)
        # alongside the flat top-level field names.
        if "." in key:
            head, _, tail = key.partition(".")
            values = self.arrow_values
            if head in values:
                container = values[head]
                if isinstance(container, Mapping):
                    if tail in container:
                        return container[tail]
                    if "." in tail:
                        # Nested deeper; recurse-by-mapping.
                        sub_head, _, sub_tail = tail.partition(".")
                        nested = container.get(sub_head)
                        if isinstance(nested, Mapping) and sub_tail in nested:
                            return nested[sub_tail]
                raise ValueError(
                    f"Unsupported request match key: {key!r}. "
                    f"Must be within: {REQUEST_ARROW_SCHEMA.names!r}"
                )

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

        return self.copy(
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
    def from_record(
        cls,
        record: "Mapping[str, Any]",
        *,
        normalize: bool = True,
    ) -> "PreparedRequest":
        """Build a :class:`PreparedRequest` from a row-shaped mapping."""
        return cls._from_get(record.get, normalize=normalize)

    @classmethod
    def _from_arrow_cols(
        cls,
        cols: dict[str, Any],
        i: int,
        *,
        normalize: bool = True,
    ) -> "PreparedRequest":
        def _arrow_get(name: str) -> Any:
            if name in cols:
                return cols[name][i].as_py()
            return None

        return cls._from_get(_arrow_get, normalize=normalize)

    @classmethod
    def _from_get(
        cls,
        get: "Callable[[str], Any]",
        *,
        normalize: bool = True,
    ) -> "PreparedRequest":
        # Single source of truth for "named-getter → PreparedRequest" —
        # used by both the Arrow-batch path and the Mapping path.
        url_value = get("url")
        headers_value = get("headers")
        sent_at_value = get("sent_at")

        return cls.from_mapping(
            {
                "method":   get("method") or "GET",
                "url":      url_value,
                "headers":  _map_as_str_dict(headers_value),
                "tags":     _map_as_str_dict(get("tags")),
                "buffer":   get("body"),
                "sent_at":  any_to_datetime(sent_at_value) if sent_at_value not in (None, "") else None,
            },
            normalize=normalize,
        )

    def apply(
        self,
        func: Callable[["PreparedRequest"], "PreparedRequest"],
    ):
        return func(self)
