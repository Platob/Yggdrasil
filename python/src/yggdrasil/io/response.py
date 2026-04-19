# yggdrasil.io.response
"""HTTP response model with Arrow, Polars, pandas, and ASGI serialisation."""
from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import MISSING, dataclass, replace, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Literal, Mapping, MutableMapping, Optional

import pyarrow as pa

import yggdrasil.pickle.json as json_module
from yggdrasil.data import any_to_datetime
from yggdrasil.data.data_field import field as schema_field
from yggdrasil.data.schema import schema
from yggdrasil.dataclasses.dataclass import get_from_dict
from .buffer import BytesIO
from .enums import Codec, MediaType, MimeTypes
from .headers import DEFAULT_HOSTNAME, PromotedHeaders, normalize_headers
from .request import PreparedRequest, REQUEST_SCHEMA

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from fastapi import Response as FastAPIResponse
    from pyspark.sql import DataFrame as SparkDataFrame, Row as SparkRow
    from starlette.responses import Response as StarletteResponse


__all__ = [
    "Response",
    "BASE_SCHEMA",
    "RESPONSE_SCHEMA",
    "RESPONSE_ARROW_SCHEMA",
]


# ---------------------------------------------------------------------------
# Private header / body helpers
# ---------------------------------------------------------------------------

def _get_header(headers: Mapping[str, str] | None, name: str) -> str | None:
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
    value = _get_header(headers, "Content-Type")
    if not value:
        return None
    bare = str(value).split(";", 1)[0].strip().lower()
    return bare or None


def _parse_content_encoding(headers: Mapping[str, str] | None) -> str | None:
    value = _get_header(headers, "Content-Encoding")
    if not value:
        return None
    parts = [p.strip().lower() for p in str(value).split(",") if p.strip()]
    return ",".join(parts) or None


def _is_probably_placeholder_content_type(value: str | None) -> bool:
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
    content_codec = Codec.parse(content_encoding, default=None)

    if content_type and not _is_probably_placeholder_content_type(content_type):
        try:
            return MediaType.parse(content_type, codec=content_codec)
        except Exception:
            pass

    try:
        sniffed = body.media_type
    except Exception:
        sniffed = MediaType(MimeTypes.OCTET_STREAM, content_codec)

    return MediaType(sniffed.mime_type, codec=content_codec) if content_encoding else sniffed


def _ensure_media_headers(
    headers: MutableMapping[str, str],
    body: BytesIO,
) -> MediaType:
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

    headers["Content-Length"] = str(body.size)
    return media


def _string_dict(arg: Optional[Mapping[Any, Any]]) -> dict[str, str]:
    if not arg:
        return {}
    return {str(k): str(v) for k, v in arg.items()}


def _map_to_str_dict(value: Any) -> dict[str, str]:
    if not value:
        return {}
    if isinstance(value, Mapping):
        return {str(k): str(v) for k, v in value.items() if k is not None and v is not None}
    try:
        return {str(k): str(v) for k, v in value if k is not None and v is not None}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_headers(obj: Mapping[str, Any], *, prefix: str) -> MutableMapping[str, str]:
    headers = get_from_dict(obj, keys=("headers", "header", "hdrs", "response_headers"), prefix=prefix)
    if headers is MISSING:
        headers = get_from_dict(obj, keys=("headers", "header", "hdrs", "response_headers"), prefix="")

    if not isinstance(headers, Mapping):
        headers = {} if headers is MISSING else dict(headers)

    headers = _string_dict(headers)

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
    }

    for k, v in dumped_headers.items():
        if v is not MISSING and v not in (None, ""):
            headers[k] = v

    return headers


def _parse_tags(obj: Mapping[str, Any], *, prefix: str) -> dict[str, str]:
    tags = get_from_dict(obj, keys=("tags", "response_tags"), prefix=prefix)
    if tags is MISSING:
        tags = get_from_dict(obj, keys=("tags", "response_tags"), prefix="")
    return _string_dict(tags if isinstance(tags, Mapping) else None)


def _parse_buffer(obj: Mapping[str, Any], *, prefix: str) -> BytesIO:
    body = get_from_dict(obj, keys=("buffer", "body", "content", "data", "response_body"), prefix=prefix)
    if body is MISSING:
        body = get_from_dict(obj, keys=("buffer", "body", "content", "data", "response_body"), prefix="")
    if body is MISSING or body is None:
        return BytesIO()
    return BytesIO.parse(obj=body)


def _parse_status_code(obj: Mapping[str, Any], *, prefix: str) -> int:
    status = get_from_dict(obj, keys=("status_code", "status", "code"), prefix=prefix)
    if status is MISSING:
        status = get_from_dict(obj, keys=("status_code", "status", "code"), prefix="")
    if status is MISSING or status in (None, ""):
        raise ValueError("Response.parse_mapping: missing status_code/status/code")
    return int(status) if isinstance(status, int) else int(float(str(status).strip()))


def _parse_received_at(obj: Mapping[str, Any], *, prefix: str) -> dt.datetime:
    keys = (
        "received_at_timestamp",
        "received_at",
        "response_received_at",
    )
    value = get_from_dict(obj, keys=keys, prefix=prefix)
    if value is MISSING:
        value = get_from_dict(obj, keys=keys, prefix="")

    if value is MISSING:
        return dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)
    return any_to_datetime(value)


# ---------------------------------------------------------------------------
# Hop-by-hop header names
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

_RESPONSE_BASE_SCHEMA_JSON_TAGS: dict[str, str] = {
    "domain": "http",
    "entity": "response",
    "layer": "bronze",
    "namespace": "yggdrasil.io.response",
}


BASE_SCHEMA = schema(
    fields=[],
    metadata={
        "comment": "Response record (single row), designed for deterministic logging and replay.",
        "time_column": "response_received_at",
    },
    tags=_RESPONSE_BASE_SCHEMA_JSON_TAGS,
)

BASE_SCHEMA["response_status_code"] = schema_field(
    "response_status_code",
    pa.int32(),
    nullable=False,
    metadata={
        "comment": "HTTP status code returned by the server",
    },
    tags={
        "entity": "response",
        "group": "status",
    },
)

BASE_SCHEMA["response_host"] = schema_field(
    "response_host",
    pa.string(),
    nullable=True,
    metadata={
        "comment": "HTTP Host header",
    },
    tags={
        "entity": "response",
        "group": "headers",
    },
)

BASE_SCHEMA["response_user_agent"] = schema_field(
    "response_user_agent",
    pa.string(),
    nullable=True,
    metadata={
        "comment": "HTTP User-Agent header",
    },
    tags={
        "entity": "response",
        "group": "headers",
    },
)

BASE_SCHEMA["response_accept"] = schema_field(
    "response_accept",
    pa.string(),
    nullable=True,
    metadata={
        "comment": "HTTP Accept header",
    },
    tags={
        "entity": "response",
        "group": "headers",
    },
)

BASE_SCHEMA["response_accept_encoding"] = schema_field(
    "response_accept_encoding",
    pa.string(),
    nullable=True,
    metadata={
        "comment": "HTTP Accept-Encoding header",
    },
    tags={
        "entity": "response",
        "group": "headers",
    },
)

BASE_SCHEMA["response_accept_language"] = schema_field(
    "response_accept_language",
    pa.string(),
    nullable=True,
    metadata={
        "comment": "HTTP Accept-Language header",
    },
    tags={
        "entity": "response",
        "group": "headers",
    },
)

BASE_SCHEMA["response_content_type"] = schema_field(
    "response_content_type",
    pa.string(),
    nullable=True,
    metadata={
        "comment": "HTTP Content-Type header",
    },
    tags={
        "entity": "response",
        "group": "headers",
    },
)

BASE_SCHEMA["response_content_length"] = schema_field(
    "response_content_length",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "HTTP Content-Length header parsed as integer",
    },
    tags={
        "entity": "response",
        "group": "headers",
    },
)

BASE_SCHEMA["response_content_encoding"] = schema_field(
    "response_content_encoding",
    pa.string(),
    nullable=True,
    metadata={
        "comment": "HTTP Content-Encoding header",
    },
    tags={
        "entity": "response",
        "group": "headers",
    },
)

BASE_SCHEMA["response_transfer_encoding"] = schema_field(
    "response_transfer_encoding",
    pa.string(),
    nullable=True,
    metadata={
        "comment": "HTTP Transfer-Encoding header",
    },
    tags={
        "entity": "response",
        "group": "headers",
    },
)

BASE_SCHEMA["response_headers"] = schema_field(
    "response_headers",
    pa.map_(pa.string(), pa.string()),
    nullable=False,
    metadata={
        "comment": "Response headers excluding promoted common headers",
        "keys_sorted": "false",
    },
    tags={
        "entity": "response",
        "group": "headers",
    },
)

BASE_SCHEMA["response_tags"] = schema_field(
    "response_tags",
    pa.map_(pa.string(), pa.string()),
    nullable=False,
    metadata={
        "comment": "Arbitrary string tags attached to this response",
    },
    tags={
        "entity": "response",
        "group": "tags",
    },
)

BASE_SCHEMA["response_body"] = schema_field(
    "response_body",
    pa.binary(),
    nullable=True,
    metadata={
        "comment": "Raw binary payload of the response",
    },
    tags={
        "entity": "response",
        "group": "payload",
    },
)

BASE_SCHEMA["response_body_hash"] = schema_field(
    "response_body_hash",
    pa.int64(),
    nullable=True,
    metadata={
        "comment": "Signed Int64 XXH3 digest of response_body",
        "algorithm": "xxh3_64",
    },
    tags={
        "entity": "response",
        "group": "payload",
        "algorithm": "xxh3_64",
    },
)

BASE_SCHEMA["response_received_at"] = schema_field(
    "response_received_at",
    pa.timestamp("us", "UTC"),
    nullable=False,
    metadata={
        "comment": "UTC timestamp when the response was captured",
        "unit": "us",
        "tz": "UTC",
    },
    tags={
        "entity": "response",
        "group": "timing",
    },
)

RESPONSE_SCHEMA = REQUEST_SCHEMA + BASE_SCHEMA
RESPONSE_ARROW_SCHEMA = RESPONSE_SCHEMA.to_arrow_schema()

_PROMOTED_RESPONSE_HEADER_FIELDS: tuple[tuple[str, str], ...] = (
    ("Host", "response_host"),
    ("User-Agent", "response_user_agent"),
    ("Accept", "response_accept"),
    ("Accept-Encoding", "response_accept_encoding"),
    ("Accept-Language", "response_accept_language"),
    ("Content-Type", "response_content_type"),
    ("Content-Length", "response_content_length"),
    ("Content-Encoding", "response_content_encoding"),
    ("Transfer-Encoding", "response_transfer_encoding"),
)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Response:
    request: PreparedRequest
    status_code: int
    headers: MutableMapping[str, str] = field(compare=False, hash=False, repr=False)
    tags: MutableMapping[str, str] = field(compare=False, hash=False, repr=False)
    buffer: BytesIO = field(compare=False, hash=False, repr=False)
    received_at: dt.datetime = field(compare=False, hash=False, repr=False)

    _id_cache: int | None = field(default=None, init=False, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        self.status_code = int(self.status_code)
        self.headers = _string_dict(self.headers)
        self.tags = _string_dict(self.tags)
        self.received_at = any_to_datetime(self.received_at)

        if not isinstance(self.buffer, BytesIO):
            self.buffer = BytesIO(self.buffer, copy=False)

        _ensure_media_headers(self.headers, self.buffer)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<r={self.request} s={self.status_code} b={self.body!r}>"

    def __str__(self) -> str:
        return self.__repr__()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, obj: Any, *, normalize: bool = True) -> "Response":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, str):
            return cls.parse_str(obj, normalize=normalize)
        if isinstance(obj, Mapping):
            return cls.parse_mapping(obj, normalize=normalize)
        return cls.parse_str(str(obj), normalize=normalize)

    @classmethod
    def parse_str(cls, raw: str, *, normalize: bool = True) -> "Response":
        s = raw.strip()
        if not s:
            raise ValueError("Response.parse_str: empty string")
        try:
            d = json_module.loads(s)
        except Exception as exc:
            raise ValueError("Response.parse_str: expected JSON object string") from exc
        if not isinstance(d, Mapping):
            raise ValueError("Response.parse_str: JSON must decode to a mapping")
        return cls.parse_mapping(d, normalize=normalize)

    @classmethod
    def parse_mapping(
        cls,
        obj: Mapping[str, Any],
        *,
        normalize: bool = True,
        prefix: str = "response_",
    ) -> "Response":
        if not obj:
            raise ValueError("Response.parse_mapping: empty mapping")

        req_obj = get_from_dict(obj, keys=("request",), prefix="")
        request = PreparedRequest.parse(
            obj if req_obj is MISSING or req_obj in (None, "") else req_obj,
            normalize=normalize,
        )

        status_code = _parse_status_code(obj, prefix=prefix)
        headers = _parse_headers(obj, prefix=prefix)
        buffer = _parse_buffer(obj, prefix=prefix)
        received_at = _parse_received_at(obj, prefix=prefix)
        tags = _parse_tags(obj, prefix=prefix)

        if normalize:
            headers = normalize_headers(headers, body=buffer, is_request=False)

        _ensure_media_headers(headers, buffer)

        if cls is Response:
            if request.url.is_http:
                from .http_ import HTTPResponse

                return HTTPResponse(
                    request=request,
                    status_code=status_code,
                    headers=headers,
                    tags=tags,
                    buffer=buffer,
                    received_at=received_at,
                )

        return cls(
            request=request,
            status_code=status_code,
            headers=headers,
            tags=tags,
            buffer=buffer,
            received_at=received_at,
        )

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def update_headers(
        self,
        headers: MutableMapping[str, str],
        normalize: bool = True,
    ) -> "Response":
        if not headers:
            return self

        if not self.headers:
            self.headers = _string_dict(headers)
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
        if not tags:
            return self

        if not self.tags:
            self.tags = _string_dict(tags)
        else:
            self.tags.update(_string_dict(tags))

        return self

    # ------------------------------------------------------------------
    # Media type
    # ------------------------------------------------------------------

    @property
    def media_type(self) -> MediaType:
        if self.headers is None:
            self.headers = {}
        return _ensure_media_headers(self.headers, self.buffer)

    @media_type.setter
    def media_type(self, value: MediaType) -> None:
        self.set_media_type(value, safe=True)

    def set_media_type(
        self,
        value: MediaType,
        *,
        safe: bool = True,
    ) -> "Response":
        if self.headers is None:
            self.headers = {}

        self.request.accept_media_type = value
        self.buffer.set_media_type(value, safe=safe)
        self.headers["Content-Type"] = value.mime_type.value

        if value.codec is not None:
            self.headers["Content-Encoding"] = value.codec.name
        else:
            self.headers.pop("Content-Encoding", None)

        self.headers["Content-Length"] = str(self.buffer.size)
        return self

    @property
    def content_disposition(self) -> str | None:
        return self.headers.get("Content-Disposition")

    @property
    def filename(self) -> str | None:
        cd = self.content_disposition

        if not cd:
            return None

        return cd.split("filename=")[-1].split(";")[0]

    # ------------------------------------------------------------------
    # Body accessors
    # ------------------------------------------------------------------

    @property
    def body(self) -> BytesIO:
        return self.buffer

    @property
    def codec(self) -> Optional[Codec]:
        return self.media_type.codec

    @property
    def content(self) -> bytes:
        codec = self.codec
        if codec is not None:
            with self.buffer.decompress(codec=codec, copy=True) as b:
                return b.to_bytes()
        return self.buffer.to_bytes()

    @property
    def text(self) -> str:
        return self.content.decode(_get_charset(self.headers), errors="replace")

    def json(
        self,
        orient: Optional[Literal["records", "split", "index", "columns", "values"]] = None,
        *,
        media_type: Optional[MediaType] = None,
    ) -> Any:
        return self.buffer.json_load(
            orient=orient,
            media_type=media_type or self.media_type,
        )

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400

    def raise_for_status(self) -> None:
        if not self.ok:
            raise self.error()

    def warn_for_status(self) -> None:
        err = self.error()
        if err is not None:
            warnings.warn(str(err), category=RuntimeWarning, stacklevel=2)

    def error(self) -> Optional[Exception]:
        if not self.ok:
            from .errors import make_for_status
            return make_for_status(self)
        return None

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    @property
    def received_at_timestamp(self) -> int:
        return int(self.received_at.timestamp() * 1000000)

    # ------------------------------------------------------------------
    # Matching / projection
    # ------------------------------------------------------------------

    @property
    def arrow_values(self) -> dict[str, Any]:
        promoted = PromotedHeaders.extract(self.headers or {}, host=DEFAULT_HOSTNAME)

        if self.buffer is not None:
            body_bytes = self.buffer.to_bytes()
            body_hash = self.buffer.xxh3_int64()
        else:
            body_bytes = None
            body_hash = None

        return {
            **self.request.arrow_values,
            "response_status_code": self.status_code,
            "response_host": promoted.host or DEFAULT_HOSTNAME,
            "response_user_agent": promoted.user_agent,
            "response_accept": promoted.accept,
            "response_accept_encoding": promoted.accept_encoding,
            "response_accept_language": promoted.accept_language,
            "response_content_type": promoted.content_type,
            "response_content_length": promoted.content_length or 0,
            "response_content_encoding": promoted.content_encoding,
            "response_transfer_encoding": promoted.transfer_encoding,
            "response_headers": promoted.remaining,
            "response_tags": _string_dict(self.tags),
            "response_body": body_bytes,
            "response_body_hash": body_hash,
            "response_received_at": self.received_at,
        }

    def match_value(self, key: str) -> Any:
        values = self.arrow_values
        if key in values:
            return values[key]
        if hasattr(self, key):
            return getattr(self, key)
        if key.startswith("request_"):
            return self.request.match_value(key)
        raise ValueError(
            f"Unsupported response match key: {key!r}. "
            f"Must be within: {RESPONSE_ARROW_SCHEMA.names!r}"
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

    # ------------------------------------------------------------------
    # Anonymisation
    # ------------------------------------------------------------------

    def anonymize(self, mode: Literal["remove", "redact"] = "remove") -> "Response":
        if not mode:
            return self

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

    def to_arrow_batches(
        self,
        parse: bool = True,
        *,
        lazy: bool = False,
        **media_options: Any,
    ) -> Iterator[pa.RecordBatch]:
        if parse:
            mio = self.buffer.media_io(media=self.media_type)
            yield from mio.read_arrow_batches(lazy=lazy, **media_options)
            return

        yield self._arrow_batch_from_values()

    def to_arrow_batch(
        self,
        parse: bool = True,
        *,
        lazy: bool = False,
        **media_options: Any,
    ) -> pa.RecordBatch:
        if not parse:
            return self._arrow_batch_from_values()

        batches = list(self.to_arrow_batches(parse=parse, lazy=lazy, **media_options))
        return pa.concat_batches(batches)

    def to_arrow_table(
        self,
        parse: bool = True,
        *,
        lazy: bool = False,
        **media_options: Any,
    ) -> pa.Table:
        if not parse:
            return pa.Table.from_batches([self._arrow_batch_from_values()])

        batches = list(self.to_arrow_batches(parse=parse, lazy=lazy, **media_options))
        return pa.Table.from_batches(batches)

    def _arrow_batch_from_values(self) -> pa.RecordBatch:
        values = self.arrow_values
        arrays = [
            pa.array([values[f.name]], type=f.type)
            for f in RESPONSE_ARROW_SCHEMA
        ]
        return pa.RecordBatch.from_arrays(arrays, schema=RESPONSE_ARROW_SCHEMA)

    # ------------------------------------------------------------------
    # Serialisation — Polars / pandas / Spark
    # ------------------------------------------------------------------

    def read_polars_frames(
        self,
        parse: bool = True,
        *,
        lazy: bool = False,
        **media_options: Any,
    ):
        if parse:
            mio = self.buffer.media_io(media=self.media_type)

            for df in mio.read_polars_frames(lazy=lazy, **media_options):
                yield df
        else:
            from yggdrasil.polars.lib import polars as _pl

            yield _pl.from_arrow(self.to_arrow_batch(parse=False))

    def to_polars(
        self,
        parse: bool = True,
        *,
        lazy: bool = False,
        **media_options: Any,
    ) -> "pl.DataFrame | pl.LazyFrame":
        from yggdrasil.polars.lib import polars as _pl

        if parse:
            mio = self.buffer.media_io(media=self.media_type)
            return mio.read_polars_frame(lazy=lazy, **media_options)

        return _pl.from_arrow(self.to_arrow_batch(parse=False))

    def to_pandas(
        self,
        parse: bool = True,
        *,
        lazy: bool = False,
        **media_options: Any,
    ) -> "pd.DataFrame":
        return self.to_arrow_table(
            parse=parse,
            lazy=lazy,
            **media_options,
        ).to_pandas()

    def to_spark(
        self,
        parse: bool = True,
        *,
        lazy: bool = False,
        **media_options: Any,
    ) -> "SparkDataFrame":
        from yggdrasil.spark.cast import arrow_table_to_spark_dataframe

        return arrow_table_to_spark_dataframe(
            self.to_arrow_table(parse=parse, lazy=lazy, **media_options)
        )

    # ------------------------------------------------------------------
    # Arrow / Spark deserialization
    # ------------------------------------------------------------------

    @classmethod
    def from_spark_frame(
        cls,
        df: "SparkDataFrame",
    ) -> Iterator["Response"]:
        for row in df.toLocalIterator():
            yield cls.from_spark_row(row)

    @classmethod
    def from_spark_row(
        cls,
        row: "SparkRow",
        *,
        normalize: bool = True,
    ) -> "Response":
        def _to_python(value: Any) -> Any:
            if value is None:
                return None
            as_dict = getattr(value, "asDict", None)
            if callable(as_dict):
                return {str(k): _to_python(v) for k, v in value.asDict(recursive=True).items()}
            if isinstance(value, Mapping):
                return {str(k): _to_python(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_to_python(v) for v in value]
            return value

        return cls.parse_mapping(_to_python(row), normalize=normalize)

    @classmethod
    def from_arrow_tabular(
        cls,
        batch: pa.RecordBatch | pa.Table | Iterator[pa.RecordBatch | pa.Table],
        *,
        normalize: bool = False,
    ) -> Iterator["Response"]:
        def _iter_batches(
            obj: pa.RecordBatch | pa.Table | Iterator[pa.RecordBatch | pa.Table]
        ) -> Iterator[pa.RecordBatch]:
            if isinstance(obj, pa.RecordBatch):
                yield obj
            elif isinstance(obj, pa.Table):
                yield from obj.to_batches()
            else:
                for inner in obj:
                    yield from _iter_batches(inner)

        response_cols = [f.name for f in RESPONSE_ARROW_SCHEMA]

        for rb in _iter_batches(batch):
            available = rb.schema.names
            cols = {
                name: rb.column(name)
                for name in response_cols
                if name in available
            }
            for i in range(rb.num_rows):
                yield cls._from_arrow_cols(cols, i, normalize=normalize)

    @classmethod
    def _from_arrow_cols(
        cls,
        cols: dict[str, Any],
        i: int,
        *,
        normalize: bool = False,
    ) -> "Response":
        def _get(name: str) -> Any:
            if name in cols:
                return cols[name][i].as_py()
            return None

        request = PreparedRequest._from_arrow_cols(cols, i, normalize=normalize)

        headers = _map_to_str_dict(_get("response_headers"))
        for header_name, field_name in _PROMOTED_RESPONSE_HEADER_FIELDS:
            value = _get(field_name)
            if value not in (None, ""):
                headers[header_name] = str(value)

        body_bytes = _get("response_body")
        buffer = BytesIO() if body_bytes is None else BytesIO(body_bytes, copy=False)

        if normalize:
            headers = normalize_headers(headers, body=buffer, is_request=False)

        out_class = cls
        if cls is Response and request.url.is_http:
            from .http_ import HTTPResponse
            out_class = HTTPResponse

        return out_class(
            request=request,
            status_code=_get("response_status_code") or 0,
            headers=headers,
            tags=_map_to_str_dict(_get("response_tags")),
            buffer=buffer,
            received_at=_get("response_received_at") or 0,
        )

    # ------------------------------------------------------------------
    # ASGI helpers
    # ------------------------------------------------------------------

    def _to_asgi_payload(self) -> tuple[bytes, dict[str, str], str]:
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
        from starlette.responses import Response as _StarletteResponse

        body, headers, media_type = self._to_asgi_payload()
        return _StarletteResponse(
            content=body,
            status_code=self.status_code,
            headers=headers,
            media_type=media_type,
        )

    def to_fastapi(self) -> "FastAPIResponse":
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
        return func(self)


# ---------------------------------------------------------------------------
# Cast registry — intercept Any->Arrow routes so Response instances (and
# engine-specific subclasses like HTTPResponse) use their own Arrow projection
# instead of falling back to the generic polars path, which doesn't know
# about Response.
#
# We patch the wildcard entries rather than registering (Response, pa.*)
# because dispatch checks `Any -> to_hint` before MRO lookup, so a plain
# subclass registration would never win for an instance of a wildcard target.
# ---------------------------------------------------------------------------

from yggdrasil.arrow import cast as _arrow_cast  # noqa: E402
from yggdrasil.arrow.cast import cast_arrow_tabular  # noqa: E402
from yggdrasil.data.cast.registry import _any_registry  # noqa: E402


_original_any_to_arrow_table = _arrow_cast.any_to_arrow_table
_original_any_to_arrow_record_batch = _arrow_cast.any_to_arrow_record_batch


def _any_to_arrow_table_with_response(obj, options=None):
    if isinstance(obj, Response):
        return cast_arrow_tabular(obj.to_arrow_table(parse=False), options)
    return _original_any_to_arrow_table(obj, options)


def _any_to_arrow_record_batch_with_response(obj, options=None):
    if isinstance(obj, Response):
        return cast_arrow_tabular(obj.to_arrow_batch(parse=False), options)
    return _original_any_to_arrow_record_batch(obj, options)


_any_registry[pa.Table] = _any_to_arrow_table_with_response
_any_registry[pa.RecordBatch] = _any_to_arrow_record_batch_with_response
