# yggdrasil.io.response
from __future__ import annotations

from dataclasses import MISSING, dataclass, replace
from typing import TYPE_CHECKING, Any, Iterator, Mapping, MutableMapping, Callable, Literal, Optional

import pyarrow as pa

import yggdrasil.pickle.json as json_module
from yggdrasil.dataclasses.dataclass import get_from_dict
from .buffer import BytesIO
from .enums import MediaType, Codec, MimeType
from .headers import PromotedHeaders, normalize_headers, DEFAULT_HOSTNAME
from .request import PreparedRequest, REQUEST_ARROW_SCHEMA

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


def _get_header(headers: Mapping[str, str] | None, name: str) -> str | None:
    if not headers:
        return None

    value = headers.get(name)
    if value is not None:
        return str(value)

    target = name.lower()
    for key, value in headers.items():
        if isinstance(key, str):
            if key == name:
                return str(value)
            if key.lower() == target:
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
            if key == name:
                return str(headers.pop(key))
            if key.lower() == target:
                return str(headers.pop(key))
        elif str(key).lower() == target:
            return str(headers.pop(key))

    return None


def _get_charset(headers: Mapping[str, str]) -> str:
    content_type = _get_header(headers, "Content-Type")
    if not content_type:
        return "utf-8"

    parts = [part.strip() for part in str(content_type).split(";")]
    for part in parts[1:]:
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

    parts = [part.strip().lower() for part in str(value).split(",") if part.strip()]
    return ",".join(parts) or None


def _parse_content_length(headers: Mapping[str, str] | None) -> int | None:
    value = _get_header(headers, "Content-Length")
    if value in (None, ""):
        return None

    try:
        return int(str(value).strip())
    except Exception:
        return None


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
        raw = content_type

        try:
            return MediaType.parse_str(raw, codec=content_codec)
        except Exception:
            pass

    try:
        sniffed = body.media_type
    except Exception:
        sniffed = MediaType(MimeType.OCTET_STREAM, content_codec)

    if content_encoding:
        return MediaType(sniffed.mime_type, codec=content_codec)

    return sniffed


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

    if _parse_content_length(headers) is None:
        headers["Content-Length"] = str(body.size)

    return media


def _parse_headers(obj: Mapping[str, Any], *, prefix: str) -> MutableMapping[str, str]:
    headers = get_from_dict(obj, keys=("headers", "header", "hdrs", "response_headers"), prefix=prefix)
    if headers is MISSING:
        headers = get_from_dict(obj, keys=("headers", "header", "hdrs", "response_headers"), prefix="")

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


def _parse_tags(obj: Mapping[str, Any], *, prefix: str) -> dict[str, str]:
    tags = get_from_dict(obj, keys=("tags", "response_tags"), prefix=prefix)
    if tags is MISSING:
        tags = get_from_dict(obj, keys=("tags", "response_tags"), prefix="")

    if not isinstance(tags, Mapping):
        return {}

    return {str(k): str(v) for k, v in tags.items()}


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
        raise ValueError("Response.parse_dict: missing status_code/status/code")

    return int(status) if isinstance(status, int) else int(float(str(status).strip()))


def _parse_received_at_timestamp(obj: Mapping[str, Any], *, prefix: str) -> int:
    value = get_from_dict(
        obj,
        keys=(
            "received_at_timestamp",
            "received_at",
            "time_us",
            "timestamp",
            "time_ns",
            "received_at_epoch",
            "response_received_at_epoch",
            "response_received_at",
        ),
        prefix=prefix,
    )
    if value is MISSING:
        value = get_from_dict(
            obj,
            keys=(
                "received_at_timestamp",
                "received_at",
                "time_us",
                "timestamp",
                "time_ns",
                "received_at_epoch",
                "response_received_at_epoch",
                "response_received_at",
            ),
            prefix="",
        )

    if value is MISSING or value in (None, ""):
        return 0

    return int(value) if isinstance(value, int) else int(float(str(value).strip()))


def _arrow_ts_col_to_us(col: pa.ChunkedArray | pa.Array, i: int) -> int:
    scalar = col[i]
    if scalar is None or not scalar.is_valid:
        return 0
    return int(scalar.value) if scalar.value is not None else 0


def _map_to_str_dict(value: Any) -> dict[str, str]:
    if not value:
        return {}

    if isinstance(value, dict):
        return {
            str(k): str(v)
            for k, v in value.items()
            if k is not None and v is not None
        }

    try:
        return {
            str(k): str(v)
            for k, v in value
            if k is not None and v is not None
        }
    except Exception:
        return {}


def _first_present(cols: Mapping[str, Any], i: int, *names: str) -> Any:
    for name in names:
        if name in cols:
            return cols[name][i].as_py()
    return None


_HOP_BY_HOP: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)


ARROW_SCHEMA = pa.schema(
    [
        pa.field(
            "response_status_code",
            pa.int32(),
            nullable=False,
            metadata={"comment": "Status code returned by the server"},
        ),
        pa.field("response_host", pa.string(), nullable=True, metadata={"comment": "Host header"}),
        pa.field("response_user_agent", pa.string(), nullable=True, metadata={"comment": "User-Agent header"}),
        pa.field("response_accept", pa.string(), nullable=True, metadata={"comment": "Accept header"}),
        pa.field("response_accept_encoding", pa.string(), nullable=True, metadata={"comment": "Accept-Encoding header"}),
        pa.field("response_accept_language", pa.string(), nullable=True, metadata={"comment": "Accept-Language header"}),
        pa.field("response_content_type", pa.string(), nullable=True, metadata={"comment": "Content-Type header"}),
        pa.field(
            "response_content_length",
            pa.int64(),
            nullable=False,
            metadata={"comment": "Content-Length header parsed as integer when possible"},
        ),
        pa.field("response_content_encoding", pa.string(), nullable=True, metadata={"comment": "Content-Encoding header"}),
        pa.field("response_transfer_encoding", pa.string(), nullable=True, metadata={"comment": "Transfer-Encoding header"}),
        pa.field(
            "response_headers",
            pa.map_(pa.string(), pa.string()),
            nullable=False,
            metadata={
                "comment": "Response headers excluding promoted common headers",
                "keys_sorted": "false",
            },
        ),
        pa.field(
            "response_tags",
            pa.map_(pa.string(), pa.string()),
            nullable=False,
            metadata={"comment": "Arbitrary string tags attached to this response"},
        ),
        pa.field(
            "response_body",
            pa.binary(),
            nullable=True,
            metadata={"comment": "Raw binary payload of the response (bytes)"},
        ),
        pa.field(
            "response_body_hash",
            pa.int64(),
            nullable=True,
            metadata={
                "comment": "Signed Int64 XXH3 digest of response_body",
                "algorithm": "xxh3_64",
            },
        ),
        pa.field(
            "response_received_at",
            pa.timestamp("us", "UTC"),
            nullable=False,
            metadata={
                "comment": "UTC timestamp when the response was captured",
                "unit": "us",
                "tz": "UTC",
            },
        ),
        pa.field(
            "response_received_at_epoch",
            pa.int64(),
            nullable=False,
            metadata={
                "comment": "UTC epoch timestamp when the response was captured",
                "unit": "us",
                "tz": "UTC",
            },
        ),
    ],
    metadata={
        "comment": "Response record (single row), designed for deterministic logging + replay.",
    },
)

RESPONSE_ARROW_SCHEMA = pa.schema(
    list(REQUEST_ARROW_SCHEMA) + list(ARROW_SCHEMA),
    metadata={
        "comment": "Prepared request and response flattened into columns for single-row logging batches.",
    },
)


@dataclass
class Response:
    request: PreparedRequest
    status_code: int
    headers: MutableMapping[str, str]
    tags: MutableMapping[str, str]
    buffer: BytesIO
    received_at_timestamp: int

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
            d = json_module.loads(s)
        except Exception as e:
            raise ValueError("Response.parse_str: expected JSON object string") from e

        if not isinstance(d, Mapping):
            raise ValueError("Response.parse_str: JSON must decode to an object")

        return cls.parse_dict(d, normalize=normalize)

    @classmethod
    def parse_dict(
        cls,
        obj: Mapping[str, Any],
        *,
        normalize: bool = True,
        prefix: str = "response_",
    ) -> "Response":
        if not obj:
            raise ValueError("Response.parse_dict: empty mapping")

        req_obj = get_from_dict(obj, keys=("request",), prefix="")
        if req_obj is MISSING or req_obj in (None, ""):
            req_obj = obj

        request = PreparedRequest.parse(req_obj, normalize=normalize)
        status_code = _parse_status_code(obj, prefix=prefix)
        headers = _parse_headers(obj, prefix=prefix)
        buffer = _parse_buffer(obj, prefix=prefix)
        received_at_timestamp = _parse_received_at_timestamp(obj, prefix=prefix)
        tags = _parse_tags(obj, prefix=prefix)

        if normalize:
            headers = normalize_headers(headers, body=buffer, is_request=False)

        _ensure_media_headers(headers, buffer)

        return cls(
            request=request,
            status_code=status_code,
            headers=headers,
            buffer=buffer,
            received_at_timestamp=received_at_timestamp,
            tags=tags,
        )

    def update_headers(
        self,
        headers: MutableMapping[str, str],
        normalize: bool = True,
    ) -> "PreparedRequest":
        if not headers:
            return self

        if not self.headers:
            self.headers = dict(headers)
            if normalize:
                _ensure_media_headers(self.headers, self.buffer)
            return self

        for k, v in headers.items():
            self.headers[str(k)] = str(v)

        if normalize:
            _ensure_media_headers(self.headers, self.buffer)

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
        safe: bool = True
    ) -> "Response":
        if self.headers is None:
            self.headers = {}

        self.request.accept_media_type = value
        self.buffer.set_media_type(value, safe=safe)

        self.headers["Content-Type"] = value.mime_type.value

        if value.codec is not None:
            self.headers["Content-Encoding"] = value.codec.name
        else:
            del self.headers["Content-Encoding"]

        self.headers["Content-Length"] = str(self.buffer.size)

    @property
    def body(self) -> BytesIO:
        return self.buffer

    @property
    def codec(self):
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
        orient: Literal["records", "columns"] = "records",
    ) -> Any:
        media_type = self.media_type

        if media_type.codec:
            with self.buffer.decompress(codec=media_type.codec, copy=True) as b:
                mio = b.media_io(media_type.without_codec())

                if orient == "records":
                    return mio.read_pylist()
                elif orient == "columns":
                    return mio.read_pydict()
                else:
                    raise ValueError(f"Unsupported orient: {orient!r}")

        mio = self.buffer.media_io(media_type)

        if orient == "records":
            return mio.read_pylist()
        elif orient == "columns":
            return mio.read_pydict()
        else:
            raise ValueError(f"Unsupported orient: {orient!r}")

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400

    def raise_for_status(self) -> None:
        if not self.ok:
            raise self.error()

    def error(self):
        if not self.ok:
            from .errors import make_for_status
            return make_for_status(self)
        return None

    def anonymize(self, mode: str = "remove") -> "Response":
        return replace(
            self,
            request=self.request.anonymize(mode=mode),
            headers=normalize_headers(
                self.headers,
                is_request=False, mode=mode, body=self.body, anonymize=True
            ),
        )

    def to_polars(
        self,
        parse: bool = True,
        *,
        lazy: bool = False,
    ) -> "pl.DataFrame | pl.LazyFrame":
        from yggdrasil.polars.lib import polars as _pl

        if parse:
            mt = self.media_type

            if mt.codec:
                with self.buffer.decompress(mt.codec, copy=True) as b:
                    mio = b.media_io(media=mt)
                    df = mio.read_polars_frame(lazy=False)
                return df.lazy() if lazy else df

            mio = self.buffer.media_io(media=mt)
            return mio.read_polars_frame(lazy=lazy)

        return _pl.from_arrow(self.to_arrow_batch(parse=False))

    def to_pandas(self, parse: bool = True) -> "pd.DataFrame":
        return self.to_polars(parse=parse).to_pandas()

    def to_arrow_batch(self, parse: bool = False) -> pa.RecordBatch:
        if parse:
            from yggdrasil.polars.cast import polars_dataframe_to_arrow_table

            return polars_dataframe_to_arrow_table(
                self.to_polars(parse=True)
            ).to_batches()[0]

        req_rb = self.request.to_arrow_batch(parse=False)
        promoted = PromotedHeaders.extract(self.headers or {}, host=DEFAULT_HOSTNAME)
        tags_v = {str(k): str(v) for k, v in (self.tags or {}).items()}

        if self.buffer is not None:
            body_bytes = self.buffer.to_bytes()
            body_hash = self.buffer.xxh3_int64()
        else:
            body_bytes = None
            body_hash = None

        values = {
            "response_status_code": self.status_code,
            "response_host": promoted.host,
            "response_user_agent": promoted.user_agent,
            "response_accept": promoted.accept,
            "response_accept_encoding": promoted.accept_encoding,
            "response_accept_language": promoted.accept_language,
            "response_content_type": promoted.content_type,
            "response_content_length": promoted.content_length,
            "response_content_encoding": promoted.content_encoding,
            "response_transfer_encoding": promoted.transfer_encoding,
            "response_headers": promoted.remaining,
            "response_tags": tags_v,
            "response_body": body_bytes,
            "response_body_hash": body_hash,
            "response_received_at": self.received_at_timestamp,
            "response_received_at_epoch": self.received_at_timestamp,
        }

        arrays = [
            pa.array([values[f.name]], type=f.type)
            for f in ARROW_SCHEMA
        ]

        return pa.RecordBatch.from_arrays(
            list(req_rb.columns) + arrays,
            schema=RESPONSE_ARROW_SCHEMA,
        )  # type: ignore[arg-type]

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

        req_cols = [f.name for f in REQUEST_ARROW_SCHEMA]
        resp_cols = [f.name for f in ARROW_SCHEMA]

        def _iter_batches(obj: pa.RecordBatch | pa.Table) -> Iterator[pa.RecordBatch]:
            if isinstance(obj, pa.RecordBatch):
                yield obj
                return
            for _rb in obj.to_batches():
                yield _rb

        for rb in _iter_batches(batch):
            cols = {
                name: rb.column(name)
                for name in list(req_cols) + list(resp_cols)
                if name in rb.schema.names
            }

            for i in range(rb.num_rows):
                method = _first_present(cols, i, "request_method") or "GET"

                url_str = _first_present(cols, i, "request_url_str")
                if url_str not in (None, ""):
                    req_url_str: str | None = str(url_str)
                    req_url_struct: Any = None
                else:
                    scheme = _first_present(cols, i, "request_url_scheme")
                    userinfo = _first_present(cols, i, "request_url_userinfo")
                    host = _first_present(cols, i, "request_url_host")
                    port = _first_present(cols, i, "request_url_port")
                    path = _first_present(cols, i, "request_url_path")
                    query = _first_present(cols, i, "request_url_query")
                    fragment = _first_present(cols, i, "request_url_fragment")

                    has_exploded = any(
                        part not in (None, "", 0)
                        for part in (scheme, userinfo, host, port, path, query, fragment)
                    )

                    if has_exploded:
                        req_url_str = None
                        req_url_struct = {
                            "scheme": scheme or "",
                            "userinfo": userinfo or "",
                            "host": host or "",
                            "port": 0 if port in (None, "") else int(port),
                            "path": path or "",
                            "query": query or "",
                            "fragment": fragment or "",
                        }
                    else:
                        legacy_struct = _first_present(cols, i, "request_url")
                        req_url_str = None
                        req_url_struct = legacy_struct if isinstance(legacy_struct, Mapping) else ""

                request_headers = _map_to_str_dict(_first_present(cols, i, "request_headers"))
                request_promoted_pairs = {
                    "Host": _first_present(cols, i, "request_host"),
                    "User-Agent": _first_present(cols, i, "request_user_agent"),
                    "Accept": _first_present(cols, i, "request_accept"),
                    "Accept-Encoding": _first_present(cols, i, "request_accept_encoding"),
                    "Accept-Language": _first_present(cols, i, "request_accept_language"),
                    "Content-Type": _first_present(cols, i, "request_content_type"),
                    "Content-Length": _first_present(cols, i, "request_content_length"),
                    "Content-Encoding": _first_present(cols, i, "request_content_encoding"),
                    "Transfer-Encoding": _first_present(cols, i, "request_transfer_encoding"),
                }
                for hk, hv in request_promoted_pairs.items():
                    if hv not in (None, ""):
                        request_headers[hk] = str(hv)

                request = PreparedRequest.parse_dict(
                    {
                        "method": method,
                        "url_str": req_url_str,
                        "url": req_url_struct,
                        "headers": request_headers,
                        "tags": _map_to_str_dict(_first_present(cols, i, "request_tags")),
                        "buffer": _first_present(cols, i, "request_body"),
                        "sent_at_timestamp": (
                            _arrow_ts_col_to_us(cols["request_sent_at"], i)
                            if "request_sent_at" in cols else 0
                        ),
                    },
                    normalize=normalize,
                )

                response_headers = _map_to_str_dict(_first_present(cols, i, "response_headers"))
                response_promoted_pairs = {
                    "Host": _first_present(cols, i, "response_host"),
                    "User-Agent": _first_present(cols, i, "response_user_agent"),
                    "Accept": _first_present(cols, i, "response_accept"),
                    "Accept-Encoding": _first_present(cols, i, "response_accept_encoding"),
                    "Accept-Language": _first_present(cols, i, "response_accept_language"),
                    "Content-Type": _first_present(cols, i, "response_content_type"),
                    "Content-Length": _first_present(cols, i, "response_content_length"),
                    "Content-Encoding": _first_present(cols, i, "response_content_encoding"),
                    "Transfer-Encoding": _first_present(cols, i, "response_transfer_encoding"),
                }
                for hk, hv in response_promoted_pairs.items():
                    if hv not in (None, ""):
                        response_headers[hk] = str(hv)

                body_bytes = _first_present(cols, i, "response_body")
                buffer = BytesIO(body_bytes) if body_bytes is not None else BytesIO()

                if normalize:
                    response_headers = normalize_headers(
                        response_headers,
                        is_request=False, body=buffer
                    )

                _ensure_media_headers(response_headers, buffer)

                received_at = (
                    _arrow_ts_col_to_us(cols["response_received_at"], i)
                    if "response_received_at" in cols else 0
                )

                yield cls(
                    request=request,
                    status_code=int(_first_present(cols, i, "response_status_code") or 0),
                    headers=response_headers,
                    buffer=buffer,
                    tags=_map_to_str_dict(_first_present(cols, i, "response_tags")),
                    received_at_timestamp=received_at,
                )

    def _to_asgi_payload(self) -> tuple[bytes, dict[str, str], str]:
        body = self.buffer.to_bytes() if self.buffer is not None else b""

        headers = {
            str(k): str(v)
            for k, v in (self.headers or {}).items()
            if str(k).lower() not in _HOP_BY_HOP
        }

        media = self.media_type

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

    def apply(
        self,
        func: Callable[["Response"], "Response"]
    ) -> "Response":
        return func(self)