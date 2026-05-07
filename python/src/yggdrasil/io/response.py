# yggdrasil.io.response
"""HTTP response model with Arrow, Polars, pandas, and ASGI serialisation."""
from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Literal, Mapping, MutableMapping, Optional

import polars as pl
import pyarrow as pa

import yggdrasil.pickle.json as json_module
from yggdrasil.data.cast import any_to_datetime
from yggdrasil.data.data_field import field as schema_field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import schema
from yggdrasil.dataclasses.dataclass import get_from_dict
from .bytes_io import BytesIO
from yggdrasil.data.enums import Codec, MediaType, MimeTypes
from .headers import normalize_headers
from .holder import Holder
from .tabular.base import Tabular
from yggdrasil.environ.userinfo import USERINFO_STRUCT, UserInfo
from .request import (
    PreparedRequest,
    REQUEST_SCHEMA,
    _coerce_userinfo,
    _default_sender,
    _map_as_str_dict,
    _string_dict,
)

if TYPE_CHECKING:
    import pandas as pd
    from fastapi import Response as FastAPIResponse
    from pyspark.sql import DataFrame as SparkDataFrame, Row as SparkRow
    from starlette.responses import Response as StarletteResponse

    from .session import Session


__all__ = [
    "Response",
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
    if content_type or content_encoding:
        return MediaType.from_many(
            mime_types=[content_type, content_encoding],
            default=MediaType.from_mime(MimeTypes.OCTET_STREAM)
        )
    declared = body.media_type
    if declared is not None:
        return declared
    return MediaType.from_mime(MimeTypes.OCTET_STREAM)


def _media_type_from_headers(
    headers: Mapping[str, str] | None,
) -> MediaType | None:
    declared_type = _parse_content_type(headers)
    declared_encoding = _parse_content_encoding(headers)

    if not declared_type and not declared_encoding:
        return None

    if not declared_encoding and _is_probably_placeholder_content_type(declared_type):
        return None

    return MediaType.from_many(
        mime_types=[declared_type, declared_encoding],
        default=MediaType.from_mime(MimeTypes.OCTET_STREAM),
    )


def _ensure_media_headers(
    headers: MutableMapping[str, str],
    body: BytesIO,
) -> MediaType:
    declared_type = _parse_content_type(headers)
    declared_encoding = _parse_content_encoding(headers)

    # Treat placeholder Content-Type (octet-stream, unknown/unknown,
    # blank) as "missing" for the sniff: otherwise a placeholder we
    # stamped during ``__init__`` (when the buffer was still empty)
    # would mask the real media after the body lands.
    type_is_placeholder = _is_probably_placeholder_content_type(declared_type)
    sniff_type = None if type_is_placeholder else declared_type

    media = _sniff_media_from_body(
        body,
        content_type=sniff_type,
        content_encoding=declared_encoding,
    )

    if type_is_placeholder:
        headers["Content-Type"] = media.mime_type.value

    if not declared_encoding and media.codec is not None:
        headers["Content-Encoding"] = media.codec.name

    headers["Content-Length"] = str(body.size)

    if body.media_type is None and not media.mime_type.is_any_bytes:
        try:
            body.with_media_type(media, copy=False)
        except Exception:
            pass

    return media


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_response_headers(obj: Mapping[str, Any]) -> MutableMapping[str, str]:
    headers = get_from_dict(obj, keys=("headers",), prefix=None)
    return _map_as_str_dict(headers)


def _parse_response_tags(obj: Mapping[str, Any]) -> dict[str, str]:
    tags = get_from_dict(obj, keys=("tags",), prefix=None)
    return _string_dict(tags if isinstance(tags, Mapping) else None)


def _parse_response_buffer(
    obj: Mapping[str, Any],
    *,
    media_type: MediaType | None = None,
) -> BytesIO:
    body = get_from_dict(obj, keys=("buffer", "body", "content", "data"), prefix=None)
    if body is MISSING or body is None:
        return BytesIO(media_type=media_type) if media_type is not None else BytesIO()
    if isinstance(body, BytesIO):
        if media_type is not None and body.media_type is None:
            try:
                body.with_media_type(media_type, copy=False)
            except Exception:
                pass
        return body
    return BytesIO.from_(obj=body, media_type=media_type) if media_type is not None else BytesIO.from_(obj=body)


def _parse_status_code(obj: Mapping[str, Any]) -> int:
    status = get_from_dict(obj, keys=("status_code", "status", "code"), prefix=None)
    if status is MISSING or status in (None, ""):
        raise ValueError("Response.parse_mapping: missing status_code/status/code")
    return int(status) if isinstance(status, int) else int(float(str(status).strip()))


def _parse_received_at(obj: Mapping[str, Any]) -> dt.datetime:
    value = get_from_dict(obj, keys=("received_at_timestamp", "received_at"), prefix=None)
    if value is MISSING:
        return dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)
    return any_to_datetime(value)


def _parse_receiver(obj: Mapping[str, Any]) -> UserInfo | None:
    value = get_from_dict(obj, keys=("receiver",), prefix=None)
    if value is MISSING or value in (None, ""):
        return _default_sender()
    if isinstance(value, UserInfo):
        return value
    if isinstance(value, Mapping):
        return UserInfo.from_struct_dict(value)
    return _default_sender()


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

_RESPONSE_SCHEMA_JSON_TAGS: dict[str, str] = {
    "domain": "http",
    "entity": "response",
    "layer": "bronze",
    "namespace": "yggdrasil.io.response",
}


RESPONSE_SCHEMA = schema(
    fields=[],
    metadata={
        "comment": "Response record (single row), designed for deterministic logging and replay.",
        "time_column": "received_at",
        # Schema-level identity / partitioning hints — ``autotag`` at
        # the bottom of this block propagates them to the matching
        # children. The primary key is composite ``(hash, body_size)``;
        # ``partition_key`` (the only ``partition_by`` column) is
        # derived from :meth:`Response.partition_values` and matches
        # the embedded request's partition_key so they co-locate.
        "primary_key": ["hash", "body_size"],
        "partition_by": ["partition_key"],
    },
    tags=_RESPONSE_SCHEMA_JSON_TAGS,
)

# Unnest the request schema directly into the response schema with a
# ``request_`` prefix. Flattening turns nested struct lookups into
# top-level column accesses, which engines (Delta, Spark, Polars,
# Arrow) can predicate-push and column-prune against without having
# to crack the struct open. ``_pkl`` is intentionally skipped — the
# response carries its own pickle slot and a per-request blob would
# duplicate the same bytes.
_REQUEST_FIELD_NAMES_FOR_UNNEST: tuple[str, ...] = tuple(
    f.name for f in REQUEST_SCHEMA.children_fields if f.name != "_pkl"
)

# Tags that must NOT cross over from REQUEST_SCHEMA. The
# response-level schema has its own ``partition_key`` (which already
# carries ``partition_by``) and its own composite ``primary_key`` —
# leaking those flags onto the unnested ``request_*`` fields would
# add a redundant partition column to the on-disk Hive layout and
# duplicate the primary-key declaration.
_REQUEST_TAG_DROP: frozenset[bytes] = frozenset({
    b"partition_by",
    b"primary_key",
    # ``Field.tags`` lives in metadata under a ``t:`` prefix — drop
    # both spellings so an autotag() call on the unnested field
    # doesn't re-stamp the schema-level partition / primary-key
    # markers we want only on the response side.
    b"t:partition_by",
    b"t:primary_key",
})


def _decode_meta_kv(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8")
    return value


for _req_field_name in _REQUEST_FIELD_NAMES_FOR_UNNEST:
    _src_field = REQUEST_SCHEMA[_req_field_name]
    _src_meta = dict(_src_field.metadata or {})
    _existing_comment = _src_meta.get(b"comment") or _src_meta.get("comment")
    _comment_str = _decode_meta_kv(_existing_comment)
    _src_meta_clean: dict[str, Any] = {}
    for _k, _v in _src_meta.items():
        # Skip schema-level partitioning / primary-key flags inherited
        # from the request — they belong to the response's own columns.
        _k_bytes = _k.encode("utf-8") if isinstance(_k, str) else bytes(_k)
        if _k_bytes in _REQUEST_TAG_DROP:
            continue
        _src_meta_clean[_decode_meta_kv(_k)] = _decode_meta_kv(_v)
    if _comment_str:
        _src_meta_clean["comment"] = f"[request] {_comment_str}"
    RESPONSE_SCHEMA[f"request_{_req_field_name}"] = schema_field(
        f"request_{_req_field_name}",
        _src_field.arrow_type,
        nullable=_src_field.nullable,
        metadata=_src_meta_clean,
    ).autotag()

RESPONSE_SCHEMA["receiver"] = schema_field(
    "receiver",
    USERINFO_STRUCT,
    nullable=True,
    metadata={
        "comment": "Snapshot of :class:`~yggdrasil.environ.UserInfo` for the receiver "
                   "— defaults to ``UserInfo.current()``. Carries identity (key, "
                   "email, hostname, product) plus a stable ``hash``. Per-process "
                   "fields (cwd, url, git_url) are lazy properties on UserInfo and "
                   "are not part of the wire contract.",
    },
).autotag()

RESPONSE_SCHEMA["hash"] = schema_field(
    "hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest over (request.hash, status_code, headers, body) — "
                   "overall response identity, including sensitive bits.",
        "algorithm": "xxh3_64",
    },
).autotag()

RESPONSE_SCHEMA["public_hash"] = schema_field(
    "public_hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest over (request.public_hash, status_code, anonymized headers, body) — "
                   "stable across cache anonymization and the right key for cross-system identity.",
        "algorithm": "xxh3_64",
    },
).autotag()

RESPONSE_SCHEMA["status_code"] = schema_field(
    "status_code",
    pa.int32(),
    nullable=False,
    metadata={"comment": "HTTP status code returned by the server"},
).autotag()

RESPONSE_SCHEMA["headers"] = schema_field(
    "headers",
    pa.map_(pa.string(), pa.string()),
    nullable=False,
    metadata={"comment": "All response headers as a name→value map"},
).autotag()

RESPONSE_SCHEMA["tags"] = schema_field(
    "tags",
    pa.map_(pa.string(), pa.string()),
    nullable=True,
    metadata={"comment": "Arbitrary string tags attached to this response"},
).autotag()

RESPONSE_SCHEMA["body"] = schema_field(
    "body",
    pa.large_binary(),
    nullable=True,
    metadata={"comment": "Raw binary payload of the response"},
).autotag()

RESPONSE_SCHEMA["body_size"] = schema_field(
    "body_size",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "Length of body in bytes; 0 when body is absent",
        "unit": "bytes",
    },
).autotag()

RESPONSE_SCHEMA["body_hash"] = schema_field(
    "body_hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest of body bytes; 0 when body is absent",
        "algorithm": "xxh3_64",
    },
).autotag()

RESPONSE_SCHEMA["received_at"] = schema_field(
    "received_at",
    pa.timestamp("us", "UTC"),
    nullable=False,
    metadata={"comment": "UTC timestamp when the response was captured"},
).autotag()

RESPONSE_SCHEMA["partition_key"] = schema_field(
    "partition_key",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest of the response's ``partition_values`` — the only "
                   "``partition_by`` column. Equal to the embedded request's "
                   "``partition_key`` (default behaviour) so request+response always "
                   "co-locate. Override :meth:`Response.partition_values` to change.",
        "algorithm": "xxh3_64",
    },
).autotag()

RESPONSE_SCHEMA["_pkl"] = schema_field(
    "_pkl",
    pa.large_binary(),
    nullable=True,
    metadata={
        "comment": "Placeholder for a full ``Response`` pickle blob — populated by the "
                   "pickle serializer for lossless round-trips, left null on the "
                   "deterministic-columns-only path.",
    },
).autotag()

# Propagate schema-level ``primary_key`` / ``partition_by`` down to
# the matching children (consumes those metadata keys in place).
RESPONSE_SCHEMA = RESPONSE_SCHEMA.autotag()

RESPONSE_ARROW_SCHEMA: pa.Schema = RESPONSE_SCHEMA.to_arrow_schema()


def _compute_response_identity_hash(
    request_hash: int,
    status_code: int,
    headers: Mapping[str, str] | None,
    body: BytesIO | None,
) -> int:
    import xxhash
    h = xxhash.xxh3_64()
    h.update(int(request_hash).to_bytes(8, "little", signed=True))
    h.update(b"\x00")
    h.update(int(status_code).to_bytes(4, "little", signed=False))
    h.update(b"\x00")
    for k in sorted(headers or {}):
        h.update(str(k).encode("utf-8"))
        h.update(b"=")
        h.update(str(headers[k]).encode("utf-8"))
        h.update(b"\x00")
    if body is not None:
        h.update(body.xxh3_64().digest())
    u = h.intdigest()
    return u if u < 2**63 else u - 2**64


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

class Response(Tabular[CastOptions]):
    """HTTP response model — paired with the originating :class:`PreparedRequest`.

    Implements :class:`Tabular` over the deterministic metadata
    projection: :meth:`read_arrow_batches` (and the engine fan-out
    derived from it) yields a single Arrow row built from
    :attr:`arrow_values`, matching :data:`RESPONSE_ARROW_SCHEMA`. To
    parse the *body* as a tabular payload (Parquet, CSV, JSON …),
    open a cursor with :meth:`open` and use that cursor's Tabular
    surface — :class:`BytesIO` dispatches to the right format leaf
    via the holder's media type — or call the existing
    :meth:`to_arrow_batches` with ``parse=True``.
    """

    __slots__ = (
        "request",
        "status_code",
        "headers",
        "tags",
        "buffer",
        "received_at",
        "_receiver",
        "_id_cache",
        "_session",
    )

    def __init__(
        self,
        request: PreparedRequest,
        status_code: int,
        headers: MutableMapping[str, str],
        tags: MutableMapping[str, str],
        buffer: BytesIO,
        received_at: dt.datetime,
        receiver: Optional[UserInfo] = None,
    ) -> None:
        super().__init__()
        self.request = request
        self.status_code = int(status_code)
        self.headers = _string_dict(headers)
        self.tags = _string_dict(tags)
        self.received_at = any_to_datetime(received_at)
        self.buffer = buffer if isinstance(buffer, BytesIO) else BytesIO(buffer, copy=False)
        self._receiver: UserInfo | None = (
            _coerce_userinfo(receiver) if receiver is not None else _default_sender()
        )

        _ensure_media_headers(self.headers, self.buffer)

        self._id_cache: int | None = None
        self._session: "Session | None" = None

    # ------------------------------------------------------------------
    # Holder access — the response's body lives on a :class:`Holder`;
    # :attr:`buffer` is the long-lived cursor over it. :meth:`open`
    # mints a fresh cursor for callers that want their own cursor
    # lifetime instead of sharing the response's.
    # ------------------------------------------------------------------

    @property
    def holder(self) -> Holder:
        """The :class:`Holder` backing the response body."""
        return self.buffer._holder

    def open(self, mode: str = "rb+") -> BytesIO:
        """Open a fresh :class:`BytesIO` cursor over the response's holder.

        The returned cursor is non-owning: closing it does not close
        the holder (the response keeps its own cursor in
        :attr:`buffer`). Use this when you need an independent
        cursor — for parallel reads, for a dedicated tabular leaf
        via the holder's media type, etc.
        """
        return BytesIO(holder=self.holder, owns_holder=False, mode=mode)

    # ------------------------------------------------------------------
    # Tabular contract — yield the deterministic single-row metadata
    # projection. The body itself is reachable via :meth:`open` (its
    # cursor is a :class:`Tabular` too) or :meth:`to_arrow_batches`
    # with ``parse=True``.
    # ------------------------------------------------------------------

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        yield options.cast_arrow_tabular(self._arrow_batch_from_values())

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} is read-only. To persist a "
            "response row, call ``response.to_arrow_batch(parse=False)`` "
            "and write that batch through a writable Tabular sink "
            "(ArrowTabular, ParquetIO, a Delta/SQL table, …)."
        )

    @property
    def receiver(self) -> UserInfo | None:
        """:class:`UserInfo` snapshot for the side that received this response.

        Defaults to ``UserInfo.current()`` at construction time. Use
        :meth:`with_receiver` to swap it.
        """
        return self._receiver

    def with_receiver(self, receiver: UserInfo | Mapping[str, Any] | None) -> "Response":
        """Mutate :attr:`receiver` in place and return ``self``.

        Mirrors :meth:`update_headers` / :meth:`update_tags` — the
        response is mutable in this codebase, so this is a fluent
        in-place setter rather than a clone.
        """
        self._receiver = _coerce_userinfo(receiver)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<r={self.request} s={self.status_code} b={self.body!r}>"

    def __str__(self) -> str:
        return self.__repr__()

    def __getstate__(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__slots__
            if name != "_session"
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        self._session = None

    # ------------------------------------------------------------------
    # Session attachment
    # ------------------------------------------------------------------

    def attach_session(self, session: "Session") -> "Response":
        self._session = session
        return self

    def detach_session(self) -> "Response":
        self._session = None
        return self

    @property
    def session(self) -> "Session | None":
        return self._session

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
    ) -> "Response":
        if not obj:
            raise ValueError("Response.parse_mapping: empty mapping")

        req_obj = get_from_dict(obj, keys=("request",), prefix=None)
        request = PreparedRequest.parse(
            obj if req_obj is MISSING or req_obj in (None, "") else req_obj,
            normalize=normalize,
        )

        status_code = _parse_status_code(obj)
        headers = _parse_response_headers(obj)
        pre_media = _media_type_from_headers(headers)
        buffer = _parse_response_buffer(obj, media_type=pre_media)
        received_at = _parse_received_at(obj)
        tags = _parse_response_tags(obj)
        receiver = _parse_receiver(obj)

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
                    receiver=receiver,
                )

        return cls(
            request=request,
            status_code=status_code,
            headers=headers,
            tags=tags,
            buffer=buffer,
            received_at=received_at,
            receiver=receiver,
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
        self.buffer.with_media_type(value, copy=False)
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

    def as_media(self, media_type: MediaType | None = None) -> BytesIO:
        return self.buffer.as_media(media_type=media_type or self.media_type)

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
            media_type=media_type or self.media_type,
            orient=orient
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
    # Timestamps / hashes
    # ------------------------------------------------------------------

    @property
    def received_at_timestamp(self) -> int:
        return int(self.received_at.timestamp() * 1000000)

    def partition_values(self) -> dict[str, Any]:
        """Hook for subclasses to define what feeds ``partition_key``.

        Default: delegates to the embedded request so request and
        response rows share the same partition_key and co-locate.
        Override to give responses an independent partition strategy
        (e.g. partition by status class for an audit table).
        """
        return self.request.partition_values()

    @property
    def partition_key(self) -> int:
        """xxh3_64 of the joined :meth:`partition_values` — int64 partition column."""
        joined = "\x00".join(str(v) for v in self.partition_values().values())
        from .request import _xxh3_int64_str
        return _xxh3_int64_str(joined)

    @property
    def body_size(self) -> int:
        return self.buffer.size if self.buffer is not None else 0

    @property
    def body_hash(self) -> int:
        """xxh3_64 digest of the body bytes; 0 when body is absent.

        Non-nullable in the schema — callers that need a "missing"
        signal should branch on :attr:`buffer` or :attr:`body_size`.
        """
        if self.buffer is None:
            return 0
        return self.buffer.xxh3_int64()

    @property
    def hash(self) -> int:
        """Overall identity over (request.hash, status_code, headers, body)."""
        return _compute_response_identity_hash(
            request_hash=self.request.hash,
            status_code=self.status_code,
            headers=self.headers,
            body=self.buffer,
        )

    @property
    def public_hash(self) -> int:
        """Cross-system identity stable across ``anonymize='remove'``."""
        return _compute_response_identity_hash(
            request_hash=self.request.public_hash,
            status_code=self.status_code,
            headers=normalize_headers(
                self.headers or {},
                is_request=False,
                add_missing=False,
                anonymize=True,
                mode="remove",
            ),
            body=self.buffer,
        )

    # ------------------------------------------------------------------
    # Matching / projection
    # ------------------------------------------------------------------

    @property
    def arrow_values(self) -> dict[str, Any]:
        body_bytes: bytes | None
        body_hash: int
        if self.buffer is not None:
            body_bytes = self.buffer.to_bytes()
            body_hash = self.buffer.xxh3_int64()
        else:
            body_bytes = None
            body_hash = 0

        request_values = self.request.arrow_values
        flat_request: dict[str, Any] = {
            f"request_{k}": v
            for k, v in request_values.items()
            if k != "_pkl"
        }

        return {
            **flat_request,
            "receiver":      self._receiver.to_struct_dict() if self._receiver is not None else None,
            "hash":          self.hash,
            "public_hash":   self.public_hash,
            "status_code":   self.status_code,
            "headers":       _string_dict(self.headers),
            "tags":          _string_dict(self.tags) or None,
            "body":          body_bytes,
            "body_size":     self.body_size,
            "body_hash":     body_hash,
            "received_at":   self.received_at,
            "partition_key": self.partition_key,
            # ``_pkl`` is a placeholder column populated externally by
            # the pickle serializer; null here keeps the deterministic
            # projection path side-effect-free.
            "_pkl":          None,
        }

    def match_value(self, key: str) -> Any:
        # Dotted-path lookup walks the nested struct shape:
        # ``request.method`` (legacy), ``request_url.host``,
        # ``headers.content_type``.
        if "." in key:
            head, _, tail = key.partition(".")
            # Legacy ``request.X`` form — pre-flatten callers used the
            # nested struct shape; route through the embedded request.
            if head == "request":
                return self.request.match_value(tail)
            values = self.arrow_values
            if head in values:
                container = values[head]
                cursor: Any = container
                for part in tail.split("."):
                    if isinstance(cursor, Mapping) and part in cursor:
                        cursor = cursor[part]
                    else:
                        raise ValueError(
                            f"Unsupported response match key: {key!r}. "
                            f"Must be within: {RESPONSE_ARROW_SCHEMA.names!r}"
                        )
                return cursor
            raise ValueError(
                f"Unsupported response match key: {key!r}. "
                f"Must be within: {RESPONSE_ARROW_SCHEMA.names!r}"
            )

        # Top-level response columns win, including the unnested
        # ``request_*`` ones populated by ``arrow_values``. Bare
        # request-side keys (``method``, ``url``, ``public_url_hash``, …)
        # still fall through to the embedded request so the cache layer
        # can match-by request identity without rewriting every
        # ``request_by`` list with the ``request_`` prefix.
        values = self.arrow_values
        if key in values:
            return values[key]
        if hasattr(self, key) and key not in REQUEST_SCHEMA.names:
            return getattr(self, key)
        if key in REQUEST_SCHEMA.names:
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

        return self.__class__(
            request=self.request.anonymize(mode=mode),
            status_code=self.status_code,
            headers=normalize_headers(
                self.headers,
                is_request=False,
                mode=mode,
                body=self.body,
                anonymize=True,
            ),
            tags=self.tags,
            buffer=self.buffer,
            received_at=self.received_at,
            receiver=self._receiver,
        )

    # ------------------------------------------------------------------
    # Serialisation — Arrow
    # ------------------------------------------------------------------

    def to_arrow_batches(
        self,
        parse: bool = True,
        **options: Any
    ) -> Iterator[pa.RecordBatch]:
        if parse:
            with self.as_media() as b:
                yield from b.read_arrow_batches(**options)
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
            yield from self.as_media().read_polars_frames(lazy=lazy, **media_options)
        else:
            yield pl.from_arrow(self.to_arrow_batch(parse=False))

    def to_polars(
        self,
        parse: bool = True,
        *,
        lazy: bool = False,
        **media_options: Any,
    ) -> "pl.DataFrame | pl.LazyFrame":
        from yggdrasil.lazy_imports import polars as _pl

        if parse:
            mio = self.buffer.as_media(media_type=self.media_type)
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

        # ``request`` is included for back-compat with pre-unnest
        # batches that still carry a single nested ``request`` struct
        # column instead of the flattened ``request_*`` columns.
        response_cols = [f.name for f in RESPONSE_ARROW_SCHEMA] + ["request"]

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
    def from_record(
        cls,
        record: "Mapping[str, Any]",
        *,
        normalize: bool = False,
    ) -> "Response":
        """Build a :class:`Response` from a row-shaped mapping."""
        return cls._from_get(record.get, normalize=normalize)

    @classmethod
    def from_records(
        cls,
        records: "Iterable[Mapping[str, Any]]",
        *,
        normalize: bool = False,
    ) -> Iterator["Response"]:
        for record in records:
            yield cls.from_record(record, normalize=normalize)

    @classmethod
    def _from_arrow_cols(
        cls,
        cols: dict[str, Any],
        i: int,
        *,
        normalize: bool = False,
    ) -> "Response":
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
        normalize: bool = False,
    ) -> "Response":
        # Three layouts are supported:
        #  1) flattened ``request_<col>`` columns (current schema);
        #  2) legacy nested ``request`` struct (pre-unnest snapshots
        #     and round-tripping through ``arrow_values``);
        #  3) flat top-level request columns (``method``, ``url`` ...) —
        #     used by ``parse_mapping`` callers that bypass the schema.
        request_value = get("request")
        if isinstance(request_value, Mapping):
            request = PreparedRequest._from_get(request_value.get, normalize=normalize)
        elif get("request_url") is not None or get("request_method") is not None:
            def _request_get(name: str) -> Any:
                return get(f"request_{name}")
            request = PreparedRequest._from_get(_request_get, normalize=normalize)
        else:
            request = PreparedRequest._from_get(get, normalize=normalize)

        headers = _map_as_str_dict(get("headers"))

        body_bytes = get("body")
        pre_media = _media_type_from_headers(headers)
        if body_bytes is None:
            buffer = BytesIO(media_type=pre_media) if pre_media is not None else BytesIO()
        elif pre_media is not None:
            buffer = BytesIO(body_bytes, copy=False, media_type=pre_media)
        else:
            buffer = BytesIO(body_bytes, copy=False)

        if normalize:
            headers = normalize_headers(headers, body=buffer, is_request=False)

        out_class = cls
        if cls is Response and request.url.is_http:
            from .http_ import HTTPResponse
            out_class = HTTPResponse

        receiver_value = get("receiver")
        receiver = _coerce_userinfo(receiver_value) if receiver_value is not None else None

        return out_class(
            request=request,
            status_code=get("status_code") or 0,
            headers=headers,
            tags=_map_as_str_dict(get("tags")),
            buffer=buffer,
            received_at=get("received_at") or 0,
            receiver=receiver,
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
# instead of falling back to the generic polars path.
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
