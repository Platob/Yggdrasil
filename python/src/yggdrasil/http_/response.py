# yggdrasil.io.response
"""HTTP response model with Arrow, Polars, pandas, and ASGI serialisation."""
from __future__ import annotations

import dataclasses
import datetime as dt
import http.client
import warnings
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterable, Iterator, Literal, Mapping, MutableMapping, Optional, Tuple

import polars as pl
import pyarrow as pa

import yggdrasil.pickle.json as json_module
from yggdrasil.data.cast import any_to_datetime
from yggdrasil.data.data_field import field as schema_field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import schema
from yggdrasil.dataclasses.dataclass import get_from_dict
from yggdrasil.io.base import IO
from yggdrasil.path.memory_stream import MemoryStream
from yggdrasil.enums import Codec, MediaType, MimeTypes
from yggdrasil.http_.headers import HTTPHeaders, HTTPHeaderDict
from yggdrasil.io.holder import Holder
from yggdrasil.path.memory import Memory
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.environ.userinfo import USERINFO_STRUCT, UserInfo
from yggdrasil.http_.request import (
    HTTPRequest,
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

    from yggdrasil.http_.session import HTTPSession
    from yggdrasil.http_.path import HTTPPath


__all__ = [
    "ResponseOptions",
    "RESPONSE_SCHEMA",
    "HTTPResponse",
]


@dataclasses.dataclass(frozen=True, slots=True)
class ResponseOptions(CastOptions):
    """:class:`CastOptions` for tabular reads off a :class:`Response`.

    ``parse`` toggles how :meth:`Response._read_arrow_batches` (and the
    engine fan-out derived from it) treats the response body:

    * ``True`` (default) — when the body's media type maps to a
      registered :class:`Tabular` leaf (Parquet, CSV, NDJSON, Arrow
      IPC, …), open the body and yield its parsed batches. Falls back
      to the deterministic single-row metadata projection when no
      tabular leaf claims the media type, preserving the historical
      auto-parse-when-possible behaviour.
    * ``False`` — always yield the deterministic metadata projection
      that matches :data:`RESPONSE_SCHEMA.to_arrow_schema()`, regardless of the
      body's media type.
    """

    parse: bool = True


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
    body: Holder,
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
    body: Holder,
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
            body.media_type = media
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
) -> Holder:
    body = get_from_dict(obj, keys=("buffer", "body", "content", "data"), prefix=None)
    return _coerce_buffer(
        None if body is MISSING else body,
        media_type=media_type,
    )


def _coerce_buffer(
    obj: Any,
    *,
    media_type: MediaType | None = None,
) -> Holder:
    """Normalize *obj* into a :class:`Holder` and stamp *media_type* if absent.

    Accepts the shapes :class:`Response` / :class:`HTTPRequest`
    constructors hand in: an existing :class:`Holder` (passed through),
    an :class:`IO` cursor (the holder is borrowed), bytes-like input
    (wrapped in a fresh :class:`Memory`), or ``None`` (empty Memory).
    """
    if obj is None:
        holder: Holder = Memory()
    elif isinstance(obj, Holder):
        holder = obj
    elif isinstance(obj, IO):
        holder = obj._parent
    else:
        holder = Holder.from_(obj)

    if media_type is not None and holder.media_type is None:
        try:
            holder.media_type = media_type
        except Exception:
            pass
    return holder


def _parse_status_code(obj: Mapping[str, Any]) -> int:
    status = get_from_dict(obj, keys=("status_code", "status", "code"), prefix=None)
    if status is MISSING or status in (None, ""):
        raise ValueError("Response.from_mapping: missing status_code/status/code")
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

from yggdrasil.http_.schemas import RESPONSE_SCHEMA


# Struct type that mirrors :data:`RESPONSE_SCHEMA.to_arrow_schema()` column-for-column.
# Building the single-row metadata projection through a struct array
# and :meth:`pa.RecordBatch.from_struct_array` keeps the per-child
# array construction inside pyarrow's C++ side, where the request
# projection benchmark shows a ~2x speedup over the per-column
# ``pa.array([value], type=…)`` loop.
_RESPONSE_ARROW_STRUCT_TYPE: pa.StructType = pa.struct(RESPONSE_SCHEMA.to_arrow_schema())


def _compute_response_identity_hash(
    request_hash: int,
    status_code: int,
    headers: "HTTPHeaders | Mapping[str, str] | None",
    body: Holder | None,
) -> int:
    """Mix (request_hash, status, headers, body) into one xxh3_64.

    Each component arrives pre-digested where possible:
    :attr:`HTTPRequest.hash` already collapses method / url /
    headers / body upstream, :class:`HTTPHeaders` caches
    :attr:`canonical_bytes`, :class:`Holder` caches
    :attr:`xxh3_64_digest`. The response identity is the request
    digest mixed with the response-side bytes.
    """
    import xxhash
    h = xxhash.xxh3_64()
    h.update(int(request_hash).to_bytes(8, "little", signed=True))
    h.update(b"\x00")
    h.update(int(status_code).to_bytes(4, "little", signed=False))
    h.update(b"\x00")
    if headers is not None:
        if not isinstance(headers, HTTPHeaders):
            headers = HTTPHeaders.from_(headers)
        h.update(headers.canonical_bytes)
    if body is not None:
        h.update(body.xxh3_64_digest)
    u = h.intdigest()
    return u if u < 2**63 else u - 2**64


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


class HTTPResponse(IO):  # IO inherits Tabular
    """HTTP response model — paired with the originating :class:`HTTPRequest`.

    Implements :class:`Tabular` over the deterministic metadata
    projection: :meth:`read_arrow_batches` (and the engine fan-out
    derived from it) yields a single Arrow row built from
    :attr:`arrow_values`, matching :data:`RESPONSE_SCHEMA.to_arrow_schema()`. To
    parse the *body* as a tabular payload (Parquet, CSV, JSON …),
    open a cursor with :meth:`open` and use that cursor's Tabular
    surface — :class:`IO` dispatches to the right format leaf
    via the holder's media type — or call the existing
    :meth:`to_arrow_batches` with ``parse=True``.

    The Tabular dispatch flips the same ``parse`` knob through
    :class:`ResponseOptions` — pass ``parse=False`` to force the
    metadata projection even when the body would otherwise auto-parse.
    """

    @classmethod
    def options_class(cls) -> "type[ResponseOptions]":
        return ResponseOptions

    # ``Response`` deliberately runs without ``__slots__``: the
    # high-level :class:`yggdrasil.http_.HTTPResponse` subclass mixes
    # the API in with :class:`yggdrasil.io.holder.IO` (which has its
    # own slot layout) so a slot-backed :class:`Response` would raise
    # the "multiple bases have instance lay-out conflict" :class:`TypeError`
    # at class creation. The per-instance ``__dict__`` overhead is
    # paid back by the merged HTTPResponse being a real
    # :class:`Tabular` + cursor in one object — no wrapper layer.
    #
    # Slot-style field set (kept here as the source of truth for
    # :meth:`__getstate__` / :meth:`__setstate__`):
    #   request, status_code, headers, tags, buffer, received_at,
    #   _receiver, _session, _cache, _cache_token.

    # Instance attributes that don't survive pickling — excluded by
    # ``__getstate__`` and reset by ``__setstate__``. Subclasses extend.
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset(
        {"_session", "_cache", "_cache_token"}
    )

    # Persisted slot-style field set — :meth:`__getstate__` reads
    # these off ``self.__dict__`` to keep the pickle payload free of
    # IO-mixin transients (``_pos``, ``_mode``, ``_parent``, …) when
    # subclasses bring those along.
    _PERSISTED_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset({
        "request",
        "status_code",
        "headers",
        "tags",
        "buffer",
        "received_at",
        "_receiver",
    })


    # ------------------------------------------------------------------
    # Holder access — :attr:`buffer` IS the durable :class:`Holder`.
    # :meth:`open` mints a fresh :class:`IO` cursor over it for
    # callers that want their own cursor lifetime; the response
    # itself keeps no cursor in slot.
    # ------------------------------------------------------------------

    @property
    def holder(self) -> Holder:
        """The :class:`Holder` backing the response body — alias for :attr:`buffer`."""
        return self.buffer

    def open(self, mode: str = "rb+") -> IO:
        """Open a fresh :class:`IO` cursor over the response's holder.

        Dispatches to the format-specific leaf via the holder's
        media type (Parquet → :class:`ParquetFile`, CSV →
        :class:`CSVFile`, …). The returned cursor is non-owning:
        closing it does not close the underlying holder.
        """
        return self.buffer.open(mode=mode, owns_holder=False)

    # ------------------------------------------------------------------
    # Tabular contract — yield the deterministic single-row metadata
    # projection. The body itself is reachable via :meth:`open` (its
    # cursor is a :class:`Tabular` too) or :meth:`to_arrow_batches`
    # with ``parse=True``.
    # ------------------------------------------------------------------

    def _read_arrow_batches(self, options: "ResponseOptions") -> Iterator[pa.RecordBatch]:
        # ``options.parse`` (default ``True``) opts in to body parsing
        # when its declared mime resolves to a registered tabular leaf
        # (Parquet, CSV, NDJSON, Arrow IPC, …). When ``parse`` is
        # False, or no tabular leaf claims the media type, fall back
        # to the deterministic single-row metadata projection that
        # matches :data:`RESPONSE_SCHEMA.to_arrow_schema()` (envelope mime
        # :attr:`MimeTypes.HTTP_RESPONSE`).
        from yggdrasil.io.holder import Holder
        if options.parse and Holder.class_for_media_type(
            self.media_type.mime_type, default=None,
        ) is not None:
            with self.open(mode="rb") as b:
                for batch in b.read_arrow_batches():
                    yield options.cast_arrow_batch(batch)
            return
        yield options.cast_arrow_batch(self._arrow_batch_from_values())

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: "ResponseOptions",
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} is read-only. To persist a "
            "response row, call ``response.to_arrow_batch(parse=False)`` "
            "and write that batch through a writable Tabular sink "
            "(ArrowTabular, ParquetFile, a Delta/SQL table, …)."
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
        # Pick the persisted-field set off ``self.__dict__`` — subclasses
        # like :class:`HTTPResponse` (which mixes in :class:`IO`) carry
        # extra cursor/transport attributes (``_pos``, ``_mode``,
        # ``_parent``, ``_url``, …) that have no place in the response
        # pickle payload. Sticking to ``_PERSISTED_STATE_ATTRS`` keeps
        # the wire format stable as the mixin set evolves.
        state: dict[str, Any] = {}
        for name in self._PERSISTED_STATE_ATTRS:
            try:
                state[name] = getattr(self, name)
            except AttributeError:
                continue
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        self._session = None
        self._cache = {}
        self._cache_token = ()
        # Re-coerce in case an old pickle carried ``headers`` as a
        # plain dict — :class:`HTTPHeaders.from_` is a no-op on an
        # already-built instance.
        self.headers = HTTPHeaders.from_(self.headers)

    # ------------------------------------------------------------------
    # Memoization — cheap fingerprint check, refresh on change
    # ------------------------------------------------------------------

    def _state_token(self) -> tuple:
        """Cheap fingerprint of the inputs that drive every cached
        derivation (hashes, arrow_values).

        Trusts each inner object to track its own state: the embedded
        request's :meth:`_state_token` rolls up its method / URL /
        headers / body fingerprint; :class:`HTTPHeaders` exposes
        ``version`` for in-place tracking; :class:`Holder` is keyed
        by ``id`` + size. No byte-length shadows — when the inner
        object changes, the token shifts on its own.
        """
        headers = self.headers
        buffer = self.buffer
        return (
            self.request._state_token(),
            self.status_code,
            id(headers),
            headers.version if headers is not None else 0,
            id(buffer),
            buffer.size if buffer is not None else -1,
        )

    def _cached(self, name: str, compute: Callable[[], Any]) -> Any:
        token = self._state_token()
        if self._cache_token != token:
            self._cache_token = token
            self._cache.clear()
        cached = self._cache.get(name, ...)
        if cached is ...:
            cached = compute()
            # Re-fingerprint after compute — some cached properties
            # (notably :attr:`media_type` via :func:`_ensure_media_headers`)
            # stamp ``Content-Length`` / ``Content-Type`` back onto the
            # headers as a side effect, which bumps ``headers.version``.
            # Without rolling the cache token forward, the next access
            # would see a shifted state token and wipe the entry we
            # just populated, so every "warm" read paid for a fresh
            # compute. Side-effect-free computes get the same token
            # back and this is a no-op.
            post = self._state_token()
            if post != token:
                self._cache_token = post
            self._cache[name] = cached
        return cached

    def _invalidate_cache(self) -> None:
        """Drop every memoized derivation — call after a value-equal
        :class:`HTTPHeaders` swap or any other mutation the inner-object
        fingerprint can't see (none of the standard setters need
        this; :class:`HTTPHeaders` already bumps its own version on
        in-place writes)."""
        self._cache.clear()
        self._cache_token = ()

    # ------------------------------------------------------------------
    # Session attachment
    # ------------------------------------------------------------------

    def attach_session(self, session: "HTTPSession") -> "Response":
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
    def from_(cls, obj: Any, *, normalize: bool = True) -> "Response":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, str):
            return cls.from_str(obj, normalize=normalize)
        if isinstance(obj, Mapping):
            return cls.from_mapping(obj, normalize=normalize)
        return cls.from_str(str(obj), normalize=normalize)

    @classmethod
    def from_str(cls, raw: str, *, normalize: bool = True) -> "Response":
        s = raw.strip()
        if not s:
            raise ValueError("Response.from_str: empty string")
        try:
            d = json_module.loads(s)
        except Exception as exc:
            raise ValueError("Response.from_str: expected JSON object string") from exc
        if not isinstance(d, Mapping):
            raise ValueError("Response.from_str: JSON must decode to a mapping")
        return cls.from_mapping(d, normalize=normalize)

    @classmethod
    def from_mapping(
        cls,
        obj: Mapping[str, Any],
        *,
        normalize: bool = True,
    ) -> "Response":
        if not obj:
            raise ValueError("Response.from_mapping: empty mapping")

        req_obj = get_from_dict(obj, keys=("request",), prefix=None)
        request = HTTPRequest.from_(
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
            headers = HTTPHeaders.from_(headers).normalized(body=buffer, is_request=False)

        _ensure_media_headers(headers, buffer)

        if False:  # removed: no separate Response class
            if request.url.is_http:
                from yggdrasil.http_ import HTTPResponse

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

        self._invalidate_cache()
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

        self._invalidate_cache()
        return self

    # ------------------------------------------------------------------
    # Media type
    # ------------------------------------------------------------------

    @property
    def media_type(self) -> MediaType:
        if self.headers is None:
            self.headers = {}
        return self._cached(
            "media_type",
            lambda: _ensure_media_headers(self.headers, self.buffer),
        )

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
        self.buffer.media_type = value
        self.headers["Content-Type"] = value.mime_type.value

        if value.codec is not None:
            self.headers["Content-Encoding"] = value.codec.name
        else:
            self.headers.pop("Content-Encoding", None)

        self.headers["Content-Length"] = str(self.buffer.size)
        self._invalidate_cache()
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
    def body(self) -> Holder:
        return self.buffer

    @property
    def codec(self) -> Optional[Codec]:
        return self.media_type.codec

    @property
    def content(self) -> bytes:
        codec = self.codec
        if codec is None:
            return self.buffer.to_bytes()
        with self.open(mode="rb") as bio:
            with bio.decompress(codec=codec, copy=True) as b:
                return b.to_bytes()

    @property
    def text(self) -> str:
        return self.content.decode(_get_charset(self.headers), errors="replace")

    def json(
        self,
        orient: Optional[Literal["records", "split", "index", "columns", "values"]] = None,
        *,
        media_type: Optional[MediaType] = None,
    ) -> Any:
        with self.open(mode="rb") as bio:
            return bio.json_load(
                media_type=media_type or self.media_type,
                orient=orient,
            )

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400

    def refresh_auth(self, force: bool = True) -> bool:
        """Refresh the Authorization header on the bound request.

        Delegates to :meth:`Session.refresh_auth` on the request's
        attached session, which re-runs the handler's refresh
        (force-minting a fresh token when ``force=True``) and
        re-stamps ``request.headers["Authorization"]``.

        Returns ``True`` when the handler ran and the header was
        stamped, ``False`` when no handler is bound.
        """
        session = self.request.session
        if session is None:
            raise RuntimeError(
                f"{type(self).__name__}.refresh_auth requires the bound "
                f"request to carry an attached session — got "
                f"{self.request!r} with session=None. Attach one via "
                "request.attach_session(session) or call "
                "session.refresh_auth(response.request, force=...) directly."
            )
        return session.refresh_auth(self.request, force=force)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise self.error()

    def warn_for_status(self) -> None:
        err = self.error()
        if err is not None:
            warnings.warn(str(err), category=RuntimeWarning, stacklevel=2)

    def error(self) -> Optional[Exception]:
        if not self.ok:
            from yggdrasil.exceptions import make_for_status
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
        from yggdrasil.http_.request import _xxh3_int64_str
        return self._cached(
            "partition_key",
            lambda: _xxh3_int64_str(
                "\x00".join(str(v) for v in self.partition_values().values())
            ),
        )

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
        return self._cached("body_hash", lambda: self.buffer.xxh3_int64())

    @property
    def hash(self) -> int:
        """Overall identity over (request.hash, status_code, headers, body)."""
        return self._cached(
            "hash",
            lambda: _compute_response_identity_hash(
                request_hash=self.request.hash,
                status_code=self.status_code,
                headers=self.headers,
                body=self.buffer,
            ),
        )

    @property
    def public_hash(self) -> int:
        """Cross-system identity stable across ``anonymize='remove'``."""
        return self._cached(
            "public_hash",
            lambda: _compute_response_identity_hash(
                request_hash=self.request.public_hash,
                status_code=self.status_code,
                headers=self.headers.anonymized(mode="remove"),
                body=self.buffer,
            ),
        )

    # ------------------------------------------------------------------
    # Matching / projection
    # ------------------------------------------------------------------

    def _arrow_value(self, key: str) -> Any:
        """Compute a single :data:`RESPONSE_SCHEMA.to_arrow_schema()` column on demand.

        Used by :meth:`match_value` so the lookup pays for one column
        instead of materializing every value via :attr:`arrow_values`.
        ``request_<col>`` keys forward to :meth:`HTTPRequest._arrow_value`
        on the embedded request — its own memoization takes care of
        repeated lookups.
        """
        if key.startswith("request_"):
            tail = key[len("request_"):]
            return self.request._arrow_value(tail)
        if key == "receiver":
            return self._receiver.to_struct_dict() if self._receiver is not None else None
        if key == "hash":
            return self.hash
        if key == "public_hash":
            return self.public_hash
        if key == "status_code":
            return self.status_code
        if key == "headers":
            return self._cached("headers_value", lambda: _string_dict(self.headers))
        if key == "tags":
            return self._cached("tags_value", lambda: _string_dict(self.tags) or None)
        if key == "body":
            return self.buffer.to_bytes() if self.buffer is not None else None
        if key == "body_size":
            return self.body_size
        if key == "body_hash":
            return self.body_hash
        if key == "received_at":
            return self.received_at
        if key == "partition_key":
            return self.partition_key
        if key == "_pkl":
            # Placeholder for the optional pickle blob; populated by
            # the pickle serializer, null on the structured path.
            return None
        raise KeyError(key)

    @property
    def arrow_values(self) -> dict[str, Any]:
        return self._cached("arrow_values", self._build_arrow_values)

    def _build_arrow_values(self) -> dict[str, Any]:
        return {name: self._arrow_value(name) for name in RESPONSE_SCHEMA.to_arrow_schema().names}

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
            try:
                container = self._arrow_value(head)
            except KeyError:
                raise ValueError(
                    f"Unsupported response match key: {key!r}. "
                    f"Must be within: {RESPONSE_SCHEMA.to_arrow_schema().names!r}"
                )
            cursor: Any = container
            for part in tail.split("."):
                if isinstance(cursor, Mapping) and part in cursor:
                    cursor = cursor[part]
                else:
                    raise ValueError(
                        f"Unsupported response match key: {key!r}. "
                        f"Must be within: {RESPONSE_SCHEMA.to_arrow_schema().names!r}"
                    )
            return cursor

        # Bare request-side keys (``method``, ``url``,
        # ``public_url_hash``, ``public_hash``, ``body_hash``, …) route
        # through the embedded request first so the cache layer's
        # ``request_by`` match-by-request-identity check lines up with
        # the SQL ``request_<col>`` predicate the same config emits
        # via :meth:`CacheConfig.sql_request_clause`. Without this,
        # collision keys (``public_hash`` / ``body_hash`` exist on
        # both schemas) would return the response's own value and
        # ``filter_response`` would reject every match. Response-only
        # keys still resolve through :meth:`_arrow_value` below.
        if key in REQUEST_SCHEMA.names:
            return self.request.match_value(key)
        try:
            return self._arrow_value(key)
        except KeyError:
            pass
        if hasattr(self, key):
            return getattr(self, key)
        raise ValueError(
            f"Unsupported response match key: {key!r}. "
            f"Must be within: {RESPONSE_SCHEMA.to_arrow_schema().names!r}"
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
        return tuple(self.match_value(str(key)) for key in keys)

    # ------------------------------------------------------------------
    # Anonymisation
    # ------------------------------------------------------------------

    def anonymize(self, mode: Literal["remove", "redact"] = "remove") -> "Response":
        if not mode:
            return self

        return self.__class__(
            request=self.request.anonymize(mode=mode),
            status_code=self.status_code,
            headers=self.headers.normalized(
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
            with self.open(mode="rb") as b:
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
        # See :data:`_RESPONSE_ARROW_STRUCT_TYPE` — building the single
        # row through a struct array keeps the per-child construction
        # in pyarrow C++ instead of paying a python ``pa.array(...)``
        # per column. The schema is reattached so callers get the same
        # metadata-rich :data:`RESPONSE_SCHEMA.to_arrow_schema()` shape.
        return self.values_to_arrow_batch([self])

    @classmethod
    def values_to_arrow_batch(
        cls,
        responses: "Iterable[Response]",
    ) -> pa.RecordBatch:
        """Build one :class:`pa.RecordBatch` from N responses in a single C++ pass.

        The per-row :meth:`to_arrow_batch` path does one
        ``pa.array([arrow_values], type=struct)`` plus
        ``RecordBatch.from_struct_array`` per call — N rows means N
        independent C++ struct walks plus an outer
        ``Table.from_batches(...).combine_chunks()`` concat. This
        classmethod folds the whole bulk into one struct walk: collect
        ``arrow_values`` once per response, hand the list of dicts to
        pyarrow, get back a single ``RecordBatch`` with N rows.

        Used by every bulk-writeback path in the cache pipeline
        (``Session._persist_remote``, ``Session._responses_to_spark``,
        :func:`responses_to_tabular`) — at 64 rows the bulk shape is
        ~30x faster than the per-row build it replaces.
        """
        values = [r.arrow_values for r in responses]
        struct_array = pa.array(values, type=_RESPONSE_ARROW_STRUCT_TYPE)
        batch = pa.RecordBatch.from_struct_array(struct_array)
        if batch.schema is not RESPONSE_SCHEMA.to_arrow_schema():
            batch = pa.RecordBatch.from_arrays(
                batch.columns, schema=RESPONSE_SCHEMA.to_arrow_schema(),
            )
        return batch

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
            with self.open(mode="rb") as b:
                yield from b.read_polars_frames(lazy=lazy, **media_options)
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
            with self.open(mode="rb") as mio:
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

        return cls.from_mapping(_to_python(row), normalize=normalize)

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
        response_cols = [f.name for f in RESPONSE_SCHEMA.to_arrow_schema()] + ["request"]

        for rb in _iter_batches(batch):
            available_set = set(rb.schema.names)
            picks = [n for n in response_cols if n in available_set]
            if not picks:
                continue
            cols = {n: rb.column(n) for n in picks}
            for i in range(rb.num_rows):
                yield cls._from_arrow_row(cols, i, normalize=normalize)

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
    def _from_arrow_row(
        cls,
        cols: dict[str, Any],
        i: int,
        *,
        normalize: bool = False,
    ) -> "Response":
        """Build one :class:`Response` from a per-batch column dict + row index.

        ``cols`` is ``{name: rb.column(name)}`` resolved once outside
        the row loop so the per-row ``get`` skips the schema-name lookup.
        """
        def _arrow_get(name: str) -> Any:
            col = cols.get(name)
            return col[i].as_py() if col is not None else None

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
        #     used by ``from_mapping`` callers that bypass the schema.
        request_value = get("request")
        if isinstance(request_value, Mapping):
            request = HTTPRequest._from_get(request_value.get, normalize=normalize)
        elif get("request_url") is not None or get("request_method") is not None:
            def _request_get(name: str) -> Any:
                return get(f"request_{name}")
            request = HTTPRequest._from_get(_request_get, normalize=normalize)
        else:
            request = HTTPRequest._from_get(get, normalize=normalize)

        headers = _map_as_str_dict(get("headers"))

        body_bytes = get("body")
        pre_media = _media_type_from_headers(headers)
        buffer = _coerce_buffer(body_bytes, media_type=pre_media)

        if normalize:
            headers = HTTPHeaders.from_(headers).normalized(body=buffer, is_request=False)


        receiver_value = get("receiver")
        receiver = _coerce_userinfo(receiver_value) if receiver_value is not None else None

        return cls(
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

    # No URL-routable scheme — the response's URL is sourced from the
    # bound :class:`HTTPRequest`, not the class-level scheme
    # registry. Keeping it ``None`` also bypasses the
    # :class:`URLBased`-on-import-time registration step.
    scheme = None

    # No additional storage; Response carries the response payload in
    # ``self.__dict__`` and :class:`IO` brings its own slot layout.
    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any) -> "HTTPResponse":
        # Bypass :meth:`IO.__new__`'s storage-parent mint — that path
        # would synthesise an empty :class:`~yggdrasil.io.memory.Memory`
        # parent which :meth:`__init__` immediately overwrites with
        # ``self.buffer``. Allocating directly off :class:`object`
        # also sidesteps :meth:`Singleton.__new__`'s cache machinery,
        # which doesn't apply (responses are unique per request).
        instance = object.__new__(cls)
        return instance

    def __init__(
        self,
        request: HTTPRequest,
        status_code: int,
        headers: MutableMapping[str, str],
        tags: MutableMapping[str, str],
        buffer: Any,
        received_at: dt.datetime,
        receiver: "Optional[UserInfo]" = None,
    ) -> None:
        # :class:`Response`'s ``super().__init__()`` chains through the
        # full IO MRO (:class:`Singleton` → :class:`URLBased` →
        # :class:`Tabular` → :class:`Disposable`) so cursor /
        # disposable / tabular_parent state lands with default values
        # before we wire the buffer in.
        super().__init__()
        self.request = request
        self.status_code = int(status_code)
        self.headers: HTTPHeaders = HTTPHeaders.from_(headers)
        self.tags = _string_dict(tags)
        self.received_at = any_to_datetime(received_at)
        self.buffer = _coerce_buffer(buffer)
        self._receiver: UserInfo | None = (
            _coerce_userinfo(receiver) if receiver is not None else _default_sender()
        )
        media = _ensure_media_headers(self.headers, self.buffer)
        self._session: "HTTPSession | None" = None
        self._cache: dict[str, Any] = {}
        self._cache_token: tuple = ()
        # Wire the IO cursor's parent to the response buffer so every
        # byte primitive (:meth:`IO.read` / :meth:`IO.seek` /
        # :meth:`IO.read_mv` / :meth:`IO.write_mv`, plus the
        # :class:`io.IOBase` surface :func:`pa.input_stream` reaches
        # for) delegates to ``self.buffer``. ``_owns_parent=False``
        # keeps the buffer alive across cursor close — the buffer's
        # lifetime is the response's, not the cursor's.
        self._parent = self.buffer
        self._owns_parent = False
        self._url = request.url
        # Wire-level connection-lifecycle slots — populated by
        # :meth:`from_wire` when the response is built straight off a
        # live socket. Default ``None`` keeps the urllib3-shaped
        # surface inert on parsed-from-record responses.
        self._raw: Optional[http.client.HTTPResponse] = None
        self._connection: Optional[http.client.HTTPConnection] = None
        self._pool_key: Optional[Tuple[str, str, int]] = None
        self._released: bool = True

    # ------------------------------------------------------------------
    # urllib3-shaped surface — keeps warehouse-style streaming consumers
    # (``pa.input_stream(resp)``, ``resp.drain_conn()``,
    # ``resp.release_conn()``, ``resp.status``) working without a
    # PoolManager intermediary. The :class:`HTTPSession` plays the pool
    # role directly — when it fills in ``_connection`` / ``_pool_key``
    # on :meth:`from_wire`, the release callback wires straight back
    # into ``session._release_connection``.
    # ------------------------------------------------------------------

    @property
    def status(self) -> int:
        """urllib3-shaped alias for :attr:`status_code`."""
        return self.status_code

    def release_conn(self) -> None:
        """Return the underlying connection to the pool, once.

        No-op on responses built from records / cache hits / test
        fixtures (``_connection`` is ``None``). On a wire-fetched
        response, defers to ``session._release_connection`` so the
        socket lands back in the per-host idle cache.
        """
        if self._released:
            return
        self._released = True
        session = self._session
        if (
            session is not None
            and self._connection is not None
            and self._pool_key is not None
            and hasattr(session, "_release_connection")
        ):
            session._release_connection(self._pool_key, self._connection)
        elif self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass

    def drain_conn(self) -> None:
        """Pull any remaining bytes from the source so the socket is reusable."""
        buffer = self.buffer
        if not isinstance(buffer, MemoryStream):
            return
        if buffer.eof or self._released:
            return
        try:
            buffer._pull_to_eof()
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self._raw is not None:
                self._raw.close()
        finally:
            self.release_conn()
            # Chain to :class:`io.IOBase` so ``closed`` flips to ``True``
            # — pyarrow's ``input_stream`` wrapper checks this after
            # streaming reads.
            try:
                super().close()
            except Exception:
                pass

    def stream(self, amt: int = 65536) -> Iterator[bytes]:
        """Iterate body bytes in ``amt``-sized chunks (urllib3-shaped).

        Drives the underlying :class:`MemoryStream` through its
        internal cursor so the iteration is forward-only and releases
        the connection once EOF lands.
        """
        buffer = self.buffer
        try:
            while True:
                chunk = bytes(buffer.read_mv(amt, cursor=True))
                if not chunk:
                    return
                yield chunk
        finally:
            self.release_conn()

    @property
    def data(self) -> bytes:
        """All body bytes from offset 0 (urllib3-shaped ``.data``)."""
        buffer = self.buffer
        if isinstance(buffer, MemoryStream) and not buffer.eof:
            buffer._pull_to_eof()
            self.release_conn()
        return bytes(buffer.read_mv(-1, 0))

    @property
    def path(self) -> "HTTPPath":
        """:class:`HTTPPath` view of the request URL.

        Bound to the response's attached :class:`HTTPSession` (when
        present) so the HTTPPath reuses the same connection pool.
        Useful for re-fetching the resource, issuing a HEAD probe via
        :meth:`HTTPPath.stat`, or doing a follow-up PUT / DELETE.
        """
        from .path import HTTPPath
        from .session import HTTPSession

        sess = self._session if isinstance(self._session, HTTPSession) else None
        return HTTPPath(url=self.request.url, session=sess)

    # ------------------------------------------------------------------
    # Construction shims
    # ------------------------------------------------------------------

    @classmethod
    def from_wire(
        cls,
        request: "HTTPRequest",
        raw: http.client.HTTPResponse,
        *,
        session: "Optional[HTTPSession]" = None,
        connection: Optional[http.client.HTTPConnection] = None,
        pool_key: Optional[Tuple[str, str, int]] = None,
        decode_content: bool = True,
        preload_content: bool = False,
        tags: Optional[Mapping[str, str]] = None,
        received_at: Optional[dt.datetime] = None,
    ) -> "HTTPResponse":
        """Build a response straight off a live ``http.client.HTTPResponse``.

        Replaces the urllib3-shim pool ``HTTPResponse`` that used to live
        in ``yggdrasil.http_._pool``
        intermediary — :class:`HTTPSession` builds the wire response
        directly through this factory. The body is wrapped in a
        :class:`MemoryStream` whose source is the
        the raw socket; the connection
        / pool-key pair lets :meth:`release_conn` return the socket
        to the session's idle cache after drain.
        """
        # Collect headers first — the Content-Encoding feeds the
        # decoder, and the high-level :class:`Response` constructor
        # uses them to stamp Content-Type / Content-Length on the
        # buffer.
        response_headers: dict[str, str] = {}
        for k, v in raw.getheaders():
            existing = response_headers.get(k)
            response_headers[k] = f"{existing}, {v}" if existing is not None else v

        encoding = response_headers.get("Content-Encoding") if decode_content else None
        from yggdrasil.http_.stream import HTTPStream
        buffer = HTTPStream(
            source=raw.read,
            content_encoding=encoding,
            request=request,
            session=session,
        )

        pre_media = _media_type_from_headers(response_headers)
        if pre_media is not None and buffer.media_type is None:
            buffer.media_type = pre_media

        resp = cls(
            request=request,
            status_code=raw.status,
            headers=response_headers,
            tags=dict(tags) if tags is not None else {},
            buffer=buffer,
            received_at=received_at or dt.datetime.now(dt.timezone.utc),
        )

        # Plug the connection-lifecycle metadata onto the response so
        # ``release_conn`` lands in the right ``session._release_connection``
        # slot. ``_released`` flips to False so the first release call
        # actually returns the socket; ``_session`` attaches eagerly
        # via :meth:`Response.attach_session` to keep the back-reference
        # consistent with the rest of the pipeline.
        resp._raw = raw
        resp._connection = connection
        resp._pool_key = pool_key
        resp._released = connection is None
        if session is not None:
            resp.attach_session(session)

        if preload_content:
            buffer.read_mv(-1, 0)
            resp.release_conn()
            actual = str(buffer.size)
            if resp.headers.get("Content-Length") != actual:
                resp.headers["Content-Length"] = actual

        return resp

    @classmethod
    def from_pool(
        cls,
        request: "HTTPRequest",
        response: Any,
        tags: Optional[Mapping[str, str]],
        received_at: dt.datetime,
        *,
        stream: bool = True,
        amt: int = 512 * 1024,
        release_conn: bool = True,
    ) -> "HTTPResponse":
        """Compat shim — promote a duck-typed pool response to high-level.

        Used by test fixtures and any legacy caller that still hands a
        urllib3-shaped object (with ``.status`` / ``.headers`` /
        ``.read`` / ``.stream`` / ``.release_conn``). When the input
        is already a high-level :class:`HTTPResponse` it round-trips
        through itself; otherwise the body is drained into a fresh
        :class:`MemoryStream` and the new high-level response wraps
        that buffer.
        """
        if isinstance(response, HTTPResponse):
            # Already the right shape — drain through the response's
            # own ``.stream`` / ``.read`` to ensure EOF lands, then
            # return as-is. Caller's ``release_conn`` knob is honored
            # via the inner ``release_conn`` call.
            if stream:
                for _ in response.stream(amt=amt):
                    pass
            else:
                response.read()
            if release_conn:
                response.release_conn()
            return response

        headers = dict(response.headers)
        pre_media = _media_type_from_headers(headers)

        try:
            if stream:
                body_bytes = b"".join(response.stream(amt=amt))
            else:
                body_bytes = response.read()
        finally:
            if release_conn:
                response.release_conn()

        buffer = MemoryStream(source=body_bytes)
        buffer.read_mv(-1, 0)
        if pre_media is not None and buffer.media_type is None:
            buffer.media_type = pre_media

        return cls(
            request=request,
            status_code=response.status,
            headers=headers,
            buffer=buffer,
            tags=tags,
            received_at=received_at,
        )





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




