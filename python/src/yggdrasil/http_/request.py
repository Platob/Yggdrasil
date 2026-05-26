# yggdrasil.io.request
from __future__ import annotations

import base64
import datetime as dt
import json as json_module
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterable, Iterator, Literal, Mapping, MutableMapping, \
    Optional

import pyarrow as pa

from yggdrasil.data.cast import any_to_datetime
from yggdrasil.enums import GZIP, Codec, MimeType
from yggdrasil.enums import MediaType, MimeTypes
from yggdrasil.dataclasses.dataclass import get_from_dict
from yggdrasil.environ.userinfo import USERINFO_STRUCT, UserInfo
from yggdrasil.http_.authorization.base import Authorization
from yggdrasil.io.base import IO
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.headers import Headers
from yggdrasil.io.holder import Holder
from yggdrasil.path.memory import Memory
from yggdrasil.url import URL

if TYPE_CHECKING:
    from yggdrasil.http_.response import HTTPResponse
    from yggdrasil.http_.cache_config import CacheConfig
    from yggdrasil.http_.send_config import SendConfig
    from yggdrasil.http_.session import HTTPSession


__all__ = [
    "HTTPRequest",
    "PreparedRequest",
    "REQUEST_SCHEMA",
    "REQUEST_URL_STRUCT",
]


from yggdrasil.http_.schemas import REQUEST_SCHEMA, REQUEST_URL_STRUCT


# Struct type that matches :data:`REQUEST_SCHEMA.to_arrow_schema()` column-for-column.
# Building the single-row :class:`pa.RecordBatch` via a struct array and
# :meth:`pa.RecordBatch.from_struct_array` is ~40% cheaper than the
# per-column ``pa.array([value], type=…)`` loop, because pyarrow walks
# the struct fields in C++ instead of dispatching N python-side
# array constructors. Precomputed once so :meth:`to_arrow_batch`
# doesn't rebuild it per call.
_REQUEST_ARROW_STRUCT_TYPE: pa.StructType = pa.struct(REQUEST_SCHEMA.to_arrow_schema())


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
    """Signed-int64 xxh3_64 digest of *text*."""
    if not text:
        return 0
    import xxhash
    u = xxhash.xxh3_64(text.encode("utf-8")).intdigest()
    return u if u < 2**63 else u - 2**64


def _build_url_struct(url: URL) -> dict[str, Any]:
    """Build the URL struct value (matches :data:`URL_STRUCT`)."""
    return url.to_struct_dict()


def _default_sender() -> UserInfo | None:
    """Resolve :class:`UserInfo` for the current process — never raises."""
    try:
        return UserInfo.current()
    except Exception:
        return None


def _coerce_request_buffer(obj: Any) -> Optional[Holder]:
    """Normalize a request buffer-shaped input into a :class:`Holder`.

    Returns ``None`` when *obj* is ``None`` (no body). Accepts an
    existing :class:`Holder` (passed through), an :class:`IO` cursor
    (the holder is borrowed), or anything :meth:`Holder.from_`
    understands (bytes, file-like, path-shaped).
    """
    if obj is None:
        return None
    if isinstance(obj, Holder):
        return obj
    if isinstance(obj, IO):
        return obj._parent
    return Holder.from_(obj)


def _coerce_userinfo(value: Any) -> UserInfo | None:
    """Best-effort cast of ``value`` to :class:`UserInfo`."""
    if value is None:
        return None
    if isinstance(value, UserInfo):
        return value
    if isinstance(value, Mapping):
        return UserInfo.from_struct_dict(value)
    return None


def _fold_cache_into_send_config(
    send_config: "SendConfig | None",
    local_cache_config: "CacheConfig | None",
    remote_cache_config: "CacheConfig | None",
) -> "SendConfig | None":
    """Fold convenience ``local_cache_config`` / ``remote_cache_config``
    kwargs into a :class:`SendConfig`, creating one when needed.

    Used by :meth:`PreparedRequest.prepare` and
    :meth:`PreparedRequest.copy` so callers can still pass per-request
    cache configs as positional sugar while the canonical storage is
    :attr:`PreparedRequest.send_config`.
    """
    if local_cache_config is None and remote_cache_config is None:
        return send_config
    import dataclasses
    from yggdrasil.http_.cache_config import CacheConfig as _CacheConfig
    from yggdrasil.http_.send_config import SendConfig as _SendConfig
    lc = _CacheConfig.from_(local_cache_config) if local_cache_config is not None else None
    rc = _CacheConfig.from_(remote_cache_config) if remote_cache_config is not None else None
    if send_config is not None:
        overrides: dict[str, Any] = {}
        if lc is not None:
            overrides["local_cache"] = lc
        if rc is not None:
            overrides["remote_cache"] = rc
        return dataclasses.replace(send_config, **overrides) if overrides else send_config
    return _SendConfig(local_cache=lc, remote_cache=rc)


class HTTPRequest:
    """Immutable-ish request descriptor — fields are normalized in __init__.

    ``_session`` is a transient back-reference used by
    :meth:`attach_session`; it's deliberately excluded from
    :meth:`__getstate__` so a request stays portable when pickled to
    Spark executors and re-binds via ``attach_session`` once it lands
    on the worker. Per-request request/response hooks live on the
    owning :class:`Session` (``_prepare_request`` / ``_prepare_response``).
    """

    # Instance attributes that don't survive pickling — excluded by
    # ``__getstate__`` and reset by ``__setstate__``. Subclasses extend.
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset(
        {"_session", "_cache", "_cache_token"}
    )

    def __init__(
        self,
        method: str,
        url: URL,
        headers: MutableMapping[str, str],
        tags: MutableMapping[str, str],
        buffer: Optional[Holder],
        sent_at: Optional[dt.datetime],
        sender: Optional[UserInfo] = None,
        auth: Optional[Authorization] = None,
        send_config: Optional["SendConfig"] = None,
        session: "Session | None" = None,
    ) -> None:
        self.method = method or "GET"
        self.url = URL.from_(url)
        self.headers: Headers = Headers.from_(headers)
        self.tags = _string_dict(tags)
        self.sent_at = (
            any_to_datetime(sent_at) if sent_at
            else dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)
        )
        self.buffer: Optional[Holder] = _coerce_request_buffer(buffer)
        from yggdrasil.http_.send_config import SendConfig as _SendConfig
        self.send_config: SendConfig | None = _SendConfig.from_(send_config, default=None)
        self._sender: UserInfo | None = (
            _coerce_userinfo(sender) if sender is not None else _default_sender()
        )
        self._auth: Authorization | None = auth
        self._session: "Session | None" = session
        # Memoization for the deterministic projection — hashes,
        # url-struct, arrow_values dict. Invalidated when
        # :meth:`_state_token` shifts (method, URL identity, header
        # / buffer ``id`` + ``len`` + byte length changed) or
        # :meth:`_invalidate_cache` was called by an in-place setter.
        self._cache: dict[str, Any] = {}
        self._cache_token: tuple = ()

    @property
    def sender(self) -> UserInfo | None:
        """:class:`UserInfo` snapshot for the side issuing this request.

        Defaults to ``UserInfo.current()`` at construction time. Use
        :meth:`with_sender` to swap it for a different snapshot
        (returns a new request — :class:`PreparedRequest` is
        immutable-ish, so the underlying field is read-only).
        """
        return self._sender

    @property
    def send_config_or_default(self) -> "SendConfig":
        """Return :attr:`send_config` or the shared default singleton."""
        sc = self.send_config
        if sc is not None:
            return sc
        from yggdrasil.http_.send_config import SendConfig as _SendConfig
        return _SendConfig.default()

    @property
    def local_cache_config(self) -> "CacheConfig | None":
        """Per-request local :class:`CacheConfig`, delegated to :attr:`send_config`."""
        sc = self.send_config
        if sc is None:
            return None
        config = sc.local_cache
        if config is None or not config.cache_enabled:
            return None
        return config

    @property
    def remote_cache_config(self) -> "CacheConfig | None":
        """Per-request remote :class:`CacheConfig`, delegated to :attr:`send_config`."""
        sc = self.send_config
        return sc.remote_cache if sc is not None else None

    def with_sender(self, sender: UserInfo | Mapping[str, Any] | None) -> "HTTPRequest":
        """Return a copy of this request with :attr:`sender` replaced.

        Accepts a :class:`UserInfo`, a struct-shaped mapping (matching
        :data:`USERINFO_STRUCT`), or ``None`` to clear the snapshot.
        """
        return self.copy(sender=sender)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self.method} {self.url.to_string()!r}>"

    def __str__(self) -> str:
        return self.url.to_string()

    def __getstate__(self) -> dict[str, Any]:
        # Generic: every ``__dict__`` entry except the transient set
        # (``_session``) survives. Re-bind the session on the worker
        # via :meth:`attach_session`.
        # ``send_config`` is omitted when it's ``None`` or the default
        # singleton — no point shipping bytes that the receiver would
        # reconstruct identically via :attr:`send_config_or_default`.
        from yggdrasil.http_.send_config import SendConfig as _SC
        state = {
            k: v for k, v in self.__dict__.items()
            if k not in self._TRANSIENT_STATE_ATTRS
        }
        sc = state.get("send_config")
        if sc is None or sc is _SC.default():
            state.pop("send_config", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._session = None
        self._cache = {}
        self._cache_token = ()
        if "send_config" not in self.__dict__:
            self.send_config = None
        # Re-coerce in case an old pickle carried ``headers`` as a
        # plain dict — :class:`Headers.from_` is a no-op on an
        # already-built instance, so the live path stays cheap.
        self.headers = Headers.from_(self.headers)

    # ------------------------------------------------------------------
    # Memoization — cheap fingerprint check, refresh on change
    # ------------------------------------------------------------------

    def _state_token(self) -> tuple:
        """Cheap fingerprint of the inputs that drive every cached
        derivation (hashes, url struct, arrow_values).

        We trust each inner object to track its own state:
        :class:`URL` is value-equal so it identifies itself,
        :class:`Headers` exposes a monotonic ``version`` (one int
        compare beats walking the dict), and :class:`Holder` is
        identified by ``id`` + size. No byte-length or stat shadows
        on :class:`PreparedRequest` itself — the inner objects are
        the source of truth.
        """
        headers = self.headers
        buffer = self.buffer
        return (
            self.method,
            self.url,
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
            # Re-fingerprint after compute — keep the cache token in
            # sync if ``compute`` had side effects on tracked state
            # (mirrors :meth:`Response._cached`). Side-effect-free
            # computes get the same token back and this is a no-op.
            post = self._state_token()
            if post != token:
                self._cache_token = post
            self._cache[name] = cached
        return cached

    def _invalidate_cache(self) -> None:
        """Drop every memoized derivation.

        :class:`Headers` already bumps its own ``version`` on every
        mutation, so header rewrites flow through :meth:`_state_token`
        without help. This entry point stays for the rare case where
        callers swap :attr:`headers` for a fresh value-equal
        :class:`Headers` (``id`` flips, contents identical) or mutate
        a slot the fingerprint doesn't watch — clearing the memo
        here forces a clean recompute on the next access.
        """
        self._cache.clear()
        self._cache_token = ()

    # ------------------------------------------------------------------
    # Session attachment — used by Spark broadcast on the worker side
    # ------------------------------------------------------------------

    def attach_session(self, session: "HTTPSession") -> "HTTPRequest":
        self._session = session
        return self

    def detach_session(self) -> "HTTPRequest":
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
    ) -> "HTTPResponse":
        return self._send(config, **kwargs)

    def _send(
        self,
        config: "SendConfig | Mapping[str, Any] | None" = None,
        **kwargs: Any,
    ) -> "HTTPResponse":
        if self._session is None:
            # Orphan request — lazy-build a default session for HTTP(S)
            # URLs so callers can ``PreparedRequest.prepare(...).send()``
            # without first instantiating a Session. The session is
            # cached on the request via :meth:`attach_session` so
            # subsequent calls reuse the same connection pool.
            if not self.url.is_http:
                raise RuntimeError(
                    f"{type(self).__name__}.send requires an attached "
                    f"session for non-HTTP URL {self.url.to_string()!r} — "
                    "call request.attach_session(session) first, or use "
                    "session.send(request) directly."
                )
            from yggdrasil.http_ import HTTPSession
            self.attach_session(HTTPSession())
        effective = config if config is not None else self.send_config
        return self.session.send(self, effective, **kwargs)

    @classmethod
    def from_(
        cls,
        obj: Any,
        *,
        normalize: bool = True,
    ) -> "HTTPRequest":
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
    ) -> "HTTPRequest":
        method = cls._parse_method(obj)
        url = cls._parse_url(obj, normalize=normalize)
        headers = cls._parse_headers(obj)
        tags = cls._parse_tags(obj)
        buffer = cls._parse_buffer(obj)
        sent_at = cls._parse_sent_at_timestamp(obj)
        sender = cls._parse_sender(obj, normalize=normalize)

        if cls is PreparedRequest:
            if url.is_http:
                from yggdrasil.http_ import HTTPRequest

                return HTTPRequest(
                    method=method,
                    url=url,
                    headers=headers,
                    tags=tags,
                    buffer=buffer,
                    sent_at=sent_at,
                    sender=sender,
                )

        return cls(
            method=method,
            url=url,
            headers=headers,
            tags=tags,
            buffer=buffer,
            sent_at=sent_at,
            sender=sender,
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
    def _parse_sender(
        obj: Mapping[str, Any],
        *,
        normalize: bool,
    ) -> UserInfo | None:
        # Accept a pre-built UserInfo, a struct dict matching
        # :data:`USERINFO_STRUCT`, or nothing — falling back to
        # ``UserInfo.current()`` when the caller didn't supply one.
        value = get_from_dict(obj, keys=("sender",), prefix=None)
        if value is MISSING or value in (None, ""):
            return _default_sender()
        if isinstance(value, UserInfo):
            return value
        if isinstance(value, Mapping):
            return UserInfo.from_struct_dict(value)
        return _default_sender()

    @staticmethod
    def _parse_tags(obj: Mapping[str, Any]) -> dict[str, str]:
        tags = get_from_dict(obj, keys=("tags",), prefix=None)
        return _string_dict(tags if isinstance(tags, Mapping) else None)

    @staticmethod
    def _parse_buffer(obj: Mapping[str, Any]) -> Optional[Holder]:
        buffer = get_from_dict(obj, keys=("buffer", "body", "content", "data"), prefix=None)
        if buffer is MISSING or buffer is None:
            return None
        return _coerce_request_buffer(buffer)

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
        *,
        json: Optional[Any] = None,
        normalize: bool = True,
        compress_threshold: Optional[int] = 4 * 1024 * 1024,
        compress_codec: Optional[Codec] = GZIP,
        sender: Optional[UserInfo | Mapping[str, Any]] = None,
        auth: Optional[Authorization] = None,
        send_config: "SendConfig | None" = None,
        session: "Session | None" = None,
    ) -> "HTTPRequest":
        parsed_url = URL.from_(url, normalize=normalize)
        out_headers: dict[str, str] = _string_dict(headers)

        request_body: Optional[Holder] = None
        if body is not None:
            request_body = Holder.from_(body)
        elif json is not None:
            request_body = Memory(binary=json_module.dumps(json).encode("utf-8"))
            out_headers["Content-Type"] = MimeTypes.JSON.value

            if (
                compress_threshold
                and compress_codec is not None
                and request_body.size > compress_threshold
            ):
                with BytesIO(holder=request_body, owns_holder=False) as src:
                    compressed = compress_codec.compress(src)
                request_body = compressed._parent
                out_headers["Content-Encoding"] = compress_codec.name

        if request_body is not None:
            out_headers["Content-Length"] = str(request_body.size)

        out_class = cls

        if cls is PreparedRequest:
            if parsed_url.is_http:
                from yggdrasil.http_ import HTTPRequest
                out_class = HTTPRequest

        return out_class(
            method=str(method),
            url=parsed_url,
            headers=(
                Headers.from_(out_headers).normalized(is_request=True, body=request_body)
                if normalize else out_headers
            ),
            tags=_string_dict(tags),
            buffer=request_body,
            sent_at=0,
            sender=_coerce_userinfo(sender),
            auth=auth,
            send_config=send_config,
            session=session
        )

    def copy(
        self,
        *,
        method: str | None = None,
        url: URL | str | None = None,
        headers: Optional[Mapping[str, str]] = None,
        buffer: Optional[Holder] = ...,
        tags: Optional[Mapping[str, str]] = None,
        sent_at: int | None = None,
        sender: Optional[UserInfo] = ...,
        auth: Optional[Authorization] = ...,
        send_config: Optional["SendConfig"] = ...,
        local_cache_config: Optional["CacheConfig"] = ...,
        remote_cache_config: Optional["CacheConfig"] = ...,
        normalize: bool = True,
        copy_buffer: bool = False,
    ) -> "HTTPRequest":
        new_url = self.url if url is None else URL.from_(url, normalize=normalize)
        # Clone the existing :class:`Headers` directly — ``dict(self.headers)``
        # would iterate via ``__getitem__`` only for ``Headers.from_`` to
        # immediately rebuild a :class:`Headers` from that dict. Going
        # ``Headers -> Headers`` short-circuits the round trip via the
        # ``isinstance(data, Headers)`` branch in ``Headers.__init__``,
        # which lifts the internal dict in one C-level shallow copy.
        new_headers = Headers(self.headers) if headers is None else _string_dict(headers)

        if buffer is ...:
            new_buffer = self.buffer
            if copy_buffer and new_buffer is not None:
                new_buffer = Memory(binary=new_buffer.to_bytes())
        else:
            new_buffer = buffer

        new_tags = dict(self.tags) if tags is None else _string_dict(tags)
        new_sender = self.sender if sender is ... else _coerce_userinfo(sender)
        new_auth = self._auth if auth is ... else auth

        base_send_config = self.send_config if send_config is ... else send_config
        effective_send_config = _fold_cache_into_send_config(
            base_send_config,
            self.local_cache_config if local_cache_config is ... else local_cache_config,
            self.remote_cache_config if remote_cache_config is ... else remote_cache_config,
        ) if local_cache_config is not ... or remote_cache_config is not ... else base_send_config

        return self.__class__(
            method=self.method if method is None else str(method),
            url=new_url,
            headers=new_headers,
            tags=new_tags,
            buffer=new_buffer,
            sent_at=self.sent_at if sent_at is None else any_to_datetime(sent_at),
            sender=new_sender,
            auth=new_auth,
            send_config=effective_send_config,
        )

    @property
    def body(self) -> Optional[Holder]:
        return self.buffer

    @property
    def holder(self) -> Optional[Holder]:
        """The :class:`Holder` backing the request body — alias for :attr:`buffer`."""
        return self.buffer

    def open(self, mode: str = "rb+") -> Optional[IO]:
        """Open a fresh :class:`IO` cursor over the request's holder.

        Returns ``None`` when the request has no body. Dispatches to
        the format-specific leaf via the holder's media type. The
        returned cursor is non-owning.
        """
        if self.buffer is None:
            return None
        return self.buffer.open(mode=mode, owns_holder=False)

    @property
    def content_length(self) -> int:
        return self.buffer.size if self.buffer is not None else 0

    @property
    def authorization(self) -> Optional[str]:
        return self.headers.get("Authorization") if self.headers else None

    @authorization.setter
    def authorization(self, value: "Authorization | str | None"):
        # Forgiving on input: accept either a static header value
        # (``str`` / ``None``) or an :class:`Authorization` handler.
        # Handlers are invoked lazily by :meth:`prepare_authorization`
        # so a single instance can mint fresh tokens for every send.
        if isinstance(value, Authorization):
            self._auth = value
            return
        if self.headers is None:
            self.headers = Headers()
        if value is None:
            self.headers.pop("Authorization", None)
        else:
            self.headers["Authorization"] = str(value)
        self._auth = None
        self._invalidate_cache()

    @property
    def auth(self) -> Optional[Authorization]:
        """The :class:`Authorization` handler bound to this request, if any.

        When set, :meth:`prepare_authorization` resolves it lazily into
        the ``Authorization`` header just before the request is sent —
        every send picks up a freshly-minted header value.
        """
        return self._auth

    @auth.setter
    def auth(self, value: Optional[Authorization]) -> None:
        if value is not None and not isinstance(value, Authorization):
            raise TypeError(
                f"auth must be an Authorization instance or None; got "
                f"{type(value).__name__}. Use ``request.authorization = '<value>'`` "
                f"to set a static header string."
            )
        self._auth = value

    def prepare_authorization(self) -> "HTTPRequest":
        """Resolve the bound :class:`Authorization` handler into the header.

        No-op when no handler is bound. Calls ``handler.authorization``
        each time, so handlers that refresh internally (e.g. MSAL) emit
        the current token on every send.
        """
        if self._auth is None:
            return self
        value = self._auth.authorization
        if self.headers is None:
            self.headers = Headers()
        self.headers["Authorization"] = value
        return self

    @property
    def x_api_key(self) -> Optional[str]:
        return self.headers.get("X-API-Key") if self.headers else None

    @x_api_key.setter
    def x_api_key(self, value: Optional[str]):
        if self.headers is None:
            self.headers = Headers()
        if value is None:
            self.headers.pop("X-API-Key", None)
        else:
            self.headers["X-API-Key"] = str(value)
        self._invalidate_cache()

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
            self.headers = Headers()
        self.headers["Accept"] = value.mime_type.value
        if value.codec:
            self.headers["Accept-Encoding"] = value.codec.name
        else:
            self.headers.pop("Accept-Encoding", None)
        self._invalidate_cache()

    @property
    def sent_at_timestamp(self) -> int:
        return int(self.sent_at.timestamp() * 1_000_000)

    def partition_values(self) -> dict[str, Any]:
        """Hook for subclasses to define what feeds ``partition_key``.

        Returns the ordered mapping that gets fed to :attr:`partition_key`'s
        xxh3_64 digest. Override to bucket requests differently —
        e.g. by tenant id, by ``url.host`` only, by a fixed shard.
        Order is stable: the helper concatenates values in iteration
        order so two configs that disagree on key ordering hash to
        different partitions.

        Default: ``{"host": url.host, "path": url.path}`` — every
        call to the same endpoint shares one partition leaf.
        """
        return {
            "host": self.url.host or "",
            "path": self.url.path or "",
        }

    @property
    def partition_key(self) -> int:
        """xxh3_64 of the joined :meth:`partition_values` — int64 partition column."""
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
        return self._cached("body_hash", lambda: self.buffer.xxh3_int64())  # type: ignore[union-attr]

    @property
    def private_url_hash(self) -> int:
        """xxh3_64 of (method, URL) exactly as captured (userinfo + full query)."""
        return self._cached(
            "private_url_hash",
            lambda: _xxh3_int64_str(f"{self.method}\x00{self.url.to_string()}"),
        )

    @property
    def public_url_hash(self) -> int:
        """xxh3_64 of (method, URL) after ``anonymize('remove')`` — userinfo and
        sensitive query params dropped, so this hash is stable across the
        cache's anonymize step. Method is mixed in so verbs don't collide."""
        return self._cached(
            "public_url_hash",
            lambda: _xxh3_int64_str(
                f"{self.method}\x00{self.url.anonymize(mode='remove').to_string()}"
            ),
        )

    @property
    def hash(self) -> int:
        """xxh3_64 over (method, url, headers, body) — overall identity,
        including sensitive bits (userinfo, Authorization, …).

        Use :attr:`public_hash` for matches that should survive cache
        anonymization (drops userinfo / sensitive query params / sensitive
        headers).
        """
        return self._cached("hash", lambda: self._compute_identity_hash(anonymize=False))

    @property
    def public_hash(self) -> int:
        """xxh3_64 over the anonymize='remove' projection of
        (method, url, headers, body) — the cross-system identity that
        survives cache anonymization."""
        return self._cached("public_hash", lambda: self._compute_identity_hash(anonymize=True))

    def _compute_identity_hash(self, *, anonymize: bool = False) -> int:
        """Mix (method, url, headers, body) into one xxh3_64 digest.

        Each component arrives in its own pre-cached form:
        :class:`URL` caches ``to_string()``, :class:`Headers` caches
        :attr:`canonical_bytes`, :class:`Holder` caches
        :attr:`xxh3_64_digest`. Repeat calls — common at the cache
        layer where ``hash`` and ``public_hash`` both fire — pay one
        xxh3 mix over already-bytes inputs.
        """
        import xxhash

        if anonymize:
            url_str = self.url.anonymize(mode="remove").to_string()
            headers = self.headers.anonymized(mode="remove")
        else:
            url_str = self.url.to_string()
            headers = self.headers

        h = xxhash.xxh3_64()
        h.update(self.method.encode("utf-8"))
        h.update(b"\x00")
        h.update(url_str.encode("utf-8"))
        h.update(b"\x00")
        h.update(headers.canonical_bytes)
        if self.buffer is not None:
            h.update(self.buffer.xxh3_64_digest)
        u = h.intdigest()
        return u if u < 2**63 else u - 2**64

    def _arrow_value(self, key: str) -> Any:
        """Compute a single :data:`REQUEST_SCHEMA.to_arrow_schema()` column on demand.

        Used by :meth:`match_value` so the lookup pays for one column
        instead of materializing every value via :attr:`arrow_values`.
        Cached scalars (hashes, partition_key) hit the memoized
        properties; the heavyweight columns (``url`` struct, ``tags``,
        ``headers`` copy) are also memoized through :meth:`_cached` so
        repeated single-key lookups don't rebuild them.
        """
        if key == "hash":
            return self.hash
        if key == "public_hash":
            return self.public_hash
        if key == "method":
            return self.method
        if key == "url":
            return self._cached("url_struct", lambda: _build_url_struct(self.url))
        if key == "sender":
            sender = self.sender
            return sender.to_struct_dict() if sender is not None else None
        if key == "private_url_hash":
            return self.private_url_hash
        if key == "public_url_hash":
            return self.public_url_hash
        if key == "headers":
            return self._cached("headers_value", lambda: _string_dict(self.headers))
        if key == "tags":
            return self._cached("tags_value", self._build_tags_value)
        if key == "body":
            return self.buffer.to_bytes() if self.buffer is not None else None
        if key == "body_size":
            return self.body_size
        if key == "body_hash":
            return self.body_hash
        if key == "sent_at":
            return self.sent_at
        if key == "partition_key":
            return self.partition_key
        if key == "_pkl":
            # ``_pkl`` is a placeholder column populated externally by
            # the pickle serializer; the deterministic projection path
            # leaves it null so writers don't pay for a pickle dump
            # when the structured columns are enough.
            return None
        raise KeyError(key)

    def _build_tags_value(self) -> dict[str, str]:
        tags_v = dict(self.url.query_items())
        if self.tags:
            tags_v.update(_string_dict(self.tags))
        return tags_v

    @property
    def arrow_values(self) -> dict[str, Any]:
        return self._cached("arrow_values", self._build_arrow_values)

    def _build_arrow_values(self) -> dict[str, Any]:
        return {name: self._arrow_value(name) for name in REQUEST_SCHEMA.to_arrow_schema().names}

    def match_value(self, key: str) -> Any:
        # Support dotted paths (``url.path``, ``headers.content_type``)
        # alongside the flat top-level field names.
        if "." in key:
            head, _, tail = key.partition(".")
            try:
                container = self._arrow_value(head)
            except KeyError:
                container = None
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
                f"Must be within: {REQUEST_SCHEMA.to_arrow_schema().names!r}"
            )

        try:
            return self._arrow_value(key)
        except KeyError:
            pass
        if hasattr(self, key):
            return getattr(self, key)
        # Accept the flattened ``request_<col>`` form too — same
        # column, different spelling. The send_config validator
        # already lets callers write either shape, so the lookup
        # path needs to round-trip both without forcing the cache
        # caller to strip the prefix before each match_value().
        if key.startswith("request_"):
            tail = key[len("request_"):]
            try:
                return self._arrow_value(tail)
            except KeyError:
                pass
            if hasattr(self, tail):
                return getattr(self, tail)
        raise ValueError(
            f"Unsupported request match key: {key!r}. "
            f"Must be within: {REQUEST_SCHEMA.to_arrow_schema().names!r}"
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
            elif isinstance(v, (IO, Holder)):
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
    ) -> "HTTPRequest":
        if not headers:
            return self

        next_headers: "Headers | Mapping[str, str]" = headers
        if normalize:
            next_headers = Headers.from_(headers).normalized(
                is_request=True,
                anonymize=False,
                add_missing=False,
            )

        if not self.headers:
            self.headers = Headers.from_(next_headers)
        else:
            self.headers.update(next_headers)

        self._invalidate_cache()
        return self

    def update_tags(
        self,
        tags: MutableMapping[str, str],
    ) -> "HTTPRequest":
        if not tags:
            return self

        if not self.tags:
            self.tags = _string_dict(tags)
        else:
            self.tags.update(_string_dict(tags))

        self._invalidate_cache()
        return self

    def anonymize(self, mode: Literal["remove", "redact"] = "remove") -> "HTTPRequest":
        if not mode:
            return self

        return self.copy(
            headers=self.headers.normalized(
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

        # One C++-side struct walk beats N python-side ``pa.array(...)``
        # calls — pyarrow infers the per-child arrays in a single pass
        # from the struct type. The schema is reattached afterwards so
        # the returned batch carries the same field metadata (comments,
        # nullability, tag annotations) callers downstream rely on.
        return self.values_to_arrow_batch([self])

    @classmethod
    def values_to_arrow_batch(
        cls,
        requests: "Iterable[PreparedRequest]",
    ) -> pa.RecordBatch:
        """Build one :class:`pa.RecordBatch` from N requests in a single C++ pass.

        Counterpart to :meth:`Response.values_to_arrow_batch` — same
        rationale: collect ``arrow_values`` once per request, hand the
        list of dicts to pyarrow, get back one ``RecordBatch`` with N
        rows. Replaces the
        ``pa.Table.from_batches([r.to_arrow_batch(...) for r in N])``
        shape with a single C++ struct walk; at 64 rows this is ~30x
        faster than per-row builds plus a downstream concat.
        """
        values = [r.arrow_values for r in requests]
        struct_array = pa.array(values, type=_REQUEST_ARROW_STRUCT_TYPE)
        batch = pa.RecordBatch.from_struct_array(struct_array)
        if batch.schema is not REQUEST_SCHEMA.to_arrow_schema():
            batch = pa.RecordBatch.from_arrays(
                batch.columns, schema=REQUEST_SCHEMA.to_arrow_schema(),
            )
        return batch

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

        req_cols = [f.name for f in REQUEST_SCHEMA.to_arrow_schema()]

        for rb in _iter_batches(batch):
            available_set = set(rb.schema.names)
            picks = [n for n in req_cols if n in available_set]
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
        normalize: bool = True,
    ) -> "HTTPRequest":
        """Build a :class:`PreparedRequest` from a row-shaped mapping."""
        return cls._from_get(record.get, normalize=normalize)

    @classmethod
    def _from_arrow_row(
        cls,
        cols: dict[str, Any],
        i: int,
        *,
        normalize: bool = True,
    ) -> "HTTPRequest":
        """Build one :class:`PreparedRequest` from a per-batch column dict + row index."""
        def _arrow_get(name: str) -> Any:
            col = cols.get(name)
            return col[i].as_py() if col is not None else None

        return cls._from_get(_arrow_get, normalize=normalize)

    @classmethod
    def _from_get(
        cls,
        get: "Callable[[str], Any]",
        *,
        normalize: bool = True,
    ) -> "HTTPRequest":
        # Single source of truth for "named-getter → PreparedRequest" —
        # used by both the Arrow-batch path and the Mapping path.
        url_value = get("url")
        headers_value = get("headers")
        sent_at_value = get("sent_at")

        return cls.from_mapping(
            {
                "method":   get("method") or "GET",
                "url":      url_value,
                "sender":   get("sender"),
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


# Backwards-compat alias — the rename from ``PreparedRequest`` to
# ``HTTPRequest`` (commit 585c977) left stale ``cls is PreparedRequest``
# guards inside :meth:`from_mapping` and :meth:`prepare`. Rather than
# papering over them with try/except, re-export the alias so the
# existing guards resolve, subclass checks work, and downstream code
# that still imports the old name keeps running.
PreparedRequest = HTTPRequest

