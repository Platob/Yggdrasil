from __future__ import annotations

import datetime as dt
import http.client
import zlib
from typing import TYPE_CHECKING, Any, Iterator, Mapping, MutableMapping, Optional, Tuple

from yggdrasil.io.holder import IO
from yggdrasil.io.memory_stream import MemoryStream
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response, _media_type_from_headers

from .headers import HTTPHeaderDict

if TYPE_CHECKING:
    from yggdrasil.environ.userinfo import UserInfo

    from .path import HTTPPath
    from .session import HTTPSession


__all__ = [
    "HTTPResponse",
    "_DecodingReader",
]


class _DecodingReader:
    """Wraps a chunked source iterator and decodes gzip/deflate on the fly.

    The :class:`MemoryStream` backing every :class:`HTTPResponse` body
    binds this reader's ``.read`` callable as its source. Each pull
    feeds the next decoded chunk into the stream — no separate buffer,
    no second copy.
    """

    def __init__(self, raw_read, content_encoding: Optional[str]) -> None:
        self._raw_read = raw_read
        self._encoding = (content_encoding or "").lower()
        self._decoder: Any = None
        if self._encoding in ("gzip", "x-gzip"):
            self._decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif self._encoding == "deflate":
            self._decoder = zlib.decompressobj()

    def read(self, amt: Optional[int] = None) -> bytes:
        chunk = self._raw_read(amt) if amt is not None else self._raw_read()
        if not chunk:
            if self._decoder is not None:
                tail = self._decoder.flush()
                self._decoder = None
                return tail
            return b""
        if self._decoder is not None:
            return self._decoder.decompress(chunk)
        return chunk


class HTTPResponse(Response, IO):
    """HTTP-shaped :class:`Response` that IS an IO cursor over its buffer.

    Single access point for every HTTP response in yggdrasil: the
    high-level :class:`Response` / :class:`Tabular` surface (status,
    headers, tags, request, parse / Arrow / pickle projections, the
    cache pipeline) plus the
    :class:`yggdrasil.io.holder.IO` cursor surface (``read(n)`` /
    ``seek`` / ``tell`` / :func:`pa.input_stream` compatibility) on
    one object. ``self.buffer`` is a
    :class:`~yggdrasil.io.memory_stream.MemoryStream`: in the
    network-fetched shape it lazily pulls decoded bytes from the
    underlying socket through a :class:`_DecodingReader`; in the
    parsed-from-record shape the source is the pre-collected bytes.

    Connection-lifecycle methods (:meth:`release_conn` /
    :meth:`drain_conn` / :attr:`status`) keep the urllib3-shaped
    surface that warehouse-style streaming consumers depend on so
    HTTPSession itself plays the role :class:`PoolManager` used to fill.

    :class:`Response` runs without ``__slots__`` so this mix-in compiles
    (slot-backed bases on both sides of MI raise "multiple bases have
    instance lay-out conflict" :class:`TypeError`).
    """

    # No URL-routable scheme — the response's URL is sourced from the
    # bound :class:`PreparedRequest`, not the class-level scheme
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
        request: PreparedRequest,
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
        Response.__init__(
            self,
            request=request,
            status_code=status_code,
            headers=headers,
            tags=tags,
            buffer=buffer,
            received_at=received_at,
            receiver=receiver,
        )
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
        request: "PreparedRequest",
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
        :class:`_DecodingReader` over the raw socket; the connection
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
        source_fn = _DecodingReader(raw.read, encoding).read

        buffer = MemoryStream(source=source_fn)

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

        return resp

    @classmethod
    def from_pool(
        cls,
        request: "PreparedRequest",
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
