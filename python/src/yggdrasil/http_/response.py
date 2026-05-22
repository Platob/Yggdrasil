from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, Optional

from yggdrasil.http_._pool import HTTPResponse as _PoolHTTPResponse
from yggdrasil.io.holder import IO
from yggdrasil.io.memory_stream import MemoryStream
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response, _media_type_from_headers

if TYPE_CHECKING:
    from yggdrasil.environ.userinfo import UserInfo

    from .path import HTTPPath

__all__ = [
    "HTTPResponse"
]


class HTTPResponse(Response, IO):
    """HTTP-shaped :class:`Response` that IS an IO cursor over its buffer.

    Mixes :class:`Response`'s tabular / pickle / hash / Arrow projection
    surface with :class:`yggdrasil.io.holder.IO`'s seekable byte-cursor
    surface. Callers can hand the response straight to anything that
    wants a binary file-like object (``pa.input_stream(response)``,
    ``response.read(n)``, ``response.seek(0)``, the stdlib zipfile
    reader, …) without an intermediate ``response.open()`` step — the
    response IS the cursor, ``self.buffer`` (a
    :class:`~yggdrasil.io.memory_stream.MemoryStream` in the
    network-fetched shape, a :class:`~yggdrasil.io.memory.Memory` in
    the parsed-from-record shape) is the backing storage.

    :class:`Response` runs without ``__slots__`` precisely so this
    mix-in compiles (slot-backed bases on both sides of MI raise the
    "multiple bases have instance lay-out conflict" :class:`TypeError`).
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
        *,
        local_cached: bool = False,
        remote_cached: bool = False,
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
            local_cached=local_cached,
            remote_cached=remote_cached,
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

    @classmethod
    def from_pool(
        cls,
        request: "PreparedRequest",
        response: _PoolHTTPResponse,
        tags: Optional[Mapping[str, str]],
        received_at: dt.datetime,
        *,
        stream: bool = True,
        amt: int = 512 * 1024,
        release_conn: bool = True,
    ) -> "HTTPResponse":
        # The pool's :class:`HTTPResponse` already owns a
        # :class:`MemoryStream` over the decoded raw body — borrow it
        # directly instead of allocating a second holder and copying.
        # Draining the pool response pulls every chunk into that same
        # MemoryStream; once it lands here the stream has mutated to
        # its "collected" state (``eof is True``).
        headers = dict(response.headers)
        pre_media = _media_type_from_headers(headers)

        buffer: MemoryStream
        if isinstance(response, _PoolHTTPResponse):
            buffer = response.buffer
            if pre_media is not None and buffer.media_type is None:
                buffer.media_type = pre_media
            try:
                if stream:
                    # Drive the pool's chunk iterator so the MemoryStream
                    # fills via the decoder pipeline — no extra copy, the
                    # bytes land in the same window the high-level
                    # response will read back through.
                    for _ in response.stream(amt=amt):
                        pass
                else:
                    response.read()
            finally:
                if release_conn:
                    response.release_conn()
        else:
            # Duck-typed pool response (test fixtures, the urllib3-shim
            # error path) — drain through .stream / .read into a fresh
            # MemoryStream seeded from the collected bytes. Single
            # ``read_mv(-1, 0)`` in __init__ runs through the bytes
            # source once and lands the MemoryStream in its collected
            # state, matching the real pool path.
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
