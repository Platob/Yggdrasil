from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Mapping, Optional

from urllib3 import BaseHTTPResponse

from ..holder import Holder
from ..memory import Memory
from ..request import PreparedRequest
from ..response import Response, _media_type_from_headers

if TYPE_CHECKING:
    from .path import HTTPPath

__all__ = [
    "HTTPResponse"
]


class HTTPResponse(Response):
    # No new fields — inherits ``Response.__slots__`` so the parent's
    # slotted layout still applies.
    __slots__ = ()

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
    def from_urllib3(
        cls,
        request: "PreparedRequest",
        response: BaseHTTPResponse,
        tags: Optional[Mapping[str, str]],
        received_at: dt.datetime,
        *,
        stream: bool = True,
        amt: int = 512 * 1024,
        release_conn: bool = True,
    ) -> "HTTPResponse":
        # Pre-infer media type from the response's Content-Type /
        # Content-Encoding so the holder carries it from the start;
        # downstream callers reach the registered leaf (ParquetIO,
        # JsonIO, ArrowIPCIO, …) via ``BytesIO(holder=self.buffer)``
        # without an extra format-detection hop.
        headers = dict(response.headers)
        pre_media = _media_type_from_headers(headers)
        buffer: Holder = Memory()
        if pre_media is not None:
            buffer.media_type = pre_media

        # Drain the body INTO the buffer before constructing the
        # ``Response``. ``Response.__init__`` runs ``_ensure_media_headers``
        # once; if we built the response on an empty buffer first and
        # drained second we'd pay that header sniff + Content-Length
        # stamp twice per request (once on the empty buffer, once on
        # the filled one). Draining first folds both into a single
        # post-fill pass.
        try:
            with buffer.open(mode="wb", owns_holder=False) as bio:
                if stream:
                    for chunk in response.stream(amt=amt):
                        bio.write(chunk)
                else:
                    bio.write(response.read())
        finally:
            if release_conn:
                response.release_conn()

        return cls(
            request=request,
            status_code=response.status,
            headers=headers,
            buffer=buffer,
            tags=tags,
            received_at=received_at,
        )