from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Mapping, Optional

from urllib3 import BaseHTTPResponse

from ..bytes_io import BytesIO
from ..tabular import Tabular
from ..request import PreparedRequest
from ..response import Response, _ensure_media_headers, _media_type_from_headers

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
        received_at: dt.datetime
    ) -> "HTTPResponse":
        # Pre-infer media type from the response's Content-Type /
        # Content-Encoding so the buffer is constructed as the
        # registered leaf (ParquetIO, JsonIO, ArrowIPCIO, …) up
        # front. Once bytes land via drain_urllib3, callers get a
        # tabular-ready buffer without an extra as_media() hop.
        headers = dict(response.headers)
        pre_media = _media_type_from_headers(headers)
        buffer_class = Tabular.class_for_media_type(pre_media, default=BytesIO)
        buffer = (
            buffer_class(media_type=pre_media)
            if pre_media is not None
            else buffer_class()
        )
        buffer.seek(0)

        return cls(
            request=request,
            status_code=response.status,
            headers=headers,
            buffer=buffer,
            tags=tags,
            received_at=received_at
        )

    def drain_urllib3(
        self,
        response: BaseHTTPResponse,
        stream: bool,
        *,
        amt: int = 512 * 1024,
        release_conn: bool = True
    ) -> BytesIO:
        buffer = self.buffer

        try:
            if stream:
                for chunk in response.stream(amt=amt):
                    buffer.write(chunk)
            else:
                buffer.write(response.read())
        finally:
            if release_conn:
                response.release_conn()

        # Rewind so callers (to_polars, json_load, as_media, paginated
        # combiners, …) read from byte 0 rather than from EOF.
        buffer.seek(0)

        # Resync Content-Length (and Content-Type/Encoding) to the
        # post-drain buffer. ``Response.__init__`` ran while the buffer
        # was still empty, so it stamped ``Content-Length: 0`` over the
        # remote's value — re-running here matches the persisted header
        # to the bytes that actually landed.
        _ensure_media_headers(self.headers, buffer)

        return self