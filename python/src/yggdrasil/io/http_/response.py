from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Mapping, Optional

from urllib3 import BaseHTTPResponse

from ..buffer import BytesIO
from ..request import PreparedRequest
from ..response import Response, _media_type_from_headers

__all__ = [
    "HTTPResponse"
]


@dataclass(slots=True)
class HTTPResponse(Response):

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
        buffer = (
            BytesIO(media_type=pre_media)
            if pre_media is not None
            else BytesIO()
        )

        response = cls(
            request=request,
            status_code=response.status,
            headers=headers,
            buffer=buffer,
            tags=tags,
            received_at=received_at
        )

        if request.prepare_response is not None:
            return request.prepare_response(response)

        return response

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

        return self