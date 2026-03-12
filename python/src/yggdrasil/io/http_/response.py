from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from urllib3 import BaseHTTPResponse

from ..buffer import BytesIO
from ..request import PreparedRequest
from ..response import Response

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
        received_at_timestamp: int
    ) -> "HTTPResponse":
        response = cls(
            request=request,
            status_code=response.status,
            headers=dict(response.headers),
            buffer=BytesIO(),
            tags=tags,
            received_at_timestamp=received_at_timestamp
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
            response.release_conn()

        return self