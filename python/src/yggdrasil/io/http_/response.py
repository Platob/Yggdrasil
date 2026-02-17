from __future__ import annotations

from dataclasses import dataclass

from urllib3 import BaseHTTPResponse

from ..dynamic_buffer import DynamicBuffer
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
        stream: bool,
        received_at_timestamp: int
    ) -> "Response":
        buffer = DynamicBuffer()

        try:
            if stream:
                for batch in response.stream(amt=4 * 1024 * 1024):
                    buffer.write(batch)
            else:
                buffer.write(response.data)
        finally:
            # important when streaming / pooling
            response.release_conn()

        buffer.seek(0)

        return cls(
            request=request,
            status_code=response.status,
            headers=response.headers,
            buffer=buffer,
            received_at_timestamp=received_at_timestamp
        )
