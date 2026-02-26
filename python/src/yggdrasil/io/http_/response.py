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
        stream: bool,
        tags: Optional[Mapping[str, str]],
        received_at_timestamp: int
    ) -> "Response":
        buffer = BytesIO()

        try:
            if stream:
                for batch in response.stream(amt=1024 * 1024):
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
            headers=dict(response.headers),
            buffer=buffer,
            tags=tags,
            received_at_timestamp=received_at_timestamp
        )
