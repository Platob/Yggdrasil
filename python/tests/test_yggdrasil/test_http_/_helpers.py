"""Shared test doubles for the ``test_http_`` package."""
from __future__ import annotations

import datetime as dt
from collections import deque
from typing import Any, MutableMapping, Optional

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.response import HTTPResponse
from yggdrasil.http_.session import HTTPSession
from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.path.memory import Memory


def make_request(
    url: str = "https://api.example.com/test",
    method: str = "GET",
    headers: Optional[MutableMapping[str, str]] = None,
) -> HTTPRequest:
    return HTTPRequest.prepare(method, url, headers=dict(headers or {}))


def make_response(
    *,
    request: Optional[HTTPRequest] = None,
    status_code: int = 200,
    body: bytes = b"",
    headers: Optional[MutableMapping[str, str]] = None,
) -> HTTPResponse:
    if request is None:
        request = make_request()
    resp_headers: dict[str, str] = {"Content-Length": str(len(body))}
    if headers:
        resp_headers.update(headers)
    return HTTPResponse(
        request=request,
        status_code=status_code,
        headers=resp_headers,
        tags={},
        buffer=Memory(binary=body),
        received_at=dt.datetime.now(dt.timezone.utc),
    )


class StubSession(HTTPSession):
    """In-memory session that records sent requests and returns queued responses."""

    _INSTANCES: dict = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> "StubSession":
        return object.__new__(cls)

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        auth: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url=base_url, auth=auth, **kwargs)
        self._initialized = True
        self.calls: list[HTTPRequest] = []
        self._queue: deque[HTTPResponse] = deque()

    def queue(self, response: HTTPResponse) -> None:
        self._queue.append(response)

    def _wire_send(
        self,
        request: HTTPRequest,
        wait_cfg: WaitingConfig,
    ) -> HTTPResponse:
        self.calls.append(request)
        if self._queue:
            return self._queue.popleft()
        return make_response(request=request)
