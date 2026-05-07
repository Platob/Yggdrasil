"""Importable helpers for the rewritten yggdrasil.io tests."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response
from yggdrasil.io.send_config import SendConfig
from yggdrasil.io.session import Session


EPOCH = dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)


def make_request(
    url: str = "https://example.com/path",
    method: str = "GET",
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    tags: dict[str, str] | None = None,
) -> PreparedRequest:
    return PreparedRequest.prepare(
        method=method,
        url=url,
        headers=headers,
        body=body,
        tags=tags,
    )


def make_response(
    request: PreparedRequest | None = None,
    *,
    status_code: int = 200,
    body: bytes = b'{"ok":true}',
    content_type: str = "application/json",
    headers: dict[str, str] | None = None,
    received_at: dt.datetime | None = None,
) -> Response:
    base_headers = {"Content-Type": content_type}
    if headers:
        base_headers.update(headers)
    return Response(
        request=request or make_request(),
        status_code=status_code,
        headers=base_headers,
        tags={},
        buffer=body,  # type: ignore[arg-type]
        received_at=received_at if received_at is not None else EPOCH,
    )


@dataclass
class StubSession(Session):
    """Concrete Session double that returns canned responses.

    Each call to ``send`` is recorded in ``calls`` so tests can assert
    the network was — or was not — touched.
    """

    _queue: list[Response] = field(default_factory=list, init=False, repr=False)
    calls: list[PreparedRequest] = field(default_factory=list, init=False, repr=False)

    def queue(self, *responses: Response) -> "StubSession":
        self._queue.extend(responses)
        return self

    def _local_send(self, request: PreparedRequest, config: SendConfig) -> Response:
        self.calls.append(request)
        if self._queue:
            return self._queue.pop(0)
        return make_response(request=request)
