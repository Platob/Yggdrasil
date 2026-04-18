"""Shared helpers for `test_io` — importable from both conftest and tests.

Kept here rather than in conftest.py because test files should not import
from a conftest module (pytest treats conftest as a plugin, not a module).
All fixtures in conftest.py delegate to these helpers.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa

from yggdrasil.io import SaveMode
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response
from yggdrasil.io.send_config import CacheConfig, SendConfig
from yggdrasil.io.session import Session


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Epoch zero in UTC. Deterministic `sent_at` for request fixtures.
EPOCH = dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)

#: 2026-01-01 00:00:00 UTC, as microseconds since epoch. Fits `received_at`.
RECEIVED_AT_2026 = int(
    dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1_000_000
)

#: Request-key columns that survive an Arrow round-trip through the response
#: schema — safe to use anywhere `CacheConfig.request_by` is needed.
KEY_COLS: list[str] = ["request_method", "request_url_host", "request_url_path"]


# ---------------------------------------------------------------------------
# Request / Response factories
# ---------------------------------------------------------------------------

def make_request(
    url: str = "https://example.com/a",
    method: str = "GET",
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    tags: dict[str, Any] | None = None,
    sent_at: dt.datetime = EPOCH,
) -> PreparedRequest:
    """Build a `PreparedRequest` with sensible defaults for cache tests."""
    req = PreparedRequest.prepare(
        method=method,
        url=url,
        headers=headers,
        body=body,
        tags=tags,
    )
    req.sent_at = sent_at
    return req


def make_response(
    request: PreparedRequest | None = None,
    *,
    status_code: int = 200,
    body: bytes = b'{"ok":true}',
    content_type: str = "application/json",
    headers: dict[str, str] | None = None,
    tags: dict[str, Any] | None = None,
    received_at: int | None = None,
) -> Response:
    """Build a `Response` whose `received_at_timestamp` is deterministic."""
    base_headers = {
        "Content-Type": content_type,
        "Content-Length": str(len(body)),
    }
    if headers:
        base_headers.update(headers)
    return Response(
        request=request or make_request(),
        status_code=status_code,
        headers=base_headers,
        tags=tags or {},
        buffer=body,  # type: ignore[arg-type]
        received_at=RECEIVED_AT_2026 if received_at is None else received_at,
    )


# ---------------------------------------------------------------------------
# Mock Databricks table
# ---------------------------------------------------------------------------

def make_table_mock(
    full_name: str = "cat.schema.tbl",
    hits: list[Response] | None = None,
) -> MagicMock:
    """Return a MagicMock that behaves like a Databricks `Table` for cache I/O.

    `table.sql.execute(...)` yields a result whose `to_arrow_batches()` is a
    fresh iterator on every call — critical, because the session iterates it
    once per lookup and must not exhaust a shared iterator.
    """
    hits = hits or []
    if hits:
        arrow_tbl = pa.Table.from_batches(
            [r.to_arrow_batch(parse=False) for r in hits]
        )
        batch_list = arrow_tbl.to_batches()
    else:
        batch_list = []

    cache_result = MagicMock()
    cache_result.to_arrow_batches.side_effect = lambda: iter(batch_list)

    table = MagicMock()
    table.full_name.return_value = full_name
    table.sql.execute.return_value = cache_result
    return table


def make_cache_config(
    table: MagicMock | None = None,
    *,
    mode: SaveMode = SaveMode.APPEND,
    received_from: dt.datetime | str | None = "2020-01-01T00:00:00Z",
    request_by: list[str] | None = None,
) -> CacheConfig:
    """Build a `CacheConfig` that enables both local *and* remote caching.

    `received_from` is set by default so `local_cache_enabled` is True; pass
    `received_from=None` explicitly to disable the local side.
    """
    return CacheConfig(
        table=table,
        received_from=received_from,
        request_by=request_by or KEY_COLS,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# MockSession — concrete Session double that queues canned responses
# ---------------------------------------------------------------------------

@dataclass
class MockSession(Session):
    """Session double that queues pre-built responses for `_local_send`.

    Records every call in `calls` so tests can assert the number of network
    round-trips without touching the wire.
    """

    _queue: list[Response] = field(default_factory=list, init=False, repr=False)
    calls: list[PreparedRequest] = field(default_factory=list, init=False, repr=False)

    def _local_send(self, request: PreparedRequest, config: SendConfig) -> Response:
        self.calls.append(request)
        if self._queue:
            return self._queue.pop(0)
        return make_response(request=request)

    def queue(self, *responses: Response) -> "MockSession":
        self._queue.extend(responses)
        return self
