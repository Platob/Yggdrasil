"""Tests for paginated response rechunking in HTTPSession._combine_paginated_pages.

Covers:

1. Single-page response (total_pages <= current_page) passes through unchanged.
2. Multi-page responses are fetched concurrently and concatenated.
3. The combined Arrow IPC output is rechunked to ~_PAGINATED_RECHUNK_BYTE_SIZE
   byte batches instead of being flushed as one oversized batch.
4. Schema is preserved across the concat + rechunk pipeline.
"""
from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from yggdrasil.http_.response import HTTPResponse
from yggdrasil.http_.session import HTTPSession, _PAGINATED_RECHUNK_BYTE_SIZE
from yggdrasil.io.memory import Memory
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.session import Session

EPOCH = dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)


@pytest.fixture(autouse=True)
def _clear_session_singletons():
    Session._INSTANCES.clear()
    yield
    Session._INSTANCES.clear()


def _make_request(url: str = "https://example.com/data") -> PreparedRequest:
    return PreparedRequest.prepare(
        method="GET",
        url=url,
        headers={"Accept": "application/json"},
    )


def _json_response(
    request: PreparedRequest,
    rows: list[dict],
    *,
    current_page: int | None = None,
    total_pages: int | None = None,
) -> HTTPResponse:
    import yggdrasil.pickle.json as json_module

    body = json_module.dumps(rows)
    headers = {"Content-Type": "application/json"}
    if current_page is not None:
        headers["X-Current-Page"] = str(current_page)
    if total_pages is not None:
        headers["X-Last-Page"] = str(total_pages)
    return HTTPResponse(
        request=request,
        status_code=200,
        headers=headers,
        tags={},
        buffer=Memory(binary=body),
        received_at=EPOCH,
    )


class _PaginatedStubSession(HTTPSession):
    """Session that returns canned per-page responses."""

    def __init__(self, pages: dict[int, list[dict]], **kw):
        if getattr(self, "_initialized", False):
            self._pages = pages
            return
        super().__init__(base_url="https://example.com", **kw)
        self._pages = pages

    def _fetch_paginated_page(
        self,
        *,
        request,
        page_num,
        body_seed,
        wait_cfg,
        stream,
        raise_error,
    ):
        rows = self._pages[page_num]
        resp = _json_response(request, rows)
        return page_num, resp


class TestCombinePaginatedPages:
    """Unit tests for _combine_paginated_pages."""

    def test_single_page_passthrough(self):
        """When total_pages <= current_page the result is returned as-is."""
        session = _PaginatedStubSession(pages={})
        request = _make_request()
        original = _json_response(
            request, [{"a": 1}], current_page=1, total_pages=1,
        )
        result = session._combine_paginated_pages(
            result=original,
            request=request,
            current_page=1,
            total_pages=1,
            wait_cfg=session.waiting,
            raise_error=True,
        )
        assert result is original

    def test_multi_page_concat(self):
        """Pages 1-3 are fetched and concatenated into a single IPC buffer."""
        page1_rows = [{"x": i, "y": f"p1_{i}"} for i in range(5)]
        page2_rows = [{"x": i, "y": f"p2_{i}"} for i in range(5, 10)]
        page3_rows = [{"x": i, "y": f"p3_{i}"} for i in range(10, 15)]

        session = _PaginatedStubSession(pages={2: page2_rows, 3: page3_rows})
        request = _make_request()
        first_resp = _json_response(
            request, page1_rows, current_page=1, total_pages=3,
        )

        result = session._combine_paginated_pages(
            result=first_resp,
            request=request,
            current_page=1,
            total_pages=3,
            wait_cfg=session.waiting,
            raise_error=True,
        )

        table = result.to_arrow_table(parse=True)
        assert table.num_rows == 15
        assert set(table.column_names) == {"x", "y"}

    def test_tags_carry_pagination_info(self):
        """Result tags contain page_start and page_total."""
        session = _PaginatedStubSession(pages={2: [{"v": 99}]})
        request = _make_request()
        first_resp = _json_response(
            request, [{"v": 1}], current_page=1, total_pages=2,
        )
        result = session._combine_paginated_pages(
            result=first_resp,
            request=request,
            current_page=1,
            total_pages=2,
            wait_cfg=session.waiting,
            raise_error=True,
        )
        assert result.tags["page_start"] == "1"
        assert result.tags["page_total"] == "2"

    def test_schema_preserved(self):
        """Column names and types survive the concat+rechunk pipeline."""
        rows = [{"id": 1, "name": "a", "value": 1.5}]
        session = _PaginatedStubSession(pages={2: rows})
        request = _make_request()
        first_resp = _json_response(request, rows, current_page=1, total_pages=2)

        result = session._combine_paginated_pages(
            result=first_resp,
            request=request,
            current_page=1,
            total_pages=2,
            wait_cfg=session.waiting,
            raise_error=True,
        )
        table = result.to_arrow_table(parse=True)
        assert set(table.column_names) == {"id", "name", "value"}
        assert table.num_rows == 2


class TestPaginatedRechunking:
    """Verify the IPC output is rechunked to the configured byte size."""

    def test_rechunk_splits_large_payload(self):
        """With a small byte_size the combined table is split into multiple IPC batches."""
        n_rows_per_page = 500
        padding = "x" * 200
        page1 = [{"id": i, "data": padding} for i in range(n_rows_per_page)]
        page2 = [{"id": i, "data": padding} for i in range(n_rows_per_page, 2 * n_rows_per_page)]

        session = _PaginatedStubSession(pages={2: page2})
        request = _make_request()
        first_resp = _json_response(
            request, page1, current_page=1, total_pages=2,
        )

        small_byte_size = 4 * 1024

        with patch(
            "yggdrasil.http_.session._PAGINATED_RECHUNK_BYTE_SIZE",
            small_byte_size,
        ):
            result = session._combine_paginated_pages(
                result=first_resp,
                request=request,
                current_page=1,
                total_pages=2,
                wait_cfg=session.waiting,
                raise_error=True,
            )

        result.buffer.seek(0)
        reader = ipc.open_file(pa.BufferReader(result.buffer.to_bytes()))
        n_batches = reader.num_record_batches
        total_rows = sum(reader.get_batch(i).num_rows for i in range(n_batches))

        assert total_rows == 2 * n_rows_per_page
        assert n_batches > 1, (
            f"Expected multiple IPC batches with byte_size={small_byte_size}, "
            f"got {n_batches}"
        )

    def test_small_payload_single_batch(self):
        """A payload smaller than the byte_size threshold stays as one batch."""
        page1 = [{"a": 1}]
        page2 = [{"a": 2}]

        session = _PaginatedStubSession(pages={2: page2})
        request = _make_request()
        first_resp = _json_response(
            request, page1, current_page=1, total_pages=2,
        )

        result = session._combine_paginated_pages(
            result=first_resp,
            request=request,
            current_page=1,
            total_pages=2,
            wait_cfg=session.waiting,
            raise_error=True,
        )

        result.buffer.seek(0)
        reader = ipc.open_file(pa.BufferReader(result.buffer.to_bytes()))
        assert reader.num_record_batches == 1
        assert reader.get_batch(0).num_rows == 2

    def test_default_byte_size_is_128_mib(self):
        """The module constant matches 128 MiB."""
        assert _PAGINATED_RECHUNK_BYTE_SIZE == 128 * 1024 * 1024
