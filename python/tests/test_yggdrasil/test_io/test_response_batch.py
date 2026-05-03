"""Unit tests for :mod:`yggdrasil.io.response_batch`.

These cover the structural surface (counts, iteration order, extend,
empty handling) without spinning up a server or Spark — the
session-side wiring lives in ``test_http_session_integration.py``.
"""

from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response
from yggdrasil.io.response_batch import ResponseBatch


def _make_response(url: str, status: int = 200) -> Response:
    """Build a minimal Response standing in for a real network result."""
    request = PreparedRequest.prepare(method="GET", url=url)
    return Response(
        request=request,
        status_code=status,
        headers={},
        tags={},
        buffer=BytesIO(b""),
        received_at=dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
    )


class TestResponseBatchPython:
    def test_empty_batch_is_falsy_and_zero_length(self):
        batch = ResponseBatch()
        assert len(batch) == 0
        assert not batch
        assert batch.counts == {"local": 0, "remote": 0, "new": 0}
        assert list(batch) == []
        assert batch.parts() == []

    def test_iteration_order_is_local_then_remote_then_new(self):
        local = _make_response("http://x/local")
        remote = _make_response("http://x/remote")
        new = _make_response("http://x/new")
        batch = ResponseBatch(
            local_hits=[local], remote_hits=[remote], new_hits=[new],
        )

        assert list(batch) == [local, remote, new]
        assert batch.responses == [local, remote, new]
        assert len(batch) == 3
        assert bool(batch)
        assert not batch.is_spark

    def test_counts_reports_per_origin_sizes(self):
        batch = ResponseBatch(
            local_hits=[_make_response("http://x/1"), _make_response("http://x/2")],
            new_hits=[_make_response("http://x/3")],
        )
        assert batch.counts == {"local": 2, "remote": 0, "new": 1}

    def test_setters_round_trip_through_private_holders(self):
        # Public setters write straight into the private holders; the
        # next read should reflect the new value.
        batch = ResponseBatch()
        r = _make_response("http://x/1")
        batch.local_hits = [r]
        assert batch.local_hits == [r]
        assert batch._local_response == [r]

    def test_extend_merges_in_place_and_returns_self(self):
        a = ResponseBatch(local_hits=[_make_response("http://x/a")])
        b = ResponseBatch(
            remote_hits=[_make_response("http://x/b")],
            new_hits=[_make_response("http://x/c")],
        )

        out = a.extend(b)
        assert out is a
        assert a.counts == {"local": 1, "remote": 1, "new": 1}

    def test_to_dataframe_without_spark_raises(self):
        # No SparkSession bound → can't synthesize an empty DF or lift
        # any list holders. Surface a clear error before Spark blows up.
        batch = ResponseBatch()
        with pytest.raises(RuntimeError, match="needs a SparkSession"):
            batch.to_dataframe()
