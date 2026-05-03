"""Unit tests for :mod:`yggdrasil.io.response_batch`.

These cover the structural surface (counts, iteration order, extend,
empty handling) without spinning up a server or Spark — the
session-side wiring lives in ``test_http_session_integration.py``.
"""

from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.buffer.base import TabularIO
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA, Response
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


def _urls(responses: list[Response]) -> list[str]:
    return [str(r.request.url) for r in responses]


class TestResponseBatchPython:
    def test_empty_batch_is_falsy_and_zero_length(self):
        # Every bucket carries a schema-bearing empty holder so the
        # batch can answer schema questions even with zero rows.
        batch = ResponseBatch()
        assert len(batch) == 0
        assert not batch
        assert batch.counts == {"local": 0, "remote": 0, "new": 0}
        assert list(batch) == []
        # All three holders are present (schema-bearing empties), not None.
        assert len(batch.parts()) == 3
        for holder in batch.parts():
            assert isinstance(holder, TabularIO)
            # Schema is preserved — RESPONSE_ARROW_SCHEMA — so callers
            # can introspect column names without ever fetching rows.
            assert holder.schema == RESPONSE_ARROW_SCHEMA

    def test_lists_are_coerced_to_tabular_holders(self):
        # The constructor lifts list[Response] into TabularIO holders so
        # all three buckets share one read contract. Verify the public
        # accessors expose the holders, not the raw lists.
        batch = ResponseBatch(
            local_hits=[_make_response("http://x/local")],
            remote_hits=[
                _make_response("http://x/remote-1"),
                _make_response("http://x/remote-2"),
            ],
        )
        assert isinstance(batch.local_hits, TabularIO)
        assert isinstance(batch.remote_hits, TabularIO)
        # The unsupplied bucket gets a schema-bearing empty holder
        # rather than ``None`` so empty results still expose a schema.
        assert isinstance(batch.new_hits, TabularIO)
        assert batch.new_hits.schema == RESPONSE_ARROW_SCHEMA

    def test_iteration_rebuilds_responses_in_pipeline_order(self):
        batch = ResponseBatch(
            local_hits=[_make_response("http://x/local")],
            remote_hits=[_make_response("http://x/remote")],
            new_hits=[_make_response("http://x/new")],
        )

        # Round-tripped Responses don't preserve identity (and an HTTP
        # URL gets rebuilt as HTTPResponse) — assert on the observable
        # contract: URL order matches local → remote → new.
        result = list(batch)
        assert _urls(result) == [
            "http://x/local",
            "http://x/remote",
            "http://x/new",
        ]
        assert all(r.status_code == 200 for r in result)
        assert len(batch) == 3
        assert bool(batch)
        assert not batch.is_spark

    def test_counts_reports_per_origin_sizes(self):
        batch = ResponseBatch(
            local_hits=[_make_response("http://x/1"), _make_response("http://x/2")],
            new_hits=[_make_response("http://x/3")],
        )
        assert batch.counts == {"local": 2, "remote": 0, "new": 1}

    def test_setters_coerce_through_private_holders(self):
        # Public setters route through `_coerce_bucket`, so a list
        # written via `batch.local_hits = [...]` shows up as a
        # TabularIO on the private holder.
        batch = ResponseBatch()
        batch.local_hits = [_make_response("http://x/1")]
        assert isinstance(batch.local_hits, TabularIO)
        assert batch._local_response is batch.local_hits

    def test_extend_merges_in_place_and_returns_self(self):
        a = ResponseBatch(local_hits=[_make_response("http://x/a")])
        b = ResponseBatch(
            local_hits=[_make_response("http://x/a2")],
            remote_hits=[_make_response("http://x/b")],
            new_hits=[_make_response("http://x/c")],
        )

        out = a.extend(b)
        assert out is a
        # Local was 1 + 1, remote/new came over from b.
        assert a.counts == {"local": 2, "remote": 1, "new": 1}
        assert _urls(list(a)) == [
            "http://x/a",
            "http://x/a2",
            "http://x/b",
            "http://x/c",
        ]

    def test_unknown_bucket_input_type_raises(self):
        # Strict on meaning: random objects shouldn't silently become
        # one of the supported shapes.
        with pytest.raises(TypeError, match="Unsupported bucket input"):
            ResponseBatch(local_hits=42)

    def test_to_dataframe_without_spark_raises(self):
        # No SparkSession bound → can't synthesize an empty DF or lift
        # any Arrow-IPC holders. Surface a clear error before Spark
        # blows up.
        batch = ResponseBatch()
        with pytest.raises(RuntimeError, match="needs a SparkSession"):
            batch.to_dataframe()
