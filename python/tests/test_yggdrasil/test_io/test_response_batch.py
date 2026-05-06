"""Unit tests for :mod:`yggdrasil.io.response_batch`.

These cover the structural surface (counts, iteration order, extend,
empty handling) without spinning up a server or Spark — the
session-side wiring lives in ``test_http_session_integration.py``.
"""

from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.tabular import Tabular
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA, Response
from yggdrasil.io.response_batch import (
    DEFAULT_BUCKET_KEY,
    DEFAULT_LOCAL_PATH_KEY,
    DEFAULT_REMOTE_TABLE_KEY,
    ResponseBatch,
)


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
        # batch can answer schema questions even with zero rows. Both
        # local and remote buckets start as single-entry dicts keyed
        # by the default placeholder so callers can still introspect
        # schema without checking for an empty mapping.
        batch = ResponseBatch()
        assert len(batch) == 0
        assert not batch
        assert batch.counts == {"local": 0, "remote": 0, "new": 0}
        assert batch.local_counts == {}
        assert batch.local_paths == []
        assert batch.remote_counts == {}
        assert batch.remote_tables == []
        assert list(batch) == []
        # parts() flattens to every local-path holder + every
        # remote-table holder + new, so an empty batch still exposes
        # three holders (one default placeholder per keyed bucket).
        assert len(batch.parts()) == 3
        for holder in batch.parts():
            assert isinstance(holder, Tabular)
            # Schema is preserved — RESPONSE_ARROW_SCHEMA — so callers
            # can introspect column names without ever fetching rows.
            assert holder.schema == RESPONSE_ARROW_SCHEMA
        # Both keyed dicts carry their placeholder default so schema
        # introspection works without checking for an empty mapping.
        assert list(batch.local_hits) == [DEFAULT_LOCAL_PATH_KEY]
        assert list(batch.remote_hits) == [DEFAULT_REMOTE_TABLE_KEY]
        # Local and remote share one default sentinel under the hood.
        assert DEFAULT_LOCAL_PATH_KEY == DEFAULT_REMOTE_TABLE_KEY == DEFAULT_BUCKET_KEY

    def test_lists_are_coerced_to_tabular_holders(self):
        # The constructor lifts list[Response] into Tabular holders so
        # all three buckets share one read contract. A bare list passed
        # for local_hits / remote_hits is treated as a single untagged
        # bucket and stored under the default placeholder key.
        batch = ResponseBatch(
            local_hits=[_make_response("http://x/local")],
            remote_hits=[
                _make_response("http://x/remote-1"),
                _make_response("http://x/remote-2"),
            ],
        )
        assert isinstance(batch.local_hits, dict)
        assert list(batch.local_hits) == [DEFAULT_LOCAL_PATH_KEY]
        assert isinstance(batch.local_hits[DEFAULT_LOCAL_PATH_KEY], Tabular)
        assert isinstance(batch.remote_hits, dict)
        assert list(batch.remote_hits) == [DEFAULT_REMOTE_TABLE_KEY]
        assert isinstance(batch.remote_hits[DEFAULT_REMOTE_TABLE_KEY], Tabular)
        # The unsupplied bucket gets a schema-bearing empty holder
        # rather than ``None`` so empty results still expose a schema.
        assert isinstance(batch.new_hits, Tabular)
        assert batch.new_hits.schema == RESPONSE_ARROW_SCHEMA

    def test_local_hits_dict_input_preserves_per_path_split(self):
        # When the caller hands a dict keyed by local-cache folder
        # path, each entry becomes its own holder so downstream
        # consumers can see which cache root answered which subset.
        batch = ResponseBatch(
            local_hits={
                "/var/cache/tenant_a": [_make_response("http://x/a1")],
                "/var/cache/tenant_b": [
                    _make_response("http://x/b1"),
                    _make_response("http://x/b2"),
                ],
            },
        )
        assert list(batch.local_hits) == [
            "/var/cache/tenant_a", "/var/cache/tenant_b",
        ]
        assert batch.local_paths == [
            "/var/cache/tenant_a", "/var/cache/tenant_b",
        ]
        assert batch.local_counts == {
            "/var/cache/tenant_a": 1, "/var/cache/tenant_b": 2,
        }
        # ``counts["local"]`` rolls the per-path breakdown into a
        # single integer so the existing aggregate API keeps working.
        assert batch.counts == {"local": 3, "remote": 0, "new": 0}
        # parts() flattens every local-path holder + remote default +
        # new, so a two-path local bucket produces four parts total.
        assert len(batch.parts()) == 4
        # Iteration walks every local path in registration order
        # before yielding remote / new hits.
        urls = _urls(list(batch))
        assert urls == ["http://x/a1", "http://x/b1", "http://x/b2"]

    def test_remote_hits_dict_input_preserves_per_table_split(self):
        # When the caller hands a dict keyed by cache-table full name,
        # each entry becomes its own holder so downstream consumers
        # can see which table answered which subset.
        batch = ResponseBatch(
            remote_hits={
                "cat.db.tbl_a": [_make_response("http://x/a1")],
                "cat.db.tbl_b": [
                    _make_response("http://x/b1"),
                    _make_response("http://x/b2"),
                ],
            },
        )
        assert list(batch.remote_hits) == ["cat.db.tbl_a", "cat.db.tbl_b"]
        assert batch.remote_tables == ["cat.db.tbl_a", "cat.db.tbl_b"]
        assert batch.remote_counts == {"cat.db.tbl_a": 1, "cat.db.tbl_b": 2}
        # ``counts["remote"]`` rolls the per-table breakdown into a
        # single integer so the existing aggregate API keeps working.
        assert batch.counts == {"local": 0, "remote": 3, "new": 0}
        # parts() flattens local + every remote-table holder + new,
        # so a two-table remote bucket produces four parts total.
        assert len(batch.parts()) == 4
        # Iteration walks every remote table in registration order
        # before yielding new hits.
        urls = _urls(list(batch))
        assert urls == ["http://x/a1", "http://x/b1", "http://x/b2"]

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
        # Public setters route through `_coerce_keyed_bucket`, so a
        # bare list written via `batch.local_hits = [...]` shows up
        # as a single-entry dict under the default placeholder key.
        batch = ResponseBatch()
        batch.local_hits = [_make_response("http://x/1")]
        assert isinstance(batch.local_hits, dict)
        assert list(batch.local_hits) == [DEFAULT_LOCAL_PATH_KEY]
        assert batch._local_responses is batch.local_hits

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

    def test_extend_merges_remote_per_table_dropping_default_placeholder(self):
        # ``a`` only carries the placeholder default remote bucket.
        # Merging in a real per-table bucket from ``b`` should drop
        # the placeholder so the merged batch reflects only real
        # cache tables — no stale empty default left behind.
        a = ResponseBatch(local_hits=[_make_response("http://x/a")])
        b = ResponseBatch(
            remote_hits={
                "cat.db.tbl_a": [_make_response("http://x/b1")],
                "cat.db.tbl_b": [_make_response("http://x/b2")],
            },
        )

        a.extend(b)
        assert a.remote_tables == ["cat.db.tbl_a", "cat.db.tbl_b"]
        assert a.remote_counts == {"cat.db.tbl_a": 1, "cat.db.tbl_b": 1}
        assert a.counts == {"local": 1, "remote": 2, "new": 0}

    def test_extend_merges_local_per_path_dropping_default_placeholder(self):
        # ``a`` only carries the placeholder default local bucket.
        # Merging in real per-path entries from ``b`` should drop the
        # placeholder so the merged batch reflects only real cache
        # roots — same invariant as the remote-side merge.
        a = ResponseBatch(new_hits=[_make_response("http://x/n")])
        b = ResponseBatch(
            local_hits={
                "/var/cache/tenant_a": [_make_response("http://x/a1")],
                "/var/cache/tenant_b": [_make_response("http://x/b1")],
            },
        )

        a.extend(b)
        assert a.local_paths == ["/var/cache/tenant_a", "/var/cache/tenant_b"]
        assert a.local_counts == {
            "/var/cache/tenant_a": 1, "/var/cache/tenant_b": 1,
        }
        assert a.counts == {"local": 2, "remote": 0, "new": 1}

    def test_extend_appends_into_matching_remote_table_keys(self):
        # Same table key on both sides → row counts accumulate on the
        # existing holder rather than producing a duplicate entry.
        a = ResponseBatch(
            remote_hits={"cat.db.tbl_a": [_make_response("http://x/a1")]},
        )
        b = ResponseBatch(
            remote_hits={
                "cat.db.tbl_a": [_make_response("http://x/a2")],
                "cat.db.tbl_b": [_make_response("http://x/b1")],
            },
        )

        a.extend(b)
        assert a.remote_tables == ["cat.db.tbl_a", "cat.db.tbl_b"]
        assert a.remote_counts == {"cat.db.tbl_a": 2, "cat.db.tbl_b": 1}

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
