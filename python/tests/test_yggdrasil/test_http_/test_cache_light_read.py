"""The remote/generic cache hit read pulls only the columns needed to rebuild a
response — the response payload plus the request join key — never the heavy
``request_*`` / ``receiver`` / ``*_hash`` / ``_pkl`` columns. The caller already
holds the live requests and reattaches them, so the projection is lossless."""
from __future__ import annotations

import datetime as dt

import pyarrow as pa

from yggdrasil.http_.cache_config import CacheConfig, MATCH_COLUMN
from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.response import HTTPResponse as Response
from yggdrasil.http_.schemas import RESPONSE_SCHEMA
from yggdrasil.http_.send_config import RESPONSE_REBUILD_COLUMNS, SendConfig


def _req(url="https://example.com/api?x=1", method="GET"):
    return HTTPRequest.prepare(method, url)


def _resp(req, *, status=200, body=b'{"k":1}'):
    return Response(
        request=req, status_code=status, headers={"Content-Type": "application/json"},
        tags={"src": "test"}, buffer=body, received_at=dt.datetime.now(dt.timezone.utc),
    )


class TestReadHitsProjection:
    def test_read_hits_projects_to_rebuild_columns(self):
        captured = {}

        class _FakeHolder:                          # a generic (non-local) cache
            def read_table(self, *, options=None):
                captured["options"] = options
                return None

        cache = CacheConfig()
        cache.tabular = _FakeHolder()
        SendConfig().read_hits(cache, [_req()])

        cols = set(captured["options"].read_columns())
        # Every rebuild column is read (plus, at most, the predicate's own cheap
        # pruning keys like partition_key) …
        assert set(RESPONSE_REBUILD_COLUMNS) <= cols
        # … and the heavy columns a full read would otherwise scan are absent.
        for heavy in ("request_headers", "request_body", "request_params",
                      "receiver", "_pkl", "hash", "body_hash"):
            assert heavy not in cols

    def test_rebuild_columns_list_is_minimal_and_present(self):
        names = set(RESPONSE_SCHEMA.to_arrow_schema().names)
        # Every rebuild column is a real response column, and the join key is in.
        assert set(RESPONSE_REBUILD_COLUMNS) <= names
        assert MATCH_COLUMN in RESPONSE_REBUILD_COLUMNS


class TestRebuildColumnsAreSufficient:
    def test_light_columns_rebuild_a_full_response(self):
        # A request whose headers contribute to its public_hash — proving the
        # join key must come from the cached column, not a recompute over the
        # light row (which lacks request_headers / request_body).
        req = HTTPRequest.prepare("GET", "https://example.com/api?x=1",
                                  headers={"Accept": "application/json"})
        resp = _resp(req, status=201, body=b"HELLO")
        full = pa.Table.from_batches([Response.values_to_arrow_batch([resp])])

        # Read back only the rebuild columns — as the projected remote read would.
        light = full.select(list(RESPONSE_REBUILD_COLUMNS))
        (rebuilt,) = list(Response.from_arrow_tabular(light))

        # The original join key is preserved verbatim in the request_public_hash
        # column even though it can't be recomputed from the (header-less) row.
        assert light.column(MATCH_COLUMN).to_pylist() == [req.match_value("public_hash")]

        # Reattaching the live request (as HTTPResponseBatch does) restores the
        # full request the projection dropped — payload + request are intact.
        rebuilt.request = req
        assert rebuilt.status_code == 201
        assert rebuilt.content == b"HELLO"
        assert dict(rebuilt.headers)["Content-Type"] == "application/json"
        assert dict(rebuilt.tags) == {"src": "test"}
        assert rebuilt.request.url == req.url
        assert rebuilt.request.method == req.method
