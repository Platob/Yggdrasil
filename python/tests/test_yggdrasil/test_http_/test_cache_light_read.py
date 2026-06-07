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

        opts = captured["options"]
        # ``column_names`` is the projection the source read pushes down — the
        # SQL ``SELECT`` list (Databricks Table) / parquet column set (Delta). It
        # must be exactly the rebuild columns, even after the backend binds the
        # full table schema as the cast source (``with_source``).
        projected = opts.with_source(source=RESPONSE_SCHEMA).column_names
        assert set(projected) == set(RESPONSE_REBUILD_COLUMNS)
        assert len(projected) < len(RESPONSE_SCHEMA.to_arrow_schema().names)

        cols = set(opts.read_columns())
        # Every rebuild column is read (plus, at most, the predicate's own cheap
        # pruning keys like partition_key) …
        assert set(RESPONSE_REBUILD_COLUMNS) <= cols
        # … and the heavy columns a full read would otherwise scan are absent.
        for heavy in ("request_headers", "request_body", "request_params",
                      "receiver", "_pkl", "hash", "body_hash"):
            assert heavy not in cols
        # ``received_at`` is not fetched — it is stamped on retrieve instead.
        assert "received_at" not in cols

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


class TestRetrieveStampsReceivedAt:
    def test_received_at_is_stamped_now_on_retrieve(self):
        from yggdrasil.arrow.tabular import ArrowTabular
        from yggdrasil.http_.response_batch import HTTPResponseBatch

        req = _req()
        # A cached response captured "yesterday" — the stored received_at is not
        # fetched, so it must not survive onto the retrieved response.
        resp = _resp(req, body=b"HELLO")
        resp.received_at = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
        full = pa.Table.from_batches([Response.values_to_arrow_batch([resp])])
        light = full.select(list(RESPONSE_REBUILD_COLUMNS))

        class _FakeCfg:                              # stand-in SendConfig
            def read_hits(self, cache, requests, *, session=None):
                return ArrowTabular(light.to_batches(), schema=light.schema)

        batch = HTTPResponseBatch(send_config=_FakeCfg(), requests=[req])
        before = dt.datetime.now(dt.timezone.utc)
        out = batch._read_cache_hits(CacheConfig(), {req.match_value("public_hash")})
        after = dt.datetime.now(dt.timezone.utc)

        (got,) = list(Response.from_arrow_tabular(out.read_arrow_batches()))
        assert before <= got.received_at <= after        # stamped now, not 2020
        assert got.content == b"HELLO"
        assert got.request.url == req.url                 # full request reattached


class TestTabularAccessorsConsistent:
    """Every ``Tabular`` accessor on the batch agrees on row count + content after
    the light cache read — a remote hit (reattached request, stamped received_at)
    plus a freshly-fetched response."""

    def _batch(self):
        from yggdrasil.arrow.tabular import ArrowTabular
        from yggdrasil.http_.response_batch import HTTPResponseBatch

        hit_req, new_req = _req("https://e.com/a"), _req("https://e.com/b")
        hit = _resp(hit_req, body=b"HIT")
        hit.received_at = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
        light = pa.Table.from_batches(
            [Response.values_to_arrow_batch([hit])]
        ).select(list(RESPONSE_REBUILD_COLUMNS))

        class _Cfg:
            local_cache = None
            remote_cache = CacheConfig()

            def read_hits(self, cache, requests, *, session=None):
                return ArrowTabular(light.to_batches(), schema=light.schema)

        b = HTTPResponseBatch(send_config=_Cfg(), requests=[hit_req])
        b._remote_hashes = {hit_req.match_value("public_hash")}
        b._split_done = True
        b._misses = []
        new = _resp(new_req, body=b"NEW")
        new.received_at = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
        b.new_tabular = [new]
        return b

    def test_all_accessors_agree_on_count(self):
        b = self._batch()
        assert b.counts == {"local": 0, "remote": 1, "new": 1}
        assert len(b) == 2
        assert b.count() == 2
        assert len(list(b.responses())) == 2
        assert len(list(b.read_records())) == 2
        assert b.read_arrow_table().num_rows == 2

    def test_accessors_carry_reattached_request_and_stamped_time(self):
        b = self._batch()
        by_url = {r.request.url.to_string(): r for r in b.responses()}
        hit = by_url["https://e.com/a"]
        new = by_url["https://e.com/b"]
        assert hit.content == b"HIT"
        assert hit.received_at.year != 2020          # stamped on retrieve, not the stored 2020
        assert new.content == b"NEW"
        assert new.received_at.year == 2021          # a fresh response is untouched
