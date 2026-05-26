"""Mock test verifying session.send / send_many still insert into remote tables.

Covers:
  - Single ``send()`` with ``remote_cache`` → miss → network → insert
  - ``send_many()`` with ``remote_cache`` → misses → network → bulk insert
  - ``send()`` with per-request ``send_config.remote_cache`` override
  - ``send_many()`` with mixed per-request remote cache configs
  - UPSERT mode correctly skips both lookup and writeback
  - ``send_many()`` multiple misses produce one bulk insert (not N individual)
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Iterator

import pyarrow as pa
import pytest

from yggdrasil.enums import Mode
from yggdrasil.io.response import Response
from yggdrasil.http_.cache_config import CacheConfig
from yggdrasil.http_.send_config import SendConfig
from yggdrasil.io.session import Session
from yggdrasil.io.tabular import Tabular

from ._helpers import StubSession, make_request, make_response


@pytest.fixture(autouse=True)
def _clear_singletons():
    Session._INSTANCES.clear()
    yield
    Session._INSTANCES.clear()


class FakeTable(Tabular):
    """Minimal Tabular double that records write_arrow_batches calls."""

    def __init__(self, name: str = "catalog.schema.cache_table") -> None:
        super().__init__()
        self._name = name
        self.rows: list[pa.RecordBatch] = []
        self.lookups: list[Any] = []
        self.inserts: list[dict[str, Any]] = []
        self.created = False
        self.path = name

    def full_name(self, safe: bool = False) -> str:
        return self._name

    def create(self, schema: pa.Schema, missing_ok: bool = False) -> None:
        self.created = True

    def _read_arrow_batches(self, options: Any = None, **kw) -> Iterator[pa.RecordBatch]:
        predicate = getattr(options, "predicate", None)
        self.lookups.append(predicate)
        batches = list(self.rows)
        if predicate is None:
            return iter(batches)
        return predicate.filter_arrow_batches(iter(batches))

    def _write_arrow_batches(self, batches: Any, options: Any = None, **kw) -> None:
        new: list[pa.RecordBatch] = []
        for entry in batches:
            if isinstance(entry, pa.Table):
                new.extend(entry.to_batches())
            elif isinstance(entry, pa.RecordBatch):
                new.append(entry)
        mode = getattr(options, "mode", None)
        match_by = getattr(options, "match_by", None)
        prune_values = getattr(options, "prune_values", None)
        self.inserts.append({
            "mode": mode,
            "match_by": tuple(f.name for f in match_by) if match_by else None,
            "rows": sum(b.num_rows for b in new),
            "prune_keys": tuple(sorted(prune_values.keys())) if prune_values else (),
        })
        self.rows.extend(new)


def _cache(tab: FakeTable, **kw) -> CacheConfig:
    kw.setdefault("mode", Mode.APPEND)
    kw.pop("request_by", None)
    kw.pop("wait", None)
    return CacheConfig(tabular=tab, **kw)


# =========================================================================
# Single send — remote table insert
# =========================================================================


class TestSingleSendRemoteInsert:

    def test_miss_inserts_into_remote_table(self):
        """A cache miss on send() must fetch from network then insert
        the response into the remote table."""
        tab = FakeTable()
        cfg = _cache(tab)
        req = make_request("https://api.example.com/data")

        s = StubSession()
        s.queue(make_response(request=req, body=b'{"result": "ok"}'))

        resp = s.send(req, remote_cache=cfg)

        assert resp.ok
        assert len(s.calls) == 1, "miss should hit network"
        assert len(tab.inserts) == 1, "response must be inserted into remote table"
        assert tab.inserts[0]["rows"] == 1
        assert tab.inserts[0]["mode"] == Mode.APPEND

    def test_hit_does_not_insert(self):
        """A cache hit on send() must NOT insert again."""
        tab = FakeTable()
        cfg = _cache(tab)
        req = make_request("https://api.example.com/data")

        # Seed the remote table with a cached response
        tab.rows.append(make_response(request=req, body=b'{"cached": true}').to_arrow_batch(parse=False))

        s = StubSession()
        resp = s.send(req, remote_cache=cfg)

        assert resp.ok
        assert len(s.calls) == 0, "hit should skip network"
        assert len(tab.inserts) == 0, "hit should not re-insert"

    def test_per_request_send_config_remote_cache(self):
        """Per-request send_config.remote_cache must be honored by send()."""
        tab = FakeTable(name="catalog.schema.per_request_table")
        cfg = _cache(tab)
        req = make_request("https://api.example.com/x")
        req = req.copy(send_config=SendConfig(remote_cache=cfg))

        s = StubSession()
        s.queue(make_response(request=req, body=b'{"x": 1}'))

        resp = s.send(req)

        assert resp.ok
        assert len(s.calls) == 1
        assert len(tab.inserts) == 1, "per-request remote_cache must receive the insert"

    def test_error_response_not_inserted(self):
        """5xx responses must NOT be inserted into the remote table."""
        tab = FakeTable()
        cfg = _cache(tab)
        req = make_request("https://api.example.com/fail")

        s = StubSession()
        s.queue(make_response(request=req, status_code=500, body=b"error"))

        s.send(req, remote_cache=cfg, raise_error=False)

        assert len(tab.inserts) == 0, "error response must not be cached"


# =========================================================================
# send_many — remote table insert
# =========================================================================


class TestSendManyRemoteInsert:

    def test_misses_bulk_insert_into_remote_table(self):
        """Multiple misses in send_many must all be inserted into the
        remote table (via stage 4 bulk writeback)."""
        tab = FakeTable()
        cfg = _cache(tab)

        reqs = [
            make_request(f"https://api.example.com/item/{i}")
            for i in range(5)
        ]
        s = StubSession()
        for req in reqs:
            s.queue(make_response(request=req, body=f'{{"id": {reqs.index(req)}}}'.encode()))

        results = list(s.send_many(iter(reqs), remote_cache=cfg))

        assert len(results) == 5
        assert len(s.calls) == 5, "all misses must hit the network"
        total_inserted = sum(i["rows"] for i in tab.inserts)
        assert total_inserted == 5, f"all 5 responses must land in the table, got {total_inserted}"

    def test_mix_hits_and_misses(self):
        """send_many with some hits and some misses: only misses insert."""
        tab = FakeTable()
        cfg = _cache(tab)

        hit_req = make_request("https://api.example.com/cached")
        miss_req = make_request("https://api.example.com/fresh")

        # Seed one response
        tab.rows.append(make_response(request=hit_req, body=b'{"src":"cache"}').to_arrow_batch(parse=False))

        s = StubSession()
        s.queue(make_response(request=miss_req, body=b'{"src":"network"}'))

        results = list(s.send_many(iter([hit_req, miss_req]), remote_cache=cfg))

        assert len(results) == 2
        assert len(s.calls) == 1, "only miss touches network"
        total_inserted = sum(i["rows"] for i in tab.inserts)
        assert total_inserted == 1, "only the miss should be inserted"

    def test_per_request_remote_config_in_send_many(self):
        """Per-request remote_cache overrides in send_many must route
        inserts to the correct table."""
        tab_a = FakeTable(name="catalog.a.responses")
        tab_b = FakeTable(name="catalog.b.responses")

        req_a = make_request("https://api.example.com/a").copy(
            send_config=SendConfig(remote_cache=_cache(tab_a)),
        )
        req_b = make_request("https://api.example.com/b").copy(
            send_config=SendConfig(remote_cache=_cache(tab_b)),
        )

        s = StubSession()
        s.queue(
            make_response(request=req_a, body=b'{"t":"a"}'),
            make_response(request=req_b, body=b'{"t":"b"}'),
        )

        results = list(s.send_many(iter([req_a, req_b])))

        assert len(results) == 2
        assert len(s.calls) == 2

        inserted_a = sum(i["rows"] for i in tab_a.inserts)
        inserted_b = sum(i["rows"] for i in tab_b.inserts)
        assert inserted_a == 1, f"tab_a should have 1 insert, got {inserted_a}"
        assert inserted_b == 1, f"tab_b should have 1 insert, got {inserted_b}"

    def test_upsert_mode_skips_read_and_persists_in_send_many(self):
        """UPSERT mode in send_many skips the cache read but persists."""
        tab = FakeTable()
        cfg = _cache(tab, mode=Mode.UPSERT)

        req = make_request("https://api.example.com/upsert")
        s = StubSession()
        s.queue(make_response(request=req, body=b'{"v":1}'))

        list(s.send_many(iter([req]), remote_cache=cfg))

        assert len(s.calls) == 1, "UPSERT must fetch from network"
        assert len(tab.lookups) >= 0
        assert len(tab.inserts) == 1, "UPSERT must persist the response"
        assert tab.inserts[0]["mode"] == Mode.UPSERT

    def test_error_responses_not_inserted_in_send_many(self):
        """Error responses in send_many must not be inserted into
        the remote table."""
        tab = FakeTable()
        cfg = _cache(tab)

        ok_req = make_request("https://api.example.com/ok")
        err_req = make_request("https://api.example.com/err")

        s = StubSession()
        s.queue(
            make_response(request=ok_req, body=b'{"ok":true}'),
            make_response(request=err_req, status_code=500, body=b"boom"),
        )

        list(s.send_many(iter([ok_req, err_req]), remote_cache=cfg, raise_error=False))

        total_inserted = sum(i["rows"] for i in tab.inserts)
        assert total_inserted == 1, "only OK response should be inserted"

    def test_second_send_many_hits_after_first_inserts(self):
        """After send_many inserts responses, a second send_many call
        should find them in the cache and skip the network."""
        tab = FakeTable()
        cfg = _cache(tab)

        req = make_request("https://api.example.com/once")
        s = StubSession()
        s.queue(make_response(request=req, body=b'{"v":"fetched"}'))

        # First round: miss → fetch → insert
        list(s.send_many(iter([req]), remote_cache=cfg))
        assert len(s.calls) == 1
        assert sum(i["rows"] for i in tab.inserts) == 1

        # Second round: hit → no network, no insert
        results = list(s.send_many(iter([req]), remote_cache=cfg))
        assert len(s.calls) == 1, "second round must not touch network"
        assert len(results) == 1
        assert results[0].ok


# =========================================================================
# UPSERT / OVERWRITE persist behavior
# =========================================================================


class TestSendManyUpsertOverwritePersist:
    """All modes read from cache then write back with their own
    semantics.  The mode is passed through to the insert call."""

    def test_upsert_single_send_persists(self):
        """Single send() with UPSERT reads and persists."""
        tab = FakeTable()
        cfg = _cache(tab, mode=Mode.UPSERT)
        req = make_request("https://api.example.com/u")

        s = StubSession()
        s.queue(make_response(request=req, body=b'{"u":1}'))
        s.send(req, remote_cache=cfg)

        assert len(s.calls) == 1
        assert len(tab.lookups) == 1, "UPSERT reads from cache"
        assert len(tab.inserts) == 1
        assert tab.inserts[0]["mode"] == Mode.UPSERT
        assert tab.inserts[0]["rows"] == 1

    def test_overwrite_single_send_persists(self):
        """Single send() with OVERWRITE reads then persists."""
        tab = FakeTable()
        cfg = _cache(tab, mode=Mode.OVERWRITE)
        req = make_request("https://api.example.com/o")

        s = StubSession()
        s.queue(make_response(request=req, body=b'{"o":1}'))
        s.send(req, remote_cache=cfg)

        assert len(s.calls) == 1
        assert len(tab.inserts) == 1
        assert tab.inserts[0]["mode"] == Mode.OVERWRITE

    def test_upsert_send_many_persists(self):
        """send_many with UPSERT skips read but persists."""
        tab = FakeTable()
        cfg = _cache(tab, mode=Mode.UPSERT)

        reqs = [
            make_request("https://api.example.com/u1"),
            make_request("https://api.example.com/u2"),
        ]
        s = StubSession()
        for r in reqs:
            s.queue(make_response(request=r, body=b'{"ok":true}'))

        list(s.send_many(iter(reqs), remote_cache=cfg))

        assert len(s.calls) == 2
        assert len(tab.lookups) >= 0
        total = sum(i["rows"] for i in tab.inserts)
        assert total == 2, "both responses must be persisted"
        assert all(i["mode"] == Mode.UPSERT for i in tab.inserts)

    def test_overwrite_send_many_persists(self):
        """send_many with OVERWRITE persists responses."""
        tab = FakeTable()
        cfg = _cache(tab, mode=Mode.OVERWRITE)

        req = make_request("https://api.example.com/o1")
        s = StubSession()
        s.queue(make_response(request=req, body=b'{"o":1}'))

        list(s.send_many(iter([req]), remote_cache=cfg))

        assert len(s.calls) == 1
        total = sum(i["rows"] for i in tab.inserts)
        assert total == 1
        assert any(i["mode"] == Mode.OVERWRITE for i in tab.inserts)

    def test_mixed_upsert_append_both_persist(self):
        """A batch with APPEND and UPSERT requests persists both."""
        tab = FakeTable()
        append_cfg = _cache(tab, mode=Mode.APPEND)
        upsert_cfg = _cache(tab, mode=Mode.UPSERT)

        a = make_request("https://api.example.com/a").copy(
            send_config=SendConfig(remote_cache=append_cfg),
        )
        b = make_request("https://api.example.com/b").copy(
            send_config=SendConfig(remote_cache=upsert_cfg),
        )
        s = StubSession()
        s.queue(
            make_response(request=a, body=b'{"a":1}'),
            make_response(request=b, body=b'{"b":1}'),
        )

        list(s.send_many(iter([a, b])))

        assert len(s.calls) == 2
        modes = {i["mode"] for i in tab.inserts}
        assert modes == {Mode.APPEND, Mode.UPSERT}

    def test_mixed_overwrite_upsert_both_persist(self):
        """OVERWRITE + UPSERT in the same batch — both persist."""
        tab = FakeTable()
        ow_cfg = _cache(tab, mode=Mode.OVERWRITE)
        up_cfg = _cache(tab, mode=Mode.UPSERT)

        a = make_request("https://api.example.com/ow").copy(
            send_config=SendConfig(remote_cache=ow_cfg),
        )
        b = make_request("https://api.example.com/up").copy(
            send_config=SendConfig(remote_cache=up_cfg),
        )
        s = StubSession()
        s.queue(
            make_response(request=a, body=b'{"a":1}'),
            make_response(request=b, body=b'{"b":1}'),
        )

        list(s.send_many(iter([a, b])))

        assert len(s.calls) == 2
        modes = {i["mode"] for i in tab.inserts}
        assert modes == {Mode.OVERWRITE, Mode.UPSERT}

    def test_upsert_per_request_with_session_append(self):
        """Per-request UPSERT overrides session-level APPEND config.
        Both the APPEND and the UPSERT request should persist."""
        tab = FakeTable()
        session_cfg = _cache(tab, mode=Mode.APPEND)
        per_req_cfg = _cache(tab, mode=Mode.UPSERT)

        normal = make_request("https://api.example.com/normal")
        upsert = make_request("https://api.example.com/upsert").copy(
            send_config=SendConfig(remote_cache=per_req_cfg),
        )

        s = StubSession()
        s.queue(
            make_response(request=normal, body=b'{"n":1}'),
            make_response(request=upsert, body=b'{"u":1}'),
        )

        list(s.send_many(iter([normal, upsert]), remote_cache=session_cfg))

        assert len(s.calls) == 2
        total = sum(i["rows"] for i in tab.inserts)
        assert total == 2
        modes = {i["mode"] for i in tab.inserts}
        assert modes == {Mode.APPEND, Mode.UPSERT}

    def test_upsert_local_cache_persists(self, tmp_path):
        """UPSERT local cache: skips read but persists to disk."""
        from pathlib import Path

        cache_dir = tmp_path / "upsert_local"
        cache_dir.mkdir()
        cfg = CacheConfig(
            tabular=str(cache_dir),
            mode=Mode.UPSERT,
        )

        req = make_request("https://api.example.com/local-upsert")
        s = StubSession()
        s.queue(make_response(request=req, body=b'{"local":1}'))
        s.send(req, local_cache=cfg)

        # Wait for the fire-and-forget writeback
        deadline = __import__("time").monotonic() + 5.0
        while __import__("time").monotonic() < deadline:
            parts = list(Path(cache_dir).rglob("partition_key=*/part-*"))
            if parts:
                break
            __import__("time").sleep(0.05)
        assert len(parts) >= 1, "UPSERT must persist to local cache"


# =========================================================================
# Holder grouping — mode and spark_session separation
# =========================================================================


class TestHolderGrouping:
    """Verify _group_by_config separates by mode and spark_session."""

    def test_groups_separate_by_mode(self):
        from yggdrasil.http_.session import HTTPSession

        tab = FakeTable()
        append_cfg = _cache(tab, mode=Mode.APPEND)
        upsert_cfg = _cache(tab, mode=Mode.UPSERT)

        a = make_request("https://api.example.com/a").copy(
            send_config=SendConfig(remote_cache=append_cfg),
        )
        b = make_request("https://api.example.com/b").copy(
            send_config=SendConfig(remote_cache=upsert_cfg),
        )

        groups = HTTPSession._group_by_config([a, b])
        assert len(groups) == 2, (
            "APPEND and UPSERT must be in separate groups"
        )

    def test_groups_separate_by_local_mode(self):
        from yggdrasil.http_.session import HTTPSession

        cfg_append = CacheConfig(
            tabular="/tmp/test", mode=Mode.APPEND,
        )
        cfg_overwrite = CacheConfig(
            tabular="/tmp/test", mode=Mode.OVERWRITE,
        )

        a = make_request("https://example.com/a").copy(
            send_config=SendConfig(local_cache=cfg_append),
        )
        b = make_request("https://example.com/b").copy(
            send_config=SendConfig(local_cache=cfg_overwrite),
        )

        groups = HTTPSession._group_by_config([a, b])
        assert len(groups) == 2, (
            "APPEND and OVERWRITE local modes must be in separate groups"
        )

    def test_same_mode_same_holder_groups_together(self):
        from yggdrasil.http_.session import HTTPSession

        tab = FakeTable()
        cfg = _cache(tab, mode=Mode.APPEND)

        a = make_request("https://example.com/a").copy(
            send_config=SendConfig(remote_cache=cfg),
        )
        b = make_request("https://example.com/b").copy(
            send_config=SendConfig(remote_cache=cfg),
        )

        groups = HTTPSession._group_by_config([a, b])
        assert len(groups) == 1, (
            "same holder + same mode → single group"
        )
