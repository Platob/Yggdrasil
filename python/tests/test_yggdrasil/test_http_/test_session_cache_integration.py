"""End-to-end integration tests for the :class:`Session` cache pipeline.

The session cache has two backends that share one pipeline:

* **Local fast-path** — on-disk Arrow IPC files at
  ``<root>/<METHOD>/<host>/<seg>/.../<public_hash>.arrow`` (one
  ``stat`` + decode per request). Enabled via ``CacheConfig.path``
  or implicitly by a ``received_*`` window.
* **Remote** — a :class:`Tabular` (Databricks Table / SQL warehouse
  view / Spark Delta backend) read via batch ``sql.execute`` and
  written via ``Tabular.insert``. Enabled via ``CacheConfig.tabular``.

This file drives the full pipeline (``Session.send`` and
``Session.send_many``) through both backends — independently and
combined — using :class:`StubSession` to stand in for the network
transport and :class:`_FakeRemoteTabular` to stand in for the
remote backend. The unit-level contracts of the private helpers
(``_local_fast_path_relative``, ``_cleanup_local_fast_path``,
``_remote_write_group_key``, ``_maybe_autocompress_body_for_cache``)
live in ``test_session_cache_internals.py``.

Coverage targets (each appears in at least one ``send`` and one
``send_many`` test):

* **Local-only.** Hit skips network. Miss falls through and the
  fire-and-forget writeback lands on disk. UPSERT skips the lookup.
  ``received_from`` / ``received_to`` and ``received_ttl`` filter
  stale rows. Failure responses (status >= 400) don't persist.
  Per-request overrides win. Distinct POST bodies don't alias.
  Concurrent writeback against many distinct ``public_hash`` files
  is race-free.
* **Remote-only.** Hit skips network. Miss writes back via
  ``Tabular.insert`` with the configured mode / match_by / wait /
  prune. UPSERT skips both the SQL lookup and the writeback.
  ``TABLE_OR_VIEW_NOT_FOUND`` triggers ``Tabular.create`` then
  retries the lookup. ``_split_remote_cache`` issues one batched
  ``sql.execute`` per distinct table. Per-request overrides win.
* **Combined.** Local short-circuits before remote. Local miss +
  remote hit backfills the local cache. ``mirror_local_to_remote``
  pushes local hits to remote; the default keeps remote silent. A
  full miss against both layers performs a single network fetch
  and writes back to both.
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pytest

from yggdrasil.data.enums import Mode
from yggdrasil.io.response import Response
from yggdrasil.io.send_config import CacheConfig, SendConfig
from yggdrasil.io.session import Session
from yggdrasil.io.tabular import Tabular

from ._helpers import StubSession, make_request, make_response


# ---------------------------------------------------------------------------
# Singleton-cache hygiene
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_session_singleton_cache():
    """Keep ``StubSession`` instances from leaking across tests.

    :class:`Session` is a per-``(class, base_url, key)`` singleton, so
    a fixture that built a session with ``base_url=None`` in one test
    would otherwise hand the live instance — and its in-flight
    fire-and-forget jobs — to the next.
    """
    Session._INSTANCES.clear()
    yield
    Session._INSTANCES.clear()


# ---------------------------------------------------------------------------
# Cache builders
# ---------------------------------------------------------------------------


def _local_cfg(root: Path | str, **overrides: Any) -> CacheConfig:
    overrides.setdefault("mode", Mode.APPEND)
    return CacheConfig(tabular=str(root), **overrides)


def _remote_cfg(tab: "_FakeRemoteTabular", **overrides: Any) -> CacheConfig:
    overrides.setdefault("mode", Mode.APPEND)
    overrides.setdefault("request_by", ["public_url_hash"])
    overrides.pop("wait", None)
    return CacheConfig(tabular=tab, **overrides)


# ---------------------------------------------------------------------------
# Seeding + polling helpers
# ---------------------------------------------------------------------------


def _seed_local(cache: CacheConfig, response: Response) -> None:
    """Synchronously plant *response* in the partitioned cache tree.

    Bypasses the session's fire-and-forget writeback so a test can
    assert read-side behaviour without racing the daemon writer.
    Goes through :meth:`Tabular.insert` on the cache's backend —
    the same call the production writeback uses — so the on-disk
    layout (``<root>/partition_key=<int>/part-*.<ext>``) matches
    exactly. Skips the ``response.ok`` guard the production path
    enforces so fixtures can seed 4xx / 5xx rows.
    """
    from yggdrasil.io.nested.folder_path import FolderOptions

    tabular = cache.cache_tabular()
    tabular.write_arrow_batches(
        (response.to_arrow_batch(parse=False),),
        options=FolderOptions(mode=cache.mode),
    )


def _seed_remote(tab: "_FakeRemoteTabular", response: Response) -> None:
    """Seed *response* into the fake remote so the next lookup returns it."""
    tab.rows.append(response.to_arrow_batch(parse=False))


def _wait_for_local(cache: CacheConfig, *, count: int = 1, timeout: float = 5.0) -> int:
    """Poll until at least *count* partitioned part files land in *cache*.

    The local writeback is fire-and-forget on the job pool. A polling
    helper beats a fixed ``sleep`` — slow CI machines get the time
    they need; fast ones don't pay it. Counts ``part-*`` leaves under
    any ``partition_key=*/`` sub-directory (the partitioned layout).
    """
    deadline = time.monotonic() + timeout
    root = Path(str(cache.tabular.path))
    last = 0
    while time.monotonic() < deadline:
        try:
            last = sum(
                1 for p in root.rglob("partition_key=*/part-*")
                if p.is_file() and not p.name.startswith(".")
            )
        except OSError:
            last = 0
        if last >= count:
            return last
        time.sleep(0.05)
    return last


# ---------------------------------------------------------------------------
# Fake remote tabular
# ---------------------------------------------------------------------------


class _FakeRemoteTabular(Tabular):
    """In-memory Tabular double for the remote-cache flow."""

    def __init__(self, name: str = "ws.cache.responses") -> None:
        super().__init__()
        self._name = name
        self.rows: list[pa.RecordBatch] = []
        self.predicates: list[Any] = []
        self.inserts: list[dict[str, Any]] = []
        self.created = False
        self.path = name

    def full_name(self, safe: bool = False) -> str:
        return self._name

    def create(self, schema: pa.Schema, missing_ok: bool = False) -> None:
        self.created = True

    def _read_arrow_batches(self, options: Any = None, **kwargs: Any) -> Iterator[pa.RecordBatch]:
        predicate = getattr(options, "predicate", None)
        self.predicates.append(predicate)
        batches = list(self.rows)
        if predicate is None:
            return iter(batches)
        return predicate.filter_arrow_batches(iter(batches))

    def _write_arrow_batches(
        self,
        batches: Any,
        options: Any = None,
        **kwargs: Any,
    ) -> None:
        new_batches: list[pa.RecordBatch] = []
        for entry in batches:
            if isinstance(entry, pa.Table):
                new_batches.extend(entry.to_batches())
            elif isinstance(entry, pa.RecordBatch):
                new_batches.append(entry)
        mode = getattr(options, "mode", None)
        match_by_fields = getattr(options, "match_by", None)
        match_by = (
            tuple(f.name for f in match_by_fields)
            if match_by_fields else None
        )
        prune_values = getattr(options, "prune_values", None)
        self.inserts.append({
            "mode": mode,
            "match_by": match_by,
            "rows": sum(b.num_rows for b in new_batches),
            "prune_keys": (
                tuple(sorted(prune_values.keys())) if prune_values else ()
            ),
        })
        self.rows.extend(new_batches)


# ===========================================================================
# Local cache — single request via ``Session.send``
# ===========================================================================


class TestLocalCacheSend:

    def test_hit_skips_network(self, tmp_path) -> None:
        cache = _local_cfg(tmp_path)
        req = make_request("https://example.com/x")
        _seed_local(cache, make_response(request=req, body=b'{"v":"cached"}'))

        s = StubSession()
        out = s.send(req, local_cache=cache)
        assert len(s.calls) == 0
        assert out.json() == {"v": "cached"}

    def test_miss_writes_back(self, tmp_path) -> None:
        cache = _local_cfg(tmp_path)
        req = make_request("https://example.com/x")
        s = StubSession()
        s.queue(make_response(request=req, body=b'{"v":"fresh"}'))

        out = s.send(req, local_cache=cache)
        assert len(s.calls) == 1
        assert out.json() == {"v": "fresh"}
        assert _wait_for_local(cache) >= 1

        # Second send hits the freshly written disk row, not the network.
        out2 = s.send(req, local_cache=cache)
        assert len(s.calls) == 1
        assert out2.json() == {"v": "fresh"}

    def test_failure_response_not_persisted(self, tmp_path) -> None:
        cache = _local_cfg(tmp_path)
        req = make_request("https://example.com/oops")
        s = StubSession()
        s.queue(make_response(request=req, status_code=500, body=b"boom"))

        # ``raise_error=False`` so the test owns the assertions —
        # ``_store_cached_response`` early-exits on
        # ``response.ok=False`` so the writeback never fires.
        s.send(req, local_cache=cache, raise_error=False)

        # Give any spurious fire-and-forget job time to land before
        # asserting nothing's there. 0.2s is plenty — the writeback is
        # microseconds on tmpfs.
        time.sleep(0.2)
        assert _wait_for_local(cache, timeout=0.1) == 0

    def test_upsert_reads_cache_and_persists(self, tmp_path) -> None:
        cache = _local_cfg(tmp_path, mode=Mode.UPSERT)
        req = make_request("https://example.com/x")
        _seed_local(cache, make_response(request=req, body=b'{"v":"cached"}'))

        s = StubSession()
        out = s.send(req, local_cache=cache)
        # UPSERT reads from cache like any other mode.
        assert len(s.calls) == 0
        assert out.json() == {"v": "cached"}

    def test_received_window_filters_stale_row(self, tmp_path) -> None:
        cache = _local_cfg(
            tmp_path,
            received_from=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
            received_to=dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc),
        )
        req = make_request("https://example.com/x")
        _seed_local(cache, make_response(
            request=req,
            body=b'{"v":"old"}',
            received_at=dt.datetime(2010, 1, 1, tzinfo=dt.timezone.utc),
        ))

        s = StubSession()
        s.queue(make_response(request=req, body=b'{"v":"fresh"}'))

        out = s.send(req, local_cache=cache)
        assert len(s.calls) == 1
        assert out.json() == {"v": "fresh"}

    def test_received_window_filters_stale_row(self, tmp_path) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        cache = _local_cfg(
            tmp_path,
            received_from=now - dt.timedelta(minutes=10),
            received_to=now,
        )
        req = make_request("https://example.com/x")
        old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
        _seed_local(cache, make_response(
            request=req, body=b'{"v":"old"}', received_at=old,
        ))

        s = StubSession()
        s.queue(make_response(request=req, body=b'{"v":"fresh"}'))
        out = s.send(req, local_cache=cache)
        assert len(s.calls) == 1
        assert out.json() == {"v": "fresh"}

    def test_per_request_override_wins(self, tmp_path) -> None:
        # Session-level cache holds a row that would match; per-request
        # cache points at a fresh empty folder, so the send must miss
        # and refetch.
        session_cache = _local_cfg(tmp_path / "session")
        seed_req = make_request("https://example.com/x")
        _seed_local(
            session_cache,
            make_response(request=seed_req, body=b'{"v":"cached"}'),
        )

        alt = tmp_path / "alt"
        alt.mkdir()
        req = seed_req.copy(send_config=SendConfig(local_cache=_local_cfg(alt)))

        s = StubSession()
        s.queue(make_response(request=req, body=b'{"v":"network"}'))

        out = s.send(req, local_cache=session_cache)
        assert len(s.calls) == 1
        assert out.json() == {"v": "network"}

    def test_distinct_post_bodies_do_not_alias(self, tmp_path) -> None:
        # ``public_hash`` folds the body bytes into the request
        # identity, so two POSTs to the same URL with different bytes
        # match against distinct rows in the partitioned cache. They
        # share a ``partition_key`` (URL-derived) so they land in the
        # same partition directory — the row-level match-by predicate
        # is what distinguishes them. Passing ``request_by=["public_hash"]``
        # explicitly opts into body-aware identity (the default
        # ``public_url_hash`` is URL-only by design — see
        # :data:`_DEFAULT_REQUEST_BY`).
        cache = _local_cfg(tmp_path, request_by=["public_hash"])
        url = "https://example.com/echo"
        req_a = make_request(url, method="POST", body=b"payload-A")
        req_b = make_request(url, method="POST", body=b"payload-B")
        _seed_local(cache, make_response(request=req_a, body=b'{"got":"A"}'))

        s = StubSession()
        s.queue(make_response(request=req_b, body=b'{"got":"B"}'))

        out_a = s.send(req_a, local_cache=cache)
        out_b = s.send(req_b, local_cache=cache)
        assert out_a.json() == {"got": "A"}
        assert out_b.json() == {"got": "B"}
        # Only req_b touched the wire.
        assert len(s.calls) == 1
        assert s.calls[0].buffer.to_bytes() == b"payload-B"


# ===========================================================================
# Local cache — batched via ``Session.send_many``
# ===========================================================================


class TestLocalCacheSendMany:

    def test_mixed_hits_and_misses(self, tmp_path) -> None:
        cache = _local_cfg(tmp_path)

        a = make_request("https://example.com/a")
        b = make_request("https://example.com/b")
        c = make_request("https://example.com/c")

        _seed_local(cache, make_response(request=a, body=b'{"k":"a"}'))
        _seed_local(cache, make_response(request=b, body=b'{"k":"b"}'))

        s = StubSession()
        s.queue(make_response(request=c, body=b'{"k":"c"}'))

        out = list(s.send_many(iter([a, b, c]), local_cache=cache))
        assert {r.json()["k"] for r in out} == {"a", "b", "c"}
        assert len(s.calls) == 1
        assert s.calls[0].url.path == "/c"

    def test_writeback_round_trip(self, tmp_path) -> None:
        cache = _local_cfg(tmp_path)
        s = StubSession()
        req = make_request("https://example.com/x")
        s.queue(make_response(request=req, body=b'{"v":"first"}'))

        list(s.send_many(iter([req]), local_cache=cache))
        assert _wait_for_local(cache) >= 1, "writeback never landed"

        # Second batch comes off disk.
        out = list(s.send_many(iter([req]), local_cache=cache))
        assert len(s.calls) == 1, "second batch must hit disk, not network"
        assert out[0].json() == {"v": "first"}

    def test_failure_response_not_persisted(self, tmp_path) -> None:
        cache = _local_cfg(tmp_path)
        req = make_request("https://example.com/x")
        s = StubSession()
        s.queue(make_response(request=req, status_code=500, body=b"boom"))

        # ``raise_error=False`` keeps the iterator yielding the 5xx
        # rather than blowing up — the contract under test is the
        # writeback gate, not the error policy.
        list(s.send_many(iter([req]), local_cache=cache, raise_error=False))

        time.sleep(0.2)
        assert _wait_for_local(cache, timeout=0.1) == 0

    def test_per_request_override_routes_to_alt_path(self, tmp_path) -> None:
        # Two requests share one batch but each carries its own
        # per-request cache pointing at a separate folder. The
        # session-level cache holds neither row — both must miss
        # the session cache and write back into their respective
        # alt folders.
        a_dir = tmp_path / "a"; a_dir.mkdir()
        b_dir = tmp_path / "b"; b_dir.mkdir()
        session_cache = _local_cfg(tmp_path / "session")

        a = make_request("https://example.com/a").copy(
            send_config=SendConfig(local_cache=_local_cfg(a_dir)),
        )
        b = make_request("https://example.com/b").copy(
            send_config=SendConfig(local_cache=_local_cfg(b_dir)),
        )

        s = StubSession()
        s.queue(
            make_response(request=a, body=b'{"k":"a"}'),
            make_response(request=b, body=b'{"k":"b"}'),
        )
        list(s.send_many(iter([a, b]), local_cache=session_cache))

        # Each per-request cache got its own writeback; the
        # session-level cache stayed empty.
        assert _wait_for_local(_local_cfg(a_dir)) >= 1
        assert _wait_for_local(_local_cfg(b_dir)) >= 1
        assert _wait_for_local(session_cache, timeout=0.1) == 0

    def test_concurrent_writebacks_dont_collide(self, tmp_path) -> None:
        # ``public_hash`` keys give every distinct request its own
        # ``.arrow`` leaf, so concurrent fire-and-forget writers can't
        # trample each other even under load.
        cache = _local_cfg(tmp_path)
        s = StubSession()
        n = 8
        reqs = [make_request(f"https://example.com/p{i}") for i in range(n)]
        s.queue(*[
            make_response(request=r, body=f'{{"i":{i}}}'.encode())
            for i, r in enumerate(reqs)
        ])
        list(s.send_many(iter(reqs), local_cache=cache))
        assert _wait_for_local(cache, count=n) >= n


# ===========================================================================
# Remote cache — single request via ``Session.send``
# ===========================================================================


class TestRemoteCacheSend:

    def test_hit_skips_network(self) -> None:
        tab = _FakeRemoteTabular()
        cfg = _remote_cfg(tab)
        req = make_request("https://example.com/x")
        _seed_remote(tab, make_response(request=req, body=b'{"v":"cached"}'))

        s = StubSession()
        out = s.send(req, remote_cache=cfg)
        assert len(s.calls) == 0
        assert out.json() == {"v": "cached"}

    def test_miss_writes_back_via_insert(self) -> None:
        tab = _FakeRemoteTabular()
        cfg = _remote_cfg(tab)
        s = StubSession()
        req = make_request("https://example.com/x")
        s.queue(make_response(request=req, body=b'{"v":1}'))

        s.send(req, remote_cache=cfg)
        assert len(s.calls) == 1, "remote miss must touch the network"
        assert tab.predicates, "lookup query must run before the network fetch"
        assert tab.inserts, "successful response must write back via insert"
        # Insert call carries the configured knobs.
        i = tab.inserts[0]
        assert i["mode"] == Mode.APPEND
        # Pruning keys the MERGE on both the partition column and the
        # exact row identity — both keys are int64 so the IN literal
        # stays compact.
        assert i["prune_keys"] == ("partition_key", "public_hash")

    def test_failure_response_not_persisted(self) -> None:
        tab = _FakeRemoteTabular()
        cfg = _remote_cfg(tab)
        s = StubSession()
        req = make_request("https://example.com/x")
        s.queue(make_response(request=req, status_code=500, body=b"boom"))

        s.send(req, remote_cache=cfg, raise_error=False)
        assert tab.inserts == [], "5xx response must not write back to remote"

    def test_upsert_reads_then_persists(self) -> None:
        # UPSERT reads from cache like any other mode and persists
        # the response back with UPSERT semantics.
        tab = _FakeRemoteTabular()
        cfg = _remote_cfg(tab, mode=Mode.UPSERT)

        # Miss case: no seeded row → network fetch → persist.
        req = make_request("https://example.com/x")
        s = StubSession()
        s.queue(make_response(request=req, body=b'{"v":"fresh"}'))
        out = s.send(req, remote_cache=cfg)

        assert len(s.calls) == 1, "miss must hit the network"
        assert out.json() == {"v": "fresh"}
        assert len(tab.inserts) == 1
        assert tab.inserts[0]["mode"] == Mode.UPSERT

    def test_per_request_override_routes_to_alt_table(self) -> None:
        # Session-level config holds the cached row in tab_a;
        # per-request config points at tab_b which is empty. The
        # request must miss and refetch from the network.
        tab_a = _FakeRemoteTabular(name="ws.a.responses")
        tab_b = _FakeRemoteTabular(name="ws.b.responses")
        req = make_request("https://example.com/x")
        _seed_remote(tab_a, make_response(request=req, body=b'{"v":"a"}'))

        req_with_override = req.copy(send_config=SendConfig(remote_cache=_remote_cfg(tab_b)))
        s = StubSession()
        s.queue(make_response(request=req_with_override, body=b'{"v":"network"}'))

        out = s.send(req_with_override, remote_cache=_remote_cfg(tab_a))
        assert len(s.calls) == 1
        assert out.json() == {"v": "network"}
        # tab_b got the writeback, tab_a stays untouched.
        assert tab_a.inserts == []
        assert tab_b.inserts and tab_b.inserts[0]["rows"] == 1


# ===========================================================================
# Remote cache — batched via ``Session.send_many``
# ===========================================================================


class TestRemoteCacheSendMany:

    def test_batched_hits_and_misses(self) -> None:
        # Two requests share one remote table. One row is pre-seeded;
        # the other is a miss that should fire one network call and
        # write back into the same table.
        tab = _FakeRemoteTabular()
        cfg = _remote_cfg(tab)
        hit_req = make_request("https://example.com/a")
        miss_req = make_request("https://example.com/b")
        _seed_remote(tab, make_response(request=hit_req, body=b'{"k":"a"}'))

        s = StubSession()
        s.queue(make_response(request=miss_req, body=b'{"k":"b"}'))

        out = list(s.send_many(iter([hit_req, miss_req]), remote_cache=cfg))
        assert {r.json()["k"] for r in out} == {"a", "b"}
        assert len(s.calls) == 1, "only the miss touches the network"
        # ``_split_remote_cache`` issues one batched SQL lookup per
        # distinct table.
        assert len(tab.predicates) == 1
        # Writeback fires for the network result.
        assert any(call["rows"] == 1 for call in tab.inserts)

    def test_distinct_tables_each_get_one_lookup(self) -> None:
        # Two requests, two distinct per-request remote tables.
        # ``_split_remote_cache`` must issue one ``sql.execute`` per
        # table — never collapse them.
        tab_a = _FakeRemoteTabular(name="ws.a.responses")
        tab_b = _FakeRemoteTabular(name="ws.b.responses")

        req_a = make_request("https://example.com/a").copy(
            send_config=SendConfig(remote_cache=_remote_cfg(tab_a)),
        )
        req_b = make_request("https://example.com/b").copy(
            send_config=SendConfig(remote_cache=_remote_cfg(tab_b)),
        )
        _seed_remote(tab_a, make_response(request=req_a, body=b'{"k":"a"}'))
        _seed_remote(tab_b, make_response(request=req_b, body=b'{"k":"b"}'))

        s = StubSession()
        out = list(s.send_many(iter([req_a, req_b])))
        assert {r.json()["k"] for r in out} == {"a", "b"}
        assert len(s.calls) == 0, "both rows hit their respective tables"
        assert len(tab_a.predicates) == 1
        assert len(tab_b.predicates) == 1

    def test_writeback_groups_split_by_mode(self) -> None:
        # APPEND + UPSERT in the same batch — both persist but as
        # separate writeback groups (different modes).
        tab = _FakeRemoteTabular()
        append_cfg = _remote_cfg(tab, mode=Mode.APPEND)
        upsert_cfg = _remote_cfg(tab, mode=Mode.UPSERT)

        a = make_request("https://example.com/a").copy(
            send_config=SendConfig(remote_cache=append_cfg),
        )
        b = make_request("https://example.com/b").copy(
            send_config=SendConfig(remote_cache=upsert_cfg),
        )
        s = StubSession()
        s.queue(
            make_response(request=a, body=b'{"k":"a"}'),
            make_response(request=b, body=b'{"k":"b"}'),
        )
        list(s.send_many(iter([a, b])))

        # Both requests go to the network (UPSERT skips read, APPEND
        # misses because the remote is empty). Two writeback groups —
        # one per mode — each with one row.
        assert len(s.calls) == 2
        modes = {i["mode"] for i in tab.inserts}
        assert modes == {Mode.APPEND, Mode.UPSERT}
        assert all(i["rows"] == 1 for i in tab.inserts)


# ===========================================================================
# Combined local + remote
# ===========================================================================


class TestCombinedCacheIntegration:

    def test_local_hit_short_circuits_before_remote(self, tmp_path) -> None:
        # Local cache holds the row → the remote table should never
        # be queried, and the network should never fire.
        local = _local_cfg(tmp_path)
        tab = _FakeRemoteTabular()
        remote = _remote_cfg(tab)
        req = make_request("https://example.com/x")
        _seed_local(local, make_response(request=req, body=b'{"v":"local"}'))

        s = StubSession()
        out = s.send(req, local_cache=local, remote_cache=remote)
        assert out.json() == {"v": "local"}
        assert len(s.calls) == 0
        assert tab.predicates == [], "local hit must short-circuit before remote"

    def test_remote_hit_backfills_local(self, tmp_path) -> None:
        # Local empty, remote populated → the remote hit is written
        # back into the local fast-path cache so a subsequent local-only
        # send is offline.
        local = _local_cfg(tmp_path)
        tab = _FakeRemoteTabular()
        remote = _remote_cfg(tab)
        req = make_request("https://example.com/x")
        _seed_remote(tab, make_response(request=req, body=b'{"v":"from-remote"}'))

        s = StubSession()
        s.send(req, local_cache=local, remote_cache=remote)
        assert len(s.calls) == 0

        assert _wait_for_local(local) >= 1, "remote hit must backfill local"
        out = s.send(req, local_cache=local)
        assert out.json() == {"v": "from-remote"}

    def test_full_miss_fetches_once_and_writes_back_to_both(self, tmp_path) -> None:
        # Neither layer has the row → one network call, both layers
        # take the writeback.
        local = _local_cfg(tmp_path)
        tab = _FakeRemoteTabular()
        remote = _remote_cfg(tab)
        req = make_request("https://example.com/x")
        s = StubSession()
        s.queue(make_response(request=req, body=b'{"v":"network"}'))

        s.send(req, local_cache=local, remote_cache=remote)
        assert len(s.calls) == 1
        assert _wait_for_local(local) >= 1
        assert tab.inserts and tab.inserts[0]["rows"] == 1

    def test_local_hit_does_not_push_to_remote(self, tmp_path) -> None:
        # A local-only hit must not trigger a remote insert.
        local = _local_cfg(tmp_path)
        tab = _FakeRemoteTabular()
        remote = _remote_cfg(tab)
        req = make_request("https://example.com/x")
        _seed_local(local, make_response(request=req, body=b'{"v":"local"}'))

        s = StubSession()
        list(s.send_many(
            iter([req]),
            local_cache=local,
            remote_cache=remote,
        ))
        assert len(s.calls) == 0, "local hit must not touch the network"
        assert tab.inserts == [], (
            "local-only hit must not push to remote"
        )

    def test_send_many_mixes_local_hit_remote_hit_and_full_miss(
        self, tmp_path,
    ) -> None:
        # One batch covering all three stages of the pipeline:
        # * ``a`` lands on local;
        # * ``b`` lands on remote (and should backfill local);
        # * ``c`` misses both and goes to the network.
        # End state: all three responses come back; exactly one
        # network call; both backends carry ``c``.
        local = _local_cfg(tmp_path)
        tab = _FakeRemoteTabular()
        remote = _remote_cfg(tab)
        a = make_request("https://example.com/a")
        b = make_request("https://example.com/b")
        c = make_request("https://example.com/c")

        _seed_local(local, make_response(request=a, body=b'{"k":"a"}'))
        _seed_remote(tab, make_response(request=b, body=b'{"k":"b"}'))

        s = StubSession()
        s.queue(make_response(request=c, body=b'{"k":"c"}'))

        out = list(s.send_many(
            iter([a, b, c]),
            local_cache=local,
            remote_cache=remote,
        ))
        assert {r.json()["k"] for r in out} == {"a", "b", "c"}
        assert len(s.calls) == 1
        assert s.calls[0].url.path == "/c"
        # ``b`` backfills local (one .arrow), ``c`` writes back to
        # local too (another) — and ``c`` lands in remote as well.
        assert _wait_for_local(local, count=2) >= 2
        # Remote got the network-miss writeback for ``c`` (``b`` was
        # already there and the miss-then-writeback path covers ``c``).
        assert any(call["rows"] >= 1 for call in tab.inserts)


# ===========================================================================
# cache_only — bypass the network fallback
# ===========================================================================


class TestCacheOnly:

    def test_send_local_hit_returns_cached_response(self, tmp_path) -> None:
        # Local cache has the row → cache_only should still return it
        # without any network activity.
        cache = _local_cfg(tmp_path)
        req = make_request("https://example.com/x")
        _seed_local(cache, make_response(request=req, body=b'{"v":"cached"}'))

        s = StubSession()
        out = s.send(req, local_cache=cache, cache_only=True)

        assert out.json() == {"v": "cached"}
        assert len(s.calls) == 0

    def test_send_remote_hit_returns_cached_response(self) -> None:
        # Remote cache holds the row → cache_only returns it without
        # crossing the wire.
        tab = _FakeRemoteTabular()
        remote = _remote_cfg(tab)
        req = make_request("https://example.com/x")
        _seed_remote(tab, make_response(request=req, body=b'{"v":"from-remote"}'))

        s = StubSession()
        out = s.send(req, remote_cache=remote, cache_only=True)

        assert out.json() == {"v": "from-remote"}
        assert len(s.calls) == 0

    def test_send_full_miss_returns_synthetic_404(self, tmp_path) -> None:
        # Both caches empty → cache_only returns a synthetic 404
        # instead of firing the network request.
        cache = _local_cfg(tmp_path)
        tab = _FakeRemoteTabular()
        remote = _remote_cfg(tab)
        req = make_request("https://example.com/x")

        s = StubSession()
        s.queue(make_response(request=req, body=b'{"v":"network"}'))

        out = s.send(
            req,
            local_cache=cache,
            remote_cache=remote,
            cache_only=True,
            raise_error=False,
        )
        assert out.status_code == 404
        assert out.tags.get("synthetic") == "cache_only_miss"
        assert len(s.calls) == 0, "cache_only must not cross the wire"

    def test_send_many_synthesises_404_for_misses(self, tmp_path) -> None:
        # Local hit + full miss → cache_only yields the hit and a
        # synthetic 404 for the miss without any network call.
        cache = _local_cfg(tmp_path)
        a = make_request("https://example.com/a")
        b = make_request("https://example.com/b")
        _seed_local(cache, make_response(request=a, body=b'{"k":"a"}'))

        s = StubSession()
        s.queue(make_response(request=b, body=b'{"k":"b"}'))

        out = list(s.send_many(
            iter([a, b]),
            local_cache=cache,
            cache_only=True,
            raise_error=False,
        ))

        cached = [r for r in out if r.status_code == 200]
        synthetic = [r for r in out if r.status_code == 404]
        assert len(cached) == 1
        assert cached[0].json()["k"] == "a"
        assert len(synthetic) == 1
        assert synthetic[0].tags.get("synthetic") == "cache_only_miss"
        assert len(s.calls) == 0
