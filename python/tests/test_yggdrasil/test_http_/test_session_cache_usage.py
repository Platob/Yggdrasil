"""Strict proofs that ``Session.send`` / ``Session.send_many`` actually
hit the local cache.

The existing integration suite in ``test_session_cache_integration.py``
covers the small-batch / single-request hit-skips-network contract.
The gap this file fills: high-cardinality / mixed-cardinality proofs
that the network call count stays at exactly ``misses`` (never one
more, never one less), and stress-shape checks that the cache short
circuits don't degrade as batches scale up. ``StubSession`` records
every call into ``self.calls`` so the assertions can be exact rather
than ``>= 0``.
"""
from __future__ import annotations

import pytest

from yggdrasil.io.session import Session

from ._helpers import StubSession, make_request, make_response
from .test_session_cache_integration import (
    _FakeRemoteTabular,
    _local_cfg,
    _remote_cfg,
    _seed_local,
    _seed_remote,
    _wait_for_local,
)


@pytest.fixture(autouse=True)
def _clear_session_singleton_cache():
    Session._INSTANCES.clear()
    yield
    Session._INSTANCES.clear()


# ===========================================================================
# Local cache — exact-count proofs
# ===========================================================================


class TestLocalCacheNetworkCount:

    def test_repeated_single_send_uses_cache_after_first(self, tmp_path) -> None:
        """Same request sent 10 times → exactly 1 network call."""
        cache = _local_cfg(tmp_path)
        req = make_request("https://example.com/repeat")
        s = StubSession()
        s.queue(make_response(request=req, body=b'{"v":1}'))

        first = s.send(req, local_cache=cache)
        assert first.local_cached is False
        assert _wait_for_local(cache) >= 1

        for _ in range(9):
            out = s.send(req, local_cache=cache)
            assert out.local_cached is True
            assert out.remote_cached is False

        assert len(s.calls) == 1, (
            f"expected exactly 1 network call across 10 sends, got {len(s.calls)}"
        )

    def test_large_batch_all_hits_zero_network(self, tmp_path) -> None:
        """200 pre-seeded requests in one batch → 0 network calls."""
        cache = _local_cfg(tmp_path)
        n = 200
        reqs = [make_request(f"https://example.com/h/{i:04d}") for i in range(n)]
        for i, r in enumerate(reqs):
            _seed_local(cache, make_response(request=r, body=f'{{"i":{i}}}'.encode()))

        s = StubSession()
        out = list(s.send_many(iter(reqs), local_cache=cache))

        assert len(s.calls) == 0, (
            f"all-hit batch must not touch the network, got {len(s.calls)} call(s)"
        )
        assert len(out) == n
        assert all(r.local_cached is True for r in out)
        bodies = {r.json()["i"] for r in out}
        assert bodies == set(range(n))

    def test_mixed_batch_network_count_equals_misses(self, tmp_path) -> None:
        """Mixed batch (73 hits, 127 misses) → exactly 127 network calls."""
        cache = _local_cfg(tmp_path)
        n_hit, n_miss = 73, 127
        hit_reqs = [
            make_request(f"https://example.com/hit/{i:04d}") for i in range(n_hit)
        ]
        miss_reqs = [
            make_request(f"https://example.com/miss/{i:04d}") for i in range(n_miss)
        ]
        for i, r in enumerate(hit_reqs):
            _seed_local(cache, make_response(request=r, body=f'{{"h":{i}}}'.encode()))

        # Interleave so the implementation can't sort-cheat its way to
        # the right count by accidentally bucketing all hits first.
        order: list = []
        for i in range(max(n_hit, n_miss)):
            if i < n_hit:
                order.append(hit_reqs[i])
            if i < n_miss:
                order.append(miss_reqs[i])

        s = StubSession()
        s.queue(*[
            make_response(request=r, body=f'{{"m":{i}}}'.encode())
            for i, r in enumerate(miss_reqs)
        ])

        out = list(s.send_many(iter(order), local_cache=cache))

        assert len(out) == n_hit + n_miss
        assert len(s.calls) == n_miss, (
            f"expected exactly {n_miss} network calls (one per miss), "
            f"got {len(s.calls)}"
        )
        cached_count = sum(1 for r in out if r.local_cached)
        assert cached_count == n_hit, (
            f"expected {n_hit} local_cached=True, got {cached_count}"
        )

    def test_second_batch_after_writeback_zero_network(self, tmp_path) -> None:
        """First batch writes back; second identical batch → 0 network."""
        cache = _local_cfg(tmp_path)
        n = 32
        reqs = [make_request(f"https://example.com/wb/{i:04d}") for i in range(n)]
        s = StubSession()
        s.queue(*[
            make_response(request=r, body=f'{{"i":{i}}}'.encode())
            for i, r in enumerate(reqs)
        ])

        first = list(s.send_many(iter(reqs), local_cache=cache))
        assert len(first) == n
        assert len(s.calls) == n, "first batch fully misses"
        assert _wait_for_local(cache, count=n) >= n

        second = list(s.send_many(iter(reqs), local_cache=cache))
        assert len(second) == n
        assert len(s.calls) == n, (
            f"second batch must reuse the disk; expected calls to stay at {n}, "
            f"got {len(s.calls)}"
        )
        assert all(r.local_cached is True for r in second)


# ===========================================================================
# cache_only — strict no-network proofs
# ===========================================================================


class TestCacheOnlyNoNetwork:

    def test_single_send_cache_only_hit(self, tmp_path) -> None:
        cache = _local_cfg(tmp_path)
        req = make_request("https://example.com/co/hit")
        _seed_local(cache, make_response(request=req, body=b'{"v":1}'))

        s = StubSession()
        out = s.send(req, local_cache=cache, cache_only=True)
        assert len(s.calls) == 0
        assert out.local_cached is True

    def test_single_send_cache_only_miss_returns_synthetic_404(self, tmp_path) -> None:
        cache = _local_cfg(tmp_path)
        req = make_request("https://example.com/co/miss")

        s = StubSession()
        out = s.send(req, local_cache=cache, cache_only=True, raise_error=False)
        assert out.status_code == 404
        assert out.tags.get("synthetic") == "cache_only_miss"
        assert len(s.calls) == 0, "cache_only miss must not fall back to network"

    def test_send_many_cache_only_synthesises_404_for_misses(self, tmp_path) -> None:
        """``send_many(cache_only=True)`` yields cached hits and synthetic
        404 responses for misses, with zero network touches."""
        cache = _local_cfg(tmp_path)
        hit = make_request("https://example.com/co/yes")
        miss = make_request("https://example.com/co/no")
        _seed_local(cache, make_response(request=hit, body=b'{"v":"hit"}'))

        s = StubSession()
        out = list(s.send_many(iter([hit, miss]), local_cache=cache, cache_only=True, raise_error=False))

        assert len(s.calls) == 0, "cache_only must never touch the network"
        assert len(out) == 2
        cached = [r for r in out if r.status_code == 200]
        synthetic = [r for r in out if r.status_code == 404]
        assert len(cached) == 1
        assert cached[0].json() == {"v": "hit"}
        assert cached[0].local_cached is True
        assert len(synthetic) == 1
        assert synthetic[0].tags.get("synthetic") == "cache_only_miss"


# ===========================================================================
# Combined local + remote — exact-count proofs
# ===========================================================================


class TestCombinedCacheNetworkCount:

    def test_local_then_remote_then_network(self, tmp_path) -> None:
        """One request: local miss → remote miss → 1 network call.
        Second send of same request: local hit (writeback landed) → 0
        network calls. Third send still 0."""
        local = _local_cfg(tmp_path)
        tab = _FakeRemoteTabular()
        remote = _remote_cfg(tab)

        req = make_request("https://example.com/combined")
        s = StubSession()
        s.queue(make_response(request=req, body=b'{"v":1}'))

        s.send(req, local_cache=local, remote_cache=remote)
        assert len(s.calls) == 1
        assert _wait_for_local(local) >= 1

        # Second send must be served by local (faster than remote).
        out2 = s.send(req, local_cache=local, remote_cache=remote)
        assert len(s.calls) == 1
        assert out2.local_cached is True
        assert out2.remote_cached is False

        # Third — same.
        s.send(req, local_cache=local, remote_cache=remote)
        assert len(s.calls) == 1

    def test_local_miss_remote_hit_backfills_local(self, tmp_path) -> None:
        """Local empty + remote seeded → 0 network calls; second send
        comes off local because the remote hit backfilled it."""
        local = _local_cfg(tmp_path)
        tab = _FakeRemoteTabular()
        remote = _remote_cfg(tab)
        req = make_request("https://example.com/backfill")
        _seed_remote(tab, make_response(request=req, body=b'{"v":"remote"}'))

        s = StubSession()
        out1 = s.send(req, local_cache=local, remote_cache=remote)
        assert len(s.calls) == 0
        assert out1.remote_cached is True
        assert _wait_for_local(local) >= 1, "remote hit must backfill local"

        out2 = s.send(req, local_cache=local, remote_cache=remote)
        assert len(s.calls) == 0
        # Now served by local — the backfill landed.
        assert out2.local_cached is True
        assert out2.remote_cached is False


# ===========================================================================
# Local cache — vectorisation proofs
# ===========================================================================


class TestLocalCacheVectorization:
    """Strict proofs that the local-cache layer batches every
    :meth:`FolderPath.read_arrow_batches` /
    :meth:`FolderPath.write_arrow_batches` call across a whole
    :meth:`Session.send_many` chunk.

    The cache layer's whole reason for being is that it amortises
    backend overhead (file open, predicate compile, schema rebuild)
    across the chunk. A regression that drops to a per-request
    read / write would silently 10x the cache-hit latency. These
    tests pin the call counts so a future refactor that breaks
    vectorisation fails loud instead of quiet.
    """

    def _patch_counter(self, monkeypatch, method_name: str) -> "list[int]":
        from yggdrasil.io.nested.folder_path import FolderPath

        calls: list[int] = []
        original = getattr(FolderPath, method_name)

        def wrapper(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(FolderPath, method_name, wrapper)
        return calls

    def test_send_many_local_hits_use_single_batched_read(
        self, tmp_path, monkeypatch,
    ) -> None:
        """64 seeded requests → ONE `FolderPath.read_arrow_batches`."""
        cache = _local_cfg(tmp_path)
        n = 64
        reqs = [make_request(f"https://example.com/v/{i:04d}") for i in range(n)]
        for i, r in enumerate(reqs):
            _seed_local(cache, make_response(request=r, body=f'{{"i":{i}}}'.encode()))

        s = StubSession()
        # Counter has to start AFTER seeding (each ``_seed_local`` calls
        # ``write_arrow_batches`` once — those aren't the ones we measure).
        read_calls = self._patch_counter(monkeypatch, "read_arrow_batches")

        out = list(s.send_many(iter(reqs), local_cache=cache))

        assert len(out) == n
        assert all(r.local_cached is True for r in out)
        assert len(s.calls) == 0, "all-hit chunk must not touch the network"
        # One folder root, one chunk → exactly one batched read across
        # all 64 partition_key values. Any per-request read would show
        # 64 here.
        assert len(read_calls) == 1, (
            f"local cache should serve {n} hits with ONE batched read; "
            f"saw {len(read_calls)} read(s) — vectorisation regressed."
        )

    def test_send_many_misses_use_single_batched_writeback(
        self, tmp_path, monkeypatch,
    ) -> None:
        """64 cache-miss writebacks → ONE `FolderPath.write_arrow_batches`."""
        cache = _local_cfg(tmp_path)
        n = 64
        reqs = [make_request(f"https://example.com/w/{i:04d}") for i in range(n)]

        s = StubSession()
        for r in reqs:
            s.queue(make_response(request=r, body=b'{"ok":true}'))

        # Counter starts BEFORE send_many so the writebacks land in the
        # counted window. We also need the read counter to confirm the
        # initial miss-scan ran exactly once.
        write_calls = self._patch_counter(monkeypatch, "write_arrow_batches")
        read_calls = self._patch_counter(monkeypatch, "read_arrow_batches")

        out = list(s.send_many(iter(reqs), local_cache=cache))

        assert len(out) == n
        assert len(s.calls) == n, "every miss must hit the network exactly once"
        # One root, one chunk: one batched miss-scan read, then one
        # batched writeback that drains every fetched response in a
        # single ``values_to_arrow_batch`` walk + one ``_insert_cache``.
        # Wait for the fire-and-forget writeback to drain before
        # asserting.
        _wait_for_local(cache, count=n)
        assert len(read_calls) == 1, (
            f"miss-scan must be ONE batched read across {n} requests; "
            f"saw {len(read_calls)}."
        )
        assert len(write_calls) == 1, (
            f"writeback must be ONE batched write across {n} responses; "
            f"saw {len(write_calls)} — vectorisation regressed."
        )

    def test_send_many_mixed_hits_misses_uses_one_read_one_write(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Mixed batch: 1 read for the lookup + 1 write for the misses' writeback."""
        cache = _local_cfg(tmp_path)
        n_hit, n_miss = 32, 32
        hit_reqs = [
            make_request(f"https://example.com/mh/{i:04d}") for i in range(n_hit)
        ]
        miss_reqs = [
            make_request(f"https://example.com/mm/{i:04d}") for i in range(n_miss)
        ]
        for i, r in enumerate(hit_reqs):
            _seed_local(cache, make_response(request=r, body=f'{{"h":{i}}}'.encode()))

        s = StubSession()
        for r in miss_reqs:
            s.queue(make_response(request=r, body=b'{"ok":true}'))

        write_calls = self._patch_counter(monkeypatch, "write_arrow_batches")
        read_calls = self._patch_counter(monkeypatch, "read_arrow_batches")

        # Pre-shuffle so the lookup predicate covers a non-trivial mix.
        all_reqs = []
        for h, m in zip(hit_reqs, miss_reqs):
            all_reqs.extend((h, m))
        out = list(s.send_many(iter(all_reqs), local_cache=cache))

        assert len(out) == n_hit + n_miss
        assert len(s.calls) == n_miss, "only misses should hit the network"

        # Wait for the writeback of the misses to drain.
        _wait_for_local(cache, count=n_hit + n_miss)

        assert len(read_calls) == 1, (
            f"split_local_cache must scan with ONE batched read; "
            f"saw {len(read_calls)}."
        )
        assert len(write_calls) == 1, (
            f"writeback must be ONE batched write across {n_miss} new responses; "
            f"saw {len(write_calls)}."
        )
