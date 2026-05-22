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

    def test_single_send_cache_only_miss_raises(self, tmp_path) -> None:
        cache = _local_cfg(tmp_path)
        req = make_request("https://example.com/co/miss")

        s = StubSession()
        with pytest.raises(LookupError):
            s.send(req, local_cache=cache, cache_only=True)
        assert len(s.calls) == 0, "cache_only miss must not fall back to network"

    def test_send_many_cache_only_drops_misses(self, tmp_path) -> None:
        """``send_many(cache_only=True)`` yields only the cached hits;
        misses fall off the stream with zero network touches."""
        cache = _local_cfg(tmp_path)
        hit = make_request("https://example.com/co/yes")
        miss = make_request("https://example.com/co/no")
        _seed_local(cache, make_response(request=hit, body=b'{"v":"hit"}'))

        s = StubSession()
        out = list(s.send_many(iter([hit, miss]), local_cache=cache, cache_only=True))

        assert len(s.calls) == 0, "cache_only must never touch the network"
        assert len(out) == 1
        assert out[0].json() == {"v": "hit"}
        assert out[0].local_cached is True


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
