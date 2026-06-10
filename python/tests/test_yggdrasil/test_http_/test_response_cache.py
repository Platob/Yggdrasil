"""HttpResponseCache — content-addressed local HTTP response cache.

Each response is one Arrow-IPC file keyed by the producing request's
public_hash. These tests round-trip real Response objects (no live server) and
exercise the CacheConfig integration with the disk root pointed at a temp dir.
"""
from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.response import HTTPResponse as Response
from yggdrasil.http_.response_cache import HttpResponseCache
from yggdrasil.http_.cache_config import CacheConfig


def _req(url="https://example.com/api?x=1", method="GET"):
    return HTTPRequest.prepare(method, url)


def _resp(req, *, status=200, body=b'{"k":1}', received_at=None,
          ctype="application/json"):
    return Response(
        request=req, status_code=status, headers={"Content-Type": ctype},
        tags={}, buffer=body,
        received_at=received_at or dt.datetime.now(dt.timezone.utc),
    )


def _cache(tmp_path):
    return HttpResponseCache(path=str(tmp_path))


# -- round-trip ------------------------------------------------------------

def test_write_then_read_hit(tmp_path):
    cache = _cache(tmp_path)
    req = _req()
    cache.write_arrow(Response.values_to_arrow_batch([_resp(req, body=b"HELLO")]))
    hits, misses = cache.read_responses([req], config=CacheConfig())
    assert len(hits) == 1 and not misses
    assert hits[0].status_code == 200
    assert hits[0].content == b"HELLO"
    # matched on the producing request's public_hash
    assert hits[0].match_value("public_hash") == req.match_value("public_hash")


def test_cache_is_serializable(tmp_path):
    # Survives both stdlib pickle and the ygg transport serializer, and stays
    # functional (same path) after a round-trip.
    import pickle
    from yggdrasil.pickle import dumps, loads

    cache = _cache(tmp_path)
    req = _req()
    cache.write_arrow(Response.values_to_arrow_batch([_resp(req, body=b"S")]))
    for round_trip in (lambda c: pickle.loads(pickle.dumps(c)), lambda c: loads(dumps(c))):
        c2 = round_trip(cache)
        assert isinstance(c2, HttpResponseCache)
        assert c2.root == cache.root
        hits, _ = c2.read_responses([req], config=CacheConfig())
        assert hits and hits[0].content == b"S"


def test_different_request_misses(tmp_path):
    cache = _cache(tmp_path)
    cache.write_arrow(Response.values_to_arrow_batch([_resp(_req("https://e.com/a?x=1"))]))
    other = _req("https://e.com/a?x=2")
    hits, misses = cache.read_responses([other], config=CacheConfig())
    assert not hits and misses == [other]


def test_file_named_by_request_public_hash(tmp_path):
    cache = _cache(tmp_path)
    req = _req()
    cache.write_arrow(Response.values_to_arrow_batch([_resp(req)]))
    key = req.match_value("public_hash")
    assert (tmp_path / f"{key & 0xFF:02x}" / f"{key}.arrow").exists()


def test_upsert_keeps_latest(tmp_path):
    cache = _cache(tmp_path)
    req = _req()
    cache.write_arrow(Response.values_to_arrow_batch([_resp(req, body=b"OLD")]))
    cache.write_arrow(Response.values_to_arrow_batch([_resp(req, body=b"NEW")]))
    hits, _ = cache.read_responses([req], config=CacheConfig())
    assert hits[0].content == b"NEW"


def test_batch_read_mixes_hits_and_misses(tmp_path):
    cache = _cache(tmp_path)
    a, b, c = _req("https://e.com/a"), _req("https://e.com/b"), _req("https://e.com/c")
    cache.write_arrow(Response.values_to_arrow_batch([_resp(a, body=b"A"), _resp(c, body=b"C")]))
    hits, misses = cache.read_responses([a, b, c], config=CacheConfig())
    bodies = sorted(h.content for h in hits)
    assert bodies == [b"A", b"C"]
    assert misses == [b]


def test_empty_requests(tmp_path):
    assert _cache(tmp_path).read_responses([], config=CacheConfig()) == ([], [])


def test_write_empty_batch_is_noop(tmp_path):
    cache = _cache(tmp_path)
    empty = Response.values_to_arrow_batch([_resp(_req())]).slice(0, 0)
    cache.write_arrow(empty)
    assert not list(tmp_path.rglob("*.arrow"))


# -- received-window filter (delegated to config.filter_response) ----------

def test_received_from_filters_out_stale(tmp_path):
    cache = _cache(tmp_path)
    req = _req()
    old = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    cache.write_arrow(Response.values_to_arrow_batch([_resp(req, received_at=old)]))
    # Require responses received from 2020 onward → the 2000 one is filtered.
    cfg = CacheConfig(received_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc))
    hits, misses = cache.read_responses([req], config=cfg)
    assert not hits and misses == [req]


# -- CacheConfig integration (disk root → temp dir) ------------------------

def test_cache_config_uses_response_cache_locally(tmp_path, monkeypatch):
    import yggdrasil.http_.cache_config as cc
    monkeypatch.setattr(cc, "_DEFAULT_CACHE_ROOT", tmp_path)
    cfg = CacheConfig()
    # The default local backend is the specialized cache, not a generic Folder.
    assert isinstance(cfg.cache_tabular(), HttpResponseCache)


def test_cache_config_round_trip(tmp_path, monkeypatch):
    import yggdrasil.http_.cache_config as cc
    monkeypatch.setattr(cc, "_DEFAULT_CACHE_ROOT", tmp_path)
    cfg = CacheConfig()
    req = _req("https://e.com/round")
    cfg.write_responses([_resp(req, body=b"RT")])
    hits, misses = cfg.read_responses([req])
    assert len(hits) == 1 and hits[0].content == b"RT" and not misses


def test_send_pipeline_split_and_read_hits(tmp_path, monkeypatch):
    # Drive the real send-pipeline cache touchpoints (split_requests + read_hits)
    # through the specialized local cache.
    import yggdrasil.http_.cache_config as cc
    monkeypatch.setattr(cc, "_DEFAULT_CACHE_ROOT", tmp_path)
    from yggdrasil.http_.send_config import SendConfig

    cache = CacheConfig()                     # default local → HttpResponseCache
    cfg = SendConfig(local_cache=cache)
    req = _req("https://e.com/sp")
    cache.write_responses([_resp(req, body=b"SP")])

    local, remote, misses = cfg.split_requests([req])
    assert req.match_value("public_hash") in local
    assert not remote and not misses

    tab = cfg.read_hits(cache, [req])
    assert tab is not None
    bodies = [r.content for r in Response.from_arrow_tabular(tab.read_arrow_batches())]
    assert b"SP" in bodies


@pytest.fixture(autouse=True)
def _fresh_ram(monkeypatch):
    """Each test gets an empty RAM tier + a frozen janitor (no loop thread)."""
    import yggdrasil.http_.response_cache as rc
    monkeypatch.setattr(rc, "_ram", rc._ByteLRU(rc._RAM_MAX_BYTES, rc._RAM_ITEM_MAX_BYTES))
    monkeypatch.setattr(rc, "_janitor_started", True)   # construction won't spawn the loop
    monkeypatch.setattr(rc, "_roots", set())
    yield


def test_ram_tier_serves_hit_without_disk(tmp_path):
    # After a read warms the RAM tier, deleting the file still yields a hit.
    cache = _cache(tmp_path)
    req = _req()
    cache.write_arrow(Response.values_to_arrow_batch([_resp(req, body=b"RAM")]))
    cache.read_responses([req], config=CacheConfig())     # warm RAM
    for f in tmp_path.rglob("*.arrow"):                    # nuke the disk copy
        f.unlink()
    hits, misses = cache.read_responses([req], config=CacheConfig())
    assert len(hits) == 1 and hits[0].content == b"RAM" and not misses


def test_byte_lru_bounded_at_max(tmp_path):
    import yggdrasil.http_.response_cache as rc
    lru = rc._ByteLRU(max_bytes=1000, item_max=1000)
    for i in range(50):
        lru.put(i, b"x" * 100)            # 50 * 100 = 5000 B into a 1000 B budget
    assert lru._bytes <= 1000             # never exceeds the cap
    assert lru.get(0) is None             # oldest evicted
    assert lru.get(49) == b"x" * 100      # newest kept


def test_byte_lru_skips_oversized(tmp_path):
    import yggdrasil.http_.response_cache as rc
    lru = rc._ByteLRU(max_bytes=1000, item_max=250)
    lru.put(1, b"x" * 251)                # bigger than per-item cap → disk only
    assert lru.get(1) is None
    assert lru._bytes == 0


def test_ram_max_is_32mb_default():
    import yggdrasil.http_.response_cache as rc
    assert rc._RAM_MAX_BYTES == 32 * 1024 * 1024


def test_prune_old_deletes_stale_keeps_fresh(tmp_path):
    import os
    import time
    from yggdrasil.http_.response_cache import _prune_old

    root = tmp_path / "c"
    (root / "ab").mkdir(parents=True)
    stale = root / "ab" / "1.arrow"
    stale.write_bytes(b"x")
    fresh = root / "ab" / "2.arrow"
    fresh.write_bytes(b"y")
    two_days_ago = time.time() - 2 * 86400
    os.utime(stale, (two_days_ago, two_days_ago))

    removed = _prune_old(root, 86400)            # 1-day TTL
    assert removed == 1
    assert not stale.exists()
    assert fresh.exists()


def test_construction_registers_root_for_janitor(tmp_path):
    import pathlib
    import yggdrasil.http_.response_cache as rc

    rc.HttpResponseCache(path=str(tmp_path))           # _janitor_started frozen True
    assert str(pathlib.Path(str(tmp_path)).expanduser()) in rc._roots


def test_byte_lru_sweep_drops_stale_keeps_fresh():
    import time
    import yggdrasil.http_.response_cache as rc

    lru = rc._ByteLRU(max_bytes=1000, item_max=1000)
    lru.put(1, b"x" * 10)
    time.sleep(0.03)
    lru.put(2, b"y" * 10)                               # fresh
    assert lru.sweep(0.02) == 1                         # key 1 is older than 0.02s
    assert lru.get(1) is None
    assert lru.get(2) == b"y" * 10
    assert lru._bytes == 10                             # byte accounting stays correct


def test_janitor_pass_prunes_old_disk_keeps_fresh(tmp_path, monkeypatch):
    import os
    import time
    import yggdrasil.http_.response_cache as rc

    root = tmp_path / "j"
    (root / "ab").mkdir(parents=True)
    stale = root / "ab" / "old.arrow"
    stale.write_bytes(b"x")
    fresh = root / "ab" / "new.arrow"
    fresh.write_bytes(b"y")
    old = time.time() - 2 * 86400
    os.utime(stale, (old, old))
    monkeypatch.setattr(rc, "_roots", {str(root)})
    rc._ram.put(7, b"z")                                # fresh RAM entry

    rc._janitor_pass()                                  # active sweep (1-day TTL)

    assert not stale.exists() and fresh.exists()        # day-old disk pruned, fresh kept
    assert rc._ram.get(7) == b"z"                        # fresh RAM survives the sweep


def test_send_pipeline_miss_for_uncached(tmp_path, monkeypatch):
    import yggdrasil.http_.cache_config as cc
    monkeypatch.setattr(cc, "_DEFAULT_CACHE_ROOT", tmp_path)
    from yggdrasil.http_.send_config import SendConfig

    cfg = SendConfig(local_cache=CacheConfig())
    req = _req("https://e.com/never-cached")
    local, remote, misses = cfg.split_requests([req])
    assert not local and misses == [req]


# -- remote-cache window probe (only-new-data) -----------------------------

def test_remote_probe_skips_stale_when_only_new_data_requested(tmp_path):
    """A stale row in a (generic Folder) remote cache must NOT count as a hit
    when the caller asks for fresh data only — otherwise the batch reads the
    full remote row just to drop it and re-fetch. With a ``received_from``
    window the presence probe excludes it, so the request stays a miss."""
    from yggdrasil.http_.send_config import SendConfig

    req = _req("https://e.com/only-new")
    old = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)

    # Generic Folder backend (not the content-addressed local cache) → exercises
    # the predicate-scanned probe path that powers the remote (Databricks) cache.
    remote = CacheConfig(tabular=str(tmp_path / "remote"))
    remote.write_responses([_resp(req, received_at=old)])

    # Sanity: with no window, the stale row is a remote hit (and no miss).
    warm = SendConfig(remote_cache=remote)
    _local, remote_h, misses = warm.split_requests([req])
    assert req.match_value("public_hash") in remote_h and not misses

    # Asking for data received from 2020 onward → the 2000 row is excluded by
    # the probe itself: no remote hit, the request is a clean miss.
    fresh = SendConfig(
        remote_cache=remote.copy(
            received_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
        ),
    )
    _local, remote_h, misses = fresh.split_requests([req])
    assert not remote_h and misses == [req]


def test_remote_cache_full_read_is_lazy_in_fetch(tmp_path):
    """``_fetch`` must NOT materialise the remote cache — its window-aware probe
    already guarantees every remote hit is valid, so the (potentially Databricks)
    full read stays lazy until the batch is consumed."""
    from yggdrasil.http_.send_config import SendConfig
    from yggdrasil.http_.response_batch import HTTPResponseBatch

    req = _req("https://e.com/lazy-remote")
    remote = CacheConfig(tabular=str(tmp_path / "remote"))
    remote.write_responses([_resp(req, body=b"REMOTE")])

    batch = HTTPResponseBatch(SendConfig(remote_cache=remote), [req])
    batch._fetch()                                  # resolve split + (no) misses

    # The request was a remote hit (no misses, no network), but the remote rows
    # are still unread — the holder sits at its lazy sentinel.
    assert not batch.misses
    assert batch._remote_tabular is ...

    # Consuming the batch is what triggers the remote read.
    bodies = [r.content for r in batch.responses()]
    assert b"REMOTE" in bodies
    assert batch._remote_tabular is not ...
