"""`Session.send()` — disk-backed local cache.

Paths exercised:
- hit: load returns a response → no `_local_send` call.
- miss: load returns (None, path) → `_local_send` + `_store_local_cached_response`.
- UPSERT: stale file is evicted *before* fetching; read path is skipped entirely.
- per-request override: `request.local_cache_config` beats the session-level arg.
- TTL boundary: `received_from` > file mtime evicts the file.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch

from yggdrasil.io import SaveMode
from yggdrasil.io.send_config import CacheConfig

from .._helpers import make_request, make_response


LOCAL_FROM = "2020-01-01T00:00:00Z"


class TestLocalCacheHit:
    def test_hit_returns_cached_and_skips_network(self, mock_session):
        req = make_request()
        cached = make_response(request=req)

        with patch.object(
            mock_session,
            "_load_local_cached_response",
            return_value=(cached, None),
        ):
            result = mock_session.send(req, local_cache=CacheConfig(received_from=LOCAL_FROM))

        assert result is cached
        assert mock_session.calls == []


class TestLocalCacheMiss:
    def test_miss_sends_then_stores(self, mock_session):
        req = make_request()
        fresh = make_response(request=req)
        mock_session.queue(fresh)

        with patch.object(mock_session, "_load_local_cached_response", return_value=(None, None)):
            with patch.object(mock_session, "_store_local_cached_response") as store:
                result = mock_session.send(
                    req,
                    local_cache=CacheConfig(received_from=LOCAL_FROM),
                )

        assert result is fresh
        assert len(mock_session.calls) == 1
        store.assert_called_once()


class TestLocalCacheUpsert:
    def test_upsert_evicts_stale_file_before_fetch(self, mock_session, tmp_path):
        req = make_request()
        cfg = CacheConfig(
            path=tmp_path,
            received_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
            mode=SaveMode.UPSERT,
        )
        # Seed a stale file at the exact path the eviction code targets.
        stale = cfg.local_cache_file(req, suffix=".ypkl", force=True)
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"old data")

        with patch.object(mock_session, "_store_local_cached_response"):
            mock_session.send(req, local_cache=cfg)

        assert not stale.exists(), "UPSERT must delete the stale local cache file"
        assert len(mock_session.calls) == 1

    def test_upsert_with_no_existing_file_still_sends_cleanly(self, mock_session, tmp_path):
        req = make_request()
        fresh = make_response(request=req)
        mock_session.queue(fresh)

        cfg = CacheConfig(
            path=tmp_path,
            received_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
            mode=SaveMode.UPSERT,
        )
        with patch.object(mock_session, "_store_local_cached_response"):
            result = mock_session.send(req, local_cache=cfg)

        assert result is fresh
        assert len(mock_session.calls) == 1


class TestLocalCachePerRequestOverride:
    def test_request_level_config_beats_session_level(self, mock_session):
        req = make_request()
        cached = make_response(request=req)
        override = CacheConfig(received_from=LOCAL_FROM)
        req.local_cache_config = override

        with patch.object(
            mock_session,
            "_load_local_cached_response",
            return_value=(cached, None),
        ) as load:
            result = mock_session.send(req)

        assert result is cached
        # Second positional arg is the effective config.
        assert load.call_args[0][1] is override


class TestLocalCacheFileTTLBoundary:
    """`CacheConfig.local_cache_file` enforces the received_from/to window."""

    def test_file_older_than_received_from_is_evicted_on_lookup(self, tmp_path):
        req = make_request()
        # Window starts "now"; any pre-existing file must be past.
        cfg = CacheConfig(
            path=tmp_path,
            received_from=dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1),
        )
        stale = cfg.local_cache_file(req, suffix=".ypkl", force=True)
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"old")

        # Without force, the lookup must reject the file *and* delete it.
        assert cfg.local_cache_file(req, suffix=".ypkl") is None
        assert not stale.exists()

    def test_file_newer_than_received_to_is_not_returned_but_kept(self, tmp_path):
        req = make_request()
        # Window ends an hour ago; a freshly-written file is past the "to".
        cfg = CacheConfig(
            path=tmp_path,
            received_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
            received_to=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1),
        )
        fresh = cfg.local_cache_file(req, suffix=".ypkl", force=True)
        fresh.parent.mkdir(parents=True, exist_ok=True)
        fresh.write_bytes(b"new")

        # Future-relative-to-the-window file: lookup misses but does NOT delete.
        assert cfg.local_cache_file(req, suffix=".ypkl") is None
        assert fresh.exists()
