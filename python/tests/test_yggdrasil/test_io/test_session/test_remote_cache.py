"""`Session.send()` — SQL-table-backed remote cache.

The `_load_remote_cached_response` / `_store_remote_cached_response` hooks
are patched here to keep each test focused on dispatch + config propagation.
Actual SQL generation is covered in `test_send_config.py`.
"""

from __future__ import annotations

from unittest.mock import patch

from yggdrasil.io import SaveMode
from yggdrasil.io.send_config import CacheConfig

from .._helpers import make_request, make_response, make_table_mock


class TestRemoteCacheHit:
    def test_hit_returns_cached_and_skips_network(self, mock_session):
        req = make_request()
        cached = make_response(request=req)
        table = make_table_mock()

        with patch.object(mock_session, "_load_remote_cached_response", return_value=cached):
            result = mock_session.send(req, remote_cache=CacheConfig(table=table))

        assert result is cached
        assert mock_session.calls == []


class TestRemoteCacheMiss:
    def test_miss_sends_then_stores(self, mock_session):
        req = make_request()
        fresh = make_response(request=req)
        mock_session.queue(fresh)
        table = make_table_mock()

        with patch.object(mock_session, "_load_remote_cached_response", return_value=None):
            with patch.object(mock_session, "_store_remote_cached_response") as store:
                result = mock_session.send(req, remote_cache=CacheConfig(table=table))

        assert result is fresh
        store.assert_called_once()


class TestRemoteCacheUpsertBypassesRead:
    def test_upsert_never_reads_cache(self, mock_session):
        req = make_request()
        req.remote_cache_config = CacheConfig(
            table=make_table_mock(),
            mode=SaveMode.UPSERT,
        )

        with patch.object(mock_session, "_load_remote_cached_response") as load:
            with patch.object(mock_session, "_store_remote_cached_response"):
                mock_session.send(req)

        load.assert_not_called()

    def test_upsert_stores_with_upsert_mode(self, mock_session):
        req = make_request()
        req.remote_cache_config = CacheConfig(
            table=make_table_mock(),
            mode=SaveMode.UPSERT,
        )
        fresh = make_response(request=req)
        mock_session.queue(fresh)

        with patch.object(mock_session, "_store_remote_cached_response") as store:
            mock_session.send(req)

        store.assert_called_once()
        # Second positional arg is cache_cfg — its mode must be UPSERT.
        stored_cfg: CacheConfig = store.call_args[0][1]
        assert stored_cfg.mode == SaveMode.UPSERT


class TestRemoteCachePerRequestOverride:
    def test_request_level_override_wins(self, mock_session):
        req = make_request()
        override_table = make_table_mock("cat.schema.override")
        req.remote_cache_config = CacheConfig(table=override_table)
        cached = make_response(request=req)

        with patch.object(
            mock_session,
            "_load_remote_cached_response",
            return_value=cached,
        ) as load:
            result = mock_session.send(req)

        assert result is cached
        used_cfg: CacheConfig = load.call_args[0][1]
        assert used_cfg.table is override_table


class TestRemoteHitBackfillsLocal:
    """When a remote hit lands and local caching is enabled, the response
    should be persisted locally so the next run skips the SQL roundtrip.
    """

    def test_remote_hit_triggers_local_store(self, mock_session):
        req = make_request()
        cached = make_response(request=req)
        table = make_table_mock()

        with patch.object(
            mock_session,
            "_load_remote_cached_response",
            return_value=cached,
        ):
            with patch.object(mock_session, "_store_local_cached_response") as local_store:
                result = mock_session.send(
                    req,
                    remote_cache=CacheConfig(table=table),
                    local_cache=CacheConfig(received_from="2020-01-01T00:00:00Z"),
                )

        assert result is cached
        local_store.assert_called_once()
        # First positional arg: the remote hit is the one written locally.
        assert local_store.call_args[0][0] is cached
