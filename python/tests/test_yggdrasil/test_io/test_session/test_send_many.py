"""`Session.send_many()` — batch cache dispatch.

Isolation strategy
------------------
`_send_many` talks directly to `table.sql.execute()` for the *batch* lookup
and `table.insert()` for the *write-back*. When requests carry a per-request
`remote_cache_config`, the per-request send() inside the miss path would also
consult that table. We patch `_load_remote_cached_response` /
`_store_remote_cached_response` on the session so only the batch-level SQL
and insert calls are counted.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch

from yggdrasil.io import SaveMode
from yggdrasil.io.send_config import CacheConfig

from .._helpers import (
    KEY_COLS,
    make_cache_config,
    make_request,
    make_response,
    make_table_mock,
)


def _run(session, requests, *, remote_cache: CacheConfig):
    """Drive `send_many` with per-request load/store patched out."""
    with patch.object(session, "_load_remote_cached_response", return_value=None):
        with patch.object(session, "_store_remote_cached_response"):
            return list(session.send_many(iter(requests), remote_cache=remote_cache))


# ---------------------------------------------------------------------------
# Batch lookup dispatch
# ---------------------------------------------------------------------------

class TestBatchLookupGrouping:
    def test_same_table_collapses_to_one_sql_call(self, mock_session):
        req_a = make_request("https://example.com/a")
        req_b = make_request("https://example.com/b")
        table = make_table_mock()
        cfg = make_cache_config(table)

        mock_session.queue(make_response(req_a), make_response(req_b))
        _run(mock_session, [req_a, req_b], remote_cache=cfg)

        assert table.sql.execute.call_count == 1

    def test_different_tables_get_one_sql_call_each(self, mock_session):
        req_a = make_request("https://example.com/a")
        req_b = make_request("https://example.com/b")
        table_a = make_table_mock("cat.schema.tbl_a")
        table_b = make_table_mock("cat.schema.tbl_b")
        req_a.remote_cache_config = make_cache_config(table_a)
        req_b.remote_cache_config = make_cache_config(table_b)

        session_table = make_table_mock("cat.schema.session")
        mock_session.queue(make_response(req_a), make_response(req_b))

        _run(
            mock_session,
            [req_a, req_b],
            remote_cache=make_cache_config(session_table),
        )

        assert table_a.sql.execute.call_count == 1
        assert table_b.sql.execute.call_count == 1


# ---------------------------------------------------------------------------
# Cache hits / misses / UPSERT bypass
# ---------------------------------------------------------------------------

class TestBatchCacheHits:
    def test_single_hit_skips_network(self, mock_session):
        req = make_request("https://example.com/a")
        cached = make_response(request=req.anonymize(mode="remove"))
        table = make_table_mock(hits=[cached])
        cfg = make_cache_config(table)

        results = list(mock_session.send_many(iter([req]), remote_cache=cfg))

        assert len(results) == 1
        assert mock_session.calls == []

    def test_all_hits_produce_zero_inserts_and_zero_network_calls(self, mock_session):
        req = make_request("https://example.com/a")
        cached = make_response(request=req.anonymize(mode="remove"))
        table = make_table_mock(hits=[cached])

        results = list(mock_session.send_many(iter([req]), remote_cache=make_cache_config(table)))

        assert len(results) == 1
        assert table.insert.call_count == 0
        assert mock_session.calls == []


class TestBatchUpsertBypass:
    def test_upsert_bypasses_batch_lookup_but_hits_are_served_for_append(self, mock_session):
        # req_a is an APPEND hit; req_b is an UPSERT forced-refresh.
        req_a = make_request("https://example.com/a")
        req_b = make_request("https://example.com/b")
        cached_a = make_response(request=req_a.anonymize(mode="remove"))

        upsert_table = make_table_mock("cat.schema.tbl_upsert")
        req_b.remote_cache_config = CacheConfig(
            table=upsert_table,
            received_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
            request_by=KEY_COLS,
            mode=SaveMode.UPSERT,
        )
        session_table = make_table_mock("cat.schema.tbl", hits=[cached_a])
        mock_session.queue(make_response(req_b))  # the UPSERT refetch

        with patch.object(mock_session, "_load_remote_cached_response", return_value=None):
            with patch.object(mock_session, "_store_remote_cached_response"):
                results = list(
                    mock_session.send_many(
                        iter([req_a, req_b]),
                        remote_cache=make_cache_config(session_table),
                    )
                )

        assert len(results) == 2
        # Only the UPSERT refetch hit the network.
        assert len(mock_session.calls) == 1


# ---------------------------------------------------------------------------
# Write-back per-table / per-mode grouping
# ---------------------------------------------------------------------------

class TestBatchWriteBack:
    def test_each_per_request_table_gets_its_own_insert(self, mock_session):
        req_a = make_request("https://example.com/a")
        req_b = make_request("https://example.com/b")
        table_a = make_table_mock("cat.schema.tbl_a")
        table_b = make_table_mock("cat.schema.tbl_b")
        req_a.remote_cache_config = make_cache_config(table_a)
        req_b.remote_cache_config = make_cache_config(table_b)

        session_table = make_table_mock("cat.schema.session")
        mock_session.queue(make_response(req_a), make_response(req_b))

        _run(
            mock_session,
            [req_a, req_b],
            remote_cache=make_cache_config(session_table),
        )

        assert table_a.insert.call_count == 1
        assert table_b.insert.call_count == 1

    def test_upsert_response_is_written_with_upsert_mode(self, mock_session):
        req = make_request("https://example.com/a")
        table = make_table_mock()
        req.remote_cache_config = CacheConfig(
            table=table,
            received_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
            request_by=KEY_COLS,
            mode=SaveMode.UPSERT,
        )
        mock_session.queue(make_response(req))
        session_table = make_table_mock("cat.schema.session")

        _run(
            mock_session,
            [req],
            remote_cache=make_cache_config(session_table),
        )

        assert table.insert.call_count == 1
        assert table.insert.call_args[1]["mode"] == SaveMode.UPSERT


class TestBatchDisabledConfig:
    def test_request_with_disabled_remote_config_is_never_queried(self, mock_session):
        """A per-request config with no table disables remote caching for
        that request — neither the request table nor the session table should
        be touched for it."""
        req = make_request("https://example.com/a")
        req.remote_cache_config = CacheConfig()  # no table

        session_table = make_table_mock()
        results = list(
            mock_session.send_many(
                iter([req]),
                remote_cache=make_cache_config(session_table),
            )
        )

        assert session_table.sql.execute.call_count == 0
        assert len(results) == 1
        assert len(mock_session.calls) == 1
