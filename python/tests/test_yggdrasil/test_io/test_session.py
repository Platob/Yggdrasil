"""Tests for yggdrasil.io.session.

Coverage
--------
- PreparedRequest.local_cache_config / remote_cache_config fields
- Session.send()     — local cache (hit/miss/UPSERT) + remote cache (hit/miss/UPSERT)
                       + per-request config overrides
- Session._send_many_remote() — per-table SQL grouping, UPSERT bypass, write-back by (table, mode)
- Session.spark_send()        — scatter/gather path + per-table prefetch
                                 (entire Spark section is skipped when pyspark is absent)
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from yggdrasil.io import SaveMode
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response, RESPONSE_ARROW_SCHEMA
from yggdrasil.io.send_config import CacheConfig, SendConfig
from yggdrasil.io.session import Session


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _req(url: str = "https://example.com/a", method: str = "GET") -> PreparedRequest:
    req = PreparedRequest.prepare(method=method, url=url)
    req.sent_at = dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)
    return req


def _resp(
    request: PreparedRequest | None = None,
    status_code: int = 200,
    body: bytes = b'{"ok":true}',
    received_at: int | None = None,
) -> Response:
    if received_at is None:
        received_at = int(
            dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1_000_000
        )
    req = request or _req()
    return Response(
        request=req,
        status_code=status_code,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
        tags={},
        buffer=body,
        received_at=received_at,
    )


# CacheConfig key columns that survive Arrow round-trip through RESPONSE_ARROW_SCHEMA.
_KEY_COLS = ["request_method", "request_url_host", "request_url_path"]


def _cache_cfg(table: MagicMock, *, mode: SaveMode = SaveMode.APPEND) -> CacheConfig:
    """Return a CacheConfig that enables both local & remote caching."""
    return CacheConfig(
        table=table,
        received_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
        request_by=_KEY_COLS,
        mode=mode,
    )


def _make_table_mock(full_name: str, hit_responses: list[Response]) -> MagicMock:
    """Build a mock cache table whose SQL execute returns *hit_responses*."""
    if hit_responses:
        arrow_tbl = pa.Table.from_batches(
            [r.to_arrow_batch(parse=False) for r in hit_responses]
        )
        batch_list = arrow_tbl.to_batches()
    else:
        batch_list = []

    cache_result = MagicMock()
    # Use side_effect so a fresh iterator is returned on every call.
    cache_result.to_arrow_batches.side_effect = lambda: iter(batch_list)

    table = MagicMock()
    table.full_name.return_value = full_name
    table.sql.execute.return_value = cache_result
    return table


# ---------------------------------------------------------------------------
# Minimal concrete Session subclass
# ---------------------------------------------------------------------------

@dataclass
class _MockSession(Session):
    """Test double that queues pre-built responses for _local_send."""

    _queue: list[Response] = field(default_factory=list, init=False, repr=False)
    _calls: list[PreparedRequest] = field(default_factory=list, init=False, repr=False)

    def _local_send(self, request: PreparedRequest, config: SendConfig) -> Response:
        self._calls.append(request)
        return self._queue.pop(0) if self._queue else _resp(request=request)

    def queue(self, *responses: Response) -> "_MockSession":
        self._queue.extend(responses)
        return self


# ===========================================================================
# 1. PreparedRequest per-request cache-config fields
# ===========================================================================

class TestPerRequestCacheConfig:

    def test_fields_default_to_none(self):
        req = _req()
        assert req.local_cache_config is None
        assert req.remote_cache_config is None

    def test_can_be_assigned_after_construction(self):
        req = _req()
        cfg = CacheConfig(mode=SaveMode.UPSERT)
        req.local_cache_config = cfg
        req.remote_cache_config = cfg
        assert req.local_cache_config is cfg
        assert req.remote_cache_config is cfg

    def test_copy_preserves_both_fields(self):
        local_cfg = CacheConfig()
        remote_cfg = CacheConfig(mode=SaveMode.UPSERT)
        req = _req()
        req.local_cache_config = local_cfg
        req.remote_cache_config = remote_cfg

        copied = req.copy()

        assert copied.local_cache_config is local_cfg
        assert copied.remote_cache_config is remote_cfg

    def test_copy_local_cache_config_override(self):
        original = CacheConfig()
        new = CacheConfig(mode=SaveMode.UPSERT)
        req = _req()
        req.local_cache_config = original

        copied = req.copy(local_cache_config=new)

        assert copied.local_cache_config is new

    def test_copy_can_clear_cache_configs(self):
        req = _req()
        req.local_cache_config = CacheConfig()
        req.remote_cache_config = CacheConfig()

        copied = req.copy(local_cache_config=None, remote_cache_config=None)

        assert copied.local_cache_config is None
        assert copied.remote_cache_config is None

    def test_excluded_from_equality_comparison(self):
        req_a = _req()
        req_b = _req()
        req_b.local_cache_config = CacheConfig()
        # compare=False means fields are ignored in __eq__
        assert req_a == req_b

    def test_anonymize_preserves_cache_configs(self):
        cfg = CacheConfig()
        req = _req()
        req.local_cache_config = cfg
        req.remote_cache_config = cfg

        anon = req.anonymize(mode="remove")

        assert anon.local_cache_config is cfg
        assert anon.remote_cache_config is cfg


# ===========================================================================
# 2. Session.send() — local cache
# ===========================================================================

class TestSendLocalCache:

    def test_hit_is_returned_without_network_call(self):
        session = _MockSession()
        req = _req()
        cached = _resp(request=req)

        with patch.object(session, "_load_local_cached_response", return_value=(cached, None)):
            result = session.send(
                req,
                local_cache=CacheConfig(received_from="2020-01-01T00:00:00Z"),
            )

        assert result is cached
        assert len(session._calls) == 0

    def test_miss_sends_request_and_stores_response(self):
        session = _MockSession()
        req = _req()
        fresh = _resp(request=req)
        session.queue(fresh)

        with patch.object(session, "_load_local_cached_response", return_value=(None, None)):
            with patch.object(session, "_store_local_cached_response") as mock_store:
                result = session.send(
                    req,
                    local_cache=CacheConfig(received_from="2020-01-01T00:00:00Z"),
                )

        assert result is fresh
        assert len(session._calls) == 1
        mock_store.assert_called_once()

    def test_upsert_evicts_existing_local_file(self, tmp_path):
        session = _MockSession()
        req = _req()
        cfg = CacheConfig(
            path=tmp_path,
            received_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
            mode=SaveMode.UPSERT,
        )
        # Compute the exact path the eviction code will look at.
        stale_file = cfg.local_cache_file(req, suffix=".ypkl", force=True)
        stale_file.parent.mkdir(parents=True, exist_ok=True)
        stale_file.write_bytes(b"old data")

        with patch.object(session, "_store_local_cached_response"):
            session.send(req, local_cache=cfg)

        assert not stale_file.exists(), "UPSERT must delete the stale local cache file"
        assert len(session._calls) == 1

    def test_upsert_with_no_existing_file_sends_cleanly(self, tmp_path):
        session = _MockSession()
        req = _req()
        fresh = _resp(request=req)
        session.queue(fresh)

        # path=tmp_path ensures local_cache_file() points to a real but non-existent path.
        cfg = CacheConfig(
            path=tmp_path,
            received_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
            mode=SaveMode.UPSERT,
        )
        with patch.object(session, "_store_local_cached_response"):
            result = session.send(req, local_cache=cfg)

        assert result is fresh
        assert len(session._calls) == 1

    def test_per_request_local_cache_config_is_used_instead_of_session_level(self):
        session = _MockSession()
        req = _req()
        cached = _resp(request=req)
        override_cfg = CacheConfig(received_from="2020-01-01T00:00:00Z")
        req.local_cache_config = override_cfg

        with patch.object(session, "_load_local_cached_response", return_value=(cached, None)) as mock_load:
            result = session.send(req)

        assert result is cached
        # Verify the override config was passed, not the default empty one.
        assert mock_load.call_args[0][1] is override_cfg


# ===========================================================================
# 3. Session.send() — remote cache
# ===========================================================================

class TestSendRemoteCache:

    def test_hit_is_returned_without_network_call(self):
        session = _MockSession()
        req = _req()
        cached = _resp(request=req)
        table = _make_table_mock("cat.schema.tbl", [])

        with patch.object(session, "_load_remote_cached_response", return_value=cached):
            result = session.send(req, remote_cache=CacheConfig(table=table))

        assert result is cached
        assert len(session._calls) == 0

    def test_miss_sends_request_and_stores(self):
        session = _MockSession()
        req = _req()
        fresh = _resp(request=req)
        session.queue(fresh)
        table = _make_table_mock("cat.schema.tbl", [])

        with patch.object(session, "_load_remote_cached_response", return_value=None):
            with patch.object(session, "_store_remote_cached_response") as mock_store:
                result = session.send(req, remote_cache=CacheConfig(table=table))

        assert result is fresh
        mock_store.assert_called_once()

    def test_upsert_skips_remote_cache_read(self):
        session = _MockSession()
        req = _req()
        req.remote_cache_config = CacheConfig(
            table=_make_table_mock("cat.schema.tbl", []),
            mode=SaveMode.UPSERT,
        )

        with patch.object(session, "_load_remote_cached_response") as mock_load:
            with patch.object(session, "_store_remote_cached_response"):
                session.send(req)

        mock_load.assert_not_called()

    def test_upsert_stores_with_upsert_mode_config(self):
        session = _MockSession()
        req = _req()
        upsert_table = _make_table_mock("cat.schema.tbl", [])
        req.remote_cache_config = CacheConfig(table=upsert_table, mode=SaveMode.UPSERT)
        fresh = _resp(request=req)
        session.queue(fresh)

        with patch.object(session, "_store_remote_cached_response") as mock_store:
            session.send(req)

        mock_store.assert_called_once()
        # Second positional arg is cache_cfg — its mode must be UPSERT.
        stored_cfg: CacheConfig = mock_store.call_args[0][1]
        assert stored_cfg.mode == SaveMode.UPSERT

    def test_per_request_remote_cache_config_overrides_session_level(self):
        session = _MockSession()
        req = _req()
        override_table = _make_table_mock("cat.schema.override", [])
        req.remote_cache_config = CacheConfig(table=override_table)
        cached = _resp(request=req)

        with patch.object(session, "_load_remote_cached_response", return_value=cached) as mock_load:
            result = session.send(req)

        assert result is cached
        used_cfg: CacheConfig = mock_load.call_args[0][1]
        assert used_cfg.table is override_table


# ===========================================================================
# 4. Session._send_many_remote() — per-table SQL grouping
# ===========================================================================

class TestSendManyRemoteGrouping:
    """
    Isolation strategy
    ------------------
    _send_many_remote calls table.sql.execute() directly for the *batch* lookup
    and table.insert() directly for the *write-back*.  When requests carry a
    per-request remote_cache_config the individual send() calls inside the miss
    path would also touch those tables.  We patch _load_remote_cached_response
    and _store_remote_cached_response on the session so that only the batch-
    level SQL / insert calls are counted in the assertions.
    """

    def _run(self, session: _MockSession, requests, *, remote_cache: CacheConfig):
        """Helper: call send_many and patch individual-send remote cache methods."""
        with patch.object(session, "_load_remote_cached_response", return_value=None):
            with patch.object(session, "_store_remote_cached_response"):
                return list(session.send_many(iter(requests), remote_cache=remote_cache))

    def test_requests_for_same_table_produce_single_sql_query(self):
        req_a = _req("https://example.com/a")
        req_b = _req("https://example.com/b")
        table = _make_table_mock("cat.schema.tbl", [])
        cfg = _cache_cfg(table)
        # No per-request override → both use session-level cfg and share one lookup.
        session = _MockSession().queue(_resp(req_a), _resp(req_b))

        self._run(session, [req_a, req_b], remote_cache=cfg)

        assert table.sql.execute.call_count == 1

    def test_requests_for_different_tables_each_get_own_sql_query(self):
        req_a = _req("https://example.com/a")
        req_b = _req("https://example.com/b")
        table_a = _make_table_mock("cat.schema.tbl_a", [])
        table_b = _make_table_mock("cat.schema.tbl_b", [])
        req_a.remote_cache_config = _cache_cfg(table_a)
        req_b.remote_cache_config = _cache_cfg(table_b)

        session_table = _make_table_mock("cat.schema.session", [])
        session = _MockSession().queue(_resp(req_a), _resp(req_b))

        self._run(session, [req_a, req_b], remote_cache=_cache_cfg(session_table))

        assert table_a.sql.execute.call_count == 1
        assert table_b.sql.execute.call_count == 1

    def test_cache_hit_is_returned_without_network_call(self):
        req = _req("https://example.com/a")
        cached = _resp(request=req.anonymize(mode="remove"))
        table = _make_table_mock("cat.schema.tbl", [cached])
        cfg = _cache_cfg(table)

        session = _MockSession()
        # No per-request config → individual sends won't touch remote cache.
        results = list(session.send_many(iter([req]), remote_cache=cfg))

        assert len(results) == 1
        assert len(session._calls) == 0

    def test_upsert_requests_always_bypass_cache_and_are_fetched(self):
        req_a = _req("https://example.com/a")
        req_b = _req("https://example.com/b")
        # req_a is a cache hit; req_b forces a refetch via UPSERT.
        cached_a = _resp(request=req_a.anonymize(mode="remove"))
        upsert_table = _make_table_mock("cat.schema.tbl_upsert", [])
        req_b.remote_cache_config = CacheConfig(
            table=upsert_table,
            received_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
            request_by=_KEY_COLS,
            mode=SaveMode.UPSERT,
        )
        session_table = _make_table_mock("cat.schema.tbl", [cached_a])
        session = _MockSession()
        session.queue(_resp(req_b))  # network response for the UPSERT miss

        with patch.object(session, "_load_remote_cached_response", return_value=None):
            with patch.object(session, "_store_remote_cached_response"):
                results = list(session.send_many(
                    iter([req_a, req_b]),
                    remote_cache=_cache_cfg(session_table),
                ))

        assert len(results) == 2
        assert len(session._calls) == 1  # only req_b hit the network

    def test_write_back_inserts_into_correct_table_per_request(self):
        req_a = _req("https://example.com/a")
        req_b = _req("https://example.com/b")
        table_a = _make_table_mock("cat.schema.tbl_a", [])
        table_b = _make_table_mock("cat.schema.tbl_b", [])
        req_a.remote_cache_config = _cache_cfg(table_a)
        req_b.remote_cache_config = _cache_cfg(table_b)

        session_table = _make_table_mock("cat.schema.session", [])
        session = _MockSession().queue(_resp(req_a), _resp(req_b))

        self._run(session, [req_a, req_b], remote_cache=_cache_cfg(session_table))

        assert table_a.insert.call_count == 1
        assert table_b.insert.call_count == 1

    def test_write_back_upsert_response_uses_upsert_mode(self):
        req = _req("https://example.com/a")
        table = _make_table_mock("cat.schema.tbl", [])
        req.remote_cache_config = CacheConfig(
            table=table,
            received_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
            request_by=_KEY_COLS,
            mode=SaveMode.UPSERT,
        )
        session = _MockSession().queue(_resp(req))
        # Session-level cfg provides the remote_cache_enabled guard.
        session_table = _make_table_mock("cat.schema.session", [])
        self._run(session, [req], remote_cache=_cache_cfg(session_table))

        assert table.insert.call_count == 1
        insert_kwargs = table.insert.call_args[1]
        assert insert_kwargs["mode"] == SaveMode.UPSERT

    def test_all_cache_hits_produce_zero_inserts_and_zero_network_calls(self):
        req = _req("https://example.com/a")
        cached = _resp(request=req.anonymize(mode="remove"))
        table = _make_table_mock("cat.schema.tbl", [cached])
        cfg = _cache_cfg(table)

        session = _MockSession()
        results = list(session.send_many(iter([req]), remote_cache=cfg))

        assert len(results) == 1
        assert table.insert.call_count == 0
        assert len(session._calls) == 0

    def test_request_with_disabled_remote_config_becomes_miss(self):
        """Per-request config with no table → request goes to network."""
        req = _req("https://example.com/a")
        req.remote_cache_config = CacheConfig()  # no table → disabled

        session_table = _make_table_mock("cat.schema.tbl", [])
        session = _MockSession()

        # Since miss_send_config disables remote cache and req's config also has
        # no table, the session table must NOT be queried for this request.
        results = list(session.send_many(
            iter([req]),
            remote_cache=_cache_cfg(session_table),
        ))

        assert session_table.sql.execute.call_count == 0
        assert len(session._calls) == 1


# ===========================================================================
# 5. Session.spark_send() — requires pyspark
# ===========================================================================

# Skip the entire section when pyspark is not installed.

import os as _os
import sys as _sys

# mapInArrow requires Hadoop native libs on Windows.  When HADOOP_HOME is not
# set the Python worker crashes before executing any Arrow code.
_MAPIN_ARROW_AVAILABLE = not (
    _sys.platform == "win32" and not _os.environ.get("HADOOP_HOME")
)
_SKIP_MAPIN_ARROW = pytest.mark.skipif(
    not _MAPIN_ARROW_AVAILABLE,
    reason="mapInArrow unavailable on Windows without HADOOP_HOME",
)


@pytest.fixture(scope="session")
def spark():
    pytest.importorskip("pyspark", reason="pyspark not installed")

    from pyspark.sql import SparkSession

    try:
        return (
            SparkSession.builder
            .master("local[1]")
            .appName("yggdrasil-session-test")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
    except Exception as exc:
        pytest.skip(f"Local SparkSession is unavailable in this environment: {exc}")


def _empty_spark_df(spark_session):
    """Return an empty Spark DataFrame with the cache-key columns only.

    Using the full RESPONSE_ARROW_SCHEMA (which has MapType/complex fields)
    causes PySpark to crash on Windows even for empty collects.  We only
    need the key columns so that ``select(*request_by_cols).collect()``
    returns an empty list, which is all spark_send needs for hit detection.
    """
    from pyspark.sql.types import StringType, StructField, StructType

    return spark_session.createDataFrame(
        [],
        schema=StructType([StructField(col, StringType(), True) for col in _KEY_COLS]),
    )


class TestSparkSend:
    """
    Tests for spark_send().

    Structural tests (SQL call counts, return types) run on all platforms.
    Execution tests that call .collect() on a DynamicFrame require mapInArrow
    to work and are skipped on Windows without HADOOP_HOME.
    """

    # ------------------------------------------------------------------ #
    # Structural tests — no .collect(), always run                        #
    # ------------------------------------------------------------------ #

    def test_no_remote_cache_returns_spark_dataframe(self, spark):
        """spark_send without schema returns SparkDataFrame with RESPONSE_SCHEMA."""
        from pyspark.sql import DataFrame as SparkDataFrame

        req_a = _req("https://example.com/a")
        req_b = _req("https://example.com/b")
        session = _MockSession().queue(_resp(req_a), _resp(req_b))

        result = session.spark_send(iter([req_a, req_b]), spark_session=spark)

        assert isinstance(result, SparkDataFrame)

    def test_no_remote_cache_schema_cast_returns_spark_dataframe(self, spark):
        """When schema= is given, spark_send returns a SparkDataFrame via cast()."""
        from pyspark.sql import DataFrame as SparkDataFrame
        from yggdrasil.spark.frame import DynamicFrame

        req = _req("https://example.com/a")
        session = _MockSession().queue(_resp(req))

        fake_df = spark.createDataFrame([{"value": 42}])
        with patch.object(DynamicFrame, "cast", return_value=fake_df):
            from yggdrasil.data.schema import schema as build_schema
            from yggdrasil.data.data_field import field as build_field

            out_schema = build_schema([build_field("value", pa.int64())])
            result = session.spark_send(
                iter([req]),
                spark_session=spark,
                schema=out_schema,
            )

        assert isinstance(result, SparkDataFrame)

    def test_remote_cache_per_table_sql_called_once_per_table(self, spark):
        """Two requests each pointing to a different cache table → 2 SQL calls.

        The Spark DataFrames returned by to_spark() are fully mocked so that
        .select().collect() returns [] (all misses) without submitting a real
        Spark job.  This is a unit test for the grouping / SQL dispatch logic.
        """
        req_a = _req("https://example.com/a")
        req_b = _req("https://example.com/b")

        def _mock_cache_df_for(req: PreparedRequest):
            """Return a MagicMock cache Spark DF with one matching hit key."""
            m = MagicMock()
            hit_row = {
                "request_method": req.method,
                "request_url_host": req.url.host,
                "request_url_path": req.url.path,
            }
            m.select.return_value.collect.return_value = [hit_row]
            # mapInArrow must return something union-able; since we never
            # call .collect() on the final result, a MagicMock is fine.
            m.mapInArrow.return_value = MagicMock()
            return m

        result_a = MagicMock()
        result_a.to_spark.return_value = _mock_cache_df_for(req_a)

        result_b = MagicMock()
        result_b.to_spark.return_value = _mock_cache_df_for(req_b)

        table_a = _make_table_mock("cat.schema.tbl_a", [])
        table_a.sql.execute.return_value = result_a

        table_b = _make_table_mock("cat.schema.tbl_b", [])
        table_b.sql.execute.return_value = result_b

        req_a.remote_cache_config = _cache_cfg(table_a)
        req_b.remote_cache_config = _cache_cfg(table_b)

        session_table = _make_table_mock("cat.schema.tbl_a", [])
        session = _MockSession().queue(_resp(req_a), _resp(req_b))

        result = session.spark_send(
            iter([req_a, req_b]),
            spark_session=spark,
            remote_cache=_cache_cfg(session_table),
        )

        # Each per-request table got exactly one SQL lookup and one to_spark call.
        assert table_a.sql.execute.call_count == 1
        assert table_b.sql.execute.call_count == 1
        result_a.to_spark.assert_called_once()
        result_b.to_spark.assert_called_once()

        # Result is a MagicMock here because to_spark()/mapInArrow are mocked;
        # this test validates SQL grouping + dispatch, not Spark execution.
        assert result is not None

    def test_remote_cache_upsert_requests_bypass_sql_lookup(self, spark):
        """UPSERT requests must not appear in any SQL query."""
        req = _req("https://example.com/upsert")
        table = _make_table_mock("cat.schema.tbl", [])
        req.remote_cache_config = _cache_cfg(table, mode=SaveMode.UPSERT)

        session_table = _make_table_mock("cat.schema.tbl", [])
        session = _MockSession()

        result = session.spark_send(
            iter([req]),
            spark_session=spark,
            remote_cache=_cache_cfg(session_table),
        )

        table.sql.execute.assert_not_called()
        from pyspark.sql import DataFrame as SparkDataFrame
        assert isinstance(result, SparkDataFrame)

    # ------------------------------------------------------------------ #
    # Execution tests — need mapInArrow to work (.collect() is called)    #
    # ------------------------------------------------------------------ #

    @_SKIP_MAPIN_ARROW
    def test_no_remote_cache_collects_correct_responses(self, spark):
        """scatter/gather path materialises one row per response."""
        from pyspark.sql import DataFrame as SparkDataFrame

        req_a = _req("https://example.com/a")
        req_b = _req("https://example.com/b")
        session = _MockSession().queue(_resp(req_a), _resp(req_b))

        result = session.spark_send(iter([req_a, req_b]), spark_session=spark)

        assert isinstance(result, SparkDataFrame)
        collected = result.collect()
        assert len(collected) == 2
        assert all(hasattr(r, "request_method") for r in collected)
        assert all(hasattr(r, "response_status_code") for r in collected)

    @_SKIP_MAPIN_ARROW
    def test_all_requests_served_from_cache_zero_network_calls(self, spark):
        """When every request is a cache hit, _local_send is never called."""
        from pyspark.sql import DataFrame as SparkDataFrame
        from yggdrasil.spark.cast import any_to_spark_schema

        req = _req("https://example.com/a")
        cached = _resp(request=req.anonymize(mode="remove"))

        arrow_tbl = pa.Table.from_batches([cached.to_arrow_batch(parse=False)])
        spark_hit_df = spark.createDataFrame(
            arrow_tbl.to_pandas(),
            schema=any_to_spark_schema(RESPONSE_ARROW_SCHEMA),
        )

        t_cache_result = MagicMock()
        t_cache_result.to_spark.return_value = spark_hit_df

        table = _make_table_mock("cat.schema.tbl", [cached])
        table.sql.execute.return_value = t_cache_result

        cfg = _cache_cfg(table)
        session = _MockSession()

        result = session.spark_send(iter([req]), spark_session=spark, remote_cache=cfg)

        assert isinstance(result, SparkDataFrame)
        collected = result.collect()
        assert len(collected) == 1
        assert len(session._calls) == 0

    @_SKIP_MAPIN_ARROW
    def test_explode_flattens_list_of_items_to_individual_rows(self, spark):
        """DynamicFrame.explode() turns DynamicFrame[list[T]] into DynamicFrame[T]."""
        from yggdrasil.spark.frame import DynamicFrame

        items = [{"x": 1}, {"x": 2}, {"x": 3}]

        def _identity(batch):
            return batch

        dyn = DynamicFrame.parallelize(
            _identity,
            [items],
            spark_session=spark,
        )

        exploded = dyn.explode()
        collected = exploded.collect()

        assert len(collected) == 3
        assert collected == items

