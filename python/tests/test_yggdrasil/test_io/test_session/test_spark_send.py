"""`Session.spark_send()` — Spark scatter/gather with optional cache prefetch.

Structural tests (SQL call counts, return types) run everywhere. Execution
tests that call `.collect()` on a DynamicFrame need `mapInArrow`, which
requires Hadoop native libs on Windows — those are skipped when the libs
are absent.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from yggdrasil.io import SaveMode
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA
from yggdrasil.io.send_config import CacheConfig

from .._helpers import KEY_COLS, make_cache_config, make_request, make_response, make_table_mock


_MAPIN_ARROW_AVAILABLE = not (
    sys.platform == "win32" and not os.environ.get("HADOOP_HOME")
)
SKIP_MAPIN_ARROW = pytest.mark.skipif(
    not _MAPIN_ARROW_AVAILABLE,
    reason="mapInArrow unavailable on Windows without HADOOP_HOME",
)


@pytest.fixture(scope="module")
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


# ---------------------------------------------------------------------------
# Structural tests — no .collect(), always run
# ---------------------------------------------------------------------------

class TestSparkStructural:
    def test_no_remote_cache_returns_spark_dataframe(self, spark, mock_session):
        from pyspark.sql import DataFrame as SparkDataFrame

        req_a = make_request("https://example.com/a")
        req_b = make_request("https://example.com/b")
        mock_session.queue(make_response(req_a), make_response(req_b))

        result = mock_session.spark_send(iter([req_a, req_b]), spark_session=spark)

        assert isinstance(result, SparkDataFrame)

    def test_schema_cast_returns_spark_dataframe(self, spark, mock_session):
        from pyspark.sql import DataFrame as SparkDataFrame
        from yggdrasil.data.data_field import field as build_field
        from yggdrasil.data.schema import schema as build_schema
        from yggdrasil.spark.frame import DynamicFrame

        req = make_request("https://example.com/a")
        mock_session.queue(make_response(req))

        fake_df = spark.createDataFrame([{"value": 42}])
        with patch.object(DynamicFrame, "cast", return_value=fake_df):
            out_schema = build_schema([build_field("value", pa.int64())])
            result = mock_session.spark_send(
                iter([req]),
                spark_session=spark,
                schema=out_schema,
            )

        assert isinstance(result, SparkDataFrame)

    def test_remote_cache_runs_one_sql_query_per_table(self, spark, mock_session):
        """Two requests pointing at two different cache tables → two SQL calls.

        The cache Spark DFs are mocked so `.select().collect()` returns a
        fabricated hit key list without submitting a real Spark job.
        """
        req_a = make_request("https://example.com/a")
        req_b = make_request("https://example.com/b")

        def _mock_cache_df(req: PreparedRequest) -> MagicMock:
            m = MagicMock()
            hit_row = {
                "request_method": req.method,
                "request_url_host": req.url.host,
                "request_url_path": req.url.path,
            }
            m.select.return_value.collect.return_value = [hit_row]
            m.mapInArrow.return_value = MagicMock()
            return m

        result_a = MagicMock()
        result_a.to_spark.return_value = _mock_cache_df(req_a)
        result_b = MagicMock()
        result_b.to_spark.return_value = _mock_cache_df(req_b)

        table_a = make_table_mock("cat.schema.tbl_a")
        table_a.sql.execute.return_value = result_a
        table_b = make_table_mock("cat.schema.tbl_b")
        table_b.sql.execute.return_value = result_b

        req_a.remote_cache_config = make_cache_config(table_a)
        req_b.remote_cache_config = make_cache_config(table_b)

        session_table = make_table_mock("cat.schema.tbl_a")
        mock_session.queue(make_response(req_a), make_response(req_b))

        result = mock_session.spark_send(
            iter([req_a, req_b]),
            spark_session=spark,
            remote_cache=make_cache_config(session_table),
        )

        assert table_a.sql.execute.call_count == 1
        assert table_b.sql.execute.call_count == 1
        result_a.to_spark.assert_called_once()
        result_b.to_spark.assert_called_once()
        assert result is not None

    def test_upsert_requests_bypass_sql_lookup(self, spark, mock_session):
        req = make_request("https://example.com/upsert")
        table = make_table_mock()
        req.remote_cache_config = make_cache_config(table, mode=SaveMode.UPSERT)

        session_table = make_table_mock()
        result = mock_session.spark_send(
            iter([req]),
            spark_session=spark,
            remote_cache=make_cache_config(session_table),
        )

        table.sql.execute.assert_not_called()
        from pyspark.sql import DataFrame as SparkDataFrame
        assert isinstance(result, SparkDataFrame)


# ---------------------------------------------------------------------------
# Execution tests — exercise mapInArrow end-to-end
# ---------------------------------------------------------------------------

class TestSparkExecution:
    @SKIP_MAPIN_ARROW
    def test_no_remote_cache_materializes_one_row_per_response(self, spark, mock_session):
        from pyspark.sql import DataFrame as SparkDataFrame

        req_a = make_request("https://example.com/a")
        req_b = make_request("https://example.com/b")
        mock_session.queue(make_response(req_a), make_response(req_b))

        result = mock_session.spark_send(iter([req_a, req_b]), spark_session=spark)

        assert isinstance(result, SparkDataFrame)
        rows = result.collect()
        assert len(rows) == 2
        assert all(hasattr(r, "request_method") for r in rows)
        assert all(hasattr(r, "response_status_code") for r in rows)

    @SKIP_MAPIN_ARROW
    def test_all_cache_hits_triggers_zero_network_calls(self, spark, mock_session):
        from pyspark.sql import DataFrame as SparkDataFrame
        from yggdrasil.spark.cast import any_to_spark_schema

        req = make_request("https://example.com/a")
        cached = make_response(request=req.anonymize(mode="remove"))

        arrow_tbl = pa.Table.from_batches([cached.to_arrow_batch(parse=False)])
        spark_hit_df = spark.createDataFrame(
            arrow_tbl.to_pandas(),
            schema=any_to_spark_schema(RESPONSE_ARROW_SCHEMA),
        )
        result_obj = MagicMock()
        result_obj.to_spark.return_value = spark_hit_df

        table = make_table_mock(hits=[cached])
        table.sql.execute.return_value = result_obj

        result = mock_session.spark_send(
            iter([req]),
            spark_session=spark,
            remote_cache=make_cache_config(table),
        )

        assert isinstance(result, SparkDataFrame)
        assert len(result.collect()) == 1
        assert mock_session.calls == []

    @SKIP_MAPIN_ARROW
    def test_explode_flattens_list_rows(self, spark):
        from yggdrasil.spark.frame import DynamicFrame

        items = [{"x": 1}, {"x": 2}, {"x": 3}]

        def _identity(batch):
            return batch

        dyn = DynamicFrame.parallelize(_identity, [items], spark_session=spark)

        assert dyn.explode().collect() == items
