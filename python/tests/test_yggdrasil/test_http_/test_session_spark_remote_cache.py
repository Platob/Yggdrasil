"""Spark stage-4 fan-out tests for the remote cache.

Tests bypass :class:`SparkTestCase` so the suite runs against vanilla
local PySpark without ``databricks-connect``. Skip the whole module
when PySpark or Java isn't reachable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yggdrasil.enums import Mode
from yggdrasil.http_.session import HTTPSession
from yggdrasil.http_.response import RESPONSE_SCHEMA, Response
from yggdrasil.http_.send_config import CacheConfig
from yggdrasil.http_.session import Session
from yggdrasil.io.tabular import Tabular

from ._helpers import StubSession, make_request, make_response


pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession  # noqa: E402


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    from yggdrasil.spark.tests import _get_test_spark

    try:
        return _get_test_spark(app_name="ygg-spark-remote-cache-test")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"local SparkSession unavailable: {exc!r}")


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    return tmp_path


class _SparkAwareFakeTabular(Tabular):
    """Tabular double that records Spark-mode inserts."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self.inserts: list[dict[str, Any]] = []

    def full_name(self, safe: bool = False) -> str:
        return self._name

    def _read_arrow_batches(self, options=None): return iter(())
    def _write_arrow_batches(self, batches, options=None): pass

    def write_spark_frame(
        self,
        df: Any,
        options: Any = None,
        **_: Any,
    ) -> None:
        rows = df.toPandas()
        match_by_fields = getattr(options, "match_by", None)
        match_by = (
            tuple(f.name for f in match_by_fields)
            if match_by_fields else None
        )
        self.inserts.append({
            "mode": getattr(options, "mode", None),
            "match_by": match_by,
            "url_hashes": [
                int(h) for h in rows["request_public_url_hash"].tolist()
            ],
            "status_codes": [int(c) for c in rows["status_code"].tolist()],
        })


def _remote_cfg(tab: _SparkAwareFakeTabular, **overrides: Any) -> CacheConfig:
    overrides.setdefault("mode", Mode.APPEND)
    overrides.pop("request_by", None)
    overrides.pop("wait", None)
    return CacheConfig(tabular=tab, **overrides)


def _responses_to_spark(
    spark: SparkSession,
    scratch: Path,
    responses: list[Response],
) -> "pyspark.sql.DataFrame":
    batches = [r.to_arrow_batch(parse=False) for r in responses]
    table = pa.Table.from_batches(batches)
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / "responses.parquet"
    pq.write_table(table, str(out))
    return spark.read.parquet(str(out))


@pytest.fixture(autouse=True)
def _clear_singletons():
    Session._INSTANCES.clear()
    yield
    Session._INSTANCES.clear()


class TestSparkPersistRemote:

    def test_per_request_override_routes_to_alt_table(
        self, spark, scratch,
    ) -> None:
        tab_a = _SparkAwareFakeTabular("cache_a")
        tab_b = _SparkAwareFakeTabular("cache_b")
        cfg_a = _remote_cfg(tab_a)
        cfg_b = _remote_cfg(tab_b)

        req_a = make_request("https://example.com/a")
        resp_a = make_response(request=req_a, body=b'{"k":"a"}')
        df_a = _responses_to_spark(spark, scratch / "a", [resp_a])

        req_b = make_request("https://example.com/b")
        resp_b = make_response(request=req_b, body=b'{"k":"b"}')
        df_b = _responses_to_spark(spark, scratch / "b", [resp_b])

        s = StubSession()
        s._spark_persist_remote(df_a, cfg_a, spark=spark)
        s._spark_persist_remote(df_b, cfg_b, spark=spark)

        assert len(tab_a.inserts) == 1
        assert len(tab_b.inserts) == 1
        assert tab_a.inserts[0]["url_hashes"] == [req_a.public_url_hash]
        assert tab_b.inserts[0]["url_hashes"] == [req_b.public_url_hash]

    def test_disabled_cfg_skips_persist(self, spark, scratch) -> None:
        disabled = CacheConfig()
        req = make_request("https://example.com/x")
        df = _responses_to_spark(
            spark, scratch,
            [make_response(request=req, body=b'{"v":1}')],
        )
        s = StubSession()
        s._spark_persist_remote(df, disabled, spark=spark)

    def test_upsert_cfg_short_circuits_writeback(self, spark, scratch) -> None:
        tab = _SparkAwareFakeTabular("cache_t")
        upsert_cfg = _remote_cfg(tab, mode=Mode.UPSERT)

        req = make_request("https://example.com/upsert")
        df = _responses_to_spark(spark, scratch, [
            make_response(request=req, body=b'{"k":"upsert"}'),
        ])

        s = StubSession()
        s._spark_persist_remote(df, upsert_cfg, spark=spark)
        assert len(tab.inserts) == 0

    def test_ok_responses_inserted(self, spark, scratch) -> None:
        tab = _SparkAwareFakeTabular("cache_t")
        cfg = _remote_cfg(tab)

        req = make_request("https://example.com/ok")
        df = _responses_to_spark(spark, scratch, [
            make_response(request=req, body=b'{"ok":true}'),
        ])

        s = StubSession()
        s._spark_persist_remote(df, cfg, spark=spark)
        assert len(tab.inserts) == 1
        assert tab.inserts[0]["url_hashes"] == [req.public_url_hash]

    def test_failed_responses_filtered_out(self, spark, scratch) -> None:
        tab = _SparkAwareFakeTabular("cache_t")
        cfg = _remote_cfg(tab)

        req_ok = make_request("https://example.com/ok")
        req_err = make_request("https://example.com/err")
        df = _responses_to_spark(spark, scratch, [
            make_response(request=req_ok, body=b'{"ok":true}'),
            make_response(request=req_err, status_code=500, body=b"boom"),
        ])

        s = StubSession()
        s._spark_persist_remote(df, cfg, spark=spark)
        assert len(tab.inserts) == 1
        assert tab.inserts[0]["url_hashes"] == [req_ok.public_url_hash]


class TestPinSparkSnapshot:

    def test_dataset_wraps_empty_frame(self, spark) -> None:
        from yggdrasil.io.tabular import Dataset

        schema = RESPONSE_SCHEMA.to_spark_schema()
        df = spark.createDataFrame([], schema=schema)
        tab = Dataset(df)
        assert tab is not None
        assert tab.frame is df
