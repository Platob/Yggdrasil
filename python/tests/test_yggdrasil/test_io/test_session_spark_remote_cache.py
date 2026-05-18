"""Spark stage-4 fan-out tests for the remote cache.

The non-Spark ``Session._persist_remote`` resolves each response's
effective :class:`CacheConfig` via ``key_to_remote_cfg`` (keyed by
``PreparedRequest.public_url_hash``) and groups inserts by
``_remote_write_group_key``. Before this test's matching change the
Spark path collapsed onto ``session_remote_cfg`` and ignored
per-request overrides — meaning a chunk targeting two distinct remote
tables (or with a per-request ``remote_cache_enabled=False``) silently
mis-routed.

These tests bypass :class:`SparkTestCase` (which routes through
:meth:`PyEnv.spark_session` with ``connect=True``) so the suite runs
against vanilla local PySpark in environments where
``databricks-connect`` isn't installed. Skip the whole module when
PySpark or Java isn't reachable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yggdrasil.data.enums import Mode
from yggdrasil.io.response import RESPONSE_SCHEMA, Response
from yggdrasil.io.send_config import CacheConfig
from yggdrasil.io.session import Session

from ._helpers import StubSession, make_request, make_response


pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession  # noqa: E402


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    # Bypass PyEnv.spark_session / databricks-connect entirely — those
    # paths require either a Databricks workspace or the connect SDK.
    os.environ.setdefault("PYSPARK_PYTHON", "python")
    try:
        session = (
            SparkSession.builder
            .master("local[2]")
            .appName("ygg-spark-remote-cache-test")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"local SparkSession unavailable: {exc!r}")
    try:
        yield session
    finally:
        session.stop()


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    return tmp_path


class _SparkAwareFakeTabular:
    """Tabular double that records Spark-mode inserts.

    ``cfg.tabular.insert(df, ..., spark_session=spark)`` is the wire
    contract on the Spark path. The fake collects the frame to a
    pyarrow table so assertions can pinpoint exactly which rows landed
    where without depending on the live Spark plan.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self.inserts: list[dict[str, Any]] = []

    def full_name(self, safe: bool = False) -> str:
        return self._name

    def insert(
        self,
        df: Any,
        *,
        mode: Mode = Mode.APPEND,
        match_by: Any = None,
        wait: bool = False,
        spark_session: Any = None,
        **_: Any,
    ) -> None:
        rows = df.toPandas()
        self.inserts.append({
            "mode": mode,
            "match_by": tuple(match_by) if match_by else None,
            "wait": bool(wait),
            "url_hashes": [
                int(h) for h in rows["request_public_url_hash"].tolist()
            ],
            "status_codes": [int(c) for c in rows["status_code"].tolist()],
        })


def _remote_cfg(tab: _SparkAwareFakeTabular, **overrides: Any) -> CacheConfig:
    overrides.setdefault("mode", Mode.APPEND)
    overrides.setdefault("request_by", ["public_url_hash"])
    overrides.setdefault("wait", False)
    return CacheConfig(tabular=tab, **overrides)


_TEST_RESPONSE_COUNTER = 0


def _responses_to_spark(
    spark: SparkSession, scratch: Path, responses: list[Response],
):
    # ``createDataFrame`` on local PySpark 3.5 rejects both raw
    # ``pa.Table`` (no ChunkedArray inference) and Arrow-backed pandas
    # frames (map<string,string> headers come out as list-of-tuples
    # which the JVM side doesn't recognise). Round-tripping through a
    # parquet file is the cleanest way to drop the canonical
    # :data:`RESPONSE_SCHEMA` rows into a Spark plan without dragging
    # in Databricks Connect.
    global _TEST_RESPONSE_COUNTER
    _TEST_RESPONSE_COUNTER += 1
    table = pa.Table.from_batches([
        Response.values_to_arrow_batch(responses),
    ])
    path = scratch / f"resp-{_TEST_RESPONSE_COUNTER}.parquet"
    pq.write_table(table, str(path))
    return spark.read.schema(RESPONSE_SCHEMA.to_spark_schema()).parquet(str(path))


@pytest.fixture(autouse=True)
def _clear_session_singleton_cache():
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
        req_b = make_request("https://example.com/b")
        resp_a = make_response(request=req_a, body=b'{"k":"a"}')
        resp_b = make_response(request=req_b, body=b'{"k":"b"}')
        df = _responses_to_spark(spark, scratch, [resp_a, resp_b])

        key_to_remote_cfg = {
            req_a.public_url_hash: cfg_a,
            req_b.public_url_hash: cfg_b,
        }
        s = StubSession()
        s._spark_persist_remote(
            df, key_to_remote_cfg, cfg_a, spark=spark,
        )

        assert len(tab_a.inserts) == 1
        assert len(tab_b.inserts) == 1
        assert tab_a.inserts[0]["url_hashes"] == [req_a.public_url_hash]
        assert tab_b.inserts[0]["url_hashes"] == [req_b.public_url_hash]

    def test_disabled_per_request_cfg_drops_row(self, spark, scratch) -> None:
        tab = _SparkAwareFakeTabular("cache_t")
        cfg = _remote_cfg(tab)
        # No tabular → remote_cache_enabled is False. The request
        # mapped to this cfg must not show up in any insert call.
        disabled_cfg = CacheConfig()

        req_keep = make_request("https://example.com/keep")
        req_drop = make_request("https://example.com/drop")
        df = _responses_to_spark(spark, scratch, [
            make_response(request=req_keep, body=b'{"k":"keep"}'),
            make_response(request=req_drop, body=b'{"k":"drop"}'),
        ])

        key_to_remote_cfg = {
            req_keep.public_url_hash: cfg,
            req_drop.public_url_hash: disabled_cfg,
        }
        s = StubSession()
        s._spark_persist_remote(
            df, key_to_remote_cfg, cfg, spark=spark,
        )

        assert len(tab.inserts) == 1
        assert tab.inserts[0]["url_hashes"] == [req_keep.public_url_hash]

    def test_all_disabled_skips_persist(self, spark, scratch) -> None:
        disabled = CacheConfig()
        req = make_request("https://example.com/x")
        df = _responses_to_spark(
            spark, scratch,
            [make_response(request=req, body=b'{"v":1}')],
        )
        key_to_remote_cfg = {req.public_url_hash: disabled}
        s = StubSession()
        # Should be a no-op — nothing to assert beyond "doesn't raise".
        s._spark_persist_remote(
            df, key_to_remote_cfg, disabled, spark=spark,
        )

    def test_upsert_cfg_short_circuits_writeback(self, spark, scratch) -> None:
        # ``CacheConfig.cache_enabled`` is False for ``Mode.UPSERT`` —
        # mirrors the non-Spark ``_persist_remote`` semantics. The
        # upsert request must drop out of stage 4 entirely while the
        # APPEND request still lands.
        tab = _SparkAwareFakeTabular("cache_t")
        append_cfg = _remote_cfg(tab, mode=Mode.APPEND)
        upsert_cfg = _remote_cfg(tab, mode=Mode.UPSERT)

        req_append = make_request("https://example.com/append")
        req_upsert = make_request("https://example.com/upsert")
        df = _responses_to_spark(spark, scratch, [
            make_response(request=req_append, body=b'{"m":"append"}'),
            make_response(request=req_upsert, body=b'{"m":"upsert"}'),
        ])

        key_to_remote_cfg = {
            req_append.public_url_hash: append_cfg,
            req_upsert.public_url_hash: upsert_cfg,
        }
        s = StubSession()
        s._spark_persist_remote(
            df, key_to_remote_cfg, append_cfg, spark=spark,
        )

        assert len(tab.inserts) == 1
        assert tab.inserts[0]["mode"] == Mode.APPEND
        assert tab.inserts[0]["url_hashes"] == [req_append.public_url_hash]

    def test_failed_responses_filtered_out(self, spark, scratch) -> None:
        tab = _SparkAwareFakeTabular("cache_t")
        cfg = _remote_cfg(tab)

        req_ok = make_request("https://example.com/ok")
        req_err = make_request("https://example.com/err")
        df = _responses_to_spark(spark, scratch, [
            make_response(request=req_ok, body=b'{"ok":true}'),
            make_response(request=req_err, status_code=500, body=b"boom"),
        ])

        key_to_remote_cfg = {
            req_ok.public_url_hash: cfg,
            req_err.public_url_hash: cfg,
        }
        s = StubSession()
        s._spark_persist_remote(
            df, key_to_remote_cfg, cfg, spark=spark,
        )

        assert len(tab.inserts) == 1
        assert tab.inserts[0]["url_hashes"] == [req_ok.public_url_hash]
