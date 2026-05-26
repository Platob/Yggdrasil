"""Spark integration tests for HTTPSession.send_many with Spark fan-out.

Requires pyspark + Java — skipped when not available.
"""
from __future__ import annotations

import json
import http.server
import shutil
import threading

import pytest

pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession

from yggdrasil.http_.cache_config import CacheConfig
from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.send_config import SendConfig
from yggdrasil.http_.session import HTTPSession
from yggdrasil.io.nested.folder_path import FolderPath
from yggdrasil.io.path.local_path import LocalPath


class _Handler(http.server.BaseHTTPRequestHandler):
    call_count: int = 0

    def do_GET(self):
        _Handler.call_count += 1
        body = json.dumps({"path": self.path, "n": _Handler.call_count}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("ygg-http-test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="module")
def server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


@pytest.fixture()
def base_url(server):
    _Handler.call_count = 0
    HTTPSession._INSTANCES.clear()
    return server


@pytest.fixture()
def cache_dir(tmp_path):
    d = tmp_path / "spark_cache"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _folder(path) -> FolderPath:
    return FolderPath(path=LocalPath.from_(str(path)))


class TestSparkSendMany:

    def test_spark_fan_out_fetches_all(self, base_url, spark):
        session = HTTPSession(base_url=base_url)
        reqs = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/a"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/b"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/c"),
        ]
        responses = list(session.send_many(reqs, spark_session=spark))
        assert len(responses) >= 3
        paths = {r.json()["path"] for r in responses}
        assert "/a" in paths
        assert "/b" in paths
        assert "/c" in paths

    def test_spark_fan_out_returns_valid_responses(self, base_url, spark):
        session = HTTPSession(base_url=base_url)
        reqs = [HTTPRequest.prepare(method="GET", url=f"{base_url}/x")]
        responses = list(session.send_many(reqs, spark_session=spark))
        assert len(responses) >= 1
        resp = responses[0]
        assert resp.status_code == 200
        assert resp.json()["path"] == "/x"

    def test_spark_fan_out_with_cache(self, base_url, spark, cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(cache_dir))

        reqs = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/cached_a"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/cached_b"),
        ]
        for r in reqs:
            r.send_config = SendConfig(local_cache=cache)

        responses1 = list(session.send_many(reqs, spark_session=spark))
        assert len(responses1) >= 2
        n_after = _Handler.call_count

        reqs2 = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/cached_a"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/cached_b"),
        ]
        for r in reqs2:
            r.send_config = SendConfig(local_cache=cache)

        responses2 = list(session.send_many(reqs2, spark_session=spark))
        assert len(responses2) >= 2
        assert _Handler.call_count == n_after


class TestSparkResponseArrow:

    def test_response_to_spark_dataframe(self, base_url, spark):
        session = HTTPSession(base_url=base_url)
        resp = session.get(f"/arrow_test")
        arrow_table = resp.read_arrow_table()
        df = spark.createDataFrame(arrow_table.to_pandas())
        assert df.count() >= 1

    def test_multiple_responses_to_dataframe(self, base_url, spark):
        session = HTTPSession(base_url=base_url)
        reqs = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/r1"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/r2"),
        ]
        from yggdrasil.http_.response import HTTPResponse
        responses = list(session.send_many(reqs))
        batch = HTTPResponse.values_to_arrow_batch(responses)
        assert batch.num_rows >= 2


# ---------------------------------------------------------------------------
# Spark + cache interactions
# ---------------------------------------------------------------------------


@pytest.fixture()
def local_cache_dir(tmp_path):
    d = tmp_path / "local"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def remote_cache_dir(tmp_path):
    d = tmp_path / "remote"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestSparkLocalCache:

    def test_spark_populates_local_cache(self, base_url, spark, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        reqs = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_local_a"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_local_b"),
        ]
        for r in reqs:
            r.send_config = SendConfig(local_cache=cache)

        list(session.send_many(reqs, spark_session=spark))

        table = _folder(local_cache_dir).read_arrow_table()
        assert table.num_rows >= 2

    def test_spark_reads_from_local_cache(self, base_url, spark, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))

        reqs1 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_read_a")]
        for r in reqs1:
            r.send_config = SendConfig(local_cache=cache)
        list(session.send_many(reqs1, spark_session=spark))
        n_after_warm = _Handler.call_count

        reqs2 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_read_a")]
        for r in reqs2:
            r.send_config = SendConfig(local_cache=cache)
        responses = list(session.send_many(reqs2, spark_session=spark))
        assert len(responses) >= 1
        assert _Handler.call_count == n_after_warm

    def test_spark_mixed_cache_hit_miss(self, base_url, spark, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))

        warm = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_mix_hit")]
        for r in warm:
            r.send_config = SendConfig(local_cache=cache)
        list(session.send_many(warm, spark_session=spark))
        n_after_warm = _Handler.call_count

        mixed = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_mix_hit"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_mix_miss"),
        ]
        for r in mixed:
            r.send_config = SendConfig(local_cache=cache)
        responses = list(session.send_many(mixed, spark_session=spark))
        assert len(responses) >= 2
        assert _Handler.call_count == n_after_warm + 1


class TestSparkRemoteCache:

    def test_spark_populates_remote_folder_cache(self, base_url, spark, remote_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(remote_cache_dir))
        reqs = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_remote_a"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_remote_b"),
        ]
        for r in reqs:
            r.send_config = SendConfig(remote_cache=cache)

        list(session.send_many(reqs, spark_session=spark))

        table = _folder(remote_cache_dir).read_arrow_table()
        assert table.num_rows >= 2

    def test_spark_reads_from_remote_cache(self, base_url, spark, remote_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(remote_cache_dir))

        reqs1 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_rread")]
        for r in reqs1:
            r.send_config = SendConfig(remote_cache=cache)
        list(session.send_many(reqs1, spark_session=spark))
        n_after_warm = _Handler.call_count

        reqs2 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_rread")]
        for r in reqs2:
            r.send_config = SendConfig(remote_cache=cache)
        responses = list(session.send_many(reqs2, spark_session=spark))
        assert len(responses) >= 1
        assert _Handler.call_count == n_after_warm


class TestSparkDualCache:

    def test_spark_local_and_remote_together(self, base_url, spark, local_cache_dir, remote_cache_dir):
        session = HTTPSession(base_url=base_url)
        cfg = SendConfig(
            local_cache=CacheConfig(tabular=_folder(local_cache_dir)),
            remote_cache=CacheConfig(tabular=_folder(remote_cache_dir)),
        )

        reqs1 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_dual")]
        for r in reqs1:
            r.send_config = cfg
        list(session.send_many(reqs1, spark_session=spark))
        n_after = _Handler.call_count

        reqs2 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_dual")]
        for r in reqs2:
            r.send_config = cfg
        list(session.send_many(reqs2, spark_session=spark))
        assert _Handler.call_count == n_after

    def test_spark_remote_backfills_local(self, base_url, spark, local_cache_dir, remote_cache_dir):
        session = HTTPSession(base_url=base_url)
        remote_cfg = SendConfig(remote_cache=CacheConfig(tabular=_folder(remote_cache_dir)))

        reqs1 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_backfill")]
        for r in reqs1:
            r.send_config = remote_cfg
        list(session.send_many(reqs1, spark_session=spark))
        n_after = _Handler.call_count

        dual_cfg = SendConfig(
            local_cache=CacheConfig(tabular=_folder(local_cache_dir)),
            remote_cache=CacheConfig(tabular=_folder(remote_cache_dir)),
        )
        reqs2 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_backfill")]
        for r in reqs2:
            r.send_config = dual_cfg
        list(session.send_many(reqs2, spark_session=spark))
        assert _Handler.call_count == n_after


class TestSparkCacheOnly:

    def test_spark_cache_only_miss_returns_404(self, base_url, spark, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        reqs = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_co_miss")]
        for r in reqs:
            r.send_config = SendConfig(local_cache=cache, cache_only=True, raise_error=False)

        responses = list(session.send_many(reqs, spark_session=spark))
        assert len(responses) >= 1
        assert responses[0].status_code == 404
        assert _Handler.call_count == 0

    def test_spark_cache_only_hit_returns_cached(self, base_url, spark, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))

        warm = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_co_hit")]
        for r in warm:
            r.send_config = SendConfig(local_cache=cache)
        list(session.send_many(warm, spark_session=spark))
        n_after = _Handler.call_count

        reqs = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_co_hit")]
        for r in reqs:
            r.send_config = SendConfig(local_cache=cache, cache_only=True)
        responses = list(session.send_many(reqs, spark_session=spark))
        assert len(responses) >= 1
        assert responses[0].status_code == 200
        assert _Handler.call_count == n_after


class TestSparkCacheFidelity:

    def test_spark_cached_response_preserves_body(self, base_url, spark, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))

        reqs1 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_fidelity")]
        for r in reqs1:
            r.send_config = SendConfig(local_cache=cache)
        responses1 = list(session.send_many(reqs1, spark_session=spark))
        original_body = responses1[0].json()

        reqs2 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_fidelity")]
        for r in reqs2:
            r.send_config = SendConfig(local_cache=cache)
        responses2 = list(session.send_many(reqs2, spark_session=spark))
        cached_body = responses2[0].json()

        assert cached_body["path"] == original_body["path"]
        assert cached_body["n"] == original_body["n"]

    def test_spark_cached_response_preserves_status(self, base_url, spark, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))

        reqs = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_status")]
        for r in reqs:
            r.send_config = SendConfig(local_cache=cache)

        list(session.send_many(reqs, spark_session=spark))

        reqs2 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_status")]
        for r in reqs2:
            r.send_config = SendConfig(local_cache=cache)
        responses = list(session.send_many(reqs2, spark_session=spark))
        assert responses[0].status_code == 200

    def test_spark_many_urls_all_cached(self, base_url, spark, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        urls = [f"{base_url}/sp_bulk_{i}" for i in range(5)]

        reqs1 = [HTTPRequest.prepare(method="GET", url=u) for u in urls]
        for r in reqs1:
            r.send_config = SendConfig(local_cache=cache)
        list(session.send_many(reqs1, spark_session=spark))
        n_after = _Handler.call_count

        reqs2 = [HTTPRequest.prepare(method="GET", url=u) for u in urls]
        for r in reqs2:
            r.send_config = SendConfig(local_cache=cache)
        responses = list(session.send_many(reqs2, spark_session=spark))
        assert len(responses) >= 5
        assert _Handler.call_count == n_after


# ---------------------------------------------------------------------------
# Spark response batch metadata consistency
# ---------------------------------------------------------------------------


class TestSparkBatchMetadata:

    def test_spark_arrow_batch_has_expected_columns(self, base_url, spark):
        session = HTTPSession(base_url=base_url)
        from yggdrasil.http_.response import HTTPResponse
        resp = session.get("/json")
        batch = HTTPResponse.values_to_arrow_batch([resp])
        names = batch.schema.names
        assert "status_code" in names
        assert "hash" in names
        assert "public_hash" in names
        assert "body" in names
        assert "body_size" in names
        assert "body_hash" in names
        assert "request_method" in names
        assert "request_hash" in names
        assert "partition_key" in names

    def test_spark_hashes_stable_across_runs(self, base_url, spark, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))

        reqs1 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_meta_hash")]
        for r in reqs1:
            r.send_config = SendConfig(local_cache=cache)
        resps1 = list(session.send_many(reqs1, spark_session=spark))
        h1 = resps1[0].arrow_values["hash"]
        ph1 = resps1[0].arrow_values["public_hash"]
        rh1 = resps1[0].arrow_values["request_hash"]

        reqs2 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_meta_hash")]
        for r in reqs2:
            r.send_config = SendConfig(local_cache=cache)
        resps2 = list(session.send_many(reqs2, spark_session=spark))
        h2 = resps2[0].arrow_values["hash"]
        ph2 = resps2[0].arrow_values["public_hash"]
        rh2 = resps2[0].arrow_values["request_hash"]

        assert h1 == h2
        assert ph1 == ph2
        assert rh1 == rh2

    def test_spark_body_hash_consistent(self, base_url, spark, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))

        reqs1 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_meta_body")]
        for r in reqs1:
            r.send_config = SendConfig(local_cache=cache)
        resps1 = list(session.send_many(reqs1, spark_session=spark))
        bh1 = resps1[0].arrow_values["body_hash"]
        bs1 = resps1[0].arrow_values["body_size"]

        reqs2 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_meta_body")]
        for r in reqs2:
            r.send_config = SendConfig(local_cache=cache)
        resps2 = list(session.send_many(reqs2, spark_session=spark))

        assert resps2[0].arrow_values["body_hash"] == bh1
        assert resps2[0].arrow_values["body_size"] == bs1
        assert bs1 > 0

    def test_spark_partition_key_stable(self, base_url, spark):
        session = HTTPSession(base_url=base_url)
        reqs = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_meta_pk"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_meta_pk"),
        ]
        resps = list(session.send_many(reqs, spark_session=spark))
        pks = [r.arrow_values["partition_key"] for r in resps]
        assert pks[0] == pks[1]

    def test_spark_partition_key_differs_by_url(self, base_url, spark):
        session = HTTPSession(base_url=base_url)
        reqs = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_pk_a"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_pk_b"),
        ]
        resps = list(session.send_many(reqs, spark_session=spark))
        assert resps[0].arrow_values["partition_key"] != resps[1].arrow_values["partition_key"]

    def test_spark_cached_metadata_matches_fresh(self, base_url, spark, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))

        reqs1 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_meta_full")]
        for r in reqs1:
            r.send_config = SendConfig(local_cache=cache)
        fresh = list(session.send_many(reqs1, spark_session=spark))[0]
        fresh_vals = {
            "hash": fresh.arrow_values["hash"],
            "public_hash": fresh.arrow_values["public_hash"],
            "status_code": fresh.arrow_values["status_code"],
            "body_size": fresh.arrow_values["body_size"],
            "body_hash": fresh.arrow_values["body_hash"],
            "request_hash": fresh.arrow_values["request_hash"],
            "request_method": fresh.arrow_values["request_method"],
            "partition_key": fresh.arrow_values["partition_key"],
        }

        reqs2 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_meta_full")]
        for r in reqs2:
            r.send_config = SendConfig(local_cache=cache)
        cached = list(session.send_many(reqs2, spark_session=spark))[0]

        for key in fresh_vals:
            assert fresh_vals[key] == cached.arrow_values[key], (
                f"{key}: {fresh_vals[key]} != {cached.arrow_values[key]}"
            )


class TestSparkBatchHolders:

    def test_spark_batch_new_tabular_is_set(self, base_url, spark):
        session = HTTPSession(base_url=base_url)
        reqs = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_holder_new")]
        for r in reqs:
            r.send_config = SendConfig()
        batches = list(session.send_many_batches(reqs, spark_session=spark))
        assert len(batches) >= 1
        batch = batches[0]
        assert batch.new_tabular is not None

    def test_spark_batch_read_arrow_batches(self, base_url, spark):
        session = HTTPSession(base_url=base_url)
        reqs = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_holder_arrow_a"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_holder_arrow_b"),
        ]
        for r in reqs:
            r.send_config = SendConfig()
        batches = list(session.send_many_batches(reqs, spark_session=spark))
        arrow_batches = list(batches[0].read_arrow_batches())
        total_rows = sum(b.num_rows for b in arrow_batches)
        assert total_rows >= 2

    def test_spark_batch_responses_iterator(self, base_url, spark):
        session = HTTPSession(base_url=base_url)
        reqs = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_holder_iter_a"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_holder_iter_b"),
        ]
        for r in reqs:
            r.send_config = SendConfig()
        batches = list(session.send_many_batches(reqs, spark_session=spark))
        responses = list(batches[0].responses())
        assert len(responses) >= 2
        assert all(r.status_code == 200 for r in responses)

    def test_spark_batch_with_cache_has_local_tabular(self, base_url, spark, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))

        reqs_warm = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_holder_cached")]
        for r in reqs_warm:
            r.send_config = SendConfig(local_cache=cache)
        list(session.send_many(reqs_warm, spark_session=spark))

        reqs = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_holder_cached")]
        for r in reqs:
            r.send_config = SendConfig(local_cache=cache)
        batches = list(session.send_many_batches(reqs, spark_session=spark))
        batch = batches[0]
        assert batch.local_tabular is not None

    def test_spark_batch_extend_merges(self, base_url, spark):
        session = HTTPSession(base_url=base_url)
        reqs_a = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_ext_a")]
        reqs_b = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_ext_b")]
        for r in reqs_a + reqs_b:
            r.send_config = SendConfig()
        batches_a = list(session.send_many_batches(reqs_a, spark_session=spark))
        batches_b = list(session.send_many_batches(reqs_b, spark_session=spark))
        merged = batches_a[0].extend(batches_b[0])
        responses = list(merged.responses())
        assert len(responses) >= 2

    def test_spark_batch_schema_matches_response_schema(self, base_url, spark):
        session = HTTPSession(base_url=base_url)
        from yggdrasil.http_.schemas import RESPONSE_SCHEMA
        reqs = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_schema")]
        for r in reqs:
            r.send_config = SendConfig()
        batches = list(session.send_many_batches(reqs, spark_session=spark))
        arrow_batches = list(batches[0].read_arrow_batches())
        assert len(arrow_batches) >= 1
        batch_schema = arrow_batches[0].schema
        expected = RESPONSE_SCHEMA.to_arrow_schema()
        assert set(batch_schema.names) == set(expected.names)


class TestSparkHolderTypes:

    def test_spark_batch_new_tabular_is_spark_dataset(self, base_url, spark):
        from yggdrasil.spark.tabular import SparkDataset
        session = HTTPSession(base_url=base_url)
        reqs = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_type_a")]
        for r in reqs:
            r.send_config = SendConfig()
        batches = list(session.send_many_batches(reqs, spark_session=spark))
        tab = batches[0].new_tabular
        assert tab is not None
        assert isinstance(tab, SparkDataset)

    def test_spark_batch_holders_are_spark_dataset(self, base_url, spark):
        from yggdrasil.spark.tabular import SparkDataset
        session = HTTPSession(base_url=base_url)
        reqs = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_type_b")]
        for r in reqs:
            r.send_config = SendConfig()
        batches = list(session.send_many_batches(reqs, spark_session=spark))
        for holder in batches[0]._holders():
            assert isinstance(holder, SparkDataset)

    def test_spark_dataset_has_spark_frame(self, base_url, spark):
        from pyspark.sql import DataFrame
        from yggdrasil.spark.tabular import SparkDataset
        session = HTTPSession(base_url=base_url)
        reqs = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_type_frame")]
        for r in reqs:
            r.send_config = SendConfig()
        batches = list(session.send_many_batches(reqs, spark_session=spark))
        tab = batches[0].new_tabular
        assert isinstance(tab, SparkDataset)
        assert isinstance(tab.frame, DataFrame)

    def test_spark_dataset_readable_as_arrow(self, base_url, spark):
        import pyarrow as pa
        session = HTTPSession(base_url=base_url)
        reqs = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_type_arrow")]
        for r in reqs:
            r.send_config = SendConfig()
        batches = list(session.send_many_batches(reqs, spark_session=spark))
        tab = batches[0].new_tabular
        table = tab.read_arrow_table()
        assert isinstance(table, pa.Table)
        assert table.num_rows >= 1

    def test_spark_cached_data_persists_in_folder(self, base_url, spark, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))

        reqs = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_type_cached")]
        for r in reqs:
            r.send_config = SendConfig(local_cache=cache)
        list(session.send_many(reqs, spark_session=spark))

        import pyarrow as pa
        batches = list(_folder(local_cache_dir).read_arrow_batches())
        total = sum(b.num_rows for b in batches)
        assert total >= 1

    def test_spark_batch_schema_matches_response_schema(self, base_url, spark):
        from yggdrasil.http_.schemas import RESPONSE_SCHEMA
        session = HTTPSession(base_url=base_url)
        reqs = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_type_schema_a"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_type_schema_b"),
        ]
        for r in reqs:
            r.send_config = SendConfig()
        batches = list(session.send_many_batches(reqs, spark_session=spark))
        arrow_batches = list(batches[0].read_arrow_batches())
        expected_names = set(RESPONSE_SCHEMA.to_arrow_schema().names)
        for ab in arrow_batches:
            assert set(ab.schema.names) == expected_names

    def test_spark_batch_holders_after_dual_cache(self, base_url, spark, local_cache_dir, remote_cache_dir):
        from yggdrasil.io.tabular.base import Tabular
        session = HTTPSession(base_url=base_url)
        cfg = SendConfig(
            local_cache=CacheConfig(tabular=_folder(local_cache_dir)),
            remote_cache=CacheConfig(tabular=_folder(remote_cache_dir)),
        )

        reqs1 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_type_dual")]
        for r in reqs1:
            r.send_config = cfg
        list(session.send_many(reqs1, spark_session=spark))

        reqs2 = [HTTPRequest.prepare(method="GET", url=f"{base_url}/sp_type_dual")]
        for r in reqs2:
            r.send_config = cfg
        batches = list(session.send_many_batches(reqs2, spark_session=spark))
        holders = batches[0]._holders()
        assert len(holders) >= 1
        assert all(isinstance(h, Tabular) for h in holders)
