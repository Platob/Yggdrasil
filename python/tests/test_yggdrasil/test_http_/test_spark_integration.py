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
