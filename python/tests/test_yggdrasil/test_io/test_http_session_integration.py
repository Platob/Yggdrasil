"""Integration tests: real HTTP requests against an in-process server.

These tests exercise the real network path of :class:`HTTPSession` —
no mocks. A small :mod:`http.server` stands in for an upstream API on
``127.0.0.1`` and counts the requests it sees, so cache-hit assertions
become "the server's hit counter did not increment."

Coverage:

- ``HTTPSession.send`` with a local :class:`CacheConfig` that takes a
  fresh fetch on the first call and serves from disk on the second.
- ``HTTPSession.send_many`` running the same flow over a batch.
- ``HTTPSession.send_many(spark_session=...)`` returning a real
  :class:`pyspark.sql.DataFrame` whose rows match the responses, with
  the local cache short-circuiting subsequent batches.

Marked ``integration`` so the suite stays opt-in for environments
where Java / urllib3 / pyspark aren't available.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pytest

from yggdrasil.io.errors import NotFoundError
from yggdrasil.io.http_ import HTTPSession
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.send_config import CacheConfig


def _wait_for_cache(tmp_path: Path, expected: int = 1, timeout: float = 10.0) -> None:
    """Wait until at least *expected* responses have landed in the cache.

    Cache writes go through ``Job.make(...).fire_and_forget()`` so
    the partitioned-folder write is in flight when the call site
    returns. Polling on the :class:`FolderIO` row count is the
    cheapest reliable barrier — it walks the partitioned tree and
    reads every leaf the way a real lookup would.
    """
    from yggdrasil.io.buffer.nested.folder_io import FolderIO

    cache_root = tmp_path
    deadline = time.time() + timeout
    while time.time() < deadline:
        n = 0
        if cache_root.exists():
            try:
                with FolderIO(path=cache_root) as folder:
                    n = folder.read_arrow_table().num_rows
            except Exception:
                n = 0
        if n >= expected:
            return
        time.sleep(0.05)
    raise AssertionError(
        f"timed out waiting for {expected} cached response(s) under {tmp_path}"
    )


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Test HTTP server
# ---------------------------------------------------------------------------


class _CountingHandler(BaseHTTPRequestHandler):
    """Echo handler that counts hits per path.

    GET / and GET /<key> return JSON ``{"path": ..., "hit": N}`` where N
    is how many times that path has been served. This makes "did the
    network actually run?" trivially observable from the test side.
    """

    # Disable the default access log — pollutes pytest output.
    def log_message(self, format, *args):  # noqa: A002 — base-class signature
        return

    def do_GET(self):  # noqa: N802 — base-class hook
        path = self.path
        counters = self.server.counters  # type: ignore[attr-defined]
        with self.server.lock:  # type: ignore[attr-defined]
            counters[path] = counters.get(path, 0) + 1
            hit = counters[path]
        if path.startswith("/error/"):
            try:
                code = int(path.rsplit("/", 1)[-1])
            except ValueError:
                code = 500
            self.send_response(code)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"error")
            return
        body = json.dumps({"path": path, "hit": hit}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802 — base-class hook
        length = int(self.headers.get("Content-Length", "0") or "0")
        body_in = self.rfile.read(length) if length else b""
        with self.server.lock:  # type: ignore[attr-defined]
            self.server.counters[self.path] = (  # type: ignore[attr-defined]
                self.server.counters.get(self.path, 0) + 1  # type: ignore[attr-defined]
            )
        body = json.dumps(
            {"path": self.path, "echo": body_in.decode("utf-8", errors="replace")}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Server:
    """Helper that owns the lifetime of the test ``ThreadingHTTPServer``."""

    def __init__(self):
        # Ephemeral port on loopback only.
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _CountingHandler)
        self._httpd.counters = {}
        self._httpd.lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address
        return f"http://{host}:{port}"

    @property
    def counters(self) -> dict[str, int]:
        return self._httpd.counters

    def reset_counters(self) -> None:
        with self._httpd.lock:
            self._httpd.counters.clear()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture(scope="module")
def http_server() -> Iterator[_Server]:
    server = _Server()
    # Make sure the bound socket is actually accepting before tests run.
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with socket.create_connection(
                ("127.0.0.1", server._httpd.server_address[1]), timeout=0.5
            ):
                break
        except OSError:
            time.sleep(0.05)
    yield server
    server.stop()


@pytest.fixture(autouse=True)
def _reset_counters(http_server: _Server):
    http_server.reset_counters()


# ---------------------------------------------------------------------------
# HTTPSession.send + local cache
# ---------------------------------------------------------------------------


class TestHttpSessionRealRequest:
    def test_get_returns_200_with_body(self, http_server: _Server):
        with HTTPSession(verify=False) as session:
            resp = session.get(f"{http_server.base_url}/probe")
        assert resp.ok
        assert resp.status_code == 200
        body = resp.json()
        assert body["path"] == "/probe"
        assert body["hit"] == 1

    def test_get_404_raises_by_default(self, http_server: _Server):
        with HTTPSession(verify=False) as session:
            with pytest.raises(NotFoundError):
                session.get(f"{http_server.base_url}/error/404")

    def test_get_404_returns_response_when_raise_error_false(
        self, http_server: _Server
    ):
        with HTTPSession(verify=False) as session:
            resp = session.get(
                f"{http_server.base_url}/error/404", raise_error=False
            )
        assert resp.status_code == 404

    def test_post_echoes_body(self, http_server: _Server):
        with HTTPSession(verify=False) as session:
            resp = session.post(
                f"{http_server.base_url}/echo",
                json={"hello": "world"},
            )
        body = resp.json()
        assert body["path"] == "/echo"
        assert "hello" in body["echo"]


# Local-cache filenames are derived via xxh3_b64 of the anonymized
# request, so the whole local-cache surface depends on the optional
# ``xxhash`` package. Skip the cache classes when the extra isn't
# installed.
class TestHttpSessionLocalCache:
    def _cache(self, tmp_path) -> CacheConfig:
        return CacheConfig(
            path=tmp_path,
            received_from="2020-01-01T00:00:00Z",
        )

    def test_first_call_hits_network_second_serves_from_cache(
        self, http_server: _Server, tmp_path
    ):
        pytest.importorskip("xxhash")
        cache = self._cache(tmp_path)
        url = f"{http_server.base_url}/cache/first"
        with HTTPSession(verify=False) as session:
            first = session.get(url, local_cache=cache)
            _wait_for_cache(tmp_path, expected=1)
            second = session.get(url, local_cache=cache)

        assert first.ok and second.ok
        # First call landed on the upstream; second was served from disk.
        assert http_server.counters.get("/cache/first") == 1
        # Both responses parse to the same "hit": 1 — proving the second
        # came from the cached payload, not a fresh server count of 2.
        assert first.json() == {"path": "/cache/first", "hit": 1}
        assert second.json() == {"path": "/cache/first", "hit": 1}

    def test_cache_files_are_written_under_configured_root(
        self, http_server: _Server, tmp_path
    ):
        pytest.importorskip("xxhash")
        cache = self._cache(tmp_path)
        with HTTPSession(verify=False) as session:
            session.get(f"{http_server.base_url}/cache/path-check", local_cache=cache)
        _wait_for_cache(tmp_path, expected=1)
        # Partitioned layout: at least one leaf lives under
        # ``<root>/request_method=GET/request_url_host=.../`` — assert
        # on the partitioned shape, not a flat glob, so a layout
        # change doesn't sneak through unnoticed.
        cache_root = tmp_path
        leaves = [p for p in cache_root.rglob("*") if p.is_file()]
        assert leaves, "expected at least one partition leaf under tmp_path"
        rel_dirs = {str(leaf.parent.relative_to(cache_root)) for leaf in leaves}
        assert any(
            d.startswith("request_method=GET") for d in rel_dirs
        ), f"expected Hive-partitioned layout, got: {sorted(rel_dirs)!r}"

    def test_distinct_urls_cache_independently(
        self, http_server: _Server, tmp_path
    ):
        pytest.importorskip("xxhash")
        cache = self._cache(tmp_path)
        with HTTPSession(verify=False) as session:
            session.get(f"{http_server.base_url}/cache/a", local_cache=cache)
            session.get(f"{http_server.base_url}/cache/b", local_cache=cache)

        assert http_server.counters.get("/cache/a") == 1
        assert http_server.counters.get("/cache/b") == 1


class TestHttpSessionSendManyLocalCache:
    def _cache(self, tmp_path) -> CacheConfig:
        return CacheConfig(
            path=tmp_path,
            received_from="2020-01-01T00:00:00Z",
        )

    def _build_requests(self, base_url: str, n: int) -> list[PreparedRequest]:
        return [
            PreparedRequest.prepare(method="GET", url=f"{base_url}/many/{i}")
            for i in range(n)
        ]

    def test_send_many_first_pass_hits_each_url_once(
        self, http_server: _Server, tmp_path
    ):
        pytest.importorskip("xxhash")
        cache = self._cache(tmp_path)
        reqs = self._build_requests(http_server.base_url, 4)
        with HTTPSession(verify=False) as session:
            responses = list(session.send_many(iter(reqs), local_cache=cache))

        assert len(responses) == len(reqs)
        for i in range(4):
            assert http_server.counters.get(f"/many/{i}") == 1

    def test_send_many_second_pass_reads_from_cache(
        self, http_server: _Server, tmp_path
    ):
        pytest.importorskip("xxhash")
        cache = self._cache(tmp_path)
        reqs = self._build_requests(http_server.base_url, 3)
        with HTTPSession(verify=False) as session:
            list(session.send_many(iter(reqs), local_cache=cache))
            _wait_for_cache(tmp_path, expected=3)
            http_server.reset_counters()
            list(session.send_many(iter(reqs), local_cache=cache))

        assert http_server.counters == {}, (
            "second pass should be served entirely from disk"
        )

    def test_send_many_batch_first_pass_all_new_hits(
        self, http_server: _Server, tmp_path
    ):
        from yggdrasil.io.buffer.base import TabularIO
        from yggdrasil.io.session import ResponseBatch

        pytest.importorskip("xxhash")
        cache = self._cache(tmp_path)
        reqs = self._build_requests(http_server.base_url, 3)
        with HTTPSession(verify=False) as session:
            batch = session.send_many_batch(iter(reqs), local_cache=cache)

        assert isinstance(batch, ResponseBatch)
        assert batch.counts == {"local": 0, "remote": 0, "new": 3}
        assert len(batch) == 3
        # Even unfilled buckets carry a schema-bearing empty holder
        # so the batch always advertises the response schema. The
        # remote bucket is a per-table dict whose default placeholder
        # entry holds the schema-bearing empty.
        # Both keyed buckets are dicts; an unfilled stage carries only
        # the default-placeholder schema-bearing empty holder so the
        # batch can still answer schema questions.
        assert isinstance(batch.local_hits, dict)
        assert all(isinstance(h, TabularIO) for h in batch.local_hits.values())
        assert isinstance(batch.remote_hits, dict)
        assert all(isinstance(h, TabularIO) for h in batch.remote_hits.values())
        assert isinstance(batch.new_hits, TabularIO)
        # No per-key hits — only the default placeholders are present,
        # so both breakdowns elide the empty defaults and read empty.
        assert batch.local_counts == {}
        assert batch.remote_counts == {}
        assert all(r.status_code == 200 for r in batch)

    def test_send_many_batch_second_pass_all_local_hits(
        self, http_server: _Server, tmp_path
    ):
        from yggdrasil.io.session import ResponseBatch

        pytest.importorskip("xxhash")
        cache = self._cache(tmp_path)
        reqs = self._build_requests(http_server.base_url, 2)
        with HTTPSession(verify=False) as session:
            session.send_many_batch(iter(reqs), local_cache=cache)
            _wait_for_cache(tmp_path, expected=2)
            http_server.reset_counters()
            batch = session.send_many_batch(iter(reqs), local_cache=cache)

        assert isinstance(batch, ResponseBatch)
        assert batch.counts == {"local": 2, "remote": 0, "new": 0}
        assert http_server.counters == {}, (
            "second-pass send_many_batch should not touch upstream"
        )


# ---------------------------------------------------------------------------
# HTTPSession.send_many with Spark
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spark_session():
    pyspark = pytest.importorskip("pyspark")
    del pyspark  # imported only to gate the suite

    from yggdrasil.environ import PyEnv

    spark = PyEnv.spark_session(
        create=True,
        install_spark=False,
        import_error=True,
    )
    yield spark
    # Don't stop the session — `PyEnv.spark_session` returns a singleton
    # shared across the test process. Stopping it here would break any
    # Spark-using test that runs later in the same pytest invocation.


class TestHttpSessionSparkSend:
    def test_send_many_with_spark_returns_dataframe(
        self, http_server: _Server, spark_session
    ):
        from pyspark.sql import DataFrame as SparkDataFrame

        reqs = [
            PreparedRequest.prepare(
                method="GET", url=f"{http_server.base_url}/spark/{i}"
            )
            for i in range(3)
        ]
        with HTTPSession(verify=False) as session:
            df = session.send_many(iter(reqs), spark_session=spark_session)

        assert isinstance(df, SparkDataFrame)
        rows = df.collect()
        assert len(rows) == 3

    def test_spark_send_many_paths_all_hit_upstream(
        self, http_server: _Server, spark_session
    ):
        reqs = [
            PreparedRequest.prepare(
                method="GET", url=f"{http_server.base_url}/spark-real/{i}"
            )
            for i in range(2)
        ]
        with HTTPSession(verify=False) as session:
            df = session.send_many(iter(reqs), spark_session=spark_session)
            df.collect()  # force evaluation of mapInArrow workers

        for i in range(2):
            assert http_server.counters.get(f"/spark-real/{i}", 0) >= 1

    def test_spark_send_many_with_local_cache_short_circuits(
        self, http_server: _Server, spark_session, tmp_path
    ):
        cache = CacheConfig(
            path=tmp_path,
            received_from="2020-01-01T00:00:00Z",
        )
        reqs = [
            PreparedRequest.prepare(
                method="GET", url=f"{http_server.base_url}/spark-cache/{i}"
            )
            for i in range(2)
        ]

        with HTTPSession(verify=False) as session:
            # Warm the cache via the local (non-Spark) path so the writes
            # go through the driver's job pool rather than ephemeral
            # Spark worker subprocesses, which can exit before
            # fire-and-forget pickle dumps complete.
            list(session.send_many(iter(reqs), local_cache=cache))
            _wait_for_cache(tmp_path, expected=2)

            http_server.reset_counters()
            df = session.send_many(
                iter(reqs),
                spark_session=spark_session,
                local_cache=cache,
            )
            rows = df.collect()

        assert len(rows) == 2
        # The Spark pass reused the warm cache: stage 1 (driver-side)
        # served every request from disk and the worker fanout never
        # touched the upstream counter.
        for i in range(2):
            assert http_server.counters.get(f"/spark-cache/{i}", 0) == 0
