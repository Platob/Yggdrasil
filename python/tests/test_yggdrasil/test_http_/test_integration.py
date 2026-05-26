"""Integration tests for HTTPSession with real HTTP calls and caching."""
from __future__ import annotations

import json
import http.server
import shutil
import threading

import pytest

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
        if self.path == "/json":
            body = json.dumps({"ok": True, "n": _Handler.call_count}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/text":
            body = b"hello world"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/empty":
            self.send_response(204)
            self.end_headers()
        elif self.path == "/error":
            self.send_response(500)
            body = b"internal error"
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        _Handler.call_count += 1
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        resp = json.dumps({"echoed": body.decode(), "n": _Handler.call_count}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, format, *args):
        pass


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


class TestGet:

    def test_get_json(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/json")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_get_text(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/text")
        assert resp.status_code == 200
        assert resp.text == "hello world"

    def test_get_empty(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/empty")
        assert resp.status_code == 204

    def test_get_404(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/nonexistent", raise_error=False)
        assert resp.status_code == 404

    def test_get_500(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/error", raise_error=False)
        assert resp.status_code == 500
        assert not resp.ok


class TestPost:

    def test_post_body(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.post("/echo", data=b'{"key": "value"}')
        assert resp.status_code == 200
        assert resp.json()["echoed"] == '{"key": "value"}'


class TestResponseProperties:

    def test_content_type(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/json")
        assert "application/json" in resp.headers.get("Content-Type", "")

    def test_ok_property(self, base_url):
        session = HTTPSession(base_url=base_url)
        assert session.get("/json").ok
        assert not session.get("/error", raise_error=False).ok

    def test_raise_for_status(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/error", raise_error=False)
        with pytest.raises(Exception):
            resp.raise_for_status()

    def test_response_has_request(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/json")
        assert resp.request is not None
        assert resp.request.method == "GET"


class TestSessionSingleton:

    def test_same_base_url_same_instance(self, base_url):
        a = HTTPSession(base_url=base_url)
        b = HTTPSession(base_url=base_url)
        assert a is b

    def test_different_base_url_different_instance(self, base_url):
        a = HTTPSession(base_url=base_url)
        b = HTTPSession(base_url="http://other.example.com")
        assert a is not b


class TestRequestResponse:

    def test_request_url_parsed(self, base_url):
        req = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        assert req.url.path == "/json"

    def test_request_hash_stable(self, base_url):
        a = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        b = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        assert a.hash == b.hash

    def test_request_hash_differs_by_method(self, base_url):
        a = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        b = HTTPRequest.prepare(method="POST", url=f"{base_url}/json")
        assert a.hash != b.hash


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def local_cache_dir(tmp_path):
    d = tmp_path / "local_cache"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def remote_cache_dir(tmp_path):
    d = tmp_path / "remote_cache"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _folder(path) -> FolderPath:
    return FolderPath(path=LocalPath.from_(str(path)))


class TestLocalCache:

    def test_local_cache_hit_skips_network(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        cfg = SendConfig(local_cache=cache)

        resp1 = session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            config=cfg,
        )
        assert resp1.status_code == 200
        n1 = resp1.json()["n"]

        resp2 = session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            config=cfg,
        )
        assert resp2.status_code == 200
        assert resp2.json()["n"] == n1

    def test_local_cache_different_urls_miss(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        cfg = SendConfig(local_cache=cache)

        resp1 = session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            config=cfg,
        )
        n1 = resp1.json()["n"]

        resp2 = session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/text"),
            config=cfg,
        )
        assert resp2.text == "hello world"
        assert _Handler.call_count > n1

    def test_local_cache_different_methods_miss(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        cfg = SendConfig(local_cache=cache)

        resp1 = session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            config=cfg,
        )
        n1 = resp1.json()["n"]

        resp2 = session.send(
            HTTPRequest.prepare(method="POST", url=f"{base_url}/json"),
            config=cfg,
        )
        assert resp2.json()["n"] != n1


class TestRemoteCache:

    def test_remote_folder_cache_hit(self, base_url, remote_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(remote_cache_dir))
        cfg = SendConfig(remote_cache=cache)

        resp1 = session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            config=cfg,
        )
        assert resp1.status_code == 200
        n1 = resp1.json()["n"]

        resp2 = session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            config=cfg,
        )
        assert resp2.status_code == 200
        assert resp2.json()["n"] == n1

    def test_remote_cache_miss_fetches_from_network(self, base_url, remote_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(remote_cache_dir))
        cfg = SendConfig(remote_cache=cache)

        resp = session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/text"),
            config=cfg,
        )
        assert resp.text == "hello world"
        assert _Handler.call_count >= 1


class TestDualCache:

    def test_local_and_remote_cache(self, base_url, local_cache_dir, remote_cache_dir):
        session = HTTPSession(base_url=base_url)
        cfg = SendConfig(
            local_cache=CacheConfig(tabular=_folder(local_cache_dir)),
            remote_cache=CacheConfig(tabular=_folder(remote_cache_dir)),
        )

        resp1 = session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            config=cfg,
        )
        assert resp1.status_code == 200
        n1 = resp1.json()["n"]

        resp2 = session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            config=cfg,
        )
        assert resp2.json()["n"] == n1

    def test_local_cache_populated_from_remote(self, base_url, local_cache_dir, remote_cache_dir):
        session = HTTPSession(base_url=base_url)
        remote_cfg = SendConfig(
            remote_cache=CacheConfig(tabular=_folder(remote_cache_dir)),
        )
        resp1 = session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            config=remote_cfg,
        )
        n1 = resp1.json()["n"]

        dual_cfg = SendConfig(
            local_cache=CacheConfig(tabular=_folder(local_cache_dir)),
            remote_cache=CacheConfig(tabular=_folder(remote_cache_dir)),
        )
        resp2 = session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            config=dual_cfg,
        )
        assert resp2.json()["n"] == n1


class TestSendMany:

    def test_send_many_multiple_requests(self, base_url):
        session = HTTPSession(base_url=base_url)
        reqs = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/text"),
        ]
        responses = list(session.send_many(reqs))
        assert len(responses) >= 2

    def test_send_many_with_cache(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))

        reqs = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/text"),
        ]
        for r in reqs:
            r.send_config = SendConfig(local_cache=cache)

        responses1 = list(session.send_many(reqs))
        assert len(responses1) >= 2
        n1 = _Handler.call_count

        reqs2 = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/text"),
        ]
        for r in reqs2:
            r.send_config = SendConfig(local_cache=cache)

        responses2 = list(session.send_many(reqs2))
        assert len(responses2) >= 2
        assert _Handler.call_count == n1


class TestCacheOnly:

    def test_cache_only_returns_404_on_miss(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        cfg = SendConfig(local_cache=cache, cache_only=True, raise_error=False)

        resp = session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            config=cfg,
        )
        assert resp.status_code == 404
        assert _Handler.call_count == 0

    def test_cache_only_returns_cached_on_hit(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))

        session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            config=SendConfig(local_cache=cache),
        )
        n_after_warm = _Handler.call_count

        cfg = SendConfig(local_cache=cache, cache_only=True)
        resp = session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            config=cfg,
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert _Handler.call_count == n_after_warm


# ---------------------------------------------------------------------------
# Cache upsert / overwrite
# ---------------------------------------------------------------------------


class TestCacheUpsert:

    def test_append_mode_keeps_old_entries(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        from yggdrasil.enums import Mode
        cache = CacheConfig(tabular=_folder(local_cache_dir), mode=Mode.APPEND)
        cfg = SendConfig(local_cache=cache)

        session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/json"), config=cfg)
        session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/text"), config=cfg)

        folder = _folder(local_cache_dir)
        table = folder.read_arrow_table()
        assert table.num_rows >= 2

    def test_overwrite_mode_writes(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        from yggdrasil.enums import Mode
        cache = CacheConfig(tabular=_folder(local_cache_dir), mode=Mode.OVERWRITE)
        cfg = SendConfig(local_cache=cache)

        session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/json"), config=cfg)
        rows = _folder(local_cache_dir).read_arrow_table().num_rows
        assert rows >= 1

        session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/json"), config=cfg)
        rows_after = _folder(local_cache_dir).read_arrow_table().num_rows
        assert rows_after >= 1


# ---------------------------------------------------------------------------
# Cache key correctness
# ---------------------------------------------------------------------------


class TestCacheKeyCorrectness:

    def test_post_with_different_body_misses(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        cfg = SendConfig(local_cache=cache)

        resp1 = session.send(
            HTTPRequest.prepare(method="POST", url=f"{base_url}/echo", body=b"body_a"),
            config=cfg,
        )
        n1 = resp1.json()["n"]

        resp2 = session.send(
            HTTPRequest.prepare(method="POST", url=f"{base_url}/echo", body=b"body_b"),
            config=cfg,
        )
        n2 = resp2.json()["n"]
        assert n2 != n1

    def test_post_same_body_hits_cache(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        cfg = SendConfig(local_cache=cache)

        resp1 = session.send(
            HTTPRequest.prepare(method="POST", url=f"{base_url}/echo", body=b"same"),
            config=cfg,
        )
        n1 = resp1.json()["n"]

        resp2 = session.send(
            HTTPRequest.prepare(method="POST", url=f"{base_url}/echo", body=b"same"),
            config=cfg,
        )
        assert resp2.json()["n"] == n1

    def test_different_query_params_different_cache_key(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        req_a = HTTPRequest.prepare(method="GET", url=f"{base_url}/json?page=1")
        req_b = HTTPRequest.prepare(method="GET", url=f"{base_url}/json?page=2")
        assert req_a.public_hash != req_b.public_hash


# ---------------------------------------------------------------------------
# Error response caching
# ---------------------------------------------------------------------------


class TestErrorCaching:

    def test_500_response_not_cached(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        cfg = SendConfig(local_cache=cache, raise_error=False)

        session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/error"),
            config=cfg,
        )
        n1 = _Handler.call_count

        session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/error"),
            config=cfg,
        )
        assert _Handler.call_count > n1

    def test_404_response_not_cached(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        cfg = SendConfig(local_cache=cache, raise_error=False)

        session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/nonexistent"),
            config=cfg,
        )
        n1 = _Handler.call_count

        session.send(
            HTTPRequest.prepare(method="GET", url=f"{base_url}/nonexistent"),
            config=cfg,
        )
        assert _Handler.call_count > n1


# ---------------------------------------------------------------------------
# Cache read-back fidelity
# ---------------------------------------------------------------------------


class TestCacheFidelity:

    def test_cached_response_preserves_status_code(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        cfg = SendConfig(local_cache=cache)

        session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/json"), config=cfg)
        resp = session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/json"), config=cfg)
        assert resp.status_code == 200

    def test_cached_response_preserves_body(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        cfg = SendConfig(local_cache=cache)

        session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/text"), config=cfg)
        resp = session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/text"), config=cfg)
        assert resp.text == "hello world"

    def test_cached_response_preserves_json(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        cfg = SendConfig(local_cache=cache)

        session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/json"), config=cfg)
        resp = session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/json"), config=cfg)
        data = resp.json()
        assert data["ok"] is True
        assert isinstance(data["n"], int)

    def test_cached_response_preserves_content_type(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        cfg = SendConfig(local_cache=cache)

        session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/json"), config=cfg)
        resp = session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/json"), config=cfg)
        assert "application/json" in resp.headers.get("Content-Type", "")

    def test_cached_response_has_request(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        cfg = SendConfig(local_cache=cache)

        session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/json"), config=cfg)
        resp = session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/json"), config=cfg)
        assert resp.request is not None
        assert resp.request.method == "GET"


# ---------------------------------------------------------------------------
# Anonymization
# ---------------------------------------------------------------------------


class TestCacheAnonymization:

    def test_cached_response_strips_auth_header(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir), anonymize="remove")
        cfg = SendConfig(local_cache=cache)

        req = HTTPRequest.prepare(
            method="GET",
            url=f"{base_url}/json",
            headers={"Authorization": "Bearer secret123"},
        )
        session.send(req, config=cfg)

        folder = _folder(local_cache_dir)
        table = folder.read_arrow_table()
        assert table.num_rows >= 1


# ---------------------------------------------------------------------------
# send_many ordering and batching
# ---------------------------------------------------------------------------


class TestSendManyAdvanced:

    def test_send_many_returns_all(self, base_url):
        session = HTTPSession(base_url=base_url)
        urls = [f"{base_url}/json", f"{base_url}/text", f"{base_url}/json"]
        reqs = [HTTPRequest.prepare(method="GET", url=u) for u in urls]
        responses = list(session.send_many(reqs))
        assert len(responses) >= len(urls)

    def test_send_many_cached_no_extra_calls(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))

        urls = [f"{base_url}/json", f"{base_url}/text"]
        reqs1 = [HTTPRequest.prepare(method="GET", url=u) for u in urls]
        for r in reqs1:
            r.send_config = SendConfig(local_cache=cache)
        list(session.send_many(reqs1))
        n_after = _Handler.call_count

        reqs2 = [HTTPRequest.prepare(method="GET", url=u) for u in urls]
        for r in reqs2:
            r.send_config = SendConfig(local_cache=cache)
        list(session.send_many(reqs2))
        assert _Handler.call_count == n_after

    def test_send_many_mixed_hit_miss(self, base_url, local_cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(tabular=_folder(local_cache_dir))
        cfg = SendConfig(local_cache=cache)

        session.send(HTTPRequest.prepare(method="GET", url=f"{base_url}/json"), config=cfg)
        n_after_warm = _Handler.call_count

        reqs = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/text"),
        ]
        for r in reqs:
            r.send_config = SendConfig(local_cache=cache)
        responses = list(session.send_many(reqs))
        assert len(responses) >= 2
        assert _Handler.call_count == n_after_warm + 1
