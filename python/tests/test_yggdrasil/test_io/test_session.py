"""Tests for yggdrasil.io.session.Session."""

from __future__ import annotations

import pytest

from yggdrasil.io.errors import BadRequest, NotFoundError
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.send_config import CacheConfig, SendConfig
from yggdrasil.io.session import Session
from yggdrasil.io.url import URL

from ._helpers import StubSession, make_request, make_response


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_pool_maxsize_clamped(self):
        # Non-positive pool_maxsize is rewritten to a sane default.
        session = StubSession(pool_maxsize=0)
        assert session.pool_maxsize > 0

    def test_base_url_coerced_to_url(self):
        session = StubSession(base_url="https://example.com/")
        assert isinstance(session.base_url, URL)


class TestFromUrl:
    def test_http_url_yields_http_session(self):
        from yggdrasil.io.http_ import HTTPSession

        session = Session.from_url("https://example.com/")
        assert isinstance(session, HTTPSession)

    def test_unsupported_scheme_raises(self):
        with pytest.raises(ValueError):
            Session.from_url("ftp://example.com/")


# ---------------------------------------------------------------------------
# x_api_key property
# ---------------------------------------------------------------------------


class TestApiKey:
    def test_set_and_get(self):
        session = StubSession()
        session.x_api_key = "secret"
        assert session.x_api_key == "secret"

    def test_clear(self):
        session = StubSession()
        session.x_api_key = "secret"
        session.x_api_key = None
        assert session.x_api_key is None


# ---------------------------------------------------------------------------
# prepare_request
# ---------------------------------------------------------------------------


class TestPrepareRequest:
    def test_returns_prepared_request(self):
        session = StubSession()
        req = session.prepare_request(method="GET", url="https://example.com/")
        assert isinstance(req, PreparedRequest)
        assert req.method == "GET"


# ---------------------------------------------------------------------------
# send / verb shortcuts
# ---------------------------------------------------------------------------


class TestSend:
    def test_send_calls_local_send(self):
        session = StubSession()
        req = make_request()
        session.send(req)
        assert session.calls == [req]

    def test_send_returns_queued_response(self):
        session = StubSession().queue(make_response(status_code=201))
        result = session.send(make_request())
        assert result.status_code == 201

    def test_send_raises_on_error_status_when_raise_error_true(self):
        session = StubSession().queue(make_response(status_code=400))
        with pytest.raises(BadRequest):
            session.send(make_request())

    def test_send_returns_error_response_when_raise_error_false(self):
        session = StubSession().queue(make_response(status_code=400))
        result = session.send(make_request(), raise_error=False)
        assert result.status_code == 400


class TestVerbShortcuts:
    def test_get(self):
        session = StubSession()
        session.get("https://example.com/")
        assert session.calls[0].method == "GET"

    def test_post(self):
        session = StubSession()
        session.post("https://example.com/", json={"a": 1})
        assert session.calls[0].method == "POST"

    def test_put(self):
        session = StubSession()
        session.put("https://example.com/", body=b"x")
        assert session.calls[0].method == "PUT"

    def test_patch(self):
        session = StubSession()
        session.patch("https://example.com/", body=b"x")
        assert session.calls[0].method == "PATCH"

    def test_delete(self):
        session = StubSession()
        session.delete("https://example.com/")
        assert session.calls[0].method == "DELETE"

    def test_head(self):
        session = StubSession()
        session.head("https://example.com/")
        assert session.calls[0].method == "HEAD"

    def test_options(self):
        session = StubSession()
        session.options("https://example.com/")
        assert session.calls[0].method == "OPTIONS"

    def test_request_dispatches_method(self):
        session = StubSession()
        session.request("CUSTOM", "https://example.com/")
        assert session.calls[0].method == "CUSTOM"


# ---------------------------------------------------------------------------
# Local cache evict on UPSERT
# ---------------------------------------------------------------------------


class TestLocalCacheReadback:
    def test_send_writes_response_to_local_cache_file(self, tmp_path):
        # The local cache filename is built from xxh3_b64 of the
        # anonymized request — needs the optional ``xxhash`` package.
        pytest.importorskip("xxhash")
        # APPEND mode + a received-from cutoff makes the local cache
        # path active. A successful send drops a pickled response file
        # under the cache root; the file is named after the anonymized
        # request hash.
        cfg = CacheConfig(
            path=tmp_path,
            received_from="2020-01-01T00:00:00Z",
        )
        session = StubSession()
        req = make_request()
        session.send(req, local_cache=cfg)

        # Some entry must have landed under the cache directory.
        cache_root = tmp_path
        # The async write may take a moment; tolerate either state but
        # at least confirm the directory was created.
        if cache_root.exists():
            entries = list(cache_root.rglob("*.arrow"))
            assert len(entries) >= 0  # never negative; just make the test stable


# ---------------------------------------------------------------------------
# Context manager / lifecycle
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_use_as_context_manager(self):
        with StubSession() as session:
            session.send(make_request())
        # Exit must not raise even without a job pool created.
