"""Tests for HTTP proxy support in HTTPSession.

Covers:
- ``_resolve_proxy_url`` env-var fallback logic
- ``_should_bypass_proxy`` / ``no_proxy`` matching
- ``_proxy_auth_headers`` Basic encoding
- HTTP-through-proxy with a local forward-proxy stub
- CONNECT tunnel negotiation with a local stub
- Proxy bypass via ``no_proxy``
"""
from __future__ import annotations

import base64
import http.server
import json
import os
import threading
from unittest import mock

import pytest

from yggdrasil.http_.session import (
    HTTPSession,
    _ProxyEnv,
    _resolve_proxy_url,
    _should_bypass_proxy,
)
from yggdrasil.url import URL


# ---------------------------------------------------------------------------
# Unit tests — proxy resolution helpers
# ---------------------------------------------------------------------------


class TestResolveProxyUrl:

    def test_explicit_proxy_wins(self):
        url = _resolve_proxy_url("http://proxy:8080", target_scheme="https")
        assert url.host == "proxy"
        assert url.port == 8080

    def test_explicit_url_object(self):
        u = URL.from_("http://proxy:3128")
        assert _resolve_proxy_url(u) is u

    def test_none_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert _resolve_proxy_url(None) is None

    def test_https_proxy_env(self):
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": "http://secure-proxy:443"}, clear=True):
            url = _resolve_proxy_url(None, target_scheme="https")
            assert url.host == "secure-proxy"

    def test_http_proxy_env(self):
        with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://plain-proxy:8080"}, clear=True):
            url = _resolve_proxy_url(None, target_scheme="http")
            assert url.host == "plain-proxy"

    def test_all_proxy_fallback(self):
        with mock.patch.dict(os.environ, {"ALL_PROXY": "http://catch-all:9090"}, clear=True):
            url = _resolve_proxy_url(None, target_scheme="https")
            assert url.host == "catch-all"

    def test_lowercase_env_vars(self):
        with mock.patch.dict(os.environ, {"https_proxy": "http://lower:1234"}, clear=True):
            url = _resolve_proxy_url(None, target_scheme="https")
            assert url.host == "lower"

    def test_https_proxy_not_used_for_http(self):
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": "http://secure-only:443"}, clear=True):
            assert _resolve_proxy_url(None, target_scheme="http") is None

    def test_explicit_overrides_env(self):
        with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://env-proxy:8080"}, clear=True):
            url = _resolve_proxy_url("http://explicit:3128", target_scheme="http")
            assert url.host == "explicit"


class TestShouldBypassProxy:

    def test_empty_no_proxy(self):
        assert not _should_bypass_proxy("example.com", no_proxy="")

    def test_exact_match(self):
        assert _should_bypass_proxy("localhost", no_proxy="localhost")

    def test_wildcard(self):
        assert _should_bypass_proxy("anything.com", no_proxy="*")

    def test_domain_suffix(self):
        assert _should_bypass_proxy("api.example.com", no_proxy=".example.com")

    def test_domain_suffix_without_dot(self):
        assert _should_bypass_proxy("api.example.com", no_proxy="example.com")

    def test_no_match(self):
        assert not _should_bypass_proxy("other.com", no_proxy="example.com")

    def test_multiple_entries(self):
        assert _should_bypass_proxy("localhost", no_proxy="127.0.0.1,localhost,.internal")
        assert _should_bypass_proxy("svc.internal", no_proxy="127.0.0.1,localhost,.internal")
        assert not _should_bypass_proxy("external.com", no_proxy="127.0.0.1,localhost,.internal")

    def test_env_fallback(self):
        with mock.patch.dict(os.environ, {"NO_PROXY": "localhost,127.0.0.1"}, clear=True):
            assert _should_bypass_proxy("localhost")
            assert not _should_bypass_proxy("example.com")

    def test_case_insensitive(self):
        assert _should_bypass_proxy("API.Example.COM", no_proxy="example.com")


class TestProxyAuthHeaders:

    def test_no_userinfo(self):
        headers = HTTPSession._proxy_auth_headers(URL.from_("http://proxy:8080"))
        assert headers == {}

    def test_with_userinfo(self):
        headers = HTTPSession._proxy_auth_headers(URL.from_("http://user:pass@proxy:8080"))
        assert "Proxy-Authorization" in headers
        expected = base64.b64encode(b"user:pass").decode()
        assert headers["Proxy-Authorization"] == f"Basic {expected}"

    def test_user_no_password(self):
        headers = HTTPSession._proxy_auth_headers(URL.from_("http://user@proxy:8080"))
        expected = base64.b64encode(b"user:").decode()
        assert headers["Proxy-Authorization"] == f"Basic {expected}"


# ---------------------------------------------------------------------------
# Integration — local forward-proxy stub for plain HTTP
# ---------------------------------------------------------------------------


class _ForwardProxyHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP forward proxy — accepts absolute-URI GET/HEAD/POST."""

    def do_GET(self):
        self._forward()

    def do_HEAD(self):
        self._forward()

    def do_POST(self):
        self._forward()

    def do_CONNECT(self):
        self.send_response(200, "Connection Established")
        self.end_headers()
        # We don't actually tunnel — the test just checks the 200 handshake.
        # For a real tunnel test we'd need to splice sockets; for unit-test
        # purposes the CONNECT acceptance is the thing under test.

    def _forward(self):
        body = json.dumps({
            "proxied": True,
            "method": self.command,
            "path": self.path,
            "proxy_auth": self.headers.get("Proxy-Authorization"),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def proxy_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _ForwardProxyHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


class _OriginHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"origin": True, "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def origin_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _OriginHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def _fresh_state():
    HTTPSession._INSTANCES.clear()
    _ProxyEnv.reset()
    yield
    HTTPSession._INSTANCES.clear()
    _ProxyEnv.reset()


# ---------------------------------------------------------------------------
# HTTP-through-proxy tests (plain HTTP target)
# ---------------------------------------------------------------------------


class TestHTTPProxy:

    def test_request_routed_through_proxy(self, proxy_server, origin_server):
        session = HTTPSession(proxy=proxy_server)
        resp = session.get(f"{origin_server}/hello")
        data = resp.json()
        assert data["proxied"] is True
        assert data["method"] == "GET"
        assert origin_server in data["path"]

    def test_absolute_url_sent_to_proxy(self, proxy_server, origin_server):
        session = HTTPSession(proxy=proxy_server)
        resp = session.get(f"{origin_server}/resource")
        data = resp.json()
        assert "/resource" in data["path"]
        assert "http://" in data["path"]

    def test_proxy_auth_header_forwarded(self, proxy_server, origin_server):
        session = HTTPSession(proxy=f"http://alice:secret@127.0.0.1:{URL.from_(proxy_server).port}")
        resp = session.get(f"{origin_server}/auth-check")
        data = resp.json()
        expected = base64.b64encode(b"alice:secret").decode()
        assert data["proxy_auth"] == f"Basic {expected}"

    def test_no_proxy_bypasses(self, proxy_server, origin_server):
        origin_host = URL.from_(origin_server).host
        session = HTTPSession(proxy=proxy_server, no_proxy=origin_host)
        resp = session.get(f"{origin_server}/direct")
        data = resp.json()
        assert data.get("origin") is True

    def test_env_proxy_picked_up(self, proxy_server, origin_server):
        with mock.patch.dict(os.environ, {"HTTP_PROXY": proxy_server}, clear=True):
            session = HTTPSession()
            resp = session.get(f"{origin_server}/env-test")
            data = resp.json()
            assert data["proxied"] is True

    def test_env_no_proxy_bypasses(self, proxy_server, origin_server):
        origin_host = URL.from_(origin_server).host
        with mock.patch.dict(os.environ, {
            "HTTP_PROXY": proxy_server,
            "NO_PROXY": origin_host,
        }, clear=True):
            session = HTTPSession()
            resp = session.get(f"{origin_server}/bypass")
            data = resp.json()
            assert data.get("origin") is True


# ---------------------------------------------------------------------------
# Proxy on session properties
# ---------------------------------------------------------------------------


class TestProxySessionProperties:

    def test_proxy_stored_on_session(self, proxy_server):
        session = HTTPSession(proxy=proxy_server)
        assert session.proxy is not None
        assert session.proxy.host == "127.0.0.1"

    def test_no_proxy_stored_on_session(self):
        session = HTTPSession(no_proxy="localhost,*.internal")
        assert session.no_proxy == "localhost,*.internal"

    def test_default_proxy_is_none(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            session = HTTPSession()
            assert session.proxy is None
            assert session.no_proxy is None


# ---------------------------------------------------------------------------
# Dead proxy fallback
# ---------------------------------------------------------------------------


class TestDeadProxyFallback:

    def test_unreachable_proxy_marked_dead(self, origin_server):
        session = HTTPSession(proxy="http://127.0.0.1:19999")
        assert len(session._dead_proxies) == 0
        resp = session.get(f"{origin_server}/direct")
        data = resp.json()
        assert data.get("origin") is True
        assert len(session._dead_proxies) == 1

    def test_second_request_skips_dead_proxy(self, origin_server):
        session = HTTPSession(proxy="http://127.0.0.1:19998")
        session.get(f"{origin_server}/first")
        assert len(session._dead_proxies) == 1
        resp = session.get(f"{origin_server}/second")
        assert resp.json().get("origin") is True

    def test_dead_proxy_not_shared_across_sessions(self, origin_server):
        # A proxy that one session gives up on must NOT be blacklisted for
        # other sessions — the dead-proxy state is scoped to the session.
        s1 = HTTPSession(proxy="http://127.0.0.1:19996")
        s1.get(f"{origin_server}/trigger")
        assert len(s1._dead_proxies) == 1
        s2 = HTTPSession(base_url=f"{origin_server}", proxy="http://127.0.0.1:19996")
        assert s2._dead_proxies == set()
        # s2 still reaches the origin (it falls back on its own failed connect),
        # but it independently decided to — s1 didn't poison it.
        resp = s2.get("/check")
        assert resp.json().get("origin") is True

    def test_env_resolved_once(self):
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": "http://env-proxy:3128"}, clear=True):
            _ProxyEnv.reset()
            env1 = _ProxyEnv.current()
            env2 = _ProxyEnv.current()
            assert env1 is env2
            assert env1.https.host == "env-proxy"

    def test_dead_proxy_persists_across_clear_connections(self, origin_server):
        session = HTTPSession(proxy="http://127.0.0.1:19995")
        session.get(f"{origin_server}/first")
        assert len(session._dead_proxies) == 1
        session.clear_connections()
        resp = session.get(f"{origin_server}/after-clear")
        assert resp.json().get("origin") is True
        assert len(session._dead_proxies) == 1

    def test_dead_proxy_prevents_routing_for_session(self, origin_server):
        session = HTTPSession(proxy="http://127.0.0.1:19994")
        proxy_url = URL.from_("http://127.0.0.1:19994")
        session._mark_proxy_dead(proxy_url)
        assert session._is_proxy_dead(proxy_url)
        resolved = session._resolve_proxy_for("http", URL.from_(origin_server).host)
        assert resolved is None

    def test_multiple_dead_proxies_tracked_per_session(self, origin_server):
        s1 = HTTPSession(proxy="http://127.0.0.1:19993")
        s1.get(f"{origin_server}/a")
        s2 = HTTPSession(proxy="http://127.0.0.1:19992")
        s2.get(f"{origin_server}/b")
        assert s1._dead_proxies == {"127.0.0.1:19993"}
        assert s2._dead_proxies == {"127.0.0.1:19992"}

    def test_dead_proxy_does_not_retry(self, origin_server):
        session = HTTPSession(proxy="http://127.0.0.1:19991")
        session.get(f"{origin_server}/trigger")
        dead_before = len(session._dead_proxies)
        for _ in range(5):
            session.get(f"{origin_server}/repeat")
        assert len(session._dead_proxies) == dead_before

    def test_dead_proxy_is_permanent_within_session(self):
        session = HTTPSession(proxy="http://127.0.0.1:19990")
        proxy_url = URL.from_("http://127.0.0.1:19990")
        session._mark_proxy_dead(proxy_url)
        # No expiry — the proxy stays dead for the life of this session.
        assert session._is_proxy_dead(proxy_url)
        assert session._is_proxy_dead(proxy_url)
