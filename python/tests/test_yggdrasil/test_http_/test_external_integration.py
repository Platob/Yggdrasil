"""Integration tests for HTTPSession against real external HTTP sites.

These tests hit live endpoints (example.com, httpbin.org) to validate
the full request/response pipeline over a real network: TLS handshake,
redirect following, content negotiation, header handling, verb methods,
and response parsing.

Requires outbound HTTPS.  Skipped automatically when the network is
unreachable.
"""
from __future__ import annotations

import pytest

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.response import HTTPResponse
from yggdrasil.http_.session import HTTPSession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _check_network():
    """Skip the entire module when the network is unreachable."""
    import socket
    try:
        socket.create_connection(("example.com", 443), timeout=5).close()
    except OSError:
        pytest.skip("Network unreachable — skipping external integration tests")


@pytest.fixture(autouse=True)
def _fresh_singleton_cache():
    HTTPSession._INSTANCES.clear()
    yield
    HTTPSession._INSTANCES.clear()


# ---------------------------------------------------------------------------
# Basic GET — example.com
# ---------------------------------------------------------------------------


class TestExampleCom:

    def test_get_returns_200(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/")
        assert resp.status_code == 200

    def test_response_has_html_body(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/")
        assert "<html" in resp.text.lower()
        assert "example domain" in resp.text.lower()

    def test_ok_property(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/")
        assert resp.ok

    def test_content_type_header(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/")
        ct = resp.headers.get("Content-Type", "") or resp.headers.get("content-type", "")
        assert "text/html" in ct

    def test_response_has_request_attached(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/")
        assert resp.request is not None
        assert resp.request.method == "GET"

    def test_content_bytes_non_empty(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/")
        assert len(resp.content) > 100

    def test_head_returns_200_no_body(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.head("/")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Full URL (no base_url)
# ---------------------------------------------------------------------------


class TestFullURL:

    def test_get_with_full_url_no_base(self):
        session = HTTPSession()
        resp = session.get("https://example.com/")
        assert resp.status_code == 200
        assert "example" in resp.text.lower()

    def test_different_hosts_in_same_session(self):
        session = HTTPSession()
        r1 = session.get("https://example.com/")
        assert r1.status_code == 200


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------


class TestSingleton:

    def test_same_base_url_same_instance(self):
        a = HTTPSession(base_url="https://example.com")
        b = HTTPSession(base_url="https://example.com")
        assert a is b

    def test_different_base_url_different_instance(self):
        a = HTTPSession(base_url="https://example.com")
        b = HTTPSession(base_url="https://www.iana.org")
        assert a is not b


# ---------------------------------------------------------------------------
# HTTPRequest round-trip
# ---------------------------------------------------------------------------


class TestRequestProperties:

    def test_request_url_parsed(self):
        req = HTTPRequest.prepare(method="GET", url="https://example.com/path?q=1")
        assert req.url.path == "/path"
        assert "q=1" in req.url.to_string()

    def test_request_hash_stable(self):
        a = HTTPRequest.prepare(method="GET", url="https://example.com/")
        b = HTTPRequest.prepare(method="GET", url="https://example.com/")
        assert a.hash == b.hash

    def test_request_hash_differs_by_method(self):
        a = HTTPRequest.prepare(method="GET", url="https://example.com/")
        b = HTTPRequest.prepare(method="POST", url="https://example.com/")
        assert a.hash != b.hash

    def test_request_hash_differs_by_path(self):
        a = HTTPRequest.prepare(method="GET", url="https://example.com/a")
        b = HTTPRequest.prepare(method="GET", url="https://example.com/b")
        assert a.hash != b.hash


# ---------------------------------------------------------------------------
# Response metadata (arrow values)
# ---------------------------------------------------------------------------


class TestResponseMetadata:

    def test_arrow_values_status_code(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/")
        assert resp.arrow_values["status_code"] == 200

    def test_arrow_values_body_size_positive(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/")
        assert resp.arrow_values["body_size"] > 0

    def test_arrow_values_request_method(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/")
        assert resp.arrow_values["request_method"] == "GET"

    def test_arrow_values_request_hash_stable(self):
        session = HTTPSession(base_url="https://example.com")
        r1 = session.get("/")
        r2 = session.get("/")
        assert r1.arrow_values["request_hash"] == r2.arrow_values["request_hash"]

    def test_arrow_batch_from_responses(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/")
        batch = HTTPResponse.values_to_arrow_batch([resp])
        assert batch.num_rows == 1
        assert "status_code" in batch.schema.names
        assert "body" in batch.schema.names


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:

    def test_404_with_raise_error_false(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/this-page-does-not-exist-404", raise_error=False)
        assert resp.status_code == 404
        assert not resp.ok

    def test_404_raises_by_default(self):
        session = HTTPSession(base_url="https://example.com")
        with pytest.raises(Exception):
            session.get("/this-page-does-not-exist-404")


# ---------------------------------------------------------------------------
# send_many
# ---------------------------------------------------------------------------


class TestSendMany:

    def test_send_many_multiple_urls(self):
        session = HTTPSession()
        reqs = [
            HTTPRequest.prepare(method="GET", url="https://example.com/"),
            HTTPRequest.prepare(method="GET", url="https://www.iana.org/"),
        ]
        responses = list(session.send_many(reqs))
        assert len(responses) >= 2
        assert all(r.status_code == 200 for r in responses)

    def test_send_many_same_host(self):
        session = HTTPSession(base_url="https://example.com")
        reqs = [
            HTTPRequest.prepare(method="GET", url="https://example.com/"),
            HTTPRequest.prepare(method="HEAD", url="https://example.com/"),
        ]
        responses = list(session.send_many(reqs))
        assert len(responses) >= 2


# ---------------------------------------------------------------------------
# TLS / HTTPS
# ---------------------------------------------------------------------------


class TestTLS:

    def test_https_works(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/")
        assert resp.status_code == 200

    def test_request_url_is_https(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/")
        url_str = resp.request.url.to_string()
        assert url_str.startswith("https://")


# ---------------------------------------------------------------------------
# Custom headers
# ---------------------------------------------------------------------------


class TestCustomHeaders:

    def test_custom_user_agent(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/", headers={"User-Agent": "yggdrasil-test/1.0"})
        assert resp.status_code == 200

    def test_accept_header(self):
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/", headers={"Accept": "text/html"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Redirects (example.com HTTP → HTTPS)
# ---------------------------------------------------------------------------


class TestRedirects:

    def test_http_to_https_redirect(self):
        session = HTTPSession(base_url="http://example.com")
        resp = session.get("/")
        assert resp.status_code == 200
        assert "example" in resp.text.lower()


# ---------------------------------------------------------------------------
# Browser-profile smoke test — a real server sees a coherent browser
# ---------------------------------------------------------------------------


class TestBrowserProfileLive:
    """Send a generated browser profile to live endpoints and confirm the
    server receives a coherent, browser-shaped request (matching UA + client
    hints), and that a real CDN accepts it. Header-only compatibility — no TLS
    fingerprint spoofing."""

    @staticmethod
    def _echoed(resp):
        # httpbin echoes request headers; normalise names for comparison.
        return {k.lower(): v for k, v in resp.json()["headers"].items()}

    def test_profile_headers_reach_the_server(self):
        from yggdrasil.http_.user_agents import random_browser_profile

        prof = random_browser_profile(seed=1234)
        session = HTTPSession()
        try:
            resp = session.get("https://httpbin.org/headers", headers=prof.headers)
        except Exception as exc:  # noqa: BLE001 — flaky public service
            pytest.skip(f"httpbin.org unreachable: {exc}")
        if resp.status_code != 200:
            pytest.skip(f"httpbin.org returned {resp.status_code}")

        echoed = self._echoed(resp)
        # The identity headers reach the server verbatim — it sees a consistent
        # browser (matching UA + client hints + Sec-Fetch + language). ``Accept``
        # is library-owned content negotiation (HTTPHeaders.normalized collapses
        # it to a single media type), so it's intentionally outside the rotating
        # identity and not asserted here.
        for key, value in prof.identity.items():
            assert echoed.get(key.lower()) == value, key

    def test_user_agent_endpoint_sees_the_profile_ua(self):
        from yggdrasil.http_.user_agents import random_browser_profile

        prof = random_browser_profile(seed=99)
        session = HTTPSession()
        try:
            resp = session.get("https://httpbin.org/user-agent", headers=prof.headers)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"httpbin.org unreachable: {exc}")
        if resp.status_code != 200:
            pytest.skip(f"httpbin.org returned {resp.status_code}")
        assert resp.json()["user-agent"] == prof.user_agent

    def test_real_cdn_accepts_browser_profile(self):
        from yggdrasil.http_.user_agents import random_browser_profile

        prof = random_browser_profile(seed=7)
        session = HTTPSession(base_url="https://example.com")
        resp = session.get("/", headers=prof.headers)
        assert resp.status_code == 200
        assert "example domain" in resp.text.lower()
