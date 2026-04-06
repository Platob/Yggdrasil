# tests/test_yggdrasil/test_io/test_sessions/test_browser.py
"""Unit tests for BrowserHTTPSession (no network required)."""
from __future__ import annotations

import socket
import ssl

import pytest

from yggdrasil.io.http_ import BrowserHTTPSession, HTTPSession
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.url import URL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(url: str = "https://example.com/page") -> PreparedRequest:
    return PreparedRequest.prepare("GET", url)


def _tcp_can_connect(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------


class TestBrowserHTTPSessionConstruction:
    def test_default_user_agent_auto_generated(self):
        b = BrowserHTTPSession()
        assert b.user_agent is not None
        assert len(b.user_agent) > 20

    def test_explicit_user_agent(self):
        ua = "Mozilla/5.0 (custom)"
        b = BrowserHTTPSession(user_agent=ua)
        assert b.user_agent == ua

    def test_deterministic_ua_with_seed(self):
        b1 = BrowserHTTPSession(ua_seed=42)
        b2 = BrowserHTTPSession(ua_seed=42)
        assert b1.user_agent == b2.user_agent

    def test_different_seed_gives_different_ua(self):
        b1 = BrowserHTTPSession(ua_seed=1)
        b2 = BrowserHTTPSession(ua_seed=2)
        assert b1.user_agent != b2.user_agent

    def test_default_accept_language(self):
        b = BrowserHTTPSession()
        assert b.accept_language == "en-US,en;q=0.9"

    def test_custom_accept_language(self):
        b = BrowserHTTPSession(accept_language="fr-FR,fr;q=0.9")
        assert b.accept_language == "fr-FR,fr;q=0.9"

    def test_cookies_empty_on_init(self):
        b = BrowserHTTPSession()
        assert b.cookies == {}

    def test_referrer_none_on_init(self):
        b = BrowserHTTPSession()
        assert b.referrer is None


# ---------------------------------------------------------------------------
# to_browser() factory on HTTPSession
# ---------------------------------------------------------------------------


class TestToBrowser:
    def test_returns_browser_session(self):
        s = HTTPSession(base_url="https://example.com")
        b = s.to_browser()
        assert isinstance(b, BrowserHTTPSession)

    def test_inherits_base_url(self):
        s = HTTPSession(base_url="https://api.example.com")
        b = s.to_browser()
        assert str(b.base_url) == str(s.base_url)

    def test_inherits_verify(self):
        s = HTTPSession(verify=False)
        b = s.to_browser()
        assert b.verify is False

    def test_inherits_send_headers(self):
        s = HTTPSession(send_headers={"X-API-Key": "secret"})
        b = s.to_browser()
        assert b.send_headers == {"X-API-Key": "secret"}

    def test_explicit_user_agent_passed(self):
        s = HTTPSession()
        ua = "Custom/1.0"
        b = s.to_browser(user_agent=ua)
        assert b.user_agent == ua

    def test_send_headers_isolated(self):
        """Mutating the original session's send_headers should not affect the browser."""
        orig = {"X-A": "1"}
        s = HTTPSession(send_headers=orig)
        b = s.to_browser()
        s.send_headers["X-A"] = "mutated"
        assert b.send_headers.get("X-A") == "1"


# ---------------------------------------------------------------------------
# UA introspection
# ---------------------------------------------------------------------------


class TestUAIntrospection:
    def test_chrome_windows(self):
        b = BrowserHTTPSession(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.6367.82 Safari/537.36"
            )
        )
        assert b.browser_name == "Chrome"
        assert b.platform == "Windows"
        assert b.is_mobile is False
        assert b.sec_ch_ua_platform == "Windows"

    def test_edge_windows(self):
        b = BrowserHTTPSession(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.6367.82 Edg/124.0.2478.51 Safari/537.36"
            )
        )
        assert b.browser_name == "Edge"
        assert b.platform == "Windows"

    def test_firefox_linux(self):
        b = BrowserHTTPSession(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) "
                "Gecko/20100101 Firefox/121.0"
            )
        )
        assert b.browser_name == "Firefox"
        assert b.platform == "Linux"
        assert b.sec_ch_ua is None  # Firefox doesn't send sec-ch-ua

    def test_safari_mac(self):
        b = BrowserHTTPSession(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0_0) "
                "AppleWebKit/605.1.15 (HTML, like Gecko) "
                "Version/17.0 Safari/605.1.15"
            )
        )
        assert b.browser_name == "Safari"
        assert b.platform == "macOS"
        assert b.sec_ch_ua is None

    def test_android_mobile(self):
        b = BrowserHTTPSession(
            user_agent=(
                "Mozilla/5.0 (Linux; Android 13; Pixel 7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.6367.82 Mobile Safari/537.36"
            )
        )
        assert b.platform == "Android"
        assert b.is_mobile is True

    def test_sec_ch_ua_chrome(self):
        b = BrowserHTTPSession(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124.0.6367.82 Safari/537.36"
            )
        )
        ch = b.sec_ch_ua
        assert ch is not None
        assert '"Google Chrome";v="124"' in ch
        assert '"Chromium";v="124"' in ch
        assert '"Not-A.Brand"' in ch

    def test_sec_ch_ua_edge(self):
        b = BrowserHTTPSession(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124.0.6367.82 "
                "Edg/124.0.2478.51 Safari/537.36"
            )
        )
        ch = b.sec_ch_ua
        assert ch is not None
        assert '"Microsoft Edge"' in ch


# ---------------------------------------------------------------------------
# Cookie jar
# ---------------------------------------------------------------------------


class TestCookieJar:
    def test_set_and_get(self):
        b = BrowserHTTPSession()
        b.set_cookie("session", "abc123")
        assert b.cookies == {"session": "abc123"}

    def test_delete_cookie(self):
        b = BrowserHTTPSession()
        b.set_cookie("a", "1")
        b.set_cookie("b", "2")
        b.delete_cookie("a")
        assert "a" not in b.cookies
        assert b.cookies["b"] == "2"

    def test_delete_missing_cookie_noop(self):
        b = BrowserHTTPSession()
        b.delete_cookie("nonexistent")  # should not raise

    def test_clear_cookies(self):
        b = BrowserHTTPSession()
        b.set_cookie("x", "1")
        b.clear_cookies()
        assert b.cookies == {}

    def test_serialize_cookies(self):
        b = BrowserHTTPSession()
        b.set_cookie("a", "1")
        b.set_cookie("b", "2")
        serialized = b._serialize_cookies()
        assert "a=1" in serialized
        assert "b=2" in serialized
        assert ";" in serialized

    def test_parse_set_cookie_simple(self):
        name, value = BrowserHTTPSession._parse_set_cookie("session=abc123")
        assert name == "session"
        assert value == "abc123"

    def test_parse_set_cookie_with_attributes(self):
        name, value = BrowserHTTPSession._parse_set_cookie(
            "session=abc123; Path=/; HttpOnly; Secure"
        )
        assert name == "session"
        assert value == "abc123"

    def test_parse_set_cookie_no_value(self):
        name, value = BrowserHTTPSession._parse_set_cookie("deleted")
        assert name == "deleted"
        assert value == ""

    def test_update_cookies_from_response(self):
        from yggdrasil.io.http_.response import HTTPResponse
        from yggdrasil.io.buffer import BytesIO
        import datetime as dt

        req = _make_request()
        resp = HTTPResponse(
            request=req,
            status_code=200,
            headers={"Set-Cookie": "session=xyz123; Path=/; HttpOnly"},
            buffer=BytesIO(),
            tags={},
            received_at=dt.datetime.now(dt.timezone.utc),
        )
        b = BrowserHTTPSession()
        b._update_cookies_from_response(resp)
        assert b.cookies.get("session") == "xyz123"


# ---------------------------------------------------------------------------
# Referrer
# ---------------------------------------------------------------------------


class TestReferrer:
    def test_set_referrer(self):
        b = BrowserHTTPSession()
        b.set_referrer("https://example.com/home")
        assert b.referrer == "https://example.com/home"

    def test_clear_referrer(self):
        b = BrowserHTTPSession()
        b.set_referrer("https://example.com/home")
        b.clear_referrer()
        assert b.referrer is None

    def test_set_referrer_none_string(self):
        b = BrowserHTTPSession()
        b.set_referrer("https://x.com")
        b.set_referrer("")
        assert b.referrer is None


# ---------------------------------------------------------------------------
# Sec-Fetch-Site computation
# ---------------------------------------------------------------------------


class TestSecFetchSite:
    def test_no_referrer_returns_none(self):
        b = BrowserHTTPSession()
        req = _make_request("https://example.com/page")
        assert b._compute_sec_fetch_site(req) == "none"

    def test_same_origin(self):
        b = BrowserHTTPSession()
        b.set_referrer("https://example.com/home")
        req = _make_request("https://example.com/page")
        assert b._compute_sec_fetch_site(req) == "same-origin"

    def test_same_site_subdomain(self):
        b = BrowserHTTPSession()
        b.set_referrer("https://www.example.com/home")
        req = _make_request("https://api.example.com/data")
        assert b._compute_sec_fetch_site(req) == "same-site"

    def test_cross_site(self):
        b = BrowserHTTPSession()
        b.set_referrer("https://other.com/page")
        req = _make_request("https://example.com/page")
        assert b._compute_sec_fetch_site(req) == "cross-site"


# ---------------------------------------------------------------------------
# Browser header construction
# ---------------------------------------------------------------------------


class TestBrowserHeaders:
    def test_contains_user_agent(self):
        b = BrowserHTTPSession(user_agent="TestAgent/1.0")
        req = _make_request()
        headers = b._browser_headers_for_request(req)
        assert headers["User-Agent"] == "TestAgent/1.0"

    def test_contains_accept(self):
        b = BrowserHTTPSession()
        req = _make_request()
        headers = b._browser_headers_for_request(req)
        assert "text/html" in headers["Accept"]

    def test_contains_accept_encoding(self):
        b = BrowserHTTPSession()
        req = _make_request()
        headers = b._browser_headers_for_request(req)
        assert "gzip" in headers["Accept-Encoding"]
        assert "br" in headers["Accept-Encoding"]

    def test_contains_sec_fetch_headers(self):
        b = BrowserHTTPSession()
        req = _make_request()
        headers = b._browser_headers_for_request(req)
        assert "Sec-Fetch-Dest" in headers
        assert "Sec-Fetch-Mode" in headers
        assert "Sec-Fetch-Site" in headers
        assert "Sec-Fetch-User" in headers

    def test_cookie_injected_when_present(self):
        b = BrowserHTTPSession()
        b.set_cookie("tok", "abc")
        req = _make_request()
        headers = b._browser_headers_for_request(req)
        assert "tok=abc" in headers["Cookie"]

    def test_no_cookie_header_when_jar_empty(self):
        b = BrowserHTTPSession()
        req = _make_request()
        headers = b._browser_headers_for_request(req)
        assert "Cookie" not in headers

    def test_referer_injected_when_set(self):
        b = BrowserHTTPSession()
        b.set_referrer("https://example.com/from")
        req = _make_request()
        headers = b._browser_headers_for_request(req)
        assert headers["Referer"] == "https://example.com/from"

    def test_send_headers_override_browser_defaults(self):
        """User-set send_headers must win over browser defaults."""
        b = BrowserHTTPSession(
            user_agent="InitialAgent/1.0",
            send_headers={"User-Agent": "CustomAgent/2.0"},
        )
        req = _make_request()
        merged = b._build_request_headers(req)
        assert merged["User-Agent"] == "CustomAgent/2.0"

    def test_chromium_client_hints_present_for_chrome(self):
        b = BrowserHTTPSession(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124.0.6367.82 Safari/537.36"
            )
        )
        req = _make_request()
        headers = b._browser_headers_for_request(req)
        assert "sec-ch-ua" in headers
        assert "sec-ch-ua-mobile" in headers
        assert "sec-ch-ua-platform" in headers

    def test_no_chromium_hints_for_firefox(self):
        b = BrowserHTTPSession(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) "
                "Gecko/20100101 Firefox/121.0"
            )
        )
        req = _make_request()
        headers = b._browser_headers_for_request(req)
        assert "sec-ch-ua" not in headers


# ---------------------------------------------------------------------------
# User-agent management
# ---------------------------------------------------------------------------


class TestUserAgentManagement:
    def test_set_user_agent(self):
        b = BrowserHTTPSession()
        b.set_user_agent("Custom/1.0")
        assert b.user_agent == "Custom/1.0"

    def test_rotate_user_agent_changes_ua(self):
        b = BrowserHTTPSession(ua_seed=0)
        old = b.user_agent
        b.rotate_user_agent(seed=999)
        assert b.user_agent != old

    def test_rotate_returns_new_ua(self):
        b = BrowserHTTPSession()
        returned = b.rotate_user_agent(seed=42)
        assert returned == b.user_agent

    def test_set_browser_preset_chrome_windows(self):
        b = BrowserHTTPSession()
        ua = b.set_browser_preset("chrome", platform="windows")
        assert "Chrome/" in ua
        assert "Edg/" not in ua
        assert "Windows NT" in ua

    def test_set_browser_preset_firefox_linux(self):
        b = BrowserHTTPSession()
        ua = b.set_browser_preset("firefox", platform="linux")
        assert "Firefox/" in ua
        assert "Linux" in ua

    def test_set_browser_preset_edge_windows(self):
        b = BrowserHTTPSession()
        ua = b.set_browser_preset("edge", platform="windows")
        assert "Edg/" in ua

    def test_set_browser_preset_returns_and_applies(self):
        b = BrowserHTTPSession()
        returned = b.set_browser_preset("chrome", platform="windows", seed=7)
        assert returned == b.user_agent


# ---------------------------------------------------------------------------
# _resolve_url — URL resolution against base_url
# ---------------------------------------------------------------------------


class TestResolveUrl:
    """_resolve_url must join relative URLs against base_url correctly."""

    # ── Absolute URLs ─────────────────────────────────────────────────────

    def test_absolute_https_parsed(self):
        b = BrowserHTTPSession()
        result = b._resolve_url("https://example.com/path")
        assert result.scheme == "https"
        assert result.host == "example.com"
        assert result.path == "/path"

    def test_absolute_http_parsed(self):
        b = BrowserHTTPSession()
        result = b._resolve_url("http://example.com/path?q=1")
        assert result.scheme == "http"
        assert result.query == "q=1"

    def test_protocol_relative_gets_https(self):
        b = BrowserHTTPSession()
        result = b._resolve_url("//example.com/path")
        assert result.scheme == "https"
        assert result.host == "example.com"
        assert result.path == "/path"

    def test_url_object_absolute_returned_as_is(self):
        b = BrowserHTTPSession()
        url = URL.parse("https://example.com/page")
        result = b._resolve_url(url)
        # absolute URL objects pass through unchanged (same object or equal)
        assert result == url

    # ── Schemaless hostnames ───────────────────────────────────────────────

    def test_schemaless_hostname_gets_https(self):
        b = BrowserHTTPSession()
        result = b._resolve_url("example.com/path")
        assert result.scheme == "https"
        assert result.host == "example.com"

    def test_schemaless_subdomain_gets_https(self):
        b = BrowserHTTPSession()
        result = b._resolve_url("api.example.com/v1/resource")
        assert result.scheme == "https"
        assert result.host == "api.example.com"
        assert result.path == "/v1/resource"

    # ── Relative paths — the user-reported bug ────────────────────────────

    def test_relative_no_dot_joined_with_base_url(self):
        """'api/controlador.cgi' must join against base_url, not become host='api'."""
        b = BrowserHTTPSession(base_url="https://api.example.com")
        result = b._resolve_url("api/controlador.cgi")
        assert result.scheme == "https"
        assert result.host == "api.example.com"
        assert result.path == "/api/controlador.cgi"

    def test_relative_single_segment_joined(self):
        b = BrowserHTTPSession(base_url="https://host.example.com/base/")
        result = b._resolve_url("resource")
        assert result.host == "host.example.com"
        assert result.path == "/base/resource"

    def test_absolute_path_joined_keeps_host(self):
        b = BrowserHTTPSession(base_url="https://api.example.com/v1/")
        result = b._resolve_url("/controlador.cgi")
        assert result.host == "api.example.com"
        assert result.path == "/controlador.cgi"

    def test_dotdot_relative_resolves(self):
        b = BrowserHTTPSession(base_url="https://api.example.com/v2/sub/")
        result = b._resolve_url("../other")
        assert result.host == "api.example.com"
        assert result.path == "/v2/other"

    def test_relative_no_base_url_raises(self):
        """Relative path with no base_url must raise, not silently produce a bad host."""
        b = BrowserHTTPSession()
        with pytest.raises(ValueError, match="base_url"):
            b._resolve_url("api/controlador.cgi")

    def test_absolute_path_no_base_url_raises(self):
        b = BrowserHTTPSession()
        with pytest.raises(ValueError, match="base_url"):
            b._resolve_url("/some/path")

    # ── Error cases ───────────────────────────────────────────────────────

    def test_windows_drive_letter_raises(self):
        b = BrowserHTTPSession()
        with pytest.raises(ValueError, match="local Windows path"):
            b._resolve_url("C:/Users/alice/page.html")

    def test_empty_string_raises(self):
        b = BrowserHTTPSession()
        with pytest.raises(ValueError):
            b._resolve_url("")


# ---------------------------------------------------------------------------
# _apply_params — query string merging
# ---------------------------------------------------------------------------


class TestApplyParams:
    def test_scalar_params_added(self):
        url = URL.parse("https://example.com/path")
        result = BrowserHTTPSession._apply_params(url, {"a": "1", "b": "2"})
        assert "a=1" in (result.query or "")
        assert "b=2" in (result.query or "")

    def test_multi_value_params(self):
        url = URL.parse("https://example.com/path")
        result = BrowserHTTPSession._apply_params(url, {"tag": ["x", "y"]})
        assert (result.query or "").count("tag=") == 2

    def test_preserves_existing_query(self):
        url = URL.parse("https://example.com/path?existing=1")
        result = BrowserHTTPSession._apply_params(url, {"new": "2"})
        assert "existing=1" in (result.query or "")
        assert "new=2" in (result.query or "")

    def test_empty_params_noop(self):
        url = URL.parse("https://example.com/path?q=1")
        result = BrowserHTTPSession._apply_params(url, {})
        assert result.query == "q=1"

    def test_numeric_value_coerced(self):
        url = URL.parse("https://example.com/")
        result = BrowserHTTPSession._apply_params(url, {"page": 3, "size": 20})
        assert "page=3" in (result.query or "")
        assert "size=20" in (result.query or "")


# ---------------------------------------------------------------------------
# get() / post() params integration
# ---------------------------------------------------------------------------


class TestGetPostWithParams:
    def test_get_params_appear_in_prepared_request(self):
        """params= must be merged into the URL's query string."""
        b = BrowserHTTPSession(base_url="https://api.example.com")
        # Patch send to capture the request without making a real connection
        captured = {}

        def fake_send(request, config=None, **kw):
            captured["url"] = request.url
            raise RuntimeError("stop")  # abort early

        b._local_send = fake_send  # type: ignore[method-assign]

        with pytest.raises(RuntimeError):
            b.get("api/controlador.cgi", params={"ac": "signin", "page": "1"})

        url = captured["url"]
        assert url.host == "api.example.com"
        assert url.path == "/api/controlador.cgi"
        assert "ac=signin" in (url.query or "")
        assert "page=1" in (url.query or "")

    def test_post_params_in_query_not_body(self):
        """params= on POST goes into the URL query, not the request body."""
        b = BrowserHTTPSession(base_url="https://api.example.com")
        captured = {}

        def fake_send(request, config=None, **kw):
            captured["url"] = request.url
            captured["buffer"] = request.buffer
            raise RuntimeError("stop")

        b._local_send = fake_send  # type: ignore[method-assign]

        with pytest.raises(RuntimeError):
            b.post("api/submit", params={"token": "abc"}, json={"value": 42})

        assert "token=abc" in (captured["url"].query or "")


# ---------------------------------------------------------------------------
# Network tests (skipped unless REAL_HTTP=1)
# ---------------------------------------------------------------------------

import os

REAL_HTTP = os.getenv("REAL_HTTP", "1") == "1"

network_mark = pytest.mark.skipif(
    not REAL_HTTP,
    reason="Real network tests disabled (set REAL_HTTP=1 to enable).",
)


@network_mark
class TestBrowserHTTPSessionNetwork:
    def test_navigate_httpbin(self):
        if not _tcp_can_connect("httpbin.org", 443):
            pytest.skip("Cannot reach httpbin.org:443")
        b = BrowserHTTPSession()
        resp = b.navigate("https://httpbin.org/get")
        assert resp.status_code == 200

    def test_browser_headers_received_by_server(self):
        if not _tcp_can_connect("httpbin.org", 443):
            pytest.skip("Cannot reach httpbin.org:443")
        b = BrowserHTTPSession(
            user_agent="Mozilla/5.0 TestBrowser/1.0"
        )
        resp = b.navigate("https://httpbin.org/get")
        data = resp.json()
        ua_echo = data.get("headers", {}).get("User-Agent", "")
        assert "TestBrowser" in ua_echo

    def test_cookie_jar_persists_across_requests(self):
        if not _tcp_can_connect("httpbin.org", 443):
            pytest.skip("Cannot reach httpbin.org:443")
        b = BrowserHTTPSession()
        b.set_cookie("mycookie", "hello")
        resp = b.navigate("https://httpbin.org/cookies")
        data = resp.json()
        assert data.get("cookies", {}).get("mycookie") == "hello"

    def test_follow_link_updates_referrer(self):
        if not _tcp_can_connect("httpbin.org", 443):
            pytest.skip("Cannot reach httpbin.org:443")
        b = BrowserHTTPSession()
        b.navigate("https://httpbin.org/get")
        assert b.referrer is not None

    def test_submit_form(self):
        if not _tcp_can_connect("httpbin.org", 443):
            pytest.skip("Cannot reach httpbin.org:443")
        b = BrowserHTTPSession()
        resp = b.submit_form(
            "https://httpbin.org/post",
            {"username": "alice", "password": "s3cr3t"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["form"]["username"] == "alice"

