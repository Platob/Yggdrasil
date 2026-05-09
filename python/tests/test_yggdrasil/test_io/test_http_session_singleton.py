"""Tests for the Session singleton cache and merged browser features."""
from __future__ import annotations

import pytest

from yggdrasil.io.http_ import Cookies, HTTPSession
from yggdrasil.io.url import URL

from ._helpers import StubSession, make_response


# ---------------------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_singleton_cache():
    """Wipe the per-class cache so tests don't leak instances into each other.

    The cache is a process-wide ``dict`` on :class:`Session`; without this
    fixture an HTTPSession built earlier in the suite would alias one built
    here under the same ``base_url`` and assertions on default state would
    pick up the leftover.
    """
    from yggdrasil.io.session import Session

    saved = dict(Session._singleton_cache)
    Session._singleton_cache.clear()
    yield
    Session._singleton_cache.clear()
    Session._singleton_cache.update(saved)


class TestSingletonCache:

    def test_same_base_url_returns_same_instance(self) -> None:
        a = HTTPSession(base_url="https://example.com")
        b = HTTPSession(base_url="https://example.com")
        assert a is b

    def test_url_object_and_string_share_singleton(self) -> None:
        a = HTTPSession(base_url="https://example.com")
        b = HTTPSession(base_url=URL.from_("https://example.com"))
        assert a is b

    def test_different_base_url_returns_distinct_instances(self) -> None:
        a = HTTPSession(base_url="https://example.com")
        b = HTTPSession(base_url="https://other.com")
        assert a is not b

    def test_no_base_url_always_returns_fresh_instance(self) -> None:
        a = HTTPSession()
        b = HTTPSession()
        assert a is not b

    def test_subclasses_have_isolated_cache_keys(self) -> None:
        s = StubSession(base_url="https://example.com")
        h = HTTPSession(base_url="https://example.com")
        assert s is not h
        assert isinstance(s, StubSession)
        assert isinstance(h, HTTPSession)

    def test_init_is_idempotent_on_cached_instance(self) -> None:
        a = HTTPSession(base_url="https://example.com")
        a.cookies["sid"] = "abc"
        a.user_agent = "MyBot/1.0"

        # Constructing again with the same URL must NOT wipe live state.
        b = HTTPSession(base_url="https://example.com")
        assert b is a
        assert b.cookies["sid"] == "abc"
        assert b.user_agent == "MyBot/1.0"


# ---------------------------------------------------------------------------
# Lazy browser-mode state
# ---------------------------------------------------------------------------


class TestLazyBrowserState:

    def test_cookies_jar_is_lazy(self) -> None:
        s = HTTPSession(base_url="https://example.com")
        assert s._cookies is None

        jar = s.cookies
        assert isinstance(jar, Cookies)
        assert s._cookies is jar
        # second access returns the same jar (no re-build).
        assert s.cookies is jar

    def test_ua_generator_is_lazy(self) -> None:
        s = HTTPSession(base_url="https://example.com")
        assert s._ua_generator is None
        assert s.user_agent is None

        ua = s.get_user_agent()
        assert isinstance(ua, str) and ua
        assert s._ua_generator is not None
        assert s.user_agent == ua

    def test_explicit_user_agent_skips_generator(self) -> None:
        s = HTTPSession(base_url="https://example.com", user_agent="MyBot/1.0")
        assert s.get_user_agent() == "MyBot/1.0"
        assert s._ua_generator is None

    def test_initial_cookies_seed_jar(self) -> None:
        s = HTTPSession(
            base_url="https://example.com",
            cookies={"sid": "abc", "lang": "en"},
        )
        assert s._cookies is not None
        assert s.cookies["sid"] == "abc"
        assert s.cookies["lang"] == "en"


# ---------------------------------------------------------------------------
# Browser headers
# ---------------------------------------------------------------------------


class TestBrowserHeaders:

    def test_chrome_ua_emits_sec_ch_ua(self) -> None:
        s = HTTPSession(
            base_url="https://example.com",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        headers = s._build_browser_headers(URL.from_("https://example.com/x"))
        assert headers["sec-ch-ua-platform"] == '"Windows"'
        assert "Chromium" in headers["sec-ch-ua"]
        assert headers["sec-ch-ua-mobile"] == "?0"
        assert s.browser_name == "Chrome"
        assert s.platform == "Windows"

    def test_firefox_ua_omits_sec_ch_ua(self) -> None:
        s = HTTPSession(
            base_url="https://example.com",
            user_agent="Mozilla/5.0 (X11; Linux x86_64) Firefox/124.0",
        )
        headers = s._build_browser_headers(URL.from_("https://example.com/x"))
        assert "sec-ch-ua" not in headers
        assert s.browser_name == "Firefox"
        assert s.platform == "Linux"

    def test_referer_header_added_when_set(self) -> None:
        s = HTTPSession(base_url="https://example.com")
        assert s._compute_sec_fetch_site(URL.from_("https://example.com/a")) == "none"
        s.set_referrer("https://example.com/from")
        assert s.referrer == "https://example.com/from"
        headers = s._build_browser_headers(URL.from_("https://example.com/x"))
        assert headers["Referer"] == "https://example.com/from"
        assert headers["Sec-Fetch-Site"] == "same-origin"

    def test_sec_fetch_site_classification(self) -> None:
        s = HTTPSession(base_url="https://example.com")
        s.set_referrer("https://api.example.com/feed")
        assert s._compute_sec_fetch_site(URL.from_("https://www.example.com/x")) == "same-site"
        assert s._compute_sec_fetch_site(URL.from_("https://api.example.com/y")) == "same-origin"
        assert s._compute_sec_fetch_site(URL.from_("https://other.com/y")) == "cross-site"

    def test_extra_headers_override_browser_defaults(self) -> None:
        s = HTTPSession(
            base_url="https://example.com",
            user_agent="Mozilla/5.0",
            send_headers={"Accept": "session/value"},
        )
        merged = s._build_browser_headers(
            URL.from_("https://example.com/x"),
            extra={"Accept": "request/value"},
        )
        # Layering: browser default < send_headers < per-request extra.
        assert merged["Accept"] == "request/value"

    def test_cookies_serialize_into_cookie_header(self) -> None:
        s = HTTPSession(base_url="https://example.com", user_agent="UA/1")
        s.cookies["sid"] = "abc"
        s.cookies["lang"] = "en"
        merged = s._build_browser_headers(URL.from_("https://example.com/x"))
        assert merged["Cookie"] == "sid=abc; lang=en"


# ---------------------------------------------------------------------------
# UA management
# ---------------------------------------------------------------------------


class TestUAManagement:

    def test_rotate_user_agent_replaces_string(self) -> None:
        s = HTTPSession(base_url="https://example.com")
        first = s.rotate_user_agent(seed=1)
        second = s.rotate_user_agent(seed=2)
        assert first
        assert second
        assert s.user_agent == second

    def test_set_browser_preset_chrome_windows(self) -> None:
        s = HTTPSession(base_url="https://example.com")
        ua = s.set_browser_preset("chrome", platform="windows", seed=42)
        assert "Chrome/" in ua
        assert s.user_agent == ua


# ---------------------------------------------------------------------------
# Cookies class manager
# ---------------------------------------------------------------------------


class TestCookies:

    def test_mapping_protocol(self) -> None:
        c = Cookies()
        c["a"] = "1"
        c["b"] = "2"
        assert c["a"] == "1"
        assert "a" in c
        assert "z" not in c
        assert len(c) == 2
        assert sorted(c) == ["a", "b"]

    def test_set_get_delete_clear(self) -> None:
        c = Cookies()
        c.set("a", "1")
        assert c.get("a") == "1"
        assert c.get("missing") is None
        assert c.get("missing", "fallback") == "fallback"
        c.delete("a")
        assert c.get("a") is None
        c.delete("a")  # idempotent
        c.set("b", "2")
        c.clear()
        assert len(c) == 0

    def test_to_header_serializes_pairs(self) -> None:
        c = Cookies({"sid": "abc", "lang": "en"})
        assert c.to_header() == "sid=abc; lang=en"

    def test_to_header_empty_jar_returns_empty_string(self) -> None:
        assert Cookies().to_header() == ""

    def test_parse_set_cookie_drops_attributes(self) -> None:
        name, value = Cookies.parse_set_cookie("sid=abc; Path=/; HttpOnly")
        assert name == "sid"
        assert value == "abc"

    def test_update_from_set_cookie_handles_multi_value(self) -> None:
        c = Cookies()
        c.update_from_set_cookie("a=1; Path=/, b=2; HttpOnly")
        assert c.get("a") == "1"
        assert c.get("b") == "2"

    def test_update_from_set_cookie_ignores_empty(self) -> None:
        c = Cookies()
        c.update_from_set_cookie("")
        assert len(c) == 0

    def test_update_from_response_pulls_set_cookie_header(self) -> None:
        from ._helpers import make_request

        request = make_request()
        response = make_response(
            request=request,
            headers={"Set-Cookie": "sid=abc; Path=/"},
        )
        c = Cookies()
        c.update_from_response(response)
        assert c["sid"] == "abc"

    def test_bool_reflects_emptiness(self) -> None:
        c = Cookies()
        assert not c
        c["x"] = "1"
        assert c

    def test_as_dict_returns_independent_copy(self) -> None:
        c = Cookies({"a": "1"})
        d = c.as_dict()
        d["a"] = "mutated"
        assert c["a"] == "1"
