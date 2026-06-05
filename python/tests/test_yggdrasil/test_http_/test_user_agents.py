"""Tests for yggdrasil.http_.user_agents."""
from __future__ import annotations

from yggdrasil.http_.user_agents import UserAgentGenerator, random_user_agent


class TestUserAgentGenerator:

    def test_random_returns_string(self):
        ua = UserAgentGenerator().random()
        assert isinstance(ua, str)
        assert len(ua) > 20

    def test_deterministic_with_seed(self):
        a = UserAgentGenerator(seed=42).random()
        b = UserAgentGenerator(seed=42).random()
        assert a == b

    def test_different_seeds_differ(self):
        a = UserAgentGenerator(seed=1).random()
        b = UserAgentGenerator(seed=2).random()
        assert a != b

    def test_contains_mozilla(self):
        ua = UserAgentGenerator(seed=0).random()
        assert "Mozilla/5.0" in ua


class TestRandomUserAgent:

    def test_convenience_function(self):
        ua = random_user_agent()
        assert isinstance(ua, str)
        assert "Mozilla" in ua

    def test_seeded(self):
        assert random_user_agent(seed=99) == random_user_agent(seed=99)


from yggdrasil.http_.user_agents import (
    BrowserProfile,
    random_browser_headers,
    random_browser_profile,
)


class TestBrowserProfile:

    def test_random_returns_unchanged_for_existing_callers(self):
        # The refactor must not move the seeded UA bytes.
        assert UserAgentGenerator(seed=42).random() == UserAgentGenerator(seed=42).random()
        assert "Mozilla/5.0" in UserAgentGenerator(seed=7).random()

    def test_profile_is_a_coherent_browser(self):
        prof = random_browser_profile(seed=3)
        assert isinstance(prof, BrowserProfile)
        h = prof.headers
        # The UA in the headers matches the profile's UA, and the always-on
        # browser headers are present.
        assert h["User-Agent"] == prof.user_agent
        for key in ("Accept", "Accept-Language", "Accept-Encoding",
                    "Sec-Fetch-Mode", "Upgrade-Insecure-Requests"):
            assert key in h

    def test_client_hints_match_the_engine(self):
        # A Chromium UA carries sec-ch-ua client hints with the matching major;
        # Firefox / Safari carry none (just like the real browsers).
        for seed in range(60):
            prof = random_browser_profile(seed=seed)
            ua, h = prof.user_agent, prof.headers
            is_chromium = ("Chrome/" in ua and "Firefox/" not in ua)
            if is_chromium:
                assert "sec-ch-ua" in h
                assert h["sec-ch-ua-mobile"] in ("?0", "?1")
                major = ua.split("Chrome/")[1].split(".")[0]
                assert f'v="{major}"' in h["sec-ch-ua"]
                # mobile flag agrees with the UA shape
                assert h["sec-ch-ua-mobile"] == ("?1" if "Mobile" in ua else "?0")
            else:
                assert "sec-ch-ua" not in h

    def test_deterministic_with_seed(self):
        assert random_browser_headers(seed=11) == random_browser_headers(seed=11)

    def test_identity_excludes_content_negotiation(self):
        ident = random_browser_profile(seed=5).identity
        assert "User-Agent" in ident
        assert "Accept" not in ident          # content negotiation preserved on a retry
        assert "Accept-Encoding" not in ident
