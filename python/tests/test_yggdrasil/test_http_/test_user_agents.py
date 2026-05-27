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
