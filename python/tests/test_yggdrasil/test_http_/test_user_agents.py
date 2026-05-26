"""Tests for yggdrasil.io.user_agents."""

from __future__ import annotations

from yggdrasil.http_.user_agents import UserAgentGenerator, random_user_agent


class TestUserAgentGenerator:
    def test_seeded_is_deterministic(self):
        first = UserAgentGenerator(seed=42).random()
        second = UserAgentGenerator(seed=42).random()
        assert first == second

    def test_different_seeds_can_differ(self):
        # Not strictly guaranteed across all seeds, but a handful of
        # picks should produce at least one distinct UA.
        seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        uas = {UserAgentGenerator(seed=s).random() for s in seeds}
        assert len(uas) > 1

    def test_unseeded_returns_string(self):
        ua = UserAgentGenerator().random()
        assert isinstance(ua, str)
        assert ua

    def test_user_agent_starts_with_mozilla(self):
        ua = UserAgentGenerator(seed=0).random()
        # All branches of the generator emit Mozilla/5.0 as the prefix.
        assert ua.startswith("Mozilla/5.0")


class TestRandomUserAgentHelper:
    def test_unseeded_returns_string(self):
        assert isinstance(random_user_agent(), str)

    def test_seeded_matches_class(self):
        assert random_user_agent(seed=7) == UserAgentGenerator(seed=7).random()
