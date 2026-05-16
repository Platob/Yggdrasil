"""Behaviors for :class:`Singleton`.

Covers the three constructor modes (``singleton_ttl`` omitted, ``False``,
seconds / ``None``) and the :meth:`to_singleton` promotion path used by
hot listing call sites that opt out of caching by default and later
decide one of their children is worth keeping around.
"""
from __future__ import annotations

import unittest
from typing import Any, ClassVar
from threading import RLock

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.dataclasses.singleton import Singleton


class _CachingPath(Singleton):
    """Singleton with a per-class cache and a 5-minute default TTL —
    same shape as the real ``DatabricksPath`` / ``S3Path``."""

    _SINGLETON_TTL: ClassVar[Any] = 300.0
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(
        default_ttl=300.0, max_size=64,
    )
    _INSTANCES_LOCK: ClassVar[RLock] = RLock()

    def __init__(self, key: str, *, singleton_ttl: Any = ...) -> None:
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return
        self.key = key
        self._initialized = True

    @classmethod
    def _singleton_key(cls, key: str, **kwargs: Any) -> Any:
        return (cls, key)


class TestSingletonTTLModes(unittest.TestCase):
    def setUp(self) -> None:
        _CachingPath._INSTANCES.clear()

    def test_default_uses_class_ttl(self) -> None:
        """Omitted ``singleton_ttl`` → cached under the class default."""
        first = _CachingPath("a")
        second = _CachingPath("a")
        assert first is second, "default constructor must collapse to cache"

    def test_false_skips_cache(self) -> None:
        """``singleton_ttl=False`` builds a fresh uncached instance."""
        cached = _CachingPath("k")
        uncached = _CachingPath("k", singleton_ttl=False)

        assert uncached is not cached, "False must not return the cached one"
        # And the cache wasn't disturbed — the original entry is still there.
        assert _CachingPath("k") is cached

    def test_false_stamps_singleton_key(self) -> None:
        """An uncached instance still carries its key so hash/eq work."""
        a = _CachingPath("k", singleton_ttl=False)
        b = _CachingPath("k", singleton_ttl=False)

        # Two distinct objects but equal-by-identity-key.
        assert a is not b
        assert hash(a) == hash(b)
        assert a == b

    def test_none_caches_forever(self) -> None:
        """``singleton_ttl=None`` registers with no expiry."""
        first = _CachingPath("forever", singleton_ttl=None)
        second = _CachingPath("forever")
        assert first is second


class TestToSingleton(unittest.TestCase):
    def setUp(self) -> None:
        _CachingPath._INSTANCES.clear()

    def test_promotes_uncached_into_cache(self) -> None:
        """``to_singleton`` registers a previously-uncached instance."""
        uncached = _CachingPath("p", singleton_ttl=False)
        assert _CachingPath._INSTANCES.get((_CachingPath, "p")) is None

        promoted = uncached.to_singleton()

        assert promoted is uncached
        # Next constructor call collapses to the same instance.
        assert _CachingPath("p") is uncached

    def test_returns_existing_when_collision(self) -> None:
        """Another instance is already cached → that one wins."""
        cached = _CachingPath("collide")
        new = _CachingPath("collide", singleton_ttl=False)
        assert new is not cached

        winner = new.to_singleton()

        assert winner is cached
        # The pre-existing cache entry is untouched.
        assert _CachingPath("collide") is cached

    def test_explicit_ttl_overrides_class_default(self) -> None:
        """Caller-supplied ``ttl`` is what lands in the cache."""
        inst = _CachingPath("ttl", singleton_ttl=False)
        inst.to_singleton(ttl=None)

        # ``None`` = no expiry. Pull the entry back out by key.
        stored = _CachingPath._INSTANCES.get((_CachingPath, "ttl"))
        assert stored is inst

    def test_skip_ttl_no_op(self) -> None:
        """``to_singleton(ttl=False)`` leaves the cache untouched."""
        inst = _CachingPath("noop", singleton_ttl=False)
        result = inst.to_singleton(ttl=False)

        assert result is inst
        assert _CachingPath._INSTANCES.get((_CachingPath, "noop")) is None
