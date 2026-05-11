"""Tests for the Singleton mixin: hash-based instance dedup."""
from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass

import pytest

from yggdrasil.dataclasses import Singleton


@dataclass(frozen=True)
class Endpoint(Singleton):
    host: str
    port: int = 443


@dataclass(frozen=True)
class OtherEndpoint(Singleton):
    host: str
    port: int = 443


@dataclass(unsafe_hash=True)
class MutableEndpoint(Singleton):
    host: str
    port: int = 443


def test_same_hash_returns_same_instance():
    a = Endpoint("api", 443)
    b = Endpoint("api", 443)
    assert a is b


def test_different_hash_returns_different_instance():
    a = Endpoint("api", 443)
    b = Endpoint("api", 80)
    assert a is not b
    assert a.port != b.port


def test_different_subclasses_do_not_collide():
    a = Endpoint("api", 443)
    b = OtherEndpoint("api", 443)
    assert a is not b
    assert type(a) is Endpoint
    assert type(b) is OtherEndpoint


def test_init_runs_once_on_the_cached_instance():
    """``__init__`` runs on every draft (Singleton needs a populated instance
    to hash), but the *kept* instance — the one every caller sees — only ever
    sees its ``__init__`` body execute once."""

    @dataclass(frozen=True)
    class Tagged(Singleton):
        name: str

        def __post_init__(self) -> None:
            # Tag the instance once, fail loudly if the wrapper lets Python
            # re-run __init__ on the cached instance.
            assert not hasattr(self, "_tagged"), "__init__ ran twice on cached instance"
            object.__setattr__(self, "_tagged", True)

    a = Tagged("x")
    b = Tagged("x")
    c = Tagged("x")
    assert a is b is c
    assert a._tagged is True


def test_unhashable_subclass_raises():
    @dataclass
    class Bad(Singleton):
        host: str

    with pytest.raises(TypeError, match="unhashable"):
        Bad("api")


def test_unsafe_hash_dataclass_works():
    a = MutableEndpoint("api", 443)
    b = MutableEndpoint("api", 443)
    assert a is b


def test_cache_is_global_across_threads():
    results: list[Endpoint] = []
    lock = threading.Lock()

    def build() -> None:
        ep = Endpoint("shared", 8080)
        with lock:
            results.append(ep)

    threads = [threading.Thread(target=build) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 16
    first = results[0]
    assert all(r is first for r in results)


def test_pickle_roundtrip_collapses_to_singleton():
    a = Endpoint("api", 443)
    blob = pickle.dumps(a)
    b = pickle.loads(blob)
    assert a is b


def test_repr_and_equality():
    a = Endpoint("api", 443)
    b = Endpoint("api", 443)
    assert a == b
    assert hash(a) == hash(b)


def test_cache_entry_present_after_construction():
    ep = Endpoint("registered", 9000)
    key = (Endpoint, hash(ep))
    assert Singleton._INSTANCES.get(key) is ep
