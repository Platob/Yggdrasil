"""Pickle-protocol tests for :class:`Session` / :class:`HTTPSession`.

Exercises the generic ``__getstate__`` / ``__setstate__`` contract:

* every non-transient ``__dict__`` entry survives a pickle round-trip,
* ``_lock`` / ``_job_pool`` (and ``_http_pool`` on HTTPSession) are
  rebuilt on the unpickle side,
* unpickling routes through ``__new__`` so a session reconstructed
  with the same ``base_url`` collapses to the live in-process
  singleton instead of cloning its connection pool / cookies,
* a freshly unpickled non-singleton session is a brand-new instance.
"""
from __future__ import annotations

import pickle
import threading

import pytest

from yggdrasil.io.http_ import HTTPSession
from yggdrasil.io.http_.cookies import Cookies
from yggdrasil.io.session import Session
from yggdrasil.io.url import URL

from ._helpers import StubSession


# ---------------------------------------------------------------------------
# Singleton-cache hygiene — clear cross-test bleed before each test.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_session_singleton_cache():
    Session._singleton_cache.clear()
    yield
    Session._singleton_cache.clear()


# ---------------------------------------------------------------------------
# Base Session — generic __getstate__ / __setstate__
# ---------------------------------------------------------------------------


class TestSessionGetState:

    def test_excludes_transient_attrs(self) -> None:
        s = StubSession(base_url="https://api.example.com")
        state = s.__getstate__()
        for transient in Session._TRANSIENT_STATE_ATTRS:
            assert transient not in state, (
                f"transient attr {transient!r} leaked into pickle state"
            )

    def test_includes_dict_attrs(self) -> None:
        s = StubSession(base_url="https://api.example.com")
        # StubSession.__init__ sets _queue and calls in __dict__ — the generic
        # getstate should pick them up without StubSession knowing about it.
        s.calls.append("marker")  # type: ignore[arg-type]
        state = s.__getstate__()
        assert "calls" in state
        assert state["calls"] == ["marker"]
        assert "_queue" in state
        assert state["base_url"] == s.base_url
        assert state["pool_maxsize"] == s.pool_maxsize


class TestSessionSetState:

    def test_rebuilds_lock_and_job_pool(self) -> None:
        s = StubSession()  # no base_url -> non-singleton path
        state = s.__getstate__()
        clone = pickle.loads(pickle.dumps(s))
        assert clone is not s
        # Lock is functional (RLock is duck-typed; check acquire/release work)
        assert clone._lock.acquire(blocking=False)
        clone._lock.release()
        # Job pool resets to None — a transient attr, lazily rebuilt on access
        assert clone._job_pool is None
        # State actually restored
        assert clone.base_url == s.base_url
        # And the transient set really is excluded from the pickle payload
        assert "_lock" not in state
        assert "_job_pool" not in state

    def test_restores_subclass_attrs(self) -> None:
        s = StubSession()
        s.queue()  # touch the subclass init path
        s._queue.append("sentinel")  # type: ignore[arg-type]
        clone = pickle.loads(pickle.dumps(s))
        assert clone._queue == ["sentinel"]
        assert clone.calls == []


# ---------------------------------------------------------------------------
# Singleton collapse on unpickle
# ---------------------------------------------------------------------------


class TestSessionSingletonRoundTrip:

    def test_unpickle_returns_live_singleton(self) -> None:
        s1 = StubSession(base_url="https://api.example.com")
        blob = pickle.dumps(s1)
        s2 = pickle.loads(blob)
        assert s2 is s1, "unpickle must collapse to the live singleton"

    def test_setstate_skips_when_singleton_already_initialized(self) -> None:
        # Pickle a snapshot, mutate the live singleton, then unpickle.
        # The old snapshot must NOT clobber the in-flight state.
        s1 = StubSession(base_url="https://api.example.com")
        blob = pickle.dumps(s1)
        s1.calls.append("mutated")  # type: ignore[arg-type]
        s2 = pickle.loads(blob)
        assert s2 is s1
        assert s2.calls == ["mutated"], (
            "live singleton state was overwritten by stale pickle payload"
        )

    def test_no_base_url_does_not_collapse(self) -> None:
        s1 = StubSession()
        s2 = pickle.loads(pickle.dumps(s1))
        assert s2 is not s1, (
            "sessions without a base_url should never share identity on unpickle"
        )

    def test_getnewargs_carries_base_url(self) -> None:
        s = StubSession(base_url="https://api.example.com")
        args = s.__getnewargs__()
        assert args == (s.base_url,)
        assert isinstance(args[0], URL)


# ---------------------------------------------------------------------------
# HTTPSession — additional non-picklable handle (_http_pool) + bonus fields
# ---------------------------------------------------------------------------


class TestHTTPSessionPickle:

    def test_transient_set_extends_base(self) -> None:
        assert "_http_pool" in HTTPSession._TRANSIENT_STATE_ATTRS
        # Base transients are still in the set
        assert Session._TRANSIENT_STATE_ATTRS.issubset(
            HTTPSession._TRANSIENT_STATE_ATTRS
        )

    def test_http_pool_rebuilt_on_fresh_instance(self) -> None:
        s = HTTPSession(user_agent="ua")  # no base_url -> fresh instance path
        clone = pickle.loads(pickle.dumps(s))
        assert clone is not s
        assert clone._http_pool is not None
        assert clone._http_pool is not s._http_pool, (
            "fresh unpickled session should get its own pool"
        )

    def test_http_pool_excluded_from_state(self) -> None:
        s = HTTPSession(base_url="https://example.com")
        state = s.__getstate__()
        assert "_http_pool" not in state

    def test_browser_mode_attrs_survive(self) -> None:
        # These were silently dropped by the old explicit-allowlist
        # __getstate__ — the generic version preserves them.
        s = HTTPSession(
            user_agent="MyAgent/1.0",
            accept="text/plain",
            accept_language="fr-FR",
            accept_encoding="identity",
            ua_seed=42,
            cookies={"sid": "abc"},
        )
        clone = pickle.loads(pickle.dumps(s))
        # Different identity (no base_url -> not a singleton)
        assert clone is not s
        assert clone.user_agent == "MyAgent/1.0"
        assert clone.accept == "text/plain"
        assert clone.accept_language == "fr-FR"
        assert clone.accept_encoding == "identity"
        assert clone.ua_seed == 42
        assert isinstance(clone._cookies, Cookies)
        assert clone._cookies.get("sid") == "abc"

    def test_singleton_preserves_live_pool(self) -> None:
        s1 = HTTPSession(base_url="https://example.com")
        live_pool = s1._http_pool
        blob = pickle.dumps(s1)
        s2 = pickle.loads(blob)
        assert s2 is s1
        assert s2._http_pool is live_pool, (
            "unpickle must not replace the live singleton's connection pool"
        )
