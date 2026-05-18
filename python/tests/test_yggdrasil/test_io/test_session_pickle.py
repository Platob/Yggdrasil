"""Pickle-protocol tests for :class:`Session` / :class:`HTTPSession`.

Exercises the generic ``__getstate__`` / ``__setstate__`` contract:

* every non-transient ``__dict__`` entry survives a pickle round-trip,
* ``_lock`` / ``_job_pool`` (and ``_http_pool`` on HTTPSession) are
  rebuilt on the unpickle side,
* unpickling routes through ``__new__`` so a session reconstructed
  with the same constructor arguments collapses to the live in-process
  singleton instead of cloning its connection pool / cookies — every
  ``__init__`` argument participates in the singleton key, including
  ``base_url=None`` callers that share the same defaults.
"""
from __future__ import annotations

import pickle
import threading

import pytest

from yggdrasil.io.http_ import HTTPSession
from yggdrasil.io.session import Session
from yggdrasil.io.url import URL

from ._helpers import StubSession


# ---------------------------------------------------------------------------
# Singleton-cache hygiene — clear cross-test bleed before each test.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_session_singleton_cache():
    Session._INSTANCES.clear()
    yield
    Session._INSTANCES.clear()


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
        # Build, snapshot, clear the cache so the unpickle path actually
        # runs ``__setstate__`` against a fresh instance instead of
        # short-circuiting onto the live singleton.
        s = StubSession(base_url="https://api.example.com/rebuild")
        state = s.__getstate__()
        blob = pickle.dumps(s)
        Session._INSTANCES.clear()
        clone = pickle.loads(blob)
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
        s = StubSession(base_url="https://api.example.com/restore")
        s.queue()  # touch the subclass init path
        s._queue.append("sentinel")  # type: ignore[arg-type]
        blob = pickle.dumps(s)
        Session._INSTANCES.clear()
        clone = pickle.loads(blob)
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

    def test_anonymous_sessions_collapse_when_args_match(self) -> None:
        # The key is derived from every ``__init__`` arg's stored value,
        # so two ``StubSession()`` calls with identical defaults map
        # onto the same instance — no more ``base_url=None`` bypass.
        s1 = StubSession()
        s2 = StubSession()
        assert s1 is s2
        clone = pickle.loads(pickle.dumps(s1))
        assert clone is s1

    def test_getnewargs_ex_carries_constructor_kwargs(self) -> None:
        s = StubSession(base_url="https://api.example.com")
        args, kwargs = s.__getnewargs_ex__()
        assert args == (s.base_url,)
        assert isinstance(args[0], URL)
        # Every ``Session.__init__`` parameter (except the positional
        # ``base_url``) round-trips through kwargs.
        for name in ("verify", "pool_maxsize", "headers", "waiting", "auth"):
            assert name in kwargs, f"missing constructor kwarg {name!r}"
        # Subclass-private attributes that aren't ``__init__`` params
        # do not leak.
        assert "_queue" not in kwargs
        assert "calls" not in kwargs

    def test_distinct_args_split_singletons(self) -> None:
        a = StubSession(base_url="https://api.example.com", verify=True)
        b = StubSession(base_url="https://api.example.com", verify=False)
        same = StubSession(base_url="https://api.example.com", verify=True)
        assert a is not b, "different args must yield different singletons"
        assert a is same, "matching args must collapse to one singleton"

    def test_equivalent_url_spellings_collapse(self) -> None:
        # ``__init__`` routes both inputs through ``URL.from_`` before
        # stashing them on ``self.base_url``; the singleton key reads
        # ``probe.base_url`` so the two callers land on the same key.
        s1 = StubSession(base_url="https://api.example.com")
        s2 = StubSession(base_url=URL.from_("https://api.example.com"))
        assert s1 is s2, "URL.from_-equivalent inputs must collapse"

    def test_unpickle_preserves_singleton_identity(self) -> None:
        s = StubSession(base_url="https://api.example.com/preserve")
        clone = pickle.loads(pickle.dumps(s))
        assert clone is s


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
        s = HTTPSession(base_url="https://example.com/rebuild", headers={"X-Tag": "ua"})
        # Touch the lazy pool so we have a live handle on the original.
        original_pool = s.http_pool
        blob = pickle.dumps(s)
        Session._INSTANCES.clear()
        clone = pickle.loads(blob)
        assert clone is not s
        # Touching the property on the clone forces a fresh build —
        # the transient attr is None on the unpickle side.
        assert clone.http_pool is not None
        assert clone.http_pool is not original_pool, (
            "fresh unpickled session should get its own pool"
        )

    def test_http_pool_excluded_from_state(self) -> None:
        s = HTTPSession(base_url="https://example.com")
        # Build the pool so it's a real object, not None.
        _ = s.http_pool
        state = s.__getstate__()
        assert "_http_pool" not in state

    def test_headers_survive_pickle(self) -> None:
        s = HTTPSession(
            base_url="https://example.com/hdrs",
            headers={"X-Tag": "v1", "Authorization": "Bearer x"},
        )
        blob = pickle.dumps(s)
        Session._INSTANCES.clear()
        clone = pickle.loads(blob)
        assert clone is not s
        assert clone.headers == {"X-Tag": "v1", "Authorization": "Bearer x"}

    def test_singleton_preserves_live_pool(self) -> None:
        s1 = HTTPSession(base_url="https://example.com")
        live_pool = s1.http_pool  # force lazy build
        blob = pickle.dumps(s1)
        s2 = pickle.loads(blob)
        assert s2 is s1
        assert s2._http_pool is live_pool, (
            "unpickle must not replace the live singleton's connection pool"
        )


# ---------------------------------------------------------------------------
# Subclass inheritance — third-party clients adding their own constructor
# knobs ride the same singleton + pickle plumbing without having to
# re-implement ``_singleton_key`` or ``__getnewargs_ex__``.
# ---------------------------------------------------------------------------


class _ApiClient(HTTPSession):
    """Stand-in for a real ``APIClient(HTTPSession)`` subclass.

    Adds positional constructor knobs (``mode``, ``catalog_name``,
    ``schema_name``) that participate in identity, plus a few internal
    slots (``_contents_by_id``, ``_credentials``, ``_token``,
    ``_databricks``) that are derived state and ride along on pickle
    via the transient list.
    """

    _TRANSIENT_STATE_ATTRS = HTTPSession._TRANSIENT_STATE_ATTRS | {
        "_contents_by_id", "_credentials", "_token", "_databricks",
    }

    def __init__(
        self,
        base_url: str | URL | None = None,
        mode: str = "",
        catalog_name: str = "trading_tgp_dev",
        schema_name: str = "src_meteologica",
        **kwargs,
    ):
        if getattr(self, "_initialized", False):
            return
        base_url = base_url or "https://api-markets.example.com/api/v1/"
        super().__init__(base_url=base_url, **kwargs)
        self.mode = mode
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self._contents_by_id: dict = {}
        self._credentials = None
        self._token = None
        self._databricks = None

    def __setstate__(self, state):
        if getattr(self, "_initialized", False):
            return
        super().__setstate__(state)
        # Transient handles re-init to the same shape ``__init__`` set.
        self._contents_by_id = {}
        self._credentials = None
        self._token = None
        self._databricks = None


@pytest.fixture(autouse=True)
def _clear_api_client_cache():
    yield
    Session._INSTANCES.clear()


class TestSubclassInheritance:
    """An :class:`HTTPSession` subclass that adds constructor knobs."""

    def test_subclass_args_split_singletons(self) -> None:
        a = _ApiClient(catalog_name="prod", schema_name="api")
        b = _ApiClient(catalog_name="dev", schema_name="api")
        same = _ApiClient(catalog_name="prod", schema_name="api")
        assert a is not b, "different catalog_name must split singletons"
        assert a is same, "matching subclass args must collapse"

    def test_default_args_collapse(self) -> None:
        a = _ApiClient()
        b = _ApiClient()
        assert a is b

    def test_pickle_round_trip_collapses_to_singleton(self) -> None:
        s = _ApiClient(catalog_name="prod", schema_name="api", mode="api")
        clone = pickle.loads(pickle.dumps(s))
        assert clone is s

    def test_pickle_rebuilds_when_cache_cleared(self) -> None:
        s = _ApiClient(catalog_name="prod", schema_name="api", mode="api")
        s._contents_by_id["loaded"] = True  # touch derived state
        blob = pickle.dumps(s)
        Session._INSTANCES.clear()
        clone = pickle.loads(blob)
        assert clone is not s
        # Constructor knobs survive the round-trip.
        assert clone.catalog_name == "prod"
        assert clone.schema_name == "api"
        assert clone.mode == "api"
        # Transients are re-initialised to their fresh-init defaults
        # (the ``__setstate__`` override drops the sender's snapshot).
        assert clone._contents_by_id == {}
        assert clone._credentials is None
        assert clone._token is None
        # And lock + http_pool come back as functional handles.
        assert clone._lock.acquire(blocking=False)
        clone._lock.release()
        assert clone.http_pool is not None

    def test_subclass_key_excludes_transient_slots(self) -> None:
        # Two clients differing only on derived state still collapse —
        # transient slots aren't part of the singleton key.
        a = _ApiClient(catalog_name="prod")
        a._contents_by_id["k"] = "v"
        a._token = "abc"
        b = _ApiClient(catalog_name="prod")
        assert a is b
        assert b._contents_by_id == {"k": "v"}  # same instance

    def test_getnewargs_ex_only_carries_constructor_kwargs(self) -> None:
        s = _ApiClient(catalog_name="prod", schema_name="api", mode="api")
        args, kwargs = s.__getnewargs_ex__()
        # Positional base_url, every subclass + parent kwarg.
        assert args == (s.base_url,)
        for name in (
            "mode", "catalog_name", "schema_name",
            "verify", "pool_maxsize", "headers", "waiting", "auth",
        ):
            assert name in kwargs, f"missing kwarg {name!r}"
        for name in ("_contents_by_id", "_credentials", "_token", "_databricks"):
            assert name not in kwargs, f"transient slot {name!r} leaked"
