"""Pickle + singleton tests for :class:`AWSClient` / :class:`AWSService`.

Same generic contract as the :class:`Session` refactor:

* per-(cls, config) singleton caching in ``__new__`` so two clients
  built with the same config share the boto session, connection pool,
  and per-service boto-client cache;
* ``__init__`` is idempotent — Python always re-enters it after
  ``__new__`` returns a cached instance, and the ``_initialized``
  guard skips the second pass;
* generic ``__getstate__`` / ``__setstate__`` excludes transient
  handles via ``_TRANSIENT_STATE_ATTRS``;
* ``__getnewargs__`` carries the cache key so a client unpickled in
  the same process collapses to the live singleton.
"""
from __future__ import annotations

import pickle

import pytest

from yggdrasil.aws import AWSClient
from yggdrasil.aws.fs.service import S3Service


@pytest.fixture(autouse=True)
def _clear_singleton_caches():
    """Drop cross-test bleed in the per-config / per-client caches."""
    AWSClient._INSTANCES.clear()
    S3Service._INSTANCES.clear()
    yield
    AWSClient._INSTANCES.clear()
    S3Service._INSTANCES.clear()


# ---------------------------------------------------------------------------
# AWSConfig is hashable
# ---------------------------------------------------------------------------


class TestConfigHashable:

    def test_default_config_hashes(self) -> None:
        assert hash(AWSClient()) == hash(AWSClient())

    def test_same_field_values_hash_equal(self) -> None:
        c1 = AWSClient(region="us-east-1", profile="prod")
        c2 = AWSClient(region="us-east-1", profile="prod")
        assert c1 == c2
        assert hash(c1) == hash(c2)

    def test_refresher_excluded_from_identity(self) -> None:
        c1 = AWSClient(region="us-east-1", refresher=lambda: {})
        c2 = AWSClient(region="us-east-1", refresher=lambda: {})
        # Two distinct callables — still equal & same hash.
        assert c1 == c2
        assert hash(c1) == hash(c2)


# ---------------------------------------------------------------------------
# AWSClient singleton-by-config
# ---------------------------------------------------------------------------


class TestClientSingleton:

    def test_same_config_same_instance(self) -> None:
        # Two identical-kwarg constructions collapse to one cached client.
        assert AWSClient(region="us-east-1") is AWSClient(region="us-east-1")

    def test_equal_configs_share_instance(self) -> None:
        # Distinct config instances with identical fields collapse to
        # the same client — that's the whole point of singleton caching.
        a = AWSClient(region="us-east-1")
        b = AWSClient(region="us-east-1")
        assert a is b

    def test_different_configs_different_instances(self) -> None:
        a = AWSClient(region="us-east-1")
        b = AWSClient(region="eu-west-1")
        assert a is not b

    def test_init_is_idempotent(self) -> None:
        # Mutate the live cached instance, then re-construct — the
        # second __init__ pass must skip and leave the mutation intact.
        c = AWSClient(region="us-east-1")
        c._client_cache["sentinel"] = object()
        again = AWSClient(region="us-east-1")
        assert again is c
        assert "sentinel" in again._client_cache


# ---------------------------------------------------------------------------
# AWSClient pickling
# ---------------------------------------------------------------------------


class TestClientPickle:

    def test_unpickle_collapses_to_live_singleton(self) -> None:
        c = AWSClient(region="us-east-1")
        restored = pickle.loads(pickle.dumps(c))
        assert restored is c, (
            "in-process unpickle must reuse the live singleton, not clone it"
        )

    def test_setstate_does_not_clobber_live_state(self) -> None:
        c = AWSClient(region="us-east-1")
        blob = pickle.dumps(c)
        # Mutate live; restore from stale blob — live state must win.
        c._client_cache["sentinel"] = "live"
        restored = pickle.loads(blob)
        assert restored is c
        assert restored._client_cache.get("sentinel") == "live"

    def test_transient_attrs_excluded_from_state(self) -> None:
        c = AWSClient(region="us-east-1")
        c._client_cache["k"] = "v"
        c._account_id = "123456789012"
        state = c.__getstate__()
        for attr in AWSClient._TRANSIENT_STATE_ATTRS:
            assert attr not in state, f"transient {attr!r} leaked into pickle"

    def test_cross_process_unpickle_rebuilds_transients(self) -> None:
        # Simulate a remote worker: pickle, drop the cache, unpickle.
        c = AWSClient(region="us-east-1", profile="prod")
        c._client_cache["k"] = object()
        c._account_id = "123456789012"
        blob = pickle.dumps(c)
        AWSClient._INSTANCES.clear()
        S3Service._INSTANCES.clear()

        restored = pickle.loads(blob)
        assert restored is not c
        assert restored.region == c.region
        assert restored.profile == c.profile
        # Transients reset to their fresh defaults.
        assert restored._session is None
        assert restored._client_cache == {}
        assert restored._s3 is None
        assert restored._account_id is None
        assert restored._was_connected is False
        # And the receiver re-registers as the singleton for its key.
        assert AWSClient._INSTANCES.get(restored._singleton_key_) is restored

    def test_getnewargs_carries_identity(self) -> None:
        # ``__getnewargs_ex__`` (not ``__getnewargs__``) is the route
        # used by pickle for the merged Singleton-backed client; it
        # carries every identity-bearing kwarg so unpickling collapses
        # to the cached singleton when one exists.
        c = AWSClient(region="us-east-1")
        args, kwargs = c.__getnewargs_ex__()
        assert args == ()
        assert kwargs["region"] == "us-east-1"
        for name in AWSClient._IDENTITY_FIELDS:
            assert name in kwargs


# ---------------------------------------------------------------------------
# AWSService singleton-by-client
# ---------------------------------------------------------------------------


class TestServiceSingleton:

    def test_same_client_same_service(self) -> None:
        c = AWSClient(region="us-east-1")
        assert S3Service(client=c) is S3Service(client=c)

    def test_different_clients_different_services(self) -> None:
        c1 = AWSClient(region="us-east-1")
        c2 = AWSClient(region="eu-west-1")
        assert S3Service(client=c1) is not S3Service(client=c2)

    def test_client_s3_property_returns_singleton(self) -> None:
        c = AWSClient(region="us-east-1")
        s_via_prop = c.s3
        s_via_ctor = S3Service(client=c)
        assert s_via_prop is s_via_ctor

    def test_service_init_is_idempotent(self) -> None:
        c = AWSClient(region="us-east-1")
        s = S3Service(client=c)
        again = S3Service(client=c)
        assert again is s


# ---------------------------------------------------------------------------
# AWSService pickling
# ---------------------------------------------------------------------------


class TestServicePickle:

    def test_unpickle_collapses_to_live_singleton(self) -> None:
        c = AWSClient(region="us-east-1")
        s = S3Service(client=c)
        restored = pickle.loads(pickle.dumps(s))
        assert restored is s

    def test_getnewargs_carries_client(self) -> None:
        c = AWSClient(region="us-east-1")
        s = S3Service(client=c)
        args = s.__getnewargs__()
        assert args == (s.client,)
