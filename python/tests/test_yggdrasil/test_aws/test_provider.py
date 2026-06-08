"""Tests for :class:`AwsCredentialsProvider`.

Generic contract:

* a subclass is a process-wide singleton keyed by ``(cls, key)``;
* ``__init__`` is idempotent — re-entry preserves the live state;
* the abstract :meth:`get_credentials` drives ``__call__`` so a
  provider doubles as an :class:`AWSConfig.refresher`;
* :meth:`aws_client` caches one :class:`AWSClient` per region so the
  boto session + ``RefreshableCredentials`` cycle are shared;
* pickling collapses to the live singleton in-process and rebuilds
  the per-region cache cross-process.
"""
from __future__ import annotations

import pickle
from typing import Optional

import pytest

from yggdrasil.aws import AWSClient
from yggdrasil.aws.config import AwsCredentials
from yggdrasil.aws.provider import AwsCredentialsProvider


class _StaticProvider(AwsCredentialsProvider):
    """Tiny concrete subclass used only by these tests.

    Returns a deterministic :class:`AwsCredentials` derived from
    :attr:`key` so the tests can assert which provider's vend each
    AWSClient is bound to without faking botocore.
    """

    def get_credentials(self, mode=None) -> AwsCredentials:
        return AwsCredentials(
            access_key_id=f"AKIA-{self.key}",
            secret_access_key=f"secret-{self.key}",
            session_token=f"token-{self.key}",
            expiration="2099-01-01T00:00:00Z",
        )


@pytest.fixture(autouse=True)
def _clear_singleton_caches():
    AwsCredentialsProvider._INSTANCES.clear()
    AWSClient._INSTANCES.clear()
    yield
    AwsCredentialsProvider._INSTANCES.clear()
    AWSClient._INSTANCES.clear()


# ---------------------------------------------------------------------------
# Singleton identity
# ---------------------------------------------------------------------------


class TestSingleton:

    def test_same_key_returns_same_instance(self) -> None:
        a = _StaticProvider("vol-1")
        b = _StaticProvider("vol-1")
        assert a is b

    def test_different_keys_yield_different_instances(self) -> None:
        a = _StaticProvider("vol-1")
        b = _StaticProvider("vol-2")
        assert a is not b

    def test_subclass_keyspaces_are_disjoint(self) -> None:
        class _Other(_StaticProvider):
            pass

        a = _StaticProvider("vol-1")
        b = _Other("vol-1")
        # Same string key, different subclass → different instances.
        assert a is not b

    def test_key_is_stringified(self) -> None:
        # Non-string keys flow through ``str(key)`` so callers can
        # pass numeric ids or enum members without pre-converting.
        a = _StaticProvider(42)
        b = _StaticProvider("42")
        assert a is b
        assert a.key == "42"

    def test_init_is_idempotent(self) -> None:
        # Singleton-cached instances are re-entered on every
        # constructor call; the guard must skip re-init so the live
        # per-region client cache survives.
        a = _StaticProvider("vol-1")
        a.aws_client(region="us-east-1")  # populate the cache
        a2 = _StaticProvider("vol-1")
        assert a is a2
        assert a2._client_cache  # not wiped


# ---------------------------------------------------------------------------
# Refresher contract
# ---------------------------------------------------------------------------


class TestRefresherContract:

    def test_call_delegates_to_get_credentials(self) -> None:
        p = _StaticProvider("vol-1")
        out = p()
        assert isinstance(out, AwsCredentials)
        assert out.access_key_id == "AKIA-vol-1"

    def test_hash_and_equality_follow_class_and_key(self) -> None:
        a = _StaticProvider("vol-1")
        b = _StaticProvider("vol-1")
        c = _StaticProvider("vol-2")
        assert hash(a) == hash(b)
        assert a == b
        assert a != c
        # Different subclass with same key compares unequal.
        class _Other(_StaticProvider):
            pass
        assert a != _Other("vol-1")

    def test_repr_includes_key(self) -> None:
        assert "vol-1" in repr(_StaticProvider("vol-1"))


# ---------------------------------------------------------------------------
# aws_client — per-region caching
# ---------------------------------------------------------------------------


class TestAWSClientBinding:

    def test_same_region_returns_same_client(self) -> None:
        p = _StaticProvider("vol-1")
        c1 = p.aws_client(region="us-east-1")
        c2 = p.aws_client(region="us-east-1")
        assert c1 is c2

    def test_different_regions_yield_different_clients(self) -> None:
        p = _StaticProvider("vol-1")
        us = p.aws_client(region="us-east-1")
        eu = p.aws_client(region="eu-central-1")
        assert us is not eu
        assert us.config.region == "us-east-1"
        assert eu.config.region == "eu-central-1"

    def test_returned_client_is_refresher_backed(self) -> None:
        # ``has_refresher`` is what AWSConfig uses to decide whether to
        # mint a ``RefreshableCredentials``-backed boto session — that
        # path is the whole point of going through a provider.
        p = _StaticProvider("vol-1")
        client = p.aws_client(region="us-east-1")
        assert client.config.has_refresher()
        assert client.config.refresher is p


# ---------------------------------------------------------------------------
# Pickling
# ---------------------------------------------------------------------------


class TestPickling:

    def test_in_process_pickle_round_trip_returns_singleton(self) -> None:
        p = _StaticProvider("vol-1")
        p.aws_client(region="us-east-1")  # populate cache
        loaded = pickle.loads(pickle.dumps(p))
        assert loaded is p

    def test_cross_process_pickle_rebuilds_transients(self) -> None:
        # Simulate the cross-process path: clear the live cache, then
        # unpickle. The receiver must rebuild ``_client_cache`` lazily
        # (the AWSClient pool can't ride along through pickle).
        p = _StaticProvider("vol-1")
        p.aws_client(region="us-east-1")
        payload = pickle.dumps(p)
        AwsCredentialsProvider._INSTANCES.clear()
        AWSClient._INSTANCES.clear()
        loaded = pickle.loads(payload)
        assert loaded.key == "vol-1"
        assert loaded._client_cache == {}
        # ``aws_client`` still works after rebuild.
        c = loaded.aws_client(region="us-east-1")
        assert c.config.region == "us-east-1"

    def test_client_cache_is_excluded_from_state(self) -> None:
        p = _StaticProvider("vol-1")
        p.aws_client(region="us-east-1")
        state = p.__getstate__()
        # State is intentionally minimal — only the singleton key.
        assert state == {"key": "vol-1"}


# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------


class TestAbstract:

    def test_cannot_instantiate_abstract_base(self) -> None:
        with pytest.raises(TypeError):
            AwsCredentialsProvider("vol-1")  # type: ignore[abstract]

    def test_subclass_must_implement_get_credentials(self) -> None:
        class _Missing(AwsCredentialsProvider):
            pass

        with pytest.raises(TypeError):
            _Missing("vol-1")
