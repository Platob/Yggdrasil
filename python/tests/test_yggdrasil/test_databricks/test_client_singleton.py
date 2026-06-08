"""Tests for :class:`DatabricksClient` singleton identity + caching.

The client is a :class:`Singleton` keyed on the canonical init kwargs
(after env defaulting + host normalisation), so two callers with the
same logical config should always collapse to the same instance — and
two callers with materially different config (host, token, profile,
…) must NOT. These tests pin the identity contract.
"""
from __future__ import annotations

import pickle
from unittest.mock import patch

import pytest

from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks import client as client_module


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Drop the singleton cache and force a fresh env-defaults
    snapshot per test so cross-test leakage can't mask identity
    bugs."""
    # Clear the singleton cache + the process-wide ``current`` slot.
    DatabricksClient._INSTANCES.clear()
    monkeypatch.setattr(DatabricksClient, "_current", None, raising=False)
    # Strip every DATABRICKS_* / ARM_* / GOOGLE_* env var so the
    # tests run against an empty env baseline. ``invalidate_env_defaults``
    # then forces the snapshot to re-read.
    for name in list(__import__("os").environ.keys()):
        if name.startswith(("DATABRICKS_", "ARM_", "GOOGLE_")):
            monkeypatch.delenv(name, raising=False)
    client_module.invalidate_env_defaults()
    yield
    DatabricksClient._INSTANCES.clear()
    client_module.invalidate_env_defaults()


class TestIdentityCollapsing:
    """Equivalent constructor calls return the same instance."""

    def test_same_host_token_collapses(self) -> None:
        a = DatabricksClient(host="https://ws.example.com", token="t")
        b = DatabricksClient(host="https://ws.example.com", token="t")
        assert a is b

    def test_host_normalization_collapses_variants(self) -> None:
        # The normalizer strips scheme + path, then re-prepends https://.
        # All three callers below resolve to the same canonical host.
        a = DatabricksClient(host="ws.example.com", token="t")
        b = DatabricksClient(host="https://ws.example.com", token="t")
        c = DatabricksClient(host="https://ws.example.com/some/path", token="t")
        assert a is b is c

    def test_account_id_defaults_to_accounts_host(self) -> None:
        """``account_id`` without an explicit ``host`` lands on the
        central accounts endpoint."""
        a = DatabricksClient(account_id="acc-123", token="t")
        b = DatabricksClient(
            host="https://accounts.cloud.databricks.com",
            account_id="acc-123", token="t",
        )
        assert a is b

    def test_kwarg_order_does_not_matter(self) -> None:
        """The singleton key is built from the resolved kwargs dict in
        a fixed order — caller-side reordering must not produce a
        different key."""
        a = DatabricksClient(host="ws.example.com", token="t", profile="p")
        b = DatabricksClient(profile="p", token="t", host="ws.example.com")
        assert a is b


class TestIdentityPartitioning:
    """Materially different config produces distinct instances."""

    def test_different_hosts_yield_distinct(self) -> None:
        a = DatabricksClient(host="ws-a.example.com", token="t")
        b = DatabricksClient(host="ws-b.example.com", token="t")
        assert a is not b

    def test_different_tokens_yield_distinct(self) -> None:
        """A token change is a credential-rotation event — the new
        identity must NOT alias to the previous credentials."""
        a = DatabricksClient(host="ws.example.com", token="t1")
        b = DatabricksClient(host="ws.example.com", token="t2")
        assert a is not b

    def test_different_profiles_yield_distinct(self) -> None:
        a = DatabricksClient(host="ws.example.com", profile="dev")
        b = DatabricksClient(host="ws.example.com", profile="prod")
        assert a is not b

    def test_different_account_ids_yield_distinct(self) -> None:
        a = DatabricksClient(account_id="acc-A", token="t")
        b = DatabricksClient(account_id="acc-B", token="t")
        assert a is not b


class TestEnvDefaults:
    """Environment variables flow into the singleton key so two
    callers with the same env see the same instance."""

    def test_env_host_collapses_with_explicit(self, monkeypatch) -> None:
        monkeypatch.setenv("DATABRICKS_HOST", "https://env-ws.example.com")
        client_module.invalidate_env_defaults()
        DatabricksClient._INSTANCES.clear()

        env_client = DatabricksClient(token="t")
        explicit = DatabricksClient(host="env-ws.example.com", token="t")
        assert env_client is explicit

    def test_env_change_yields_distinct_instance(
        self, monkeypatch,
    ) -> None:
        """A cached instance keyed off the previous env snapshot
        survives an env rotation until the cache is cleared. The
        next constructor call sees the new env."""
        monkeypatch.setenv("DATABRICKS_HOST", "https://env-1.example.com")
        client_module.invalidate_env_defaults()
        first = DatabricksClient(token="t")

        monkeypatch.setenv("DATABRICKS_HOST", "https://env-2.example.com")
        client_module.invalidate_env_defaults()
        second = DatabricksClient(token="t")
        # Different env -> different singleton key -> distinct instance.
        # Read ``.host`` directly — going through ``.config`` would
        # walk the SDK auth chain (env, profile, ~/.databrickscfg, …)
        # and can hang in test envs without auth configured.
        assert first is not second
        assert first.host.endswith("env-1.example.com")
        assert second.host.endswith("env-2.example.com")


class TestCurrentProcessWideSlot:
    """``DatabricksClient.current()`` returns the process-wide current
    client; ``set_current`` swaps it."""

    def test_current_returns_set_instance(self) -> None:
        c = DatabricksClient(host="ws.example.com", token="t")
        DatabricksClient.set_current(c)
        assert DatabricksClient.current() is c

    def test_set_current_to_none_clears(self) -> None:
        c = DatabricksClient(host="ws.example.com", token="t")
        DatabricksClient.set_current(c)
        DatabricksClient.set_current(None)
        # The next current() call materialises whatever the env / defaults
        # resolve to. With our scrubbed env that's still a valid singleton
        # — just not the one we set above.
        next_current = DatabricksClient.current()
        assert next_current is not c


class TestPickle:
    """Pickle round-trip preserves the singleton key so the unpickled
    client collapses onto the live instance when one is already
    cached."""

    def test_in_process_collapses_to_live(self) -> None:
        c = DatabricksClient(host="ws.example.com", token="t")
        unpickled = pickle.loads(pickle.dumps(c))
        # Live singleton survives the round trip → identity holds.
        assert unpickled is c

    def test_cross_process_rebuilds_with_same_config(self) -> None:
        """Simulate a fresh receiving process by dropping the cache
        before unpickling — the new client has different identity but
        the same config."""
        c = DatabricksClient(host="ws.example.com", token="t")
        payload = pickle.dumps(c)
        DatabricksClient._INSTANCES.clear()

        unpickled = pickle.loads(payload)
        assert unpickled is not c
        assert unpickled.host.endswith("ws.example.com")


class TestInitIsIdempotentUnderSingletonReentry:
    """A second ``DatabricksClient(...)`` call with the same key
    routes through ``__new__`` (cache hit) then re-enters ``__init__``.
    The re-entry must NOT reset live SDK handles or lazy caches —
    ``_initialized`` short-circuits the second pass."""

    def test_no_double_init(self, monkeypatch) -> None:
        calls = {"n": 0}
        original = DatabricksClient.__init__

        def counting(self, **kwargs):
            calls["n"] += 1
            return original(self, **kwargs)

        monkeypatch.setattr(DatabricksClient, "__init__", counting)

        a = DatabricksClient(host="ws.example.com", token="t")
        b = DatabricksClient(host="ws.example.com", token="t")
        assert a is b
        # __init__ fires on both calls (Python always runs it after
        # __new__), but the second call's body short-circuits via the
        # ``_initialized`` guard. The guard's evidence: calls["n"] == 2
        # but the instance state is what the FIRST call established.
        assert calls["n"] == 2
        assert getattr(a, "_initialized", False) is True
