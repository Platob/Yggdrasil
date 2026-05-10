"""Smoke tests for :class:`DatabricksClient` pickle + singleton machinery.

Covers the contract laid down on :class:`DatabricksClient`:

- ``__new__`` returns the per-process singleton for a given resolved
  init-kwarg tuple, so two ``DatabricksClient(host="X")`` calls share
  one instance and the cached SDK / sub-service handles survive.
- ``__getstate__`` only carries non-default values, so a pickle round
  trip on the same host stays compact.
- ``__reduce__`` re-resolves env-bound defaults from the local
  environment on the receiving side — env_field defaults adapt.
- ``__reduce__`` adapts to a Databricks runtime: if
  ``DATABRICKS_RUNTIME_VERSION`` is set on the receiving side, the
  rehydrated client drops sender-bound credentials and forces
  ``auth_type='runtime'``.
- :class:`DatabricksService` (now a plain class, no ``@dataclass``)
  pickles by carrying its ``client`` reference and rebuilds against the
  receiving environment.
"""

from __future__ import annotations

import os
import pickle
import unittest

from yggdrasil.databricks.client import (
    DatabricksClient,
    DatabricksService,
    _reconstruct_databricks_client,
)


def _clear_singleton_pool() -> None:
    """Wipe the per-class singleton pool between tests."""
    with DatabricksClient._SINGLETONS_LOCK:
        DatabricksClient._SINGLETONS.clear()


def _scrub_env() -> dict[str, str]:
    """Remove DATABRICKS_* env vars and return them for restoration."""
    saved: dict[str, str] = {}
    for key in list(os.environ):
        if key.startswith(("DATABRICKS_", "ARM_", "GOOGLE_")):
            saved[key] = os.environ.pop(key)
    return saved


def _restore_env(saved: dict[str, str]) -> None:
    for key, value in saved.items():
        os.environ[key] = value


class DatabricksClientPickleCase(unittest.TestCase):

    def setUp(self) -> None:
        self._saved_env = _scrub_env()
        _clear_singleton_pool()

    def tearDown(self) -> None:
        _clear_singleton_pool()
        # Drop anything tests may have left behind, then restore originals.
        for key in list(os.environ):
            if key.startswith(("DATABRICKS_", "ARM_", "GOOGLE_")):
                os.environ.pop(key, None)
        _restore_env(self._saved_env)

    # ------------------------------------------------------------------
    # Singleton dispatch
    # ------------------------------------------------------------------

    def test_same_kwargs_return_same_instance(self) -> None:
        a = DatabricksClient(host="https://ws.example.com", token="t1")
        b = DatabricksClient(host="https://ws.example.com", token="t1")
        self.assertIs(a, b)

    def test_different_host_returns_different_instance(self) -> None:
        a = DatabricksClient(host="https://ws-a.example.com")
        b = DatabricksClient(host="https://ws-b.example.com")
        self.assertIsNot(a, b)

    def test_recycled_instance_skips_init(self) -> None:
        a = DatabricksClient(host="https://ws.example.com")
        # Stash sentinel on a non-init slot — recycling must not clobber.
        a.__dict__["_sentinel"] = "kept"
        b = DatabricksClient(host="https://ws.example.com")
        self.assertIs(a, b)
        self.assertEqual(b.__dict__.get("_sentinel"), "kept")

    def test_positional_and_keyword_resolve_to_same_singleton(self) -> None:
        a = DatabricksClient("https://ws.example.com")
        b = DatabricksClient(host="https://ws.example.com")
        self.assertIs(a, b)

    # ------------------------------------------------------------------
    # __getstate__ compression
    # ------------------------------------------------------------------

    def test_getstate_drops_fields_at_local_default(self) -> None:
        client = DatabricksClient(host="https://ws.example.com", token="t")
        state = client.__getstate__()
        init = state["init"]
        # Carried: explicitly-set, non-env values.
        self.assertEqual(init["host"], "https://ws.example.com")
        self.assertEqual(init["token"], "t")
        # Dropped: anything still equal to its (env-or-static) default.
        self.assertNotIn("account_id", init)
        self.assertNotIn("client_id", init)
        self.assertNotIn("azure_use_msi", init)
        self.assertNotIn("rate_limit", init)

    def test_getstate_drops_explicit_value_matching_env(self) -> None:
        # When the local env already supplies the same value, drop it
        # from the carried state — that's what makes the receiving side
        # env-adaptive.
        os.environ["DATABRICKS_HOST"] = "https://ws.example.com"
        client = DatabricksClient(host="https://ws.example.com")
        init = client.__getstate__()["init"]
        self.assertNotIn("host", init)

    # ------------------------------------------------------------------
    # Pickle round-trip
    # ------------------------------------------------------------------

    def test_pickle_round_trip_same_process_returns_singleton(self) -> None:
        original = DatabricksClient(
            host="https://ws.example.com", token="dapi-1", profile="dev",
        )
        rehydrated = pickle.loads(pickle.dumps(original))
        self.assertIs(rehydrated, original)

    def test_pickle_round_trip_after_pool_clear_rebuilds(self) -> None:
        original = DatabricksClient(host="https://ws.example.com", token="t")
        payload = pickle.dumps(original)
        _clear_singleton_pool()
        rebuilt = pickle.loads(payload)
        self.assertIsNot(rebuilt, original)
        self.assertEqual(rebuilt.host, "https://ws.example.com")
        self.assertEqual(rebuilt.token, "t")

    def test_unpickle_re_resolves_env_defaults_on_receiving_side(self) -> None:
        # Source side: explicit host (no env var set in setUp).
        source = DatabricksClient(host="https://ws-source.example.com")
        payload = pickle.dumps(source)

        # Simulate landing on a host with a different DATABRICKS_HOST —
        # env_field default re-evaluates from the *new* environment for
        # any field the sender didn't override. Here the sender DID set
        # host, so it carries through; the rest fill from local env.
        _clear_singleton_pool()
        os.environ["DATABRICKS_TOKEN"] = "env-token"
        rebuilt = pickle.loads(payload)
        self.assertEqual(rebuilt.host, "https://ws-source.example.com")
        self.assertEqual(rebuilt.token, "env-token")

    def test_unpickle_in_databricks_runtime_drops_credentials(self) -> None:
        source = DatabricksClient(
            host="https://ws.example.com",
            token="dapi-secret",
            client_id="cid",
            client_secret="csec",
            profile="dev",
        )
        payload = pickle.dumps(source)

        _clear_singleton_pool()
        os.environ["DATABRICKS_RUNTIME_VERSION"] = "15.4"
        try:
            rebuilt = pickle.loads(payload)
            self.assertEqual(rebuilt.auth_type, "runtime")
            self.assertIsNone(rebuilt.token)
            self.assertIsNone(rebuilt.client_id)
            self.assertIsNone(rebuilt.client_secret)
            self.assertIsNone(rebuilt.profile)
        finally:
            os.environ.pop("DATABRICKS_RUNTIME_VERSION", None)

    def test_reconstruct_function_used_directly(self) -> None:
        # Smoke check on the module-level reconstructor — pickle calls
        # this; importable so external code (e.g. ``copy.copy``) can
        # also reach it.
        rebuilt = _reconstruct_databricks_client(
            DatabricksClient,
            {"init": {"host": "https://ws.example.com", "token": "t"}},
        )
        self.assertEqual(rebuilt.host, "https://ws.example.com")
        self.assertEqual(rebuilt.token, "t")

    # ------------------------------------------------------------------
    # DatabricksService — now a plain class, no @dataclass
    # ------------------------------------------------------------------

    def test_service_init_default_uses_current_client(self) -> None:
        DatabricksClient(host="https://default.example.com")
        DatabricksClient.set_current(
            DatabricksClient(host="https://default.example.com"),
        )
        try:
            from yggdrasil.databricks.sql.tables import Tables

            svc = Tables()
            self.assertIs(svc.client, DatabricksClient.current())
        finally:
            DatabricksClient.set_current(None)

    def test_service_explicit_client_wins(self) -> None:
        from yggdrasil.databricks.sql.tables import Tables

        client = DatabricksClient(host="https://ws.example.com")
        svc = Tables(client=client, catalog_name="main", schema_name="sales")
        self.assertIs(svc.client, client)
        self.assertEqual(svc.catalog_name, "main")
        self.assertEqual(svc.schema_name, "sales")

    def test_service_pickle_round_trip(self) -> None:
        from yggdrasil.databricks.sql.tables import Tables

        client = DatabricksClient(
            host="https://ws.example.com", token="t",
        )
        svc = Tables(
            client=client,
            catalog_name="main",
            schema_name="sales",
            table_name="orders",
        )
        rebuilt = pickle.loads(pickle.dumps(svc))
        self.assertIsInstance(rebuilt, Tables)
        self.assertIs(rebuilt.client, client)  # singleton on same process
        self.assertEqual(rebuilt.catalog_name, "main")
        self.assertEqual(rebuilt.schema_name, "sales")
        self.assertEqual(rebuilt.table_name, "orders")

    def test_no_field_service_pickle_round_trip(self) -> None:
        from yggdrasil.databricks.sql.catalogs import Catalogs

        client = DatabricksClient(host="https://ws.example.com")
        svc = Catalogs(client=client)
        rebuilt = pickle.loads(pickle.dumps(svc))
        self.assertIsInstance(rebuilt, Catalogs)
        self.assertIs(rebuilt.client, client)


if __name__ == "__main__":
    unittest.main()
