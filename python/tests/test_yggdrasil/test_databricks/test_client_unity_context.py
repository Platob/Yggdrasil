"""Default Unity Catalog context on DatabricksClient + global broadcast,
and the "drop fields equal to default" serialization behavior."""
from __future__ import annotations

import pickle

import pytest

from yggdrasil.databricks.client import (
    DatabricksClient,
    UNITY_CATALOG_NAME,
    UNITY_SCHEMA_NAME,
    current_unity_catalog,
    current_unity_schema,
)


@pytest.fixture(autouse=True)
def _reset_unity_context():
    cat = UNITY_CATALOG_NAME.set(None)
    sch = UNITY_SCHEMA_NAME.set(None)
    try:
        yield
    finally:
        UNITY_CATALOG_NAME.reset(cat)
        UNITY_SCHEMA_NAME.reset(sch)


class TestUnityDefaults:
    def test_defaults_to_none(self):
        c = DatabricksClient(host="ws0.example.com", token="t")
        assert c.unity_catalog_name is None
        assert c.unity_schema_name is None

    def test_fields_set_from_kwargs(self):
        c = DatabricksClient(
            host="ws.example.com", token="t",
            unity_catalog_name="main", unity_schema_name="sales",
        )
        assert c.unity_catalog_name == "main"
        assert c.unity_schema_name == "sales"

    def test_distinct_context_yields_distinct_singleton(self):
        a = DatabricksClient(host="ws.example.com", token="t", unity_catalog_name="main")
        b = DatabricksClient(host="ws.example.com", token="t", unity_catalog_name="other")
        assert a is not b  # catalog is part of identity


class TestBroadcast:
    def test_construction_broadcasts_to_context_vars(self):
        DatabricksClient(
            host="b.example.com", token="t",
            unity_catalog_name="main", unity_schema_name="sales",
        )
        assert current_unity_catalog() == "main"
        assert current_unity_schema() == "sales"

    def test_credential_only_client_does_not_wipe_active_default(self):
        DatabricksClient(host="b.example.com", token="t", unity_catalog_name="main")
        # A later client with no catalog leaves the active default in place.
        DatabricksClient(host="c.example.com", token="t")
        assert current_unity_catalog() == "main"

    def test_set_current_broadcasts(self):
        c = DatabricksClient(host="d.example.com", token="t", unity_catalog_name="cat2")
        UNITY_CATALOG_NAME.set(None)
        DatabricksClient.set_current(c)
        assert current_unity_catalog() == "cat2"

    def test_sub_service_reads_client_default(self):
        c = DatabricksClient(host="e.example.com", token="t", unity_catalog_name="main", unity_schema_name="s")
        assert c.external.unity_catalog_name == "main"
        assert c.external.unity_schema_name == "s"

    def test_sub_service_falls_back_to_global_context(self):
        # active global default…
        DatabricksClient(host="f.example.com", token="t", unity_catalog_name="globalcat")
        # …a client with no catalog of its own surfaces it through its services.
        bare = DatabricksClient(host="g.example.com", token="t")
        assert bare.external.unity_catalog_name == "globalcat"


class TestSerializationDropsDefaults:
    def test_getstate_omits_fields_at_default(self):
        c = DatabricksClient(host="ws.example.com", token="t", unity_catalog_name="main")
        state = c.__getstate__()
        # defaults dropped …
        assert "product" not in state
        assert "unity_schema_name" not in state
        # … non-defaults kept.
        assert state["host"].endswith("ws.example.com")
        assert state["token"] == "t"
        assert state["unity_catalog_name"] == "main"

    def test_cross_process_restore_reseeds_defaults(self):
        c = DatabricksClient(
            host="ws.example.com", token="t",
            unity_catalog_name="main", unity_schema_name="sales",
        )
        payload = pickle.dumps(c)
        DatabricksClient._INSTANCES.clear()
        r = pickle.loads(payload)
        assert r is not c
        assert r.host.endswith("ws.example.com")
        assert r.unity_catalog_name == "main" and r.unity_schema_name == "sales"
        assert r.product == "yggdrasil"          # default re-seeded
        assert r.max_connection_pools == c.max_connection_pools
