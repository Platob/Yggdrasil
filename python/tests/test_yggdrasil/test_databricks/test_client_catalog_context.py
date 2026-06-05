"""Default catalog / schema context on DatabricksClient + global broadcast,
the env-var defaults, and the "drop fields equal to default" serialization."""
from __future__ import annotations

import pickle

import pytest

from yggdrasil.databricks.client import (
    CATALOG_NAME,
    SCHEMA_NAME,
    DatabricksClient,
    current_catalog,
    current_schema,
    invalidate_env_defaults,
)


@pytest.fixture(autouse=True)
def _reset_catalog_context():
    cat = CATALOG_NAME.set(None)
    sch = SCHEMA_NAME.set(None)
    try:
        yield
    finally:
        CATALOG_NAME.reset(cat)
        SCHEMA_NAME.reset(sch)


class TestDefaults:
    def test_defaults_to_none(self):
        c = DatabricksClient(host="ws0.example.com", token="t")
        assert c.catalog_name is None
        assert c.schema_name is None

    def test_fields_set_from_kwargs(self):
        c = DatabricksClient(
            host="ws.example.com", token="t",
            catalog_name="main", schema_name="sales",
        )
        assert c.catalog_name == "main"
        assert c.schema_name == "sales"

    def test_distinct_context_yields_distinct_singleton(self):
        a = DatabricksClient(host="ws.example.com", token="t", catalog_name="main")
        b = DatabricksClient(host="ws.example.com", token="t", catalog_name="other")
        assert a is not b  # catalog is part of identity


class TestEnvDefaults:
    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_CATALOG_NAME", "envcat")
        monkeypatch.setenv("DATABRICKS_SCHEMA_NAME", "envsch")
        invalidate_env_defaults()
        DatabricksClient._INSTANCES.clear()
        try:
            c = DatabricksClient(host="envhost.example.com", token="t")
            assert c.catalog_name == "envcat"
            assert c.schema_name == "envsch"
            # explicit kwargs still override the env default.
            c2 = DatabricksClient(host="envhost.example.com", token="t", catalog_name="explicit")
            assert c2.catalog_name == "explicit"
        finally:
            invalidate_env_defaults()


class TestBroadcast:
    def test_construction_broadcasts_to_context_vars(self):
        DatabricksClient(
            host="b.example.com", token="t", catalog_name="main", schema_name="sales",
        )
        assert current_catalog() == "main"
        assert current_schema() == "sales"

    def test_credential_only_client_does_not_wipe_active_default(self):
        DatabricksClient(host="b.example.com", token="t", catalog_name="main")
        DatabricksClient(host="c.example.com", token="t")  # no catalog
        assert current_catalog() == "main"

    def test_set_current_broadcasts(self):
        c = DatabricksClient(host="d.example.com", token="t", catalog_name="cat2")
        CATALOG_NAME.set(None)
        DatabricksClient.set_current(c)
        assert current_catalog() == "cat2"


class TestVendedSubServices:
    def test_sql_engine_inherits_client_context(self):
        c = DatabricksClient(
            host="h.example.com", token="t", catalog_name="main", schema_name="sales",
        )
        assert c.sql.catalog_name == "main"
        assert c.sql.schema_name == "sales"

    def test_schemas_and_volumes_inherit_client_context(self):
        c = DatabricksClient(
            host="i.example.com", token="t", catalog_name="main", schema_name="sales",
        )
        assert c.schemas.catalog_name == "main"
        assert c.volumes.catalog_name == "main"
        assert c.volumes.schema_name == "sales"

    def test_defaults_stay_none_without_context(self):
        c = DatabricksClient(host="j.example.com", token="t")
        assert c.sql.catalog_name is None
        assert c.volumes.catalog_name is None


class TestSerializationDropsDefaults:
    def test_getstate_omits_fields_at_default(self):
        c = DatabricksClient(host="ws.example.com", token="t", catalog_name="main")
        state = c.__getstate__()
        assert "product" not in state            # defaults dropped …
        assert "schema_name" not in state
        assert state["host"].endswith("ws.example.com")  # … non-defaults kept.
        assert state["token"] == "t"
        assert state["catalog_name"] == "main"

    def test_cross_process_restore_reseeds_defaults(self):
        c = DatabricksClient(
            host="ws.example.com", token="t", catalog_name="main", schema_name="sales",
        )
        payload = pickle.dumps(c)
        DatabricksClient._INSTANCES.clear()
        r = pickle.loads(payload)
        assert r is not c
        assert r.host.endswith("ws.example.com")
        assert r.catalog_name == "main" and r.schema_name == "sales"
        assert r.product == "yggdrasil"          # default re-seeded
        assert r.max_connection_pools == c.max_connection_pools
