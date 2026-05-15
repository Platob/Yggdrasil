"""Tests for per-client singleton caching across UC resources.

Catalog / Schema / Table / Volume are all keyed on ``(client, …)`` so
two callers asking for the same UC resource under the *same* client
collapse to one instance, while two callers asking for the same UC
resource under *different* clients each get their own instance. The
``DatabricksClient`` is itself a :class:`Singleton`, so two clients
built from the same kwargs are also the same instance.

These tests pin those invariants without touching a real workspace —
all SDK boundaries are out of scope; only ``__new__`` / cache identity
is exercised.
"""
from __future__ import annotations

import unittest

from yggdrasil.databricks.catalog.catalog import Catalog
from yggdrasil.databricks.catalog.catalogs import Catalogs
from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.schema.schema import Schema
from yggdrasil.databricks.schema.schemas import Schemas
from yggdrasil.databricks.table.table import Table
from yggdrasil.databricks.table.tables import Tables
from yggdrasil.databricks.volume.volume import Volume
from yggdrasil.databricks.volume.volumes import Volumes


def _reset_caches() -> None:
    DatabricksClient._INSTANCES.clear()
    Catalog._INSTANCES.clear()
    Schema._INSTANCES.clear()
    Table._INSTANCES.clear()
    Volume._INSTANCES.clear()


class TestDatabricksClientSingleton(unittest.TestCase):

    def setUp(self) -> None:
        _reset_caches()

    def test_same_kwargs_same_instance(self) -> None:
        a = DatabricksClient(host="https://x.com", token="t1")
        b = DatabricksClient(host="https://x.com", token="t1")
        self.assertIs(a, b)

    def test_different_tokens_different_instances(self) -> None:
        a = DatabricksClient(host="https://x.com", token="tA")
        b = DatabricksClient(host="https://x.com", token="tB")
        self.assertIsNot(a, b)

    def test_different_hosts_different_instances(self) -> None:
        a = DatabricksClient(host="https://x.com", token="t")
        b = DatabricksClient(host="https://y.com", token="t")
        self.assertIsNot(a, b)


class TestCatalogSingleton(unittest.TestCase):

    def setUp(self) -> None:
        _reset_caches()

    def test_same_client_same_name(self) -> None:
        client = DatabricksClient(host="https://x.com", token="t")
        svc = Catalogs(client=client)
        a = Catalog(service=svc, catalog_name="main")
        b = Catalog(service=svc, catalog_name="main")
        self.assertIs(a, b)

    def test_different_clients_different_instances(self) -> None:
        client_a = DatabricksClient(host="https://x.com", token="tA")
        client_b = DatabricksClient(host="https://x.com", token="tB")
        # Different clients on the same host — must own distinct catalogs.
        a = Catalog(service=Catalogs(client=client_a), catalog_name="main")
        b = Catalog(service=Catalogs(client=client_b), catalog_name="main")
        self.assertIsNot(a, b)
        self.assertIs(a.client, client_a)
        self.assertIs(b.client, client_b)

    def test_different_names_different_instances(self) -> None:
        client = DatabricksClient(host="https://x.com", token="t")
        svc = Catalogs(client=client)
        a = Catalog(service=svc, catalog_name="main")
        b = Catalog(service=svc, catalog_name="other")
        self.assertIsNot(a, b)
        self.assertEqual(a.catalog_name, "main")
        self.assertEqual(b.catalog_name, "other")


class TestSchemaSingleton(unittest.TestCase):

    def setUp(self) -> None:
        _reset_caches()

    def test_same_client_same_names(self) -> None:
        client = DatabricksClient(host="https://x.com", token="t")
        svc = Schemas(client=client)
        a = Schema(service=svc, catalog_name="main", schema_name="sales")
        b = Schema(service=svc, catalog_name="main", schema_name="sales")
        self.assertIs(a, b)

    def test_different_clients_different_instances(self) -> None:
        client_a = DatabricksClient(host="https://x.com", token="tA")
        client_b = DatabricksClient(host="https://x.com", token="tB")
        a = Schema(
            service=Schemas(client=client_a),
            catalog_name="main", schema_name="sales",
        )
        b = Schema(
            service=Schemas(client=client_b),
            catalog_name="main", schema_name="sales",
        )
        self.assertIsNot(a, b)
        self.assertIs(a.client, client_a)
        self.assertIs(b.client, client_b)

    def test_different_schema_names_different_instances(self) -> None:
        client = DatabricksClient(host="https://x.com", token="t")
        svc = Schemas(client=client)
        a = Schema(service=svc, catalog_name="main", schema_name="sales")
        b = Schema(service=svc, catalog_name="main", schema_name="reporting")
        self.assertIsNot(a, b)


class TestTableSingleton(unittest.TestCase):

    def setUp(self) -> None:
        _reset_caches()

    def test_same_client_same_three_part_name(self) -> None:
        client = DatabricksClient(host="https://x.com", token="t")
        svc = Tables(client=client)
        a = Table(
            service=svc, catalog_name="main",
            schema_name="sales", table_name="orders",
        )
        b = Table(
            service=svc, catalog_name="main",
            schema_name="sales", table_name="orders",
        )
        self.assertIs(a, b)

    def test_different_clients_different_instances(self) -> None:
        client_a = DatabricksClient(host="https://x.com", token="tA")
        client_b = DatabricksClient(host="https://x.com", token="tB")
        a = Table(
            service=Tables(client=client_a),
            catalog_name="main", schema_name="sales", table_name="orders",
        )
        b = Table(
            service=Tables(client=client_b),
            catalog_name="main", schema_name="sales", table_name="orders",
        )
        self.assertIsNot(a, b)
        self.assertIs(a.client, client_a)
        self.assertIs(b.client, client_b)

    def test_different_table_names_different_instances(self) -> None:
        client = DatabricksClient(host="https://x.com", token="t")
        svc = Tables(client=client)
        a = Table(
            service=svc, catalog_name="main",
            schema_name="sales", table_name="orders",
        )
        b = Table(
            service=svc, catalog_name="main",
            schema_name="sales", table_name="customers",
        )
        self.assertIsNot(a, b)


class TestVolumeSingleton(unittest.TestCase):

    def setUp(self) -> None:
        _reset_caches()

    def test_same_client_same_three_part_name(self) -> None:
        client = DatabricksClient(host="https://x.com", token="t")
        svc = Volumes(client=client)
        a = Volume(
            service=svc, catalog_name="main",
            schema_name="sales", volume_name="staging",
        )
        b = Volume(
            service=svc, catalog_name="main",
            schema_name="sales", volume_name="staging",
        )
        self.assertIs(a, b)

    def test_different_clients_different_instances(self) -> None:
        client_a = DatabricksClient(host="https://x.com", token="tA")
        client_b = DatabricksClient(host="https://x.com", token="tB")
        a = Volume(
            service=Volumes(client=client_a),
            catalog_name="main", schema_name="sales", volume_name="staging",
        )
        b = Volume(
            service=Volumes(client=client_b),
            catalog_name="main", schema_name="sales", volume_name="staging",
        )
        self.assertIsNot(a, b)
        self.assertIs(a.client, client_a)
        self.assertIs(b.client, client_b)

    def test_different_volume_names_different_instances(self) -> None:
        client = DatabricksClient(host="https://x.com", token="t")
        svc = Volumes(client=client)
        a = Volume(
            service=svc, catalog_name="main",
            schema_name="sales", volume_name="staging",
        )
        b = Volume(
            service=svc, catalog_name="main",
            schema_name="sales", volume_name="raw",
        )
        self.assertIsNot(a, b)


class TestParentNavigationReturnsSingleton(unittest.TestCase):
    """Parent navigation reuses the same singleton — no redundant clones.

    ``Table.schema`` / ``Table.catalog`` / ``Volume.schema`` /
    ``Volume.catalog`` should all hand back the same instance
    callers would get by going directly through ``client.schemas`` /
    ``client.catalogs``.
    """

    def setUp(self) -> None:
        _reset_caches()

    def test_table_catalog_is_singleton_catalog(self) -> None:
        client = DatabricksClient(host="https://x.com", token="t")
        table = Table(
            service=Tables(client=client),
            catalog_name="main", schema_name="sales", table_name="orders",
        )
        direct = Catalog(service=client.catalogs, catalog_name="main")
        self.assertIs(table.catalog, direct)

    def test_table_schema_is_singleton_schema(self) -> None:
        client = DatabricksClient(host="https://x.com", token="t")
        table = Table(
            service=Tables(client=client),
            catalog_name="main", schema_name="sales", table_name="orders",
        )
        direct = Schema(
            service=client.schemas, catalog_name="main", schema_name="sales",
        )
        self.assertIs(table.schema, direct)

    def test_volume_catalog_is_singleton_catalog(self) -> None:
        client = DatabricksClient(host="https://x.com", token="t")
        vol = Volume(
            service=Volumes(client=client),
            catalog_name="main", schema_name="sales", volume_name="staging",
        )
        direct = Catalog(service=client.catalogs, catalog_name="main")
        self.assertIs(vol.catalog, direct)

    def test_volume_schema_is_singleton_schema(self) -> None:
        client = DatabricksClient(host="https://x.com", token="t")
        vol = Volume(
            service=Volumes(client=client),
            catalog_name="main", schema_name="sales", volume_name="staging",
        )
        direct = Schema(
            service=client.schemas, catalog_name="main", schema_name="sales",
        )
        self.assertIs(vol.schema, direct)


if __name__ == "__main__":
    unittest.main()
