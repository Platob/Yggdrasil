"""SQL emitted by :meth:`Table.rename` and :meth:`Table.clone`.

These pin the DDL surface without touching a real Databricks workspace
— the ``sql`` property is patched onto a recording stub so the SQL
text the methods would submit is captured directly.
"""
from __future__ import annotations

import datetime as _dt
import time
import unittest
from unittest.mock import patch

from databricks.sdk.service.catalog import TableInfo, TableType

from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.table.table import Table
from yggdrasil.databricks.table.tables import Tables


def _normalize_ws(sql: str) -> str:
    return " ".join(sql.split())


class _RecordingSql:
    """Stand-in for :class:`Table.sql` that records executed statements.

    Routes ``.tables`` to a real :class:`Tables` service so the parse
    helpers behave like production (1/2/3-part names, defaults from the
    bound catalog/schema).
    """

    def __init__(self, client: DatabricksClient, catalog_name: str, schema_name: str):
        self.client = client
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.tables = Tables(client=client, catalog_name=catalog_name, schema_name=schema_name)
        self.executed: list[str] = []

    def execute(self, statement: str, *args, **kwargs) -> None:
        self.executed.append(statement)


def _table(catalog: str = "main", schema: str = "sales", name: str = "orders") -> tuple[Table, _RecordingSql]:
    client = DatabricksClient(host="https://ws.example.com")
    service = Tables(client=client, catalog_name=catalog, schema_name=schema)
    table = Table(service=service, catalog_name=catalog, schema_name=schema, table_name=name)
    sql = _RecordingSql(client=client, catalog_name=catalog, schema_name=schema)
    # Skip cache cleanup so we don't need to mock entity_tags / Tables.invalidate.
    object.__setattr__(table, "_reset_cache", lambda invalidate_cache=False: None)
    # Seed managed-table infos so ``clone`` reads the type from cache and
    # doesn't reach out to the (fake) workspace to resolve view-ness.
    object.__setattr__(table, "_infos", TableInfo(table_type=TableType.MANAGED))
    object.__setattr__(table, "_infos_fetched_at", time.time())
    return table, sql


def _clear_singleton_cache() -> None:
    """Reset the per-class :class:`Table` singleton cache.

    ``Table`` is now a :class:`Singleton` keyed by
    ``(client, catalog, schema, name)``; rename/clone tests
    mutate ``catalog/schema/table_name`` after construction, so
    we need a fresh instance per test to keep state from
    leaking across the suite.
    """
    from yggdrasil.databricks.client import DatabricksClient
    DatabricksClient._INSTANCES.clear()
    Table._INSTANCES.clear()


class TestRename(unittest.TestCase):

    def setUp(self) -> None:
        _clear_singleton_cache()

    def test_simple_rename_within_schema(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            t.rename("new_orders")

        self.assertEqual(len(sql.executed), 1)
        self.assertEqual(
            _normalize_ws(sql.executed[0]),
            "ALTER TABLE `main`.`sales`.`orders` RENAME TO `new_orders`",
        )
        self.assertEqual(t.table_name, "new_orders")
        self.assertEqual(t.schema_name, "sales")
        self.assertEqual(t.catalog_name, "main")

    def test_cross_schema_rename(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            t.rename("archive.old_orders")

        self.assertEqual(
            _normalize_ws(sql.executed[0]),
            "ALTER TABLE `main`.`sales`.`orders` RENAME TO `archive`.`old_orders`",
        )
        self.assertEqual(t.schema_name, "archive")
        self.assertEqual(t.table_name, "old_orders")

    def test_three_part_target_same_catalog(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            t.rename("main.archive.old_orders")

        self.assertIn(
            "RENAME TO `archive`.`old_orders`",
            _normalize_ws(sql.executed[0]),
        )

    def test_explicit_schema_kwarg_wins(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            t.rename("new_orders", schema_name="archive")

        self.assertIn(
            "RENAME TO `archive`.`new_orders`",
            _normalize_ws(sql.executed[0]),
        )

    def test_cross_catalog_rename_rejected(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError) as cm:
                t.rename("other.sales.orders")
        self.assertIn("across catalogs", str(cm.exception))
        self.assertEqual(sql.executed, [])

    def test_noop_same_name(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            t.rename("orders")  # same as current
        self.assertEqual(sql.executed, [])

    def test_empty_name_raises(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError):
                t.rename("")


class TestClone(unittest.TestCase):

    def setUp(self) -> None:
        _clear_singleton_cache()

    def test_deep_clone_default(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            cloned = t.clone("backup.orders_copy")

        self.assertEqual(
            _normalize_ws(sql.executed[0]),
            "CREATE TABLE `main`.`backup`.`orders_copy` DEEP CLONE"
            " `main`.`sales`.`orders`",
        )
        self.assertEqual(cloned.catalog_name, "main")
        self.assertEqual(cloned.schema_name, "backup")
        self.assertEqual(cloned.table_name, "orders_copy")

    def test_shallow_clone(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            t.clone("backup.orders_copy", deep=False)

        self.assertIn("SHALLOW CLONE", _normalize_ws(sql.executed[0]))

    def test_replace(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            t.clone("backup.orders_copy", replace=True)

        self.assertTrue(
            _normalize_ws(sql.executed[0]).startswith(
                "CREATE OR REPLACE TABLE `main`.`backup`.`orders_copy`"
            )
        )

    def test_missing_ok(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            t.clone("backup.orders_copy", missing_ok=True)

        self.assertTrue(
            _normalize_ws(sql.executed[0]).startswith(
                "CREATE TABLE IF NOT EXISTS `main`.`backup`.`orders_copy`"
            )
        )

    def test_replace_and_missing_ok_conflict(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError):
                t.clone("backup.orders_copy", replace=True, missing_ok=True)
        self.assertEqual(sql.executed, [])

    def test_version_as_of(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            t.clone("backup.orders_copy", version=42)

        self.assertIn(
            "DEEP CLONE `main`.`sales`.`orders` VERSION AS OF 42",
            _normalize_ws(sql.executed[0]),
        )

    def test_timestamp_as_of_datetime(self) -> None:
        t, sql = _table()
        ts = _dt.datetime(2025, 1, 2, 3, 4, 5)
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            t.clone("backup.orders_copy", timestamp=ts)

        self.assertIn(
            "TIMESTAMP AS OF '2025-01-02T03:04:05'",
            _normalize_ws(sql.executed[0]),
        )

    def test_version_and_timestamp_mutex(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError):
                t.clone("backup.orders_copy", version=1, timestamp="2025-01-01")
        self.assertEqual(sql.executed, [])

    def test_tblproperties_and_location(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            t.clone(
                "backup.orders_copy",
                properties={"delta.enableChangeDataFeed": True, "owner": "data-eng"},
                location="s3://bucket/cloned/orders",
            )

        norm = _normalize_ws(sql.executed[0])
        self.assertIn(
            "TBLPROPERTIES ('delta.enableChangeDataFeed' = TRUE, 'owner' = 'data-eng')",
            norm,
        )
        self.assertIn("LOCATION 's3://bucket/cloned/orders'", norm)

    def test_target_can_be_table_instance(self) -> None:
        t, sql = _table()
        target = Table(
            service=t.service,
            catalog_name="main",
            schema_name="backup",
            table_name="orders_copy",
        )
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            cloned = t.clone(target)

        self.assertIn(
            "`main`.`backup`.`orders_copy` DEEP CLONE",
            _normalize_ws(sql.executed[0]),
        )
        self.assertEqual(cloned.full_name(), "main.backup.orders_copy")

    def test_self_clone_rejected(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError):
                t.clone("main.sales.orders")
        self.assertEqual(sql.executed, [])

    def test_missing_target_raises(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError):
                t.clone()
        self.assertEqual(sql.executed, [])

    def test_explicit_three_part_kwargs(self) -> None:
        t, sql = _table()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            t.clone(
                catalog_name="other",
                schema_name="archive",
                table_name="orders_copy",
            )

        self.assertIn(
            "CREATE TABLE `other`.`archive`.`orders_copy` DEEP CLONE"
            " `main`.`sales`.`orders`",
            _normalize_ws(sql.executed[0]),
        )


if __name__ == "__main__":
    unittest.main()
