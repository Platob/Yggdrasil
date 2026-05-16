"""SQL emitted by :meth:`Table.rename` and :meth:`Table.clone` for view-shaped tables.

These pin the DDL surface without touching a real Databricks workspace
— the ``sql`` property is patched onto a recording stub so the SQL
text the methods would submit is captured directly. Unity Catalog
stores views in the same ``tables`` API as managed/external tables,
so :class:`Table` covers both shapes; ``Table.is_view`` dispatches
into the ``ALTER VIEW`` / ``CREATE VIEW`` codepath when
``infos.table_type`` is one of ``VIEW`` / ``MATERIALIZED_VIEW`` /
``METRIC_VIEW``.
"""
from __future__ import annotations

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
    """Stand-in for :class:`Table.sql` that records executed statements."""

    def __init__(self, client: DatabricksClient, catalog_name: str, schema_name: str):
        self.client = client
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.executed: list[str] = []
        # ``Table.clone[view]`` reads ``self.sql.tables`` to resolve the
        # target service — proxy back to the same Tables singleton.
        self.tables = Tables(client=client, catalog_name=catalog_name, schema_name=schema_name)

    def execute(self, statement: str, *args, **kwargs) -> None:
        self.executed.append(statement)


def _view(
    catalog: str = "main",
    schema: str = "reporting",
    name: str = "orders_v",
    view_definition: str | None = "SELECT id, amount FROM main.sales.orders",
) -> tuple[Table, _RecordingSql]:
    client = DatabricksClient(host="https://ws.example.com")
    service = Tables(client=client, catalog_name=catalog, schema_name=schema)
    t = Table(service=service, catalog_name=catalog, schema_name=schema, table_name=name)
    sql = _RecordingSql(client=client, catalog_name=catalog, schema_name=schema)
    # Stub out cache invalidation so the rename / clone methods don't
    # try to talk to a real workspace when refreshing afterwards.
    object.__setattr__(t, "_invalidate_singleton", lambda *a, **k: None)
    infos = TableInfo(
        view_definition=view_definition,
        table_type=TableType.VIEW,
    )
    object.__setattr__(t, "_infos", infos)
    object.__setattr__(t, "_infos_fetched_at", time.time())
    return t, sql


def _clear_singleton_cache() -> None:
    """Reset the per-class :class:`Table` singleton cache between tests."""
    from yggdrasil.databricks.client import DatabricksClient
    DatabricksClient._INSTANCES.clear()
    Table._INSTANCES.clear()


class TestRename(unittest.TestCase):

    def setUp(self) -> None:
        _clear_singleton_cache()

    def test_simple_rename_within_schema(self) -> None:
        v, sql = _view()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.rename("new_orders_v")

        self.assertEqual(len(sql.executed), 1)
        self.assertEqual(
            _normalize_ws(sql.executed[0]),
            "ALTER VIEW `main`.`reporting`.`orders_v` RENAME TO `new_orders_v`",
        )
        self.assertEqual(v.table_name, "new_orders_v")
        self.assertEqual(v.schema_name, "reporting")

    def test_cross_schema_rename(self) -> None:
        v, sql = _view()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.rename("archive.old_orders_v")

        self.assertEqual(
            _normalize_ws(sql.executed[0]),
            "ALTER VIEW `main`.`reporting`.`orders_v` RENAME TO"
            " `archive`.`old_orders_v`",
        )
        self.assertEqual(v.schema_name, "archive")
        self.assertEqual(v.table_name, "old_orders_v")

    def test_three_part_target_same_catalog(self) -> None:
        v, sql = _view()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.rename("main.archive.old_v")

        self.assertIn(
            "RENAME TO `archive`.`old_v`",
            _normalize_ws(sql.executed[0]),
        )

    def test_explicit_schema_kwarg_wins(self) -> None:
        v, sql = _view()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.rename("new_orders_v", schema_name="archive")

        self.assertIn(
            "RENAME TO `archive`.`new_orders_v`",
            _normalize_ws(sql.executed[0]),
        )

    def test_cross_catalog_rename_rejected(self) -> None:
        v, sql = _view()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError) as cm:
                v.rename("other.reporting.orders_v")
        self.assertIn("across catalogs", str(cm.exception))
        self.assertEqual(sql.executed, [])

    def test_noop_same_name(self) -> None:
        v, sql = _view()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.rename("orders_v")
        self.assertEqual(sql.executed, [])

    def test_empty_name_raises(self) -> None:
        v, sql = _view()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError):
                v.rename("")


class TestClone(unittest.TestCase):

    def setUp(self) -> None:
        _clear_singleton_cache()

    def test_default_clone_reuses_source_definition(self) -> None:
        v, sql = _view()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            cloned = v.clone("backup.orders_v_copy")

        norm = _normalize_ws(sql.executed[0])
        self.assertTrue(
            norm.startswith("CREATE VIEW `main`.`backup`.`orders_v_copy`"),
            norm,
        )
        self.assertIn(
            "AS SELECT id, amount FROM main.sales.orders",
            norm,
        )
        self.assertEqual(cloned.catalog_name, "main")
        self.assertEqual(cloned.schema_name, "backup")
        self.assertEqual(cloned.table_name, "orders_v_copy")

    def test_replace(self) -> None:
        v, sql = _view()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.clone("backup.orders_v_copy", replace=True)

        self.assertTrue(
            _normalize_ws(sql.executed[0]).startswith(
                "CREATE OR REPLACE VIEW `main`.`backup`.`orders_v_copy`"
            )
        )

    def test_if_not_exists(self) -> None:
        v, sql = _view()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.clone("backup.orders_v_copy", if_not_exists=True)

        self.assertTrue(
            _normalize_ws(sql.executed[0]).startswith(
                "CREATE VIEW IF NOT EXISTS `main`.`backup`.`orders_v_copy`"
            )
        )

    def test_replace_and_if_not_exists_conflict(self) -> None:
        v, sql = _view()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError):
                v.clone("backup.orders_v_copy", replace=True, if_not_exists=True)
        self.assertEqual(sql.executed, [])

    def test_clone_without_definition_raises(self) -> None:
        v, sql = _view(view_definition=None)
        object.__setattr__(v, "_infos", TableInfo(
            view_definition=None, table_type=TableType.VIEW,
        ))
        object.__setattr__(v, "_infos_fetched_at", time.time())
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError) as cm:
                v.clone("backup.orders_v_copy")
        self.assertIn("view_definition", str(cm.exception))

    def test_self_clone_rejected(self) -> None:
        v, sql = _view()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError):
                v.clone("main.reporting.orders_v")
        self.assertEqual(sql.executed, [])

    def test_missing_target_raises(self) -> None:
        v, sql = _view()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError):
                v.clone()
        self.assertEqual(sql.executed, [])

    def test_explicit_three_part_kwargs(self) -> None:
        v, sql = _view()
        with patch.object(Table, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.clone(
                catalog_name="other",
                schema_name="archive",
                table_name="orders_v_copy",
            )

        self.assertIn(
            "CREATE VIEW `other`.`archive`.`orders_v_copy`",
            _normalize_ws(sql.executed[0]),
        )


if __name__ == "__main__":
    unittest.main()
