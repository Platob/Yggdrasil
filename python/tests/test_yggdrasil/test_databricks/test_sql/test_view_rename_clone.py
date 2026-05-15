"""SQL emitted by :meth:`View.rename` and :meth:`View.clone`.

These pin the DDL surface without touching a real Databricks workspace
— the ``sql`` property is patched onto a recording stub so the SQL
text the methods would submit is captured directly.
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from databricks.sdk.service.catalog import TableInfo

from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.view.view import View
from yggdrasil.databricks.view.views import Views


def _normalize_ws(sql: str) -> str:
    return " ".join(sql.split())


class _RecordingSql:
    """Stand-in for :class:`View.sql` that records executed statements."""

    def __init__(self, client: DatabricksClient, catalog_name: str, schema_name: str):
        self.client = client
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.executed: list[str] = []

    def execute(self, statement: str, *args, **kwargs) -> None:
        self.executed.append(statement)


def _view(
    catalog: str = "main",
    schema: str = "reporting",
    name: str = "orders_v",
    view_definition: str | None = "SELECT id, amount FROM main.sales.orders",
) -> tuple[View, _RecordingSql]:
    client = DatabricksClient(host="https://ws.example.com")
    service = Views(client=client, catalog_name=catalog, schema_name=schema)
    v = View(service=service, catalog_name=catalog, schema_name=schema, view_name=name)
    sql = _RecordingSql(client=client, catalog_name=catalog, schema_name=schema)
    object.__setattr__(v, "_reset_cache", lambda invalidate_cache=False: None)
    if view_definition is not None:
        infos = TableInfo(view_definition=view_definition)
        object.__setattr__(v, "_infos", infos)
        object.__setattr__(v, "_infos_fetched_at", time.time())
    return v, sql


class TestRename(unittest.TestCase):

    def test_simple_rename_within_schema(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.rename("new_orders_v")

        self.assertEqual(len(sql.executed), 1)
        self.assertEqual(
            _normalize_ws(sql.executed[0]),
            "ALTER VIEW `main`.`reporting`.`orders_v` RENAME TO `new_orders_v`",
        )
        self.assertEqual(v.view_name, "new_orders_v")
        self.assertEqual(v.schema_name, "reporting")

    def test_cross_schema_rename(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.rename("archive.old_orders_v")

        self.assertEqual(
            _normalize_ws(sql.executed[0]),
            "ALTER VIEW `main`.`reporting`.`orders_v` RENAME TO"
            " `archive`.`old_orders_v`",
        )
        self.assertEqual(v.schema_name, "archive")
        self.assertEqual(v.view_name, "old_orders_v")

    def test_three_part_target_same_catalog(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.rename("main.archive.old_v")

        self.assertIn(
            "RENAME TO `archive`.`old_v`",
            _normalize_ws(sql.executed[0]),
        )

    def test_explicit_schema_kwarg_wins(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.rename("new_orders_v", schema_name="archive")

        self.assertIn(
            "RENAME TO `archive`.`new_orders_v`",
            _normalize_ws(sql.executed[0]),
        )

    def test_cross_catalog_rename_rejected(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError) as cm:
                v.rename("other.reporting.orders_v")
        self.assertIn("across catalogs", str(cm.exception))
        self.assertEqual(sql.executed, [])

    def test_noop_same_name(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.rename("orders_v")
        self.assertEqual(sql.executed, [])

    def test_empty_name_raises(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError):
                v.rename("")


class TestClone(unittest.TestCase):

    def test_default_clone_reuses_source_definition(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
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
        self.assertEqual(cloned.view_name, "orders_v_copy")

    def test_replace(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.clone("backup.orders_v_copy", replace=True)

        self.assertTrue(
            _normalize_ws(sql.executed[0]).startswith(
                "CREATE OR REPLACE VIEW `main`.`backup`.`orders_v_copy`"
            )
        )

    def test_if_not_exists(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.clone("backup.orders_v_copy", if_not_exists=True)

        self.assertTrue(
            _normalize_ws(sql.executed[0]).startswith(
                "CREATE VIEW IF NOT EXISTS `main`.`backup`.`orders_v_copy`"
            )
        )

    def test_replace_and_if_not_exists_conflict(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError):
                v.clone("backup.orders_v_copy", replace=True, if_not_exists=True)
        self.assertEqual(sql.executed, [])

    def test_query_override(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.clone(
                "backup.orders_v_copy",
                query="SELECT id FROM main.sales.orders WHERE status = 'OPEN'",
            )

        norm = _normalize_ws(sql.executed[0])
        self.assertIn(
            "AS SELECT id FROM main.sales.orders WHERE status = 'OPEN'",
            norm,
        )
        # Original view_definition is no longer in the SQL.
        self.assertNotIn("amount", norm)

    def test_clone_without_definition_or_query_raises(self) -> None:
        v, sql = _view(view_definition=None)
        # Drop cached infos so view_definition access would try to fetch — but
        # the clone path reads ``self.view_definition`` directly, returning
        # ``None`` when ``_infos`` is missing a definition. To exercise the
        # error path, leave _infos as a TableInfo with no view_definition.
        object.__setattr__(v, "_infos", TableInfo(view_definition=None))
        object.__setattr__(v, "_infos_fetched_at", time.time())
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError) as cm:
                v.clone("backup.orders_v_copy")
        self.assertIn("view_definition", str(cm.exception))

    def test_comment_and_properties(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.clone(
                "backup.orders_v_copy",
                comment="Snapshot of reporting.orders_v",
                properties={"owner": "data-eng"},
            )

        norm = _normalize_ws(sql.executed[0])
        self.assertIn("COMMENT 'Snapshot of reporting.orders_v'", norm)
        self.assertIn("TBLPROPERTIES ('owner' = 'data-eng')", norm)

    def test_target_can_be_view_instance(self) -> None:
        v, sql = _view()
        target = View(
            service=v.service,
            catalog_name="main",
            schema_name="backup",
            view_name="orders_v_copy",
        )
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            cloned = v.clone(target)

        self.assertIn(
            "CREATE VIEW `main`.`backup`.`orders_v_copy`",
            _normalize_ws(sql.executed[0]),
        )
        self.assertEqual(cloned.full_name(), "main.backup.orders_v_copy")

    def test_self_clone_rejected(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError):
                v.clone("main.reporting.orders_v")
        self.assertEqual(sql.executed, [])

    def test_missing_target_raises(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            with self.assertRaises(ValueError):
                v.clone()
        self.assertEqual(sql.executed, [])

    def test_explicit_three_part_kwargs(self) -> None:
        v, sql = _view()
        with patch.object(View, "sql", new_callable=lambda: property(lambda _s: sql)):
            v.clone(
                catalog_name="other",
                schema_name="archive",
                view_name="orders_v_copy",
            )

        self.assertIn(
            "CREATE VIEW `other`.`archive`.`orders_v_copy`",
            _normalize_ws(sql.executed[0]),
        )


if __name__ == "__main__":
    unittest.main()
