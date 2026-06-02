"""Live-integration tests for :meth:`UCSchema.clone` — parallel child clone.

Exercises the schema-level clone against a real Unity Catalog endpoint:

- **Create** — every child (tables + a view) lands in a fresh target schema.
- **Skip vs overwrite** — a second clone under the default ``IGNORE`` skips the
  now-present children; ``OVERWRITE`` recreates them.
- **Kind drift (both directions)** — when a target already exists with the
  *other* shape, the stale target is dropped and recreated as the source's
  current kind: a target **view** is replaced by the source **table**
  (view→table) and a target **table** is replaced by the source **view**
  (table→view). Delta can't cross-replace those, so this is the path the
  drop-then-recreate logic guards.
- **include_views=False** — clones only tables, leaving the view behind.

Gated by :class:`DatabricksIntegrationCase` (skipped unless ``DATABRICKS_HOST``
is set) and needs a SQL warehouse + CREATE on the shared
``trading_tgp_dev``.``ygg_integration`` catalog. Source content lives in one
read-only scratch schema; each test mints its own throw-away target schema, and
all scratch schemas are dropped in ``tearDownClass``.
"""
from __future__ import annotations

import unittest
from typing import ClassVar

import pytest

from yggdrasil.enums.mode import Mode

from . import DatabricksIntegrationCase


@pytest.mark.integration
class TestSchemaCloneIntegration(DatabricksIntegrationCase):
    src: ClassVar
    _scratch: ClassVar[list]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._scratch = []
        # One read-only source schema shared by every test: two managed tables
        # and a view over the first.
        cls.src = cls._track(cls.scratch_schema())
        cat, sch = cls.INTEGRATION_CATALOG, cls.src.schema_name
        run = cls.client.sql.execute
        run(f"CREATE TABLE {cat}.{sch}.t_a AS "
            f"SELECT * FROM VALUES (1, 'x'), (2, 'y') AS t(id, label)")
        run(f"CREATE TABLE {cat}.{sch}.t_b AS "
            f"SELECT * FROM VALUES (3, 'z') AS t(id, label)")
        run(f"CREATE VIEW {cat}.{sch}.v_c AS SELECT id FROM {cat}.{sch}.t_a")

    @classmethod
    def tearDownClass(cls) -> None:
        for schema in getattr(cls, "_scratch", []):
            cls.safe_drop_schema(schema)
        super().tearDownClass()

    # -- helpers -------------------------------------------------------
    @classmethod
    def _track(cls, schema):
        cls._scratch.append(schema)
        return schema

    @classmethod
    def _new_target(cls):
        """A fresh, tracked throw-away target schema."""
        return cls._track(cls.scratch_schema())

    def _kind(self, schema: str, name: str) -> "str | None":
        """``'table'`` / ``'view'`` / ``None`` for a child — read cache-free
        from ``information_schema`` so a just-dropped/recreated kind is exact."""
        rows = self.client.sql.execute(
            f"SELECT table_type FROM {self.INTEGRATION_CATALOG}."
            f"information_schema.tables WHERE table_schema = '{schema}' "
            f"AND table_name = '{name}'"
        ).to_arrow_table()
        if rows.num_rows == 0:
            return None
        table_type = (rows.column("table_type").to_pylist()[0] or "").upper()
        return "view" if "VIEW" in table_type else "table"

    def _count(self, schema: str, name: str) -> int:
        rows = self.client.sql.execute(
            f"SELECT count(*) AS n FROM {self.INTEGRATION_CATALOG}.{schema}.{name}"
        ).to_arrow_table()
        return int(rows.column("n").to_pylist()[0])

    # -- tests ---------------------------------------------------------
    def test_clone_creates_all_children(self):
        dst = self._new_target()
        d = dst.schema_name
        result = self.src.clone(d)  # bare name → same (integration) catalog

        self.assertEqual(set(result), {"t_a", "t_b", "v_c"})
        self.assertTrue(all(v == "created" for v in result.values()), result)
        self.assertEqual(self._kind(d, "t_a"), "table")
        self.assertEqual(self._kind(d, "t_b"), "table")
        self.assertEqual(self._kind(d, "v_c"), "view")
        # data carried over (deep clone) + the view resolves
        self.assertEqual(self._count(d, "t_a"), 2)
        self.assertEqual(self._count(d, "v_c"), 2)

    def test_clone_skips_then_overwrites(self):
        dst = self._new_target()
        d = dst.schema_name
        self.src.clone(d)  # first pass — all created

        again = self.src.clone(d)  # default IGNORE → all present → skipped
        self.assertTrue(all(v == "skipped" for v in again.values()), again)

        overwritten = self.src.clone(d, mode=Mode.OVERWRITE)
        self.assertTrue(all(v == "created" for v in overwritten.values()), overwritten)

    def test_clone_view_to_table_drift(self):
        # Target 't_a' pre-exists as a VIEW; the source 't_a' is a TABLE. The
        # mismatched kind can't be cross-replaced, so the stale view is dropped
        # and recreated as a table — even under the default IGNORE policy.
        dst = self._new_target()
        d = dst.schema_name
        self.client.sql.execute(
            f"CREATE VIEW {self.INTEGRATION_CATALOG}.{d}.t_a AS "
            f"SELECT 99 AS id, 'old' AS label"
        )
        self.assertEqual(self._kind(d, "t_a"), "view")

        result = self.src.clone(d)  # default IGNORE
        self.assertEqual(result["t_a"], "created")     # recreated, not skipped
        self.assertEqual(self._kind(d, "t_a"), "table")
        self.assertEqual(self._count(d, "t_a"), 2)     # source rows, not the old view

    def test_clone_table_to_view_drift(self):
        # Target 'v_c' pre-exists as a TABLE; the source 'v_c' is a VIEW.
        dst = self._new_target()
        d = dst.schema_name
        self.client.sql.execute(
            f"CREATE TABLE {self.INTEGRATION_CATALOG}.{d}.v_c AS SELECT 7 AS id"
        )
        self.assertEqual(self._kind(d, "v_c"), "table")

        result = self.src.clone(d)
        self.assertEqual(result["v_c"], "created")
        self.assertEqual(self._kind(d, "v_c"), "view")

    def test_clone_error_if_exists_does_not_drop_drifted_target(self):
        # ERROR_IF_EXISTS must surface a clash as a failure and leave the
        # existing (drifted) target untouched — never silently drop it.
        dst = self._new_target()
        d = dst.schema_name
        self.client.sql.execute(
            f"CREATE VIEW {self.INTEGRATION_CATALOG}.{d}.t_a AS SELECT 1 AS id, 'v' AS label"
        )
        result = self.src.clone(d, mode=Mode.ERROR_IF_EXISTS)
        self.assertTrue(result["t_a"].startswith("failed: "), result)
        self.assertEqual(self._kind(d, "t_a"), "view")  # left as the pre-existing view

    def test_clone_tables_only_excludes_views(self):
        dst = self._new_target()
        d = dst.schema_name
        result = self.src.clone(d, include_views=False)
        self.assertEqual(set(result), {"t_a", "t_b"})
        self.assertIsNone(self._kind(d, "v_c"))  # the view was not cloned


if __name__ == "__main__":
    unittest.main()
