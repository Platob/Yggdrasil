"""Live-Postgres integration tests for the :class:`Schema` resource."""

from __future__ import annotations

import pytest

from yggdrasil.postgres.tests import PostgresTestCase

pytestmark = pytest.mark.postgres_integration


class TestSchemaLifecycle(PostgresTestCase):
    """``CREATE`` / ``DROP`` / ``RENAME SCHEMA`` against a live database."""

    def test_setup_schema_exists(self) -> None:
        # The base class created ``self.test_schema_name`` in setUp.
        self.assertTrue(
            self.engine.schema(self.test_schema_name).exists,
            f"setUp should have created {self.test_schema_name!r}",
        )

    def test_create_if_not_exists_is_idempotent(self) -> None:
        sch = self.engine.schema(self.test_schema_name)
        # Second call with if_not_exists=True should not raise.
        sch.create(if_not_exists=True)
        self.assertTrue(sch.exists)

    def test_rename_schema_roundtrip(self) -> None:
        original = f"{self.test_schema_name}_renamed_src"
        renamed = f"{self.test_schema_name}_renamed_dst"
        sch = self.engine.schema(original).create(if_not_exists=True)
        try:
            sch.rename(renamed)
            self.assertEqual(sch.schema_name, renamed)
            self.assertTrue(self.engine.schema(renamed).exists)
            self.assertFalse(self.engine.schema(original).exists)
        finally:
            self.engine.schema(renamed).delete(if_exists=True, cascade=True)
            self.engine.schema(original).delete(if_exists=True, cascade=True)

    def test_drop_with_cascade_removes_tables(self) -> None:
        # Create a transient schema with a table inside, then drop with
        # cascade — tearing down the schema should also drop the table.
        scratch = f"{self.test_schema_name}_cascade"
        self.engine.schema(scratch).create(if_not_exists=True)
        try:
            self.engine.execute(
                f'CREATE TABLE "{scratch}"."t" (id int)',
                prefer_arrow=False,
            )
            self.engine.schema(scratch).delete(if_exists=True, cascade=True)
            self.assertFalse(self.engine.schema(scratch).exists)
        finally:
            self.engine.schema(scratch).delete(if_exists=True, cascade=True)


class TestSchemaListing(PostgresTestCase):
    """Walk the schemas / tables collections."""

    def test_per_test_schema_appears_in_list(self) -> None:
        names = {s.schema_name for s in self.engine.schemas.list()}
        self.assertIn(self.test_schema_name, names)
        # System schemas are excluded by default.
        self.assertNotIn("information_schema", names)
        self.assertFalse(any(n.startswith("pg_") for n in names))

    def test_include_system_returns_pg_catalog(self) -> None:
        names = {s.schema_name for s in self.engine.schemas.list(include_system=True)}
        self.assertIn("pg_catalog", names)
        self.assertIn("information_schema", names)

    def test_iter_tables_in_schema(self) -> None:
        # Empty schema → empty iterator.
        sch = self.engine.schema(self.test_schema_name)
        self.assertEqual(list(sch.tables()), [])
        # Add a table; it should appear.
        self.engine.execute(
            f'CREATE TABLE "{self.test_schema_name}"."alpha" (id int)',
            prefer_arrow=False,
        )
        self.engine.execute(
            f'CREATE TABLE "{self.test_schema_name}"."beta" (id int)',
            prefer_arrow=False,
        )
        names = sorted(t.table_name for t in sch.tables())
        self.assertEqual(names, ["alpha", "beta"])
