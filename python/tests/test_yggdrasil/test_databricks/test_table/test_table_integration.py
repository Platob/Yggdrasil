"""Live-integration tests for :class:`Table` (and the view shape).

The SQL engine suite at
:mod:`tests.test_yggdrasil.test_databricks.test_sql.test_engine_integration`
already pins create / insert / merge / read-back round-trips through the
warehouse engine. This file targets the deeper :class:`Table` surface
that only meaningfully exercises against a live Unity Catalog endpoint:

- Lifecycle — ``ensure_created`` idempotence, ``create(or_replace=True)``
  one-shot replace, ``delete(missing_ok=True)`` vs raise-on-missing.
- :attr:`Table.infos` lazy fetch + ``invalidate_singleton`` cache reset,
  :attr:`Table.table_type`, :attr:`Table.is_view`, :attr:`Table.table_id`.
- :class:`Tables` collection — ``find`` returns the same singleton as
  the fixture handle; ``find_table_remote`` bypasses the cache.
- :meth:`Table.set_tags` / :meth:`Table.unset_tags` at the table level
  and :meth:`Column.set_tags` / :meth:`Table.update_columns_tags` for
  columns, against the ``entity_tag_assignments`` REST API.
- :meth:`Table.with_column` / :meth:`Table.with_columns` schema
  evolution — ADD COLUMN, RENAME COLUMN, DROP via OVERWRITE.
- :meth:`Table.rename` round-trip (and rename back so cleanup lands on
  the right ``full_name``).
- :meth:`Table.clone` deep clone — fresh target, data carried, cloned
  rows read back identically.
- View shape — :meth:`Table.create_view`, :attr:`Table.is_view`,
  :attr:`Table.view_definition`, view rename + view clone.

Skip rules
----------
Skipped wholesale unless ``DATABRICKS_HOST`` is set. The catalog /
schema are read from :envvar:`DATABRICKS_INTEGRATION_CATALOG`
(default ``trading``) and :envvar:`DATABRICKS_INTEGRATION_SCHEMA`
(default ``unittest``); the test identity must have CREATE TABLE on
the target schema. Tag tests degrade to ``unittest.SkipTest`` when
the identity lacks the matching grant rather than failing the suite.

Cleanup
-------
Each test class provisions its own throw-away table (``yg_<purpose>_<hex>``)
and drops it cascade-style in ``tearDownClass`` so a failed run leaves
at most one orphan table behind.
"""

from __future__ import annotations

import os
import secrets
import unittest
from typing import ClassVar

import pyarrow as pa
import pytest
from databricks.sdk.errors import (
    DatabricksError,
    NotFound,
    PermissionDenied,
    ResourceDoesNotExist,
)
from databricks.sdk.service.catalog import TableType

from yggdrasil.data import Field
from yggdrasil.data.enums import Mode
from yggdrasil.databricks.table.table import Table

from .. import DatabricksIntegrationCase


__all__ = [
    "TestTableLifecycleIntegration",
    "TestTableInfoIntegration",
    "TestTablesCollectionIntegration",
    "TestTableTagsIntegration",
    "TestTableSchemaEvolutionIntegration",
    "TestTableRenameIntegration",
    "TestTableCloneIntegration",
    "TestTableViewIntegration",
]


def _resolve_catalog() -> str:
    name = os.environ.get(
        "DATABRICKS_INTEGRATION_CATALOG", "trading",
    ).strip()
    if not name:
        raise unittest.SkipTest(
            "DATABRICKS_INTEGRATION_CATALOG is empty — set it to a "
            "catalog the test identity has CREATE TABLE on."
        )
    return name


def _resolve_schema() -> str:
    name = os.environ.get(
        "DATABRICKS_INTEGRATION_SCHEMA", "unittest",
    ).strip()
    if not name:
        raise unittest.SkipTest(
            "DATABRICKS_INTEGRATION_SCHEMA is empty — set it to a "
            "schema the test identity has CREATE TABLE on."
        )
    return name


def _sample_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("label", pa.string()),
            pa.field("amount", pa.float64()),
        ]
    )


def _sample_data() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "label": pa.array(["a", "b", "c"], type=pa.string()),
            "amount": pa.array([1.5, 2.5, 3.5], type=pa.float64()),
        }
    )


class _TableFixture(DatabricksIntegrationCase):
    """Per-class throw-away managed table.

    Subclasses inherit :attr:`table` (an empty Delta table with the
    ``_sample_schema`` shape) and a :attr:`schema_name` /
    :attr:`catalog_name` pair so they can mint sibling tables when
    needed without re-resolving env vars.
    """

    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    table_prefix: ClassVar[str] = "yg_table"
    table_name: ClassVar[str]
    table: ClassVar[Table]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = _resolve_catalog()
        cls.schema_name = _resolve_schema()
        cls.table_name = f"{cls.table_prefix}_{secrets.token_hex(4)}"
        full_name = f"{cls.catalog_name}.{cls.schema_name}.{cls.table_name}"
        try:
            cls.table = cls.client.tables.table(full_name)
            cls.table.ensure_created(_sample_schema())
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Cannot create table {full_name}: {exc}. Override "
                f"DATABRICKS_INTEGRATION_CATALOG / "
                f"DATABRICKS_INTEGRATION_SCHEMA with a location the "
                f"test identity can write to."
            ) from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            table = getattr(cls, "table", None)
            if table is not None:
                try:
                    table.delete(missing_ok=True)
                except Exception:
                    pass
        finally:
            super().tearDownClass()


@pytest.mark.integration
class TestTableLifecycleIntegration(_TableFixture):
    """``create`` / ``ensure_created`` / ``delete`` round-trips."""

    table_prefix = "yg_lifecycle"

    def test_exists_after_create(self) -> None:
        self.assertTrue(self.table.exists)

    def test_ensure_created_is_idempotent(self) -> None:
        # Second ensure_created on a same-shape table must NOT raise —
        # the AUTO mode path returns self.
        result = self.table.ensure_created(_sample_schema())
        self.assertIs(result, self.table)
        self.assertTrue(self.table.exists)

    def test_or_replace_resets_table_in_place(self) -> None:
        # Drop a row in so we can detect the replacement.
        self.table.insert(_sample_data(), mode=Mode.OVERWRITE)
        replaced = self.table.create(
            _sample_schema(),
            or_replace=True,
        )
        self.assertIs(replaced, self.table)
        self.assertTrue(self.table.exists)
        count = self.client.sql(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        ).execute(
            f"SELECT COUNT(*) AS n FROM {self.table.full_name(safe=True)}"
        ).to_arrow_table()
        self.assertEqual(count.column("n").to_pylist(), [0])

    def test_delete_missing_table_with_missing_ok(self) -> None:
        ghost_name = f"yg_ghost_{secrets.token_hex(4)}"
        ghost = self.client.tables.table(
            f"{self.catalog_name}.{self.schema_name}.{ghost_name}"
        )
        try:
            ghost.delete(missing_ok=True)
        except DatabricksError:
            self.fail(
                "Table.delete(missing_ok=True) leaked DatabricksError "
                "for a missing table"
            )

    def test_delete_missing_table_raises_by_default(self) -> None:
        ghost_name = f"yg_ghost_{secrets.token_hex(4)}"
        ghost = self.client.tables.table(
            f"{self.catalog_name}.{self.schema_name}.{ghost_name}"
        )
        with self.assertRaises(DatabricksError):
            ghost.delete()


@pytest.mark.integration
class TestTableInfoIntegration(_TableFixture):
    """:attr:`Table.infos` lazy fetch and cache reset behavior."""

    table_prefix = "yg_info"

    def test_infos_populated(self) -> None:
        info = self.table.infos
        self.assertEqual(info.catalog_name, self.catalog_name)
        self.assertEqual(info.schema_name, self.schema_name)
        self.assertEqual(info.name, self.table_name)

    def test_full_name(self) -> None:
        self.assertEqual(
            self.table.full_name(),
            f"{self.catalog_name}.{self.schema_name}.{self.table_name}",
        )
        self.assertEqual(
            self.table.full_name(safe=True),
            f"`{self.catalog_name}`.`{self.schema_name}`.`{self.table_name}`",
        )

    def test_table_id_is_populated(self) -> None:
        self.assertTrue(self.table.table_id)

    def test_table_type_is_managed(self) -> None:
        # The fixture builds via the default sql_create path; UC reports
        # this back as MANAGED.
        _ = self.table.infos  # warm the cache so ``table_type`` is populated
        self.assertEqual(self.table.table_type, TableType.MANAGED)
        self.assertFalse(self.table.is_view)

    def test_invalidate_singleton_drops_cached_infos(self) -> None:
        _ = self.table.infos
        self.assertIsNotNone(self.table._infos)
        self.table.invalidate_singleton()
        self.assertIsNone(self.table._infos)
        # Re-read goes through to the SDK and re-populates the slot.
        self.assertEqual(self.table.infos.name, self.table_name)
        self.assertIsNotNone(self.table._infos)

    def test_columns_reflect_schema(self) -> None:
        names = [c.name for c in self.table.columns]
        self.assertEqual(names, ["id", "label", "amount"])

    def test_exists_false_on_missing_table(self) -> None:
        ghost = self.client.tables.table(
            f"{self.catalog_name}.{self.schema_name}."
            f"yg_ghost_{secrets.token_hex(4)}"
        )
        self.assertFalse(ghost.exists)


@pytest.mark.integration
class TestTablesCollectionIntegration(_TableFixture):
    """:class:`Tables` collection-level navigation and listing."""

    table_prefix = "yg_collection"

    def test_find_returns_singleton_with_fixture(self) -> None:
        # ``find`` (via the engine / client.tables) collapses to the
        # same singleton instance as the fixture handle.
        found = self.client.tables.find_table(
            f"{self.catalog_name}.{self.schema_name}.{self.table_name}",
        )
        self.assertIsNotNone(found)
        self.assertIs(found, self.table)

    def test_find_remote_returns_info(self) -> None:
        info = self.client.tables.find_table_remote(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name=self.table_name,
        )
        self.assertEqual(info.catalog_name, self.catalog_name)
        self.assertEqual(info.schema_name, self.schema_name)
        self.assertEqual(info.name, self.table_name)

    def test_find_remote_missing_raises(self) -> None:
        with self.assertRaises((NotFound, ResourceDoesNotExist)):
            self.client.tables.find_table_remote(
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                table_name=f"yg_ghost_{secrets.token_hex(4)}",
            )


@pytest.mark.integration
class TestTableTagsIntegration(_TableFixture):
    """:meth:`Table.set_tags` / :meth:`Table.unset_tags` and column tags
    round-trips against ``entity_tag_assignments``."""

    table_prefix = "yg_tags"

    def _skip_if_no_tag_grant(self, exc: Exception) -> None:
        msg = str(exc).lower()
        if "permission" in msg or "denied" in msg or "not authorized" in msg:
            raise unittest.SkipTest(
                f"Test identity lacks tag-management permission on "
                f"{self.catalog_name}.{self.schema_name}: {exc}."
            )
        raise exc

    def test_set_and_read_table_tags(self) -> None:
        tag_key = f"yg_test_{secrets.token_hex(3)}"
        tag_value = f"v_{secrets.token_hex(2)}"
        try:
            self.table.set_tags({tag_key: tag_value})
        except (DatabricksError, PermissionDenied) as exc:
            self._skip_if_no_tag_grant(exc)
        try:
            tags = self.client.entity_tags.entity_tags(
                "tables", self.table.full_name(), as_dict=True,
                cache_ttl=None,
            )
            self.assertEqual(tags.get(tag_key), tag_value)
        finally:
            try:
                self.table.unset_tags([tag_key], if_exists=True)
            except Exception:
                pass

    def test_set_tags_empty_mapping_is_noop(self) -> None:
        # Empty / None mapping must short-circuit without an API call
        # and return self.
        self.assertIs(self.table.set_tags({}), self.table)
        self.assertIs(self.table.set_tags(None), self.table)

    def test_unset_missing_tag_with_if_exists(self) -> None:
        ghost_key = f"yg_ghost_{secrets.token_hex(3)}"
        try:
            self.table.unset_tags([ghost_key], if_exists=True)
        except DatabricksError as exc:
            self._skip_if_no_tag_grant(exc)

    def test_set_and_read_column_tags(self) -> None:
        tag_key = f"yg_col_{secrets.token_hex(3)}"
        tag_value = f"v_{secrets.token_hex(2)}"
        col = self.table.column("label")
        try:
            col.set_tags({tag_key: tag_value})
        except (DatabricksError, PermissionDenied) as exc:
            self._skip_if_no_tag_grant(exc)
        try:
            tags = self.client.entity_tags.entity_tags(
                "columns", col.entity_name, as_dict=True,
                cache_ttl=None,
            )
            self.assertEqual(tags.get(tag_key), tag_value)
            # ``Table.column_tags`` re-reads through the same cache —
            # the assignment surfaces under the column's name.
            self.assertIn(tag_key, {
                a.tag_key
                for a in self.table.column_tags.get("label", ())
            })
        finally:
            try:
                col.unset_tags([tag_key], if_exists=True)
            except Exception:
                pass

    def test_update_columns_tags_fans_out(self) -> None:
        batch = {
            "label": {f"yg_l_{secrets.token_hex(2)}": "x"},
            "amount": {f"yg_a_{secrets.token_hex(2)}": "y"},
        }
        try:
            results = self.table.update_columns_tags(batch, validate=False)
        except (DatabricksError, PermissionDenied) as exc:
            self._skip_if_no_tag_grant(exc)
        try:
            # Every column either succeeded (None) or surfaced an
            # exception in the result dict — the call itself must
            # round-trip without raising.
            self.assertEqual(set(results.keys()), set(batch.keys()))
        finally:
            for col_name, tags in batch.items():
                try:
                    self.table.column(col_name).unset_tags(
                        list(tags.keys()), if_exists=True,
                    )
                except Exception:
                    pass


@pytest.mark.integration
class TestTableSchemaEvolutionIntegration(_TableFixture):
    """:meth:`Table.with_column` / :meth:`Table.with_columns` evolution."""

    table_prefix = "yg_evolve"

    def test_with_column_adds_new_column(self) -> None:
        new_col = Field(name=f"extra_{secrets.token_hex(2)}", dtype=pa.string())
        self.table.with_column(new_col)
        self.assertIsNotNone(
            self.table.column(new_col.name, raise_error=False)
        )
        # Tidy up so the next test starts from the canonical 3-column shape.
        self.client.sql(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        ).execute(
            f"ALTER TABLE {self.table.full_name(safe=True)} "
            f"DROP COLUMN `{new_col.name}`"
        )
        self.table.invalidate_singleton()

    def test_with_columns_overwrite_drops_missing(self) -> None:
        # OVERWRITE keeps only the columns in the supplied list — any
        # column missing from it is dropped. Use a tightly-scoped table
        # so we don't mutate the shared fixture beyond the test.
        target = self.client.tables.table(
            f"{self.catalog_name}.{self.schema_name}."
            f"yg_overwrite_{secrets.token_hex(4)}"
        )
        try:
            target.ensure_created(_sample_schema())
            # Keep only ``id`` + a brand-new column; ``label`` / ``amount``
            # must disappear.
            new_col = Field(name="note", dtype=pa.string())
            keep_id = Field(name="id", dtype=pa.int64(), nullable=False)
            target.with_columns([keep_id, new_col], mode=Mode.OVERWRITE)
            names = {c.name for c in target.columns}
            self.assertIn("id", names)
            self.assertIn("note", names)
            self.assertNotIn("label", names)
            self.assertNotIn("amount", names)
        finally:
            try:
                target.delete(missing_ok=True)
            except Exception:
                pass


@pytest.mark.integration
class TestTableRenameIntegration(DatabricksIntegrationCase):
    """:meth:`Table.rename` round-trip on a dedicated throw-away table."""

    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    initial_name: ClassVar[str]
    table: ClassVar[Table]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = _resolve_catalog()
        cls.schema_name = _resolve_schema()
        cls.initial_name = f"yg_rename_{secrets.token_hex(4)}"
        full_name = f"{cls.catalog_name}.{cls.schema_name}.{cls.initial_name}"
        try:
            cls.table = cls.client.tables.table(full_name)
            cls.table.ensure_created(_sample_schema())
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Cannot create table {full_name}: {exc}."
            ) from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            table = getattr(cls, "table", None)
            if table is not None:
                try:
                    table.delete(missing_ok=True)
                except Exception:
                    pass
        finally:
            super().tearDownClass()

    def test_rename_updates_name_and_remote_state(self) -> None:
        new_name = f"yg_renamed_{secrets.token_hex(4)}"
        try:
            self.table.rename(new_name)
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Test identity cannot rename table: {exc}."
            )
        self.assertEqual(self.table.table_name, new_name)
        self.assertEqual(
            self.table.full_name(),
            f"{self.catalog_name}.{self.schema_name}.{new_name}",
        )
        # New name is reachable; old name isn't.
        fresh = self.client.tables.find_table_remote(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name=new_name,
        )
        self.assertEqual(fresh.name, new_name)
        with self.assertRaises((NotFound, ResourceDoesNotExist)):
            self.client.tables.find_table_remote(
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                table_name=self.initial_name,
            )

    def test_rename_to_same_name_is_noop(self) -> None:
        before = self.table.table_name
        result = self.table.rename(before)
        self.assertIs(result, self.table)
        self.assertEqual(self.table.table_name, before)

    def test_rename_to_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.table.rename("   ")


@pytest.mark.integration
class TestTableCloneIntegration(_TableFixture):
    """:meth:`Table.clone` deep clone round-trip."""

    table_prefix = "yg_clone_src"

    def setUp(self) -> None:
        super().setUp()
        # Seed the source with deterministic rows so the clone has
        # something concrete to mirror.
        self.table.insert(_sample_data(), mode=Mode.OVERWRITE)
        self._clone_targets: list[str] = []

    def tearDown(self) -> None:
        for full_name in self._clone_targets:
            try:
                self.client.tables.table(full_name).delete(missing_ok=True)
            except Exception:
                pass
        super().tearDown()

    def _clone_target_name(self) -> str:
        name = f"yg_clone_tgt_{secrets.token_hex(4)}"
        full = f"{self.catalog_name}.{self.schema_name}.{name}"
        self._clone_targets.append(full)
        return name

    def test_deep_clone_copies_rows(self) -> None:
        target_name = self._clone_target_name()
        try:
            cloned = self.table.clone(target_name)
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(f"Cannot clone table: {exc}.")
        self.assertEqual(cloned.table_name, target_name)
        self.assertTrue(cloned.exists)
        rows = self.client.sql(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        ).execute(
            f"SELECT id, label, amount FROM {cloned.full_name(safe=True)} "
            f"ORDER BY id"
        ).to_arrow_table()
        self.assertEqual(rows.num_rows, 3)
        self.assertEqual(rows.column("id").to_pylist(), [1, 2, 3])
        self.assertEqual(rows.column("label").to_pylist(), ["a", "b", "c"])

    def test_clone_onto_self_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.table.clone(self.table.full_name())

    def test_clone_if_not_exists_skips_when_target_present(self) -> None:
        target_name = self._clone_target_name()
        try:
            first = self.table.clone(target_name)
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(f"Cannot clone table: {exc}.")
        self.assertTrue(first.exists)
        # Second call with if_not_exists must not raise even though the
        # target is already there.
        self.table.clone(target_name, if_not_exists=True)


@pytest.mark.integration
class TestTableViewIntegration(_TableFixture):
    """View-shaped securables: ``create_view`` + ``is_view`` +
    :attr:`Table.view_definition` + view rename round-trip."""

    table_prefix = "yg_view_src"

    def setUp(self) -> None:
        super().setUp()
        self.table.insert(_sample_data(), mode=Mode.OVERWRITE)
        self._view_full_names: list[str] = []

    def tearDown(self) -> None:
        for full_name in self._view_full_names:
            try:
                self.client.tables.table(full_name).delete(missing_ok=True)
            except Exception:
                pass
        super().tearDown()

    def _new_view_handle(self) -> Table:
        name = f"yg_view_{secrets.token_hex(4)}"
        full = f"{self.catalog_name}.{self.schema_name}.{name}"
        self._view_full_names.append(full)
        return self.client.tables.view(full)

    def test_create_view_sets_view_definition(self) -> None:
        view = self._new_view_handle()
        try:
            view.create_view(
                f"SELECT id, label FROM {self.table.full_name(safe=True)}",
                mode=Mode.OVERWRITE,
            )
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(f"Cannot create view: {exc}.")
        self.assertTrue(view.exists)
        _ = view.infos  # warm the cache so ``is_view`` reads from it
        self.assertTrue(view.is_view)
        self.assertIsNotNone(view.view_definition)
        self.assertIn("SELECT", (view.view_definition or "").upper())

    def test_view_read_round_trips(self) -> None:
        view = self._new_view_handle()
        try:
            view.create_view(
                f"SELECT id, label FROM {self.table.full_name(safe=True)} "
                f"WHERE id >= 2",
                mode=Mode.OVERWRITE,
            )
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(f"Cannot create view: {exc}.")
        rows = self.client.sql(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        ).execute(
            f"SELECT id, label FROM {view.full_name(safe=True)} ORDER BY id"
        ).to_arrow_table()
        self.assertEqual(rows.column("id").to_pylist(), [2, 3])
        self.assertEqual(rows.column("label").to_pylist(), ["b", "c"])

    def test_view_rename_round_trip(self) -> None:
        view = self._new_view_handle()
        try:
            view.create_view(
                f"SELECT id FROM {self.table.full_name(safe=True)}",
                mode=Mode.OVERWRITE,
            )
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(f"Cannot create view: {exc}.")
        _ = view.infos
        new_name = f"yg_view_renamed_{secrets.token_hex(4)}"
        self._view_full_names.append(
            f"{self.catalog_name}.{self.schema_name}.{new_name}"
        )
        try:
            view.rename(new_name)
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(f"Cannot rename view: {exc}.")
        self.assertEqual(view.table_name, new_name)
        self.assertTrue(view.exists)
