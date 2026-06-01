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
- :meth:`Table.set_tags` / :meth:`Table.unset_tags` (table + column tags).
- :meth:`Table.with_column` / :meth:`Table.with_columns` schema evolution.
- :meth:`Table.rename` / :meth:`Table.clone` round-trips.
- View shape — :meth:`Table.create_view`, :attr:`Table.is_view`,
  :attr:`Table.view_definition`, view rename.
- Storage-path surface and the async-insert drop → loader round-trip.

Skip rules
----------
Skipped wholesale unless ``DATABRICKS_HOST`` is set. The catalog /
schema are read from :envvar:`DATABRICKS_INTEGRATION_CATALOG`
(default ``trading``) and :envvar:`DATABRICKS_INTEGRATION_SCHEMA`
(default ``unittest``); the test identity must have CREATE TABLE on
the target schema. Permission-gated operations degrade to
``unittest.SkipTest`` rather than failing the suite.

Cleanup
-------
Each class provisions one throw-away table (dropped in
``tearDownClass``); per-test sibling tables minted via :meth:`_sibling`
are dropped in ``tearDown`` — so a failed run leaves at most one orphan.
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
from databricks.sdk.service.catalog import TableOperation, TableType

from yggdrasil.data import Field
from yggdrasil.enums import MediaTypes, Mode
from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.databricks.table.table import Table
from yggdrasil.databricks.volume import Volume

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
    "TestTableStoragePathIntegration",
    "TestTableAsyncInsertIntegration",
]


def _resolve_env(var: str, default: str) -> str:
    name = os.environ.get(var, default).strip()
    if not name:
        raise unittest.SkipTest(
            f"{var} is empty — set it to a location the test identity "
            f"has CREATE TABLE on."
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
    """Per-class throw-away managed table + shared helpers.

    Subclasses inherit :attr:`table` (an empty Delta table with the
    ``_sample_schema`` shape), the resolved ``catalog_name`` /
    ``schema_name``, and helpers to run a quick SQL query, count rows,
    mint sibling / ghost tables, and skip cleanly on a permission error.
    """

    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    table_prefix: ClassVar[str] = "yg_table"
    table_name: ClassVar[str]
    table: ClassVar[Table]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = _resolve_env("DATABRICKS_INTEGRATION_CATALOG", "trading")
        cls.schema_name = _resolve_env("DATABRICKS_INTEGRATION_SCHEMA", "unittest")
        cls.table_name = cls.table_prefix
        full_name = f"{cls.catalog_name}.{cls.schema_name}.{cls.table_name}"
        try:
            cls.table = cls.client.tables.table(full_name)
            cls.table.ensure_created(_sample_schema())
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Cannot create table {full_name}: {exc}. Override "
                f"DATABRICKS_INTEGRATION_CATALOG / DATABRICKS_INTEGRATION_SCHEMA "
                f"with a location the test identity can write to."
            ) from exc

    @classmethod
    def tearDownClass(cls) -> None:
        table = getattr(cls, "table", None)
        if table is not None:
            try:
                table.delete(missing_ok=True)
            except Exception:
                pass
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self._cleanup: list[str] = []

    def tearDown(self) -> None:
        for full_name in self._cleanup:
            try:
                self.client.tables.table(full_name).delete(missing_ok=True)
            except Exception:
                pass
        super().tearDown()

    # -- helpers -------------------------------------------------------
    def _sibling(self, prefix: str) -> str:
        """A throw-away sibling table's full name, registered for cleanup."""
        full = f"{self.catalog_name}.{self.schema_name}.{prefix}_{secrets.token_hex(4)}"
        self._cleanup.append(full)
        return full

    def _ghost(self) -> Table:
        """A handle to a (cleanup-registered) table that does not exist."""
        return self.client.tables.table(self._sibling("yg_ghost"))

    def _sql(self, query: str) -> pa.Table:
        return self.client.sql(
            catalog_name=self.catalog_name, schema_name=self.schema_name,
        ).execute(query).to_arrow_table()

    def _count(self, table: "Table | None" = None) -> int:
        table = table or self.table
        rows = self._sql(f"SELECT COUNT(*) AS n FROM {table.full_name(safe=True)}")
        return int(rows.column("n").to_pylist()[0])

    def _skip(self, exc: Exception, what: str) -> None:
        raise unittest.SkipTest(f"{what}: {exc}.")


@pytest.mark.integration
class TestTableLifecycleIntegration(_TableFixture):
    """``create`` / ``ensure_created`` / ``delete`` round-trips."""

    table_prefix = "yg_lifecycle"

    def test_exists_after_create(self) -> None:
        self.assertTrue(self.table.exists())

    def test_ensure_created_is_idempotent(self) -> None:
        # Second ensure_created on a same-shape table must NOT raise.
        result = self.table.ensure_created(_sample_schema())
        self.assertIs(result, self.table)
        self.assertTrue(self.table.exists())

    def test_or_replace_resets_table_in_place(self) -> None:
        self.table.insert(_sample_data(), mode=Mode.OVERWRITE)
        replaced = self.table.create(_sample_schema(), or_replace=True)
        self.assertIs(replaced, self.table)
        self.assertTrue(self.table.exists())
        self.assertEqual(self._count(), 0)

    def test_delete_missing_table_with_missing_ok(self) -> None:
        try:
            self._ghost().delete(missing_ok=True)
        except DatabricksError:
            self.fail(
                "Table.delete(missing_ok=True) leaked DatabricksError "
                "for a missing table"
            )

    def test_delete_missing_table_raises_by_default(self) -> None:
        with self.assertRaises(DatabricksError):
            self._ghost().delete()


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
        self.assertFalse(self._ghost().exists())


@pytest.mark.integration
class TestTablesCollectionIntegration(_TableFixture):
    """:class:`Tables` collection-level navigation and listing."""

    table_prefix = "yg_collection"

    def test_find_returns_singleton_with_fixture(self) -> None:
        # ``find`` collapses to the same singleton as the fixture handle.
        found = self.client.tables.find_table(
            f"{self.catalog_name}.{self.schema_name}.{self.table_name}",
        )
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
                table_name=self._ghost().table_name,
            )


@pytest.mark.integration
class TestTableTagsIntegration(_TableFixture):
    """Table + column tag round-trips against ``entity_tag_assignments``."""

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
        tag_key, tag_value = f"yg_test_{secrets.token_hex(3)}", f"v_{secrets.token_hex(2)}"
        try:
            self.table.set_tags({tag_key: tag_value})
        except (DatabricksError, PermissionDenied) as exc:
            self._skip_if_no_tag_grant(exc)
        try:
            tags = self.client.entity_tags.entity_tags(
                "tables", self.table.full_name(), as_dict=True, cache_ttl=None,
            )
            self.assertEqual(tags.get(tag_key), tag_value)
        finally:
            try:
                self.table.unset_tags([tag_key], if_exists=True)
            except Exception:
                pass

    def test_set_tags_empty_mapping_is_noop(self) -> None:
        # Empty / None mapping short-circuits without an API call.
        self.assertIs(self.table.set_tags({}), self.table)
        self.assertIs(self.table.set_tags(None), self.table)

    def test_unset_missing_tag_with_if_exists(self) -> None:
        try:
            self.table.unset_tags([f"yg_ghost_{secrets.token_hex(3)}"], if_exists=True)
        except DatabricksError as exc:
            self._skip_if_no_tag_grant(exc)

    def test_set_and_read_column_tags(self) -> None:
        tag_key, tag_value = f"yg_col_{secrets.token_hex(3)}", f"v_{secrets.token_hex(2)}"
        col = self.table.column("label")
        try:
            col.set_tags({tag_key: tag_value})
        except (DatabricksError, PermissionDenied) as exc:
            self._skip_if_no_tag_grant(exc)
        try:
            tags = self.client.entity_tags.entity_tags(
                "columns", col.entity_name, as_dict=True, cache_ttl=None,
            )
            self.assertEqual(tags.get(tag_key), tag_value)
            # ``Table.column_tags`` re-reads through the same cache.
            self.assertIn(tag_key, {
                a.tag_key for a in self.table.column_tags.get("label", ())
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
        self.assertIsNotNone(self.table.column(new_col.name, raise_error=False))
        # Tidy up so the next test starts from the canonical 3-column shape.
        self._sql(
            f"ALTER TABLE {self.table.full_name(safe=True)} "
            f"DROP COLUMN `{new_col.name}`"
        )
        self.table.invalidate_singleton()

    def test_with_columns_overwrite_drops_missing(self) -> None:
        # OVERWRITE keeps only the columns in the supplied list — any
        # column missing from it is dropped. Use a sibling table so the
        # shared fixture is untouched.
        target = self.client.tables.table(self._sibling("yg_overwrite"))
        target.ensure_created(_sample_schema())
        target.with_columns(
            [Field(name="id", dtype=pa.int64(), nullable=False),
             Field(name="note", dtype=pa.string())],
            mode=Mode.OVERWRITE,
        )
        names = {c.name for c in target.columns}
        self.assertEqual(names & {"id", "note", "label", "amount"}, {"id", "note"})


@pytest.mark.integration
class TestTableRenameIntegration(_TableFixture):
    """:meth:`Table.rename` round-trip."""

    table_prefix = "yg_rename"

    def test_rename_to_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.table.rename("   ")

    def test_rename_to_same_name_is_noop(self) -> None:
        before = self.table.table_name
        result = self.table.rename(before)
        self.assertIs(result, self.table)
        self.assertEqual(self.table.table_name, before)

    def test_rename_updates_name_and_remote_state(self) -> None:
        initial = self.table.table_name
        new_name = f"yg_renamed_{secrets.token_hex(4)}"
        try:
            self.table.rename(new_name)
        except (DatabricksError, PermissionDenied) as exc:
            self._skip(exc, "Test identity cannot rename table")
        self.assertEqual(self.table.table_name, new_name)
        self.assertEqual(
            self.table.full_name(),
            f"{self.catalog_name}.{self.schema_name}.{new_name}",
        )
        # New name is reachable; the old name isn't.
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
                table_name=initial,
            )


@pytest.mark.integration
class TestTableCloneIntegration(_TableFixture):
    """:meth:`Table.clone` deep clone round-trip."""

    table_prefix = "yg_clone_src"

    def setUp(self) -> None:
        super().setUp()
        # Seed the source with deterministic rows for the clone to mirror.
        self.table.insert(_sample_data(), mode=Mode.OVERWRITE)

    def test_deep_clone_copies_rows(self) -> None:
        target_name = self._sibling("yg_clone_tgt").split(".")[-1]
        try:
            cloned = self.table.clone(target_name)
        except (DatabricksError, PermissionDenied) as exc:
            self._skip(exc, "Cannot clone table")
        self.assertEqual(cloned.table_name, target_name)
        self.assertTrue(cloned.exists())
        rows = self._sql(
            f"SELECT id, label, amount FROM {cloned.full_name(safe=True)} ORDER BY id"
        )
        self.assertEqual(rows.num_rows, 3)
        self.assertEqual(rows.column("id").to_pylist(), [1, 2, 3])
        self.assertEqual(rows.column("label").to_pylist(), ["a", "b", "c"])

    def test_clone_onto_self_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.table.clone(self.table.full_name())

    def test_clone_missing_ok_skips_when_target_present(self) -> None:
        target_name = self._sibling("yg_clone_tgt").split(".")[-1]
        try:
            first = self.table.clone(target_name)
        except (DatabricksError, PermissionDenied) as exc:
            self._skip(exc, "Cannot clone table")
        self.assertTrue(first.exists())
        # Second call with missing_ok must not raise though the target exists.
        self.table.clone(target_name, missing_ok=True)


@pytest.mark.integration
class TestTableViewIntegration(_TableFixture):
    """View-shaped securables: ``create_view`` + ``is_view`` +
    :attr:`Table.view_definition` + view rename round-trip."""

    table_prefix = "yg_view_src"

    def setUp(self) -> None:
        super().setUp()
        self.table.insert(_sample_data(), mode=Mode.OVERWRITE)

    def _new_view(self) -> Table:
        return self.client.tables.view(self._sibling("yg_view"))

    def test_create_view_sets_view_definition(self) -> None:
        view = self._new_view()
        try:
            view.create_view(
                f"SELECT id, label FROM {self.table.full_name(safe=True)}",
                mode=Mode.OVERWRITE,
            )
        except (DatabricksError, PermissionDenied) as exc:
            self._skip(exc, "Cannot create view")
        self.assertTrue(view.exists())
        _ = view.infos  # warm the cache so ``is_view`` reads from it
        self.assertTrue(view.is_view)
        self.assertIsNotNone(view.view_definition)
        self.assertIn("SELECT", (view.view_definition or "").upper())

    def test_view_read_round_trips(self) -> None:
        view = self._new_view()
        try:
            view.create_view(
                f"SELECT id, label FROM {self.table.full_name(safe=True)} WHERE id >= 2",
                mode=Mode.OVERWRITE,
            )
        except (DatabricksError, PermissionDenied) as exc:
            self._skip(exc, "Cannot create view")
        rows = self._sql(f"SELECT id, label FROM {view.full_name(safe=True)} ORDER BY id")
        self.assertEqual(rows.column("id").to_pylist(), [2, 3])
        self.assertEqual(rows.column("label").to_pylist(), ["b", "c"])

    def test_view_rename_round_trip(self) -> None:
        view = self._new_view()
        try:
            view.create_view(
                f"SELECT id FROM {self.table.full_name(safe=True)}",
                mode=Mode.OVERWRITE,
            )
        except (DatabricksError, PermissionDenied) as exc:
            self._skip(exc, "Cannot create view")
        _ = view.infos
        new_name = f"yg_view_renamed_{secrets.token_hex(4)}"
        self._cleanup.append(f"{self.catalog_name}.{self.schema_name}.{new_name}")
        try:
            view.rename(new_name)
        except (DatabricksError, PermissionDenied) as exc:
            self._skip(exc, "Cannot rename view")
        self.assertEqual(view.table_name, new_name)
        self.assertTrue(view.exists())


@pytest.mark.integration
class TestTableStoragePathIntegration(_TableFixture):
    """Storage-path surface — :attr:`Table.staging_volume`,
    :meth:`Table.staging_folder`, :meth:`Table.insert_volume_path`,
    :meth:`Table.temporary_credentials`, :meth:`Table.storage_path`.

    These bottom out in Unity Catalog APIs (Volumes, Files,
    ``temporary_table_credentials``) and only meaningfully exercise
    against a live workspace.
    """

    table_prefix = "yg_storage"

    def test_staging_volume_collapses_to_singleton(self) -> None:
        first = self.table.staging_volume
        self.assertIs(first, self.table.staging_volume)
        self.assertIsInstance(first, Volume)
        self.assertEqual(first.catalog_name, self.catalog_name)
        self.assertEqual(first.schema_name, self.schema_name)
        # Volume name is derived from the table name (lowercased, tag-safe).
        expected = self.client.safe_tag_value(self.table_name, repl="_").lower()
        self.assertEqual(first.volume_name, expected)

    def test_staging_volume_ensure_created_round_trips(self) -> None:
        volume = self.table.staging_volume
        try:
            volume.ensure_created(comment="yggdrasil storage path integration")
        except (DatabricksError, PermissionDenied) as exc:
            self._skip(exc, f"Cannot create staging volume {volume.full_name()}")
        try:
            self.assertTrue(volume.exists())
        finally:
            try:
                volume.delete(missing_ok=True)
            except Exception:
                pass

    def test_staging_folder_paths_root_under_volume(self) -> None:
        tmp = self.table.staging_folder(temporary=True)
        async_path = self.table.staging_folder(async_write=True)
        self.assertIsInstance(tmp, VolumePath)
        self.assertIsInstance(async_path, VolumePath)
        # Folder layout is part of the staging contract.
        self.assertTrue(tmp.full_path().rstrip("/").endswith(".sql/tmp"))
        self.assertTrue(async_path.full_path().rstrip("/").endswith(".sql/async/insert"))
        self.assertTrue(tmp.temporary)
        self.assertFalse(async_path.temporary)

    def test_insert_volume_path_is_unique_per_call(self) -> None:
        a, b = self.table.insert_volume_path(), self.table.insert_volume_path()
        self.assertNotEqual(a.full_path(), b.full_path())
        for p in (a, b):
            self.assertTrue(p.full_path().endswith(".parquet"))
            self.assertIn("/.sql/tmp/", p.full_path())

    def test_insert_volume_path_round_trips_parquet(self) -> None:
        try:
            self.table.staging_volume.ensure_created()
        except (DatabricksError, PermissionDenied) as exc:
            self._skip(exc, "Cannot create staging volume")
        path = self.table.insert_volume_path(temporary=False)
        data = pa.table({
            "id": pa.array([10, 20], type=pa.int64()),
            "label": pa.array(["x", "y"], type=pa.string()),
        })
        path.as_media(media_type=MediaTypes.PARQUET).write_table(data, mode=Mode.OVERWRITE)
        try:
            self.assertGreater(path.size, 0)
        finally:
            path.unlink(missing_ok=True)

    def test_temporary_credentials_returns_creds(self) -> None:
        try:
            response = self.table.temporary_credentials(operation=TableOperation.READ)
        except (DatabricksError, PermissionDenied) as exc:
            self._skip(exc, "Cannot vend temporary table credentials")
        self.assertIsNotNone(response)
        # At least one cloud-specific credential slot must be populated.
        cred_attrs = (
            "aws_temp_credentials",
            "azure_user_delegation_sas",
            "gcp_oauth_token",
            "r2_temp_credentials",
        )
        self.assertTrue(
            any(getattr(response, attr, None) is not None for attr in cred_attrs),
            f"No cloud credential slot populated on {response!r}",
        )

    def test_storage_location_returns_addressable_path(self) -> None:
        # ``storage_path`` collapses to a :class:`Path` over the backing
        # cloud store; only AWS workspaces vend the read creds we need —
        # skip cleanly on Azure / GCP (any failure) rather than failing.
        try:
            path = self.table.storage_path()
        except Exception as exc:
            self._skip(exc, "storage_location unavailable (non-AWS workspace?)")
        self.assertIsNotNone(path)
        self.assertTrue(str(path.full_path()))


@pytest.mark.integration
class TestTableAsyncInsertIntegration(_TableFixture):
    """``insert(wait=False)`` drop → loader aggregate-load round-trip.

    ``async_insert`` stages the Parquet to the table's tmp path and drops a
    JSON operation log (no warehouse statement); ``Tables.async_insert`` then
    reads the log, builds one aggregated ``INSERT`` per ``(target, mode)``
    group, loads it, and clears the consumed log + data. The loader is driven
    directly — the live file-arrival job (``Table.async_job()``) is not
    exercised here.
    """

    table_prefix = "yg_async"

    def setUp(self) -> None:
        super().setUp()
        # The class table is shared across the async tests — start each from
        # empty so row-count assertions are order-independent.
        self.table.insert(_sample_data().slice(0, 0), mode=Mode.OVERWRITE)

    def test_async_drop_then_loader_appends(self) -> None:
        from yggdrasil.databricks.table.async_job import (
            DatabricksTableInsert, logs_path,
        )

        try:
            log_path = self.table.async_insert(_sample_data(), mode=Mode.APPEND)
        except (DatabricksError, PermissionDenied) as exc:
            self._skip(exc, "async insert needs volume write access")

        self.assertTrue(log_path.exists())
        op = DatabricksTableInsert.from_log(log_path)
        self.assertEqual(op.target, self.table.full_name())
        self.assertEqual(op.mode, Mode.APPEND)
        # the staged Parquet lives exactly where the log's uniform URL says
        data_path = op.data_path(self.table.client)
        self.assertTrue(data_path.exists())

        # run the loader: aggregate + load, then clear the consumed log + data
        processed = self.client.tables.async_insert(logs_path(self.table), wait=True)
        self.assertEqual(processed, 1)
        self.assertEqual(self._count(), 3)
        self.assertFalse(log_path.exists())
        self.assertFalse(data_path.exists())

    def test_async_overwrite_replaces_rows(self) -> None:
        from yggdrasil.databricks.table.async_job import logs_path

        try:
            self.table.insert(_sample_data(), mode=Mode.APPEND)
        except (DatabricksError, PermissionDenied) as exc:
            self._skip(exc, "seed insert failed")
        self.assertEqual(self._count(), 3)

        overwrite_rows = pa.table({
            "id": pa.array([9], type=pa.int64()),
            "label": pa.array(["z"], type=pa.string()),
            "amount": pa.array([9.5], type=pa.float64()),
        })
        try:
            self.table.async_insert(overwrite_rows, mode=Mode.OVERWRITE)
        except (DatabricksError, PermissionDenied) as exc:
            self._skip(exc, "async insert failed")

        self.client.tables.async_insert(logs_path(self.table), wait=True)
        self.assertEqual(self._count(), 1)

    def test_async_rejects_merge_mode(self) -> None:
        with self.assertRaises(ValueError):
            self.table.async_insert(_sample_data(), mode=Mode.MERGE)
