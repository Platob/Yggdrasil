"""
Unit tests for :meth:`yggdrasil.databricks.sql.table.Table.update_columns`.

Covers:

- case-insensitive rename of matching columns,
- ``UPSERT`` data-type updates,
- ``OVERWRITE`` dropping of columns not in the input.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from databricks.sdk.service.catalog import ColumnInfo, TableInfo

from yggdrasil.data import DataType, Field
from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.sql import Table, Tables
from yggdrasil.databricks.sql.column import Column
from yggdrasil.io.enums.save_mode import SaveMode


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_client():
    client = MagicMock(spec=DatabricksClient)
    client.base_url.to_string.return_value = "https://adb-123.azuredatabricks.net"
    client.base_url.with_path.side_effect = lambda p: MagicMock(
        to_string=lambda: f"https://adb-123.azuredatabricks.net{p}"
    )
    client.sql = MagicMock()
    client.sql.return_value = client.sql
    return client


@pytest.fixture()
def table(mock_client):
    """A table primed with two existing columns: ``id`` BIGINT, ``name`` STRING."""
    service = Tables(client=mock_client, catalog_name="main", schema_name="sales")
    tb = Table(
        service=service,
        catalog_name="main",
        schema_name="sales",
        table_name="customers",
    )
    infos = TableInfo(
        catalog_name="main",
        schema_name="sales",
        name="customers",
        columns=[
            ColumnInfo(name="id", position=0, type_text="bigint"),
            ColumnInfo(name="name", position=1, type_text="string"),
        ],
    )
    object.__setattr__(tb, "_infos", infos)
    object.__setattr__(tb, "_infos_fetched_at", 9e18)
    object.__setattr__(
        tb, "_columns",
        [Column.from_api(table=tb, infos=c) for c in infos.columns],
    )
    return tb


def _capture_execute_many(mock_client) -> list[list[str]]:
    """Wire ``execute_many`` to record each statement batch passed in."""
    batches: list[list[str]] = []

    def fake_execute_many(stmts, *a, **kw):
        batches.append(list(stmts))

    mock_client.sql.execute_many.side_effect = fake_execute_many
    return batches


# ── case-insensitive rename ───────────────────────────────────────────────────


class TestCaseInsensitiveRename:
    def test_upsert_renames_existing_case_insensitive_match(self, table, mock_client):
        batches = _capture_execute_many(mock_client)

        table.update_columns(
            [Field(name="ID", dtype=DataType.from_any("bigint"))],
            mode=SaveMode.UPSERT,
        )

        flat = [s for batch in batches for s in batch]
        renames = [s for s in flat if "RENAME COLUMN" in s]
        assert len(renames) == 1
        assert "RENAME COLUMN `id` TO `ID`" in renames[0]

    def test_overwrite_also_renames_case_insensitive(self, table, mock_client):
        batches = _capture_execute_many(mock_client)

        table.update_columns(
            [
                Field(name="ID", dtype=DataType.from_any("bigint")),
                Field(name="Name", dtype=DataType.from_any("string")),
            ],
            mode=SaveMode.OVERWRITE,
        )

        flat = [s for batch in batches for s in batch]
        rename_targets = {s for s in flat if "RENAME COLUMN" in s}
        assert any("RENAME COLUMN `id` TO `ID`" in s for s in rename_targets)
        assert any("RENAME COLUMN `name` TO `Name`" in s for s in rename_targets)

    def test_same_casing_does_not_rename(self, table, mock_client):
        batches = _capture_execute_many(mock_client)

        table.update_columns(
            [Field(name="id", dtype=DataType.from_any("bigint"))],
            mode=SaveMode.UPSERT,
        )

        flat = [s for batch in batches for s in batch]
        assert not any("RENAME COLUMN" in s for s in flat)


# ── UPSERT dtype updates ──────────────────────────────────────────────────────


class TestUpsertDtypeUpdate:
    def test_upsert_emits_alter_type_when_dtype_differs(self, table, mock_client):
        batches = _capture_execute_many(mock_client)

        table.update_columns(
            [Field(name="id", dtype=DataType.from_any("string"))],
            mode=SaveMode.UPSERT,
        )

        flat = [s for batch in batches for s in batch]
        alter_types = [s for s in flat if "ALTER COLUMN" in s and "TYPE" in s]
        assert len(alter_types) == 1
        assert "ALTER COLUMN `id` TYPE STRING" in alter_types[0]

    def test_upsert_skips_alter_type_when_dtype_matches(self, table, mock_client):
        batches = _capture_execute_many(mock_client)

        table.update_columns(
            [Field(name="id", dtype=DataType.from_any("bigint"))],
            mode=SaveMode.UPSERT,
        )

        flat = [s for batch in batches for s in batch]
        assert not any("ALTER COLUMN" in s and "TYPE" in s for s in flat)

    def test_upsert_rename_then_alter_type_runs_in_two_phases(
        self, table, mock_client
    ):
        """Rename must finish before ALTER COLUMN TYPE refers to the new name."""
        batches = _capture_execute_many(mock_client)

        table.update_columns(
            [Field(name="ID", dtype=DataType.from_any("string"))],
            mode=SaveMode.UPSERT,
        )

        assert len(batches) == 2
        assert any("RENAME COLUMN `id` TO `ID`" in s for s in batches[0])
        assert any(
            "ALTER COLUMN `ID` TYPE STRING" in s for s in batches[1]
        )


# ── OVERWRITE drops missing columns ───────────────────────────────────────────


class TestOverwriteDropsMissing:
    def test_overwrite_drops_columns_not_in_input(self, table, mock_client):
        batches = _capture_execute_many(mock_client)

        table.update_columns(
            [Field(name="id", dtype=DataType.from_any("bigint"))],
            mode=SaveMode.OVERWRITE,
        )

        flat = [s for batch in batches for s in batch]
        drops = [s for s in flat if "DROP COLUMNS" in s]
        assert len(drops) == 1
        assert "DROP COLUMNS (`name`)" in drops[0]

    def test_upsert_does_not_drop(self, table, mock_client):
        batches = _capture_execute_many(mock_client)

        table.update_columns(
            [Field(name="id", dtype=DataType.from_any("bigint"))],
            mode=SaveMode.UPSERT,
        )

        flat = [s for batch in batches for s in batch]
        assert not any("DROP COLUMN" in s for s in flat)

    def test_overwrite_full_cycle_drops_renames_and_adds(self, table, mock_client):
        batches = _capture_execute_many(mock_client)

        table.update_columns(
            [
                Field(name="ID", dtype=DataType.from_any("bigint")),
                Field(name="created_at", dtype=DataType.from_any("timestamp")),
            ],
            mode=SaveMode.OVERWRITE,
        )

        flat = [s for batch in batches for s in batch]
        assert any("RENAME COLUMN `id` TO `ID`" in s for s in flat)
        assert any("DROP COLUMNS (`name`)" in s for s in flat)
        assert any(
            "ADD COLUMNS (`created_at` TIMESTAMP" in s for s in flat
        )


# ── default AUTO mode still updates dtype and does not drop ──────────────────


class TestAutoModeBehavior:
    def test_auto_updates_dtype_but_does_not_drop(self, table, mock_client):
        batches = _capture_execute_many(mock_client)

        table.update_columns(
            [Field(name="id", dtype=DataType.from_any("string"))],
            # no mode → defaults to AUTO
        )

        flat = [s for batch in batches for s in batch]
        assert any(
            "ALTER COLUMN `id` TYPE STRING" in s for s in flat
        )
        assert not any("DROP COLUMN" in s for s in flat)
