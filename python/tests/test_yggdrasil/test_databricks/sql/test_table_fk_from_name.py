"""
Unit tests for foreign-key detection via the ``catalog.schema.table.column``
field-name pattern in :meth:`Table.sql_create` and :meth:`Table.update_columns`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from databricks.sdk.service.catalog import ColumnInfo, ColumnTypeName, TableInfo

from yggdrasil.data import DataType, Field, Schema
from yggdrasil.data.types.primitive import IntegerType
from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.sql import Table, Tables
from yggdrasil.databricks.sql.columns import Columns
from yggdrasil.databricks.sql.types import ForeignKeySpec


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_client():
    client = MagicMock(spec=DatabricksClient)
    client.base_url.to_string.return_value = "https://adb-123.azuredatabricks.net"
    client.base_url.with_path.side_effect = lambda p: MagicMock(
        to_string=lambda: f"https://adb-123.azuredatabricks.net{p}"
    )
    # Table.sql delegates through ``self.client.sql(...)``: fold back to the same mock.
    client.sql = MagicMock()
    client.sql.return_value = client.sql
    return client


@pytest.fixture()
def ref_table(mock_client):
    """A stub referenced table whose ``id`` column is a BIGINT."""
    tbl = MagicMock()
    tbl.catalog_name = "main"
    tbl.schema_name = "sales"
    tbl.table_name = "customers"
    tbl.column.return_value = MagicMock(
        name="id",
        field=Field(name="id", dtype=DataType.from_any("bigint"), nullable=False),
    )
    # The ``.name`` attr on a MagicMock doesn't auto-set from the keyword —
    # override on the returned column.
    tbl.column.return_value.name = "id"
    tbl.column.return_value.field = Field(name="id", dtype=DataType.from_any("bigint"), nullable=False)
    return tbl


@pytest.fixture()
def columns_service_patch(mock_client, ref_table, monkeypatch):
    """Route ``Columns._find_table`` to our stub instead of the real SDK."""
    monkeypatch.setattr(
        Columns,
        "_find_table",
        lambda self, catalog_name, schema_name, table_name: ref_table,
    )
    # Route ``client.columns`` to a scoped Columns service for the top-level case.
    mock_client.columns = Columns(client=mock_client)
    return mock_client.columns


@pytest.fixture()
def table(mock_client, columns_service_patch):
    service = Tables(client=mock_client, catalog_name="main", schema_name="sales")
    tb = Table(
        service=service,
        catalog_name="main",
        schema_name="sales",
        table_name="orders",
    )
    return tb


# ── _resolve_fk_field ─────────────────────────────────────────────────────────


class TestResolveFkField:
    def test_plain_name_is_left_untouched(self, table):
        field = Field(name="amount", dtype=DataType.from_any("bigint"))
        resolved, fk = table._resolve_fk_field(field)
        assert resolved is field
        assert fk is None

    def test_two_part_name_resolves_to_same_schema_fk(self, table):
        field = Field(name="customers.id", dtype=DataType.from_any("string"))
        resolved, fk = table._resolve_fk_field(field)

        # Local column renamed to the leaf.
        assert resolved.name == "id"
        # Dtype inherited from the referenced column (BIGINT / Int64).
        assert isinstance(resolved.dtype, IntegerType)
        assert resolved.dtype.to_databricks_ddl() == "BIGINT"
        # FK spec points at the fully qualified ref.
        assert fk is not None
        assert fk.column == "id"
        assert fk.ref == "main.sales.customers.id"
        assert fk.constraint_name and "_fk" in fk.constraint_name

    def test_four_part_name_uses_exact_qualification(self, table):
        field = Field(name="warehouse.core.customers.id", dtype=DataType.from_any("string"))
        resolved, fk = table._resolve_fk_field(field)

        assert resolved.name == "id"
        assert fk is not None
        assert fk.ref == "warehouse.core.customers.id"

    def test_self_reference_is_not_treated_as_fk(self, table):
        # orders is this table — dotted form pointing at self → no FK, just rename.
        field = Field(name="orders.amount", dtype=DataType.from_any("bigint"))
        resolved, fk = table._resolve_fk_field(field)
        assert resolved.name == "amount"
        assert fk is None

    def test_lookup_failure_falls_back_to_rename_only(self, table, monkeypatch):
        def boom(self, **_):
            raise RuntimeError("ref not found")

        monkeypatch.setattr(Columns, "column", boom)

        field = Field(name="unknown_table.id", dtype=DataType.from_any("bigint"))
        resolved, fk = table._resolve_fk_field(field)
        assert resolved.name == "id"
        assert fk is None


# ── sql_create applies resolved FKs via the table_constraints API ────────────


class _TableConstraintsSpy:
    """Captures ``table_constraints.create`` calls for assertions."""

    def __init__(self) -> None:
        self.created: list = []

    def create(self, *, full_name_arg, constraint):
        self.created.append((full_name_arg, constraint))
        return constraint


class TestSqlCreateInlinesInferredFk:
    def test_dotted_field_name_produces_fk_constraint_in_create_ddl(
        self, table, mock_client
    ):
        # Capture the DDL executed by sql_create.
        captured: list[str] = []

        def fake_execute(ddl, *args, **kwargs):
            captured.append(ddl)
            return MagicMock()

        mock_client.sql.execute.side_effect = fake_execute
        # After CREATE, sql_create iterates fields to apply tags via
        # ``self.column(...)``, which re-reads the table info — feed a stub back.
        mock_client.tables.find_table_remote.return_value = TableInfo(
            catalog_name="main",
            schema_name="sales",
            name="orders",
            columns=[
                ColumnInfo(name="id", position=0, type_text="bigint"),
            ],
        )

        # Wire the table_constraints spy onto the mocked workspace client.
        constraints_spy = _TableConstraintsSpy()
        mock_client.workspace_client.return_value.table_constraints = constraints_spy

        schema = Schema.from_any_fields([
            Field(name="id", dtype=DataType.from_any("bigint"), nullable=False),
            Field(name="customers.id", dtype=DataType.from_any("string")),
        ])
        table.sql_create(schema, if_not_exists=False)

        create_ddls = [s for s in captured if s.startswith("CREATE TABLE")]
        assert len(create_ddls) == 1
        ddl = create_ddls[0]

        # Leaf name is used as the local column.
        assert "`id` BIGINT" in ddl
        # CREATE TABLE no longer carries inline constraints.
        assert "FOREIGN KEY" not in ddl
        assert "REFERENCES" not in ddl

        # FK is applied via the UC table_constraints API after CREATE.
        assert len(constraints_spy.created) == 1
        full_name, constraint = constraints_spy.created[0]
        assert full_name == "main.sales.orders"
        fk = constraint.foreign_key_constraint
        assert fk is not None
        assert fk.child_columns == ["id"]
        assert fk.parent_table == "main.sales.customers"
        assert fk.parent_columns == ["id"]


# ── update_columns applies FK after ADD COLUMNS ──────────────────────────────


class TestUpdateColumnsAppliesInferredFk:
    def test_new_column_with_dotted_name_adds_fk_after_alter(
        self, table, mock_client
    ):
        # Pretend the table currently has no columns so the dotted field adds.
        object.__setattr__(table, "_columns", [])
        object.__setattr__(
            table, "_infos",
            TableInfo(
                catalog_name="main",
                schema_name="sales",
                name="orders",
                columns=[],
            ),
        )
        object.__setattr__(table, "_infos_fetched_at", 9e18)

        executed_many: list[list[str]] = []

        def fake_execute_many(stmts, *a, **kw):
            executed_many.append(list(stmts))

        mock_client.sql.execute_many.side_effect = fake_execute_many

        # Wire the table_constraints spy onto the mocked workspace client.
        constraints_spy = _TableConstraintsSpy()
        mock_client.workspace_client.return_value.table_constraints = constraints_spy

        table.update_columns([Field(name="customers.id", dtype=DataType.from_any("string"))])

        # ADD COLUMNS still goes through execute_many; FK no longer does.
        assert len(executed_many) == 1
        stmts = executed_many[0]

        add_sqls = [s for s in stmts if "ADD COLUMNS" in s]
        fk_sqls = [s for s in stmts if "FOREIGN KEY" in s]

        assert len(add_sqls) == 1
        assert "ADD COLUMNS (`id` BIGINT)" in add_sqls[0]
        assert fk_sqls == []

        # FK applied via the UC API instead.
        assert len(constraints_spy.created) == 1
        full_name, constraint = constraints_spy.created[0]
        assert full_name == "main.sales.orders"
        fk = constraint.foreign_key_constraint
        assert fk is not None
        assert fk.child_columns == ["id"]
        assert fk.parent_table == "main.sales.customers"
        assert fk.parent_columns == ["id"]
