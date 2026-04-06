"""
Tests for :class:`~yggdrasil.databricks.sql.column.Column`.

Structure
---------
Unit tests (no live workspace)
    All DDL-building methods and ``from_api`` are exercised with lightweight
    mock / stub objects.  No network calls are made.

Integration tests (``requires_databricks``)
    :class:`TestColumnIntegration` inherits :class:`DatabricksCase`, creates
    two scratch Delta tables in ``trading.unittest`` (a *main* table and a
    *reference* table for FK tests), exercises the live constraint/tag APIs,
    and drops both tables in teardown.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
from databricks.sdk.service.catalog import ColumnInfo as CatalogColumnInfo, ColumnTypeName
from databricks.sdk.service.sql import ColumnInfo as SQLColumnInfo

from yggdrasil.data import Field
from yggdrasil.databricks.sql import Table, Tables, Column, Columns, SQLEngine

from ..conftest import DatabricksCase, requires_databricks

# ─────────────────────────────────────────────────────────────────────────────
# Shared test helpers
# ─────────────────────────────────────────────────────────────────────────────


def _stub_table(
    catalog: str = "main",
    schema: str = "sales",
    name: str = "orders",
) -> MagicMock:
    """Return a lightweight Table stub with a working ``full_name()`` and ``_safe_str()``."""
    tbl = MagicMock(spec=Table)
    tbl.name = name
    tbl.full_name.side_effect = lambda safe=None: (
        f"`{catalog}`.`{schema}`.`{name}`" if safe else f"{catalog}.{schema}.{name}"
    )
    tbl._safe_str.side_effect = lambda v: v if isinstance(v, str) else (v.decode() if v else "")
    return tbl


def _col(
    name: str,
    arrow_type: pa.DataType = pa.int64(),
    nullable: bool = True,
    metadata: dict | None = None,
    table: MagicMock | None = None,
) -> Column:
    """Construct a ``Column`` backed by a stub table (or the one provided)."""
    if table is None:
        table = _stub_table()
    meta_bytes = (
        {
            (k if isinstance(k, bytes) else k.encode()): (v if isinstance(v, bytes) else v.encode())
            for k, v in metadata.items()
        }
        if metadata
        else None
    )
    af = pa.field(name, arrow_type, nullable=nullable, metadata=meta_bytes)
    return Column(table=table, name=name, dfield=Field.from_arrow(af))


# ─────────────────────────────────────────────────────────────────────────────
# Unit — Column properties
# ─────────────────────────────────────────────────────────────────────────────


class TestColumnProperties:
    def test_arrow_field_round_trip(self):
        col = _col("price", pa.float64(), nullable=False)
        assert col.arrow_field == pa.field("price", pa.float64(), nullable=False)

    def test_metadata_empty_when_none(self):
        col = _col("price")
        assert col.metadata == {}

    def test_metadata_from_field(self):
        col = _col("c", metadata={"comment": "my comment"})
        assert col.metadata.get(b"comment") == b"my comment"

    def test_engine_delegates_to_table_sql(self):
        tbl = _stub_table()
        col = _col("c", table=tbl)
        _ = col.engine
        # ``engine`` is a property that returns ``self.table.sql``
        assert col.engine is tbl.sql

    def test_qcol_simple_name(self):
        assert _col("price")._qcol() == "`price`"

    def test_qcol_name_with_spaces(self):
        assert _col("my col")._qcol() == "`my col`"

    def test_safe_constraint_name_uses_provided(self):
        col = _col("price")
        assert col._safe_constraint_name("custom_pk", "fallback") == "custom_pk"

    def test_safe_constraint_name_falls_back(self):
        col = _col("price")
        assert col._safe_constraint_name(None, "orders_price_pk") == "orders_price_pk"


# ─────────────────────────────────────────────────────────────────────────────
# Unit — set_tags_ddl / set_tags
# ─────────────────────────────────────────────────────────────────────────────


class TestSetTagsDDL:
    def test_single_tag(self):
        col = _col("price")
        ddl = col.set_tags_ddl({"env": "prod"})
        assert ddl == (
            "ALTER TABLE `main`.`sales`.`orders` "
            "ALTER COLUMN `price` SET TAGS ('env' = 'prod')"
        )

    def test_multiple_tags_all_present(self):
        col = _col("price")
        ddl = col.set_tags_ddl({"a": "1", "b": "2"})
        assert "'a' = '1'" in ddl
        assert "'b' = '2'" in ddl

    def test_empty_mapping_returns_none(self):
        col = _col("price")
        assert col.set_tags_ddl({}) is None

    def test_tag_with_empty_value_skipped(self):
        col = _col("price")
        # value is empty string → entire entry is skipped → returns None
        assert col.set_tags_ddl({"key": ""}) is None

    def test_tag_with_empty_key_skipped(self):
        col = _col("price")
        assert col.set_tags_ddl({"": "value"}) is None

    def test_set_tags_executes_query(self):
        col = _col("price")
        col.table.sql.execute = MagicMock()
        col.set_tags({"env": "prod"})
        col.table.sql.execute.assert_called_once()

    def test_set_tags_none_skips_execute(self):
        col = _col("price")
        col.table.sql.execute = MagicMock()
        col.set_tags(None)
        col.table.sql.execute.assert_not_called()

    def test_set_tags_empty_dict_skips_execute(self):
        col = _col("price")
        col.table.sql.execute = MagicMock()
        col.set_tags({})
        col.table.sql.execute.assert_not_called()

    def test_set_tags_returns_self(self):
        col = _col("price")
        col.table.sql.execute = MagicMock()
        assert col.set_tags({"k": "v"}) is col


# ─────────────────────────────────────────────────────────────────────────────
# Unit — primary key DDL
# ─────────────────────────────────────────────────────────────────────────────


class TestPrimaryKeyDDL:
    def test_basic_structure(self):
        col = _col("id")
        ddl = col.add_primary_key_ddl()
        assert "ALTER TABLE `main`.`sales`.`orders`" in ddl
        assert "ADD CONSTRAINT" in ddl
        assert "PRIMARY KEY (`id`)" in ddl

    def test_auto_constraint_name(self):
        col = _col("id")
        assert "`orders_id_pk`" in col.add_primary_key_ddl()

    def test_custom_constraint_name(self):
        col = _col("id")
        assert "`my_pk`" in col.add_primary_key_ddl(constraint_name="my_pk")

    def test_rely_clause_present(self):
        col = _col("id")
        assert "RELY" in col.add_primary_key_ddl(rely=True)

    def test_no_rely_by_default(self):
        col = _col("id")
        assert "RELY" not in col.add_primary_key_ddl()

    def test_timeseries_clause(self):
        col = _col("ts")
        ddl = col.add_primary_key_ddl(timeseries=True)
        assert "`ts` TIMESERIES" in ddl

    def test_no_timeseries_by_default(self):
        col = _col("ts")
        assert "TIMESERIES" not in col.add_primary_key_ddl()

    def test_set_primary_key_executes(self):
        col = _col("id")
        col.table.sql.execute = MagicMock()
        col.set_primary_key()
        col.table.sql.execute.assert_called_once()

    def test_set_primary_key_returns_self(self):
        col = _col("id")
        col.table.sql.execute = MagicMock()
        assert col.set_primary_key() is col

    def test_drop_ddl_if_exists_default(self):
        col = _col("id")
        ddl = col.drop_primary_key_ddl()
        assert "DROP PRIMARY KEY IF EXISTS" in ddl
        assert "CASCADE" not in ddl

    def test_drop_ddl_cascade(self):
        col = _col("id")
        assert "CASCADE" in col.drop_primary_key_ddl(cascade=True)

    def test_drop_ddl_no_if_exists(self):
        col = _col("id")
        ddl = col.drop_primary_key_ddl(if_exists=False)
        assert "IF EXISTS" not in ddl

    def test_unset_primary_key_executes(self):
        col = _col("id")
        col.table.sql.execute = MagicMock()
        col.unset_primary_key()
        col.table.sql.execute.assert_called_once()

    def test_unset_primary_key_returns_self(self):
        col = _col("id")
        col.table.sql.execute = MagicMock()
        assert col.unset_primary_key() is col


# ─────────────────────────────────────────────────────────────────────────────
# Unit — foreign key DDL
# ─────────────────────────────────────────────────────────────────────────────


class TestForeignKeyDDL:
    def _ref(self) -> MagicMock:
        return _stub_table(catalog="main", schema="sales", name="customers")

    def test_basic_structure(self):
        col = _col("customer_id")
        ddl = col.add_foreign_key_ddl(ref_table=self._ref(), ref_column="id")
        assert "FOREIGN KEY (`customer_id`)" in ddl
        assert "REFERENCES `main`.`sales`.`customers` (`id`)" in ddl

    def test_auto_constraint_name(self):
        col = _col("customer_id")
        ddl = col.add_foreign_key_ddl(ref_table=self._ref(), ref_column="id")
        assert "`orders_customer_id__customers_id_fk`" in ddl

    def test_custom_constraint_name(self):
        col = _col("customer_id")
        ddl = col.add_foreign_key_ddl(
            ref_table=self._ref(), ref_column="id", constraint_name="my_fk"
        )
        assert "`my_fk`" in ddl

    def test_ref_column_as_column_object(self):
        ref = self._ref()
        col = _col("customer_id")
        ref_col = _col("id", table=ref)
        ddl = col.add_foreign_key_ddl(ref_table=ref, ref_column=ref_col)
        assert "(`id`)" in ddl

    def test_rely(self):
        col = _col("customer_id")
        ddl = col.add_foreign_key_ddl(ref_table=self._ref(), ref_column="id", rely=True)
        assert "RELY" in ddl

    def test_match_full(self):
        col = _col("customer_id")
        ddl = col.add_foreign_key_ddl(ref_table=self._ref(), ref_column="id", match_full=True)
        assert "MATCH FULL" in ddl

    def test_on_update_no_action(self):
        col = _col("customer_id")
        ddl = col.add_foreign_key_ddl(
            ref_table=self._ref(), ref_column="id", on_update_no_action=True
        )
        assert "ON UPDATE NO ACTION" in ddl

    def test_on_delete_no_action(self):
        col = _col("customer_id")
        ddl = col.add_foreign_key_ddl(
            ref_table=self._ref(), ref_column="id", on_delete_no_action=True
        )
        assert "ON DELETE NO ACTION" in ddl

    def test_no_options_by_default(self):
        col = _col("customer_id")
        ddl = col.add_foreign_key_ddl(ref_table=self._ref(), ref_column="id")
        assert "RELY" not in ddl
        assert "MATCH" not in ddl
        assert "ON UPDATE" not in ddl
        assert "ON DELETE" not in ddl

    def test_all_options_combined(self):
        col = _col("customer_id")
        ddl = col.add_foreign_key_ddl(
            ref_table=self._ref(),
            ref_column="id",
            rely=True,
            match_full=True,
            on_update_no_action=True,
            on_delete_no_action=True,
        )
        for clause in ("RELY", "MATCH FULL", "ON UPDATE NO ACTION", "ON DELETE NO ACTION"):
            assert clause in ddl

    def test_set_foreign_key_executes(self):
        col = _col("customer_id")
        col.table.sql.execute = MagicMock()
        col.set_foreign_key(ref_table=self._ref(), ref_column="id")
        col.table.sql.execute.assert_called_once()

    def test_set_foreign_key_returns_self(self):
        col = _col("customer_id")
        col.table.sql.execute = MagicMock()
        assert col.set_foreign_key(ref_table=self._ref(), ref_column="id") is col

    def test_drop_foreign_key_ddl_if_exists(self):
        col = _col("customer_id")
        ddl = col.drop_foreign_key_ddl()
        assert "DROP FOREIGN KEY IF EXISTS (`customer_id`)" in ddl

    def test_drop_foreign_key_ddl_no_if_exists(self):
        col = _col("customer_id")
        ddl = col.drop_foreign_key_ddl(if_exists=False)
        assert "IF EXISTS" not in ddl
        assert "DROP FOREIGN KEY (`customer_id`)" in ddl

    def test_unset_foreign_key_executes(self):
        col = _col("customer_id")
        col.table.sql.execute = MagicMock()
        col.unset_foreign_key()
        col.table.sql.execute.assert_called_once()

    def test_unset_foreign_key_returns_self(self):
        col = _col("customer_id")
        col.table.sql.execute = MagicMock()
        assert col.unset_foreign_key() is col


# ─────────────────────────────────────────────────────────────────────────────
# Unit — Column.from_api
# ─────────────────────────────────────────────────────────────────────────────


class TestColumnFromApi:
    def test_from_catalog_column_info_decimal(self):
        ci = CatalogColumnInfo(
            name="price",
            type_text="DECIMAL(18,6)",
            type_name=ColumnTypeName.DECIMAL,
            type_json=json.dumps({
                "name": "price",
                "type": "decimal(18,6)",
                "nullable": False,
                "metadata": {"comment": "USD price"},
            }),
            nullable=False,
            position=0,
        )
        col = Column.from_api(_stub_table(), ci)
        assert col.name == "price"
        assert col.arrow_field.type == pa.decimal128(18, 6)
        assert col.arrow_field.nullable is False
        assert col.metadata.get(b"comment") == b"USD price"

    def test_from_sql_column_info_int(self):
        ci = SQLColumnInfo(name="qty", type_text="INT")
        col = Column.from_api(_stub_table(), ci)
        assert col.name == "qty"
        assert col.arrow_field.type == pa.int32()
        assert col.arrow_field.nullable is True  # SQL results always nullable

    def test_from_sql_column_info_string(self):
        ci = SQLColumnInfo(name="label", type_text="STRING")
        col = Column.from_api(_stub_table(), ci)
        # Arrow maps STRING to utf8 or large_utf8
        assert col.arrow_field.type in (pa.utf8(), pa.large_utf8())

    def test_from_api_sets_table_ref(self):
        tbl = _stub_table()
        ci = SQLColumnInfo(name="c", type_text="BIGINT")
        col = Column.from_api(tbl, ci)
        assert col.table is tbl

    def test_from_catalog_col_without_metadata(self):
        ci = CatalogColumnInfo(
            name="ts",
            type_text="TIMESTAMP",
            type_name=ColumnTypeName.TIMESTAMP,
            type_json=json.dumps({
                "name": "ts",
                "type": "timestamp",
                "nullable": True,
                "metadata": {},
            }),
            nullable=True,
            position=1,
        )
        col = Column.from_api(_stub_table(), ci)
        assert col.name == "ts"
        assert col.metadata == {}

    def test_from_catalog_col_comment_promoted(self):
        """comment field is promoted into metadata when type_json metadata is absent."""
        ci = CatalogColumnInfo(
            name="note",
            type_text="STRING",
            type_name=ColumnTypeName.STRING,
            type_json=json.dumps({
                "name": "note",
                "type": "string",
                "nullable": True,
                "metadata": {},
            }),
            nullable=True,
            comment="the note column",
            position=2,
        )
        col = Column.from_api(_stub_table(), ci)
        assert col.metadata.get(b"comment") == b"the note column"


# ─────────────────────────────────────────────────────────────────────────────
# Unit — Columns service: parse_location
# ─────────────────────────────────────────────────────────────────────────────


def _svc(
    catalog: str | None = "main",
    schema: str | None = "sales",
    table: str | None = "orders",
    column: str | None = None,
) -> Columns:
    """Build a Columns service with stub defaults (no client needed for parse tests)."""
    mock_client = MagicMock()
    return Columns(
        client=mock_client,
        catalog_name=catalog,
        schema_name=schema,
        table_name=table,
        column_name=column,
    )


class TestColumnsParseLocation:
    def test_one_part_uses_all_defaults(self):
        svc = _svc()
        c, s, t, col = svc.parse_location("price")
        assert (c, s, t, col) == ("main", "sales", "orders", "price")

    def test_two_parts_overrides_table(self):
        svc = _svc()
        c, s, t, col = svc.parse_location("inventory.qty")
        assert (c, s, t, col) == ("main", "sales", "inventory", "qty")

    def test_three_parts_overrides_schema_and_table(self):
        svc = _svc()
        c, s, t, col = svc.parse_location("finance.accounts.balance")
        assert (c, s, t, col) == ("main", "finance", "accounts", "balance")

    def test_four_parts_fully_qualified(self):
        svc = _svc()
        c, s, t, col = svc.parse_location("cat.sch.tbl.col")
        assert (c, s, t, col) == ("cat", "sch", "tbl", "col")

    def test_five_parts_uses_rightmost_four(self):
        svc = _svc()
        c, s, t, col = svc.parse_location("extra.cat.sch.tbl.col")
        assert (c, s, t, col) == ("cat", "sch", "tbl", "col")

    def test_backtick_stripped(self):
        svc = _svc()
        c, s, t, col = svc.parse_location("`cat`.`sch`.`tbl`.`col`")
        assert (c, s, t, col) == ("cat", "sch", "tbl", "col")

    def test_no_defaults_one_part(self):
        svc = _svc(catalog=None, schema=None, table=None)
        c, s, t, col = svc.parse_location("price")
        assert (c, s, t, col) == (None, None, None, "price")

    def test_two_parts_no_defaults(self):
        svc = _svc(catalog=None, schema=None, table=None)
        c, s, t, col = svc.parse_location("orders.price")
        assert (c, s, t, col) == (None, None, "orders", "price")


class TestColumnsResolveTableParts:
    def test_location_provides_all_four(self):
        svc = _svc(catalog=None, schema=None, table=None)
        cat, sch, tbl, col = svc._resolve_table_parts("a.b.c.d", None, None, None, None)
        assert (cat, sch, tbl, col) == ("a", "b", "c", "d")

    def test_explicit_kwargs_override_location(self):
        svc = _svc()
        cat, sch, tbl, _ = svc._resolve_table_parts(
            "cat.sch.tbl.col",
            catalog_name="override_cat",
            schema_name=None,
            table_name=None,
            column_name=None,
        )
        assert cat == "override_cat"

    def test_service_defaults_fill_gaps(self):
        svc = _svc(catalog="my_cat", schema="my_sch", table="my_tbl")
        cat, sch, tbl, col = svc._resolve_table_parts(None, None, None, None, "col")
        assert (cat, sch, tbl, col) == ("my_cat", "my_sch", "my_tbl", "col")

    def test_missing_catalog_raises(self):
        svc = _svc(catalog=None, schema="s", table="t")
        with pytest.raises(AssertionError, match="catalog"):
            svc._resolve_table_parts(None, None, None, None, None)

    def test_missing_schema_raises(self):
        svc = _svc(catalog="c", schema=None, table="t")
        with pytest.raises(AssertionError, match="schema"):
            svc._resolve_table_parts(None, None, None, None, None)

    def test_missing_table_raises(self):
        svc = _svc(catalog="c", schema="s", table=None)
        with pytest.raises(AssertionError, match="table"):
            svc._resolve_table_parts(None, None, None, None, None)


class TestColumnsColumn:
    def test_delegates_to_table_column(self):
        """Columns.column() looks up the table then calls table.column()."""
        mock_client = MagicMock()
        mock_table = MagicMock(spec=Table)
        mock_col = MagicMock(spec=Column)
        mock_table.column.return_value = mock_col
        mock_client.tables.find_table.return_value = mock_table

        svc = Columns(
            client=mock_client,
            catalog_name="main",
            schema_name="sales",
            table_name="orders",
        )
        result = svc.column("price")

        mock_client.tables.find_table.assert_called_once_with(
            catalog_name="main", schema_name="sales", table_name="orders"
        )
        mock_table.column.assert_called_once_with("price")
        assert result is mock_col

    def test_four_part_location(self):
        mock_client = MagicMock()
        mock_table = MagicMock(spec=Table)
        mock_table.column.return_value = MagicMock(spec=Column)
        mock_client.tables.find_table.return_value = mock_table

        svc = Columns(client=mock_client)
        svc.column("cat.sch.tbl.col")

        mock_client.tables.find_table.assert_called_once_with(
            catalog_name="cat", schema_name="sch", table_name="tbl"
        )
        mock_table.column.assert_called_once_with("col")

    def test_missing_column_name_raises(self):
        svc = Columns(
            client=MagicMock(),
            catalog_name="c",
            schema_name="s",
            table_name="t",
        )
        # No location, no column_name — must raise
        with pytest.raises(AssertionError, match="column"):
            svc.column()

    def test_column_name_kwarg(self):
        mock_client = MagicMock()
        mock_table = MagicMock(spec=Table)
        mock_table.column.return_value = MagicMock(spec=Column)
        mock_client.tables.find_table.return_value = mock_table

        svc = Columns(client=mock_client, catalog_name="c", schema_name="s", table_name="t")
        svc.column(column_name="my_col")
        mock_table.column.assert_called_once_with("my_col")

    def test_service_default_column_name(self):
        mock_client = MagicMock()
        mock_table = MagicMock(spec=Table)
        mock_table.column.return_value = MagicMock(spec=Column)
        mock_client.tables.find_table.return_value = mock_table

        svc = Columns(
            client=mock_client,
            catalog_name="c",
            schema_name="s",
            table_name="t",
            column_name="default_col",
        )
        svc.column()
        mock_table.column.assert_called_once_with("default_col")


class TestColumnsListColumns:
    def test_returns_table_columns(self):
        mock_client = MagicMock()
        mock_table = MagicMock(spec=Table)
        fake_cols = [MagicMock(spec=Column), MagicMock(spec=Column)]
        mock_table.columns = fake_cols
        mock_client.tables.find_table.return_value = mock_table

        svc = Columns(client=mock_client, catalog_name="c", schema_name="s", table_name="t")
        result = svc.list_columns()

        assert result is fake_cols
        mock_client.tables.find_table.assert_called_once_with(
            catalog_name="c", schema_name="s", table_name="t"
        )

    def test_location_overrides_defaults(self):
        mock_client = MagicMock()
        mock_table = MagicMock(spec=Table)
        mock_table.columns = []
        mock_client.tables.find_table.return_value = mock_table

        svc = Columns(client=mock_client, catalog_name="c", schema_name="s", table_name="t")
        svc.list_columns("x.y.z.ignored_col")

        mock_client.tables.find_table.assert_called_once_with(
            catalog_name="x", schema_name="y", table_name="z"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Unit — _resolve_ref_args
# ─────────────────────────────────────────────────────────────────────────────


def _stub_client_with_table(found_table: MagicMock) -> MagicMock:
    """Return a mock client whose tables.find_table() returns *found_table*."""
    cl = MagicMock()
    cl.tables.find_table.return_value = found_table
    return cl


def _col_with_client(
    name: str = "price",
    tbl_catalog: str = "main",
    tbl_schema: str = "sales",
    tbl_name: str = "orders",
    mock_client: MagicMock | None = None,
) -> Column:
    """Build a Column whose table has a real mock client attached."""
    tbl = MagicMock(spec=Table)
    tbl.name = tbl_name
    tbl.catalog_name = tbl_catalog
    tbl.schema_name = tbl_schema
    tbl.table_name = tbl_name
    tbl.full_name.side_effect = lambda safe=None: (
        f"`{tbl_catalog}`.`{tbl_schema}`.`{tbl_name}`" if safe
        else f"{tbl_catalog}.{tbl_schema}.{tbl_name}"
    )
    tbl._safe_str.side_effect = lambda v: v if isinstance(v, str) else v.decode()
    tbl.client = mock_client or MagicMock()
    af = pa.field(name, pa.int64())
    return Column(table=tbl, name=name, dfield=Field.from_arrow(af))


class TestResolveRefArgs:
    def test_column_object_sets_ref_table_and_column(self):
        ref_tbl = _stub_table("main", "sales", "customers")
        ref_col = _col("id", table=ref_tbl)
        src_col = _col_with_client()

        tbl_out, col_out = src_col._resolve_ref_args(ref_col, None, None)
        assert tbl_out is ref_tbl
        assert col_out is ref_col

    def test_column_object_does_not_override_explicit_ref_table(self):
        other_tbl = _stub_table("x", "y", "z")
        ref_col = _col("id")
        src_col = _col_with_client()

        tbl_out, col_out = src_col._resolve_ref_args(ref_col, other_tbl, None)
        assert tbl_out is other_tbl   # explicit ref_table wins
        assert col_out is ref_col

    def test_column_object_does_not_override_explicit_ref_column(self):
        ref_tbl = _stub_table()
        ref_col_obj = _col("id", table=ref_tbl)
        src_col = _col_with_client()

        tbl_out, col_out = src_col._resolve_ref_args(ref_col_obj, None, "explicit_col")
        assert col_out == "explicit_col"   # explicit ref_column wins

    def test_string_one_part_defaults_to_own_table(self):
        own_tbl_mock = MagicMock(spec=Table)
        own_tbl_mock.catalog_name = "main"
        own_tbl_mock.schema_name = "sales"
        own_tbl_mock.table_name = "orders"
        own_tbl_mock.client = _stub_client_with_table(own_tbl_mock)

        af = pa.field("price", pa.int64())
        src_col = Column(table=own_tbl_mock, name="price", dfield=Field.from_arrow(af))

        tbl_out, col_out = src_col._resolve_ref_args("price", None, None)
        # table defaults to self.table (parsed table_name == "orders" → resolved to own_tbl_mock)
        assert col_out == "price"
        assert tbl_out is own_tbl_mock  # find_table returned own_tbl_mock

    def test_string_four_part_looks_up_foreign_table(self):
        foreign_tbl = MagicMock(spec=Table)
        cl = _stub_client_with_table(foreign_tbl)
        src_col = _col_with_client(mock_client=cl)

        tbl_out, col_out = src_col._resolve_ref_args("cat.sch.tbl.id", None, None)

        cl.tables.find_table.assert_called_once_with(
            catalog_name="cat", schema_name="sch", table_name="tbl"
        )
        assert tbl_out is foreign_tbl
        assert col_out == "id"

    def test_string_ref_table_is_resolved(self):
        foreign_tbl = MagicMock(spec=Table)
        cl = _stub_client_with_table(foreign_tbl)
        src_col = _col_with_client(mock_client=cl)

        tbl_out, col_out = src_col._resolve_ref_args(None, "cat.sch.tbl", "id")

        cl.tables.find_table.assert_called_once_with(location="cat.sch.tbl")
        assert tbl_out is foreign_tbl
        assert col_out == "id"

    def test_none_ref_table_defaults_to_own_table(self):
        own_tbl = _stub_table()
        af = pa.field("id", pa.int64())
        src_col = Column(table=own_tbl, name="id", dfield=Field.from_arrow(af))

        tbl_out, col_out = src_col._resolve_ref_args(None, None, "id")
        assert tbl_out is own_tbl

    def test_table_object_ref_table_not_resolved_via_client(self):
        explicit_tbl = _stub_table("a", "b", "c")
        cl = MagicMock()
        src_col = _col_with_client(mock_client=cl)

        src_col._resolve_ref_args(None, explicit_tbl, "id")
        cl.tables.find_table.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Integration — live Databricks workspace
# ─────────────────────────────────────────────────────────────────────────────


class TestColumnIntegration(DatabricksCase):
    pytestmark = [requires_databricks, pytest.mark.integration]
    """
    Integration tests that create two scratch Delta tables in
    ``trading.unittest``:

    - **test_col_ref** — single ``id BIGINT NOT NULL`` column (PK target)
    - **test_col_main** — ``id BIGINT NOT NULL, name STRING`` columns

    All constraint/tag operations are exercised against the live workspace and
    cleaned up in ``tearDownClass``.
    """

    _CATALOG = "trading"
    _SCHEMA_NAME = "unittest"

    # Arrow schemas — NOT NULL on ``id`` is required by Databricks PK constraints
    _REF_SCHEMA = pa.schema([pa.field("id", pa.int64(), nullable=False)])
    _MAIN_SCHEMA = pa.schema([
        pa.field("id", pa.int64(), nullable=False),
        pa.field("name", pa.string(), nullable=True),
    ])

    engine: SQLEngine = None
    table: Table = None
    ref_table: Table = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.engine = cls.workspace.sql(
            catalog_name=cls._CATALOG,
            schema_name=cls._SCHEMA_NAME,
        )

        # Reference table (created first so FK can reference it)
        ref_data = pa.table(
            [pa.array([1, 2, 3], type=pa.int64())],
            schema=cls._REF_SCHEMA,
        )
        cls.ref_table = cls.engine.table("test_col_ref").create(
            ref_data, if_not_exists=True
        )

        # Main table
        data = pa.table(
            [
                pa.array([1, 2, 3], type=pa.int64()),
                pa.array(["a", "b", "c"]),
            ],
            schema=cls._MAIN_SCHEMA,
        )
        cls.table = cls.engine.table("test_col_main").create(
            data, if_not_exists=True
        )

    @classmethod
    def tearDownClass(cls) -> None:
        for tbl in (cls.table, cls.ref_table):
            try:
                if tbl is not None:
                    tbl.delete()
            except Exception:
                pass
        super().tearDownClass()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _col(self, name: str) -> Column:
        return self.table.column(name)

    def _ref_col(self, name: str) -> Column:
        return self.ref_table.column(name)

    # ── column access ─────────────────────────────────────────────────────────

    def test_column_access_by_name(self):
        col = self._col("id")
        assert isinstance(col, Column)
        assert col.name == "id"

    def test_column_arrow_field_type(self):
        col = self._col("id")
        assert col.arrow_field.type == pa.int64()

    def test_columns_list_contains_schema_names(self):
        names = [c.name for c in self.table.columns]
        assert "id" in names
        assert "name" in names

    def test_table_getitem_returns_column(self):
        col = self.table["id"]
        assert isinstance(col, Column)
        assert col.name == "id"

    # ── DDL targets the correct table ─────────────────────────────────────────

    def test_add_pk_ddl_targets_table(self):
        col = self._col("id")
        ddl = col.add_primary_key_ddl(constraint_name="it_pk_check")
        assert self.table.full_name(safe=True) in ddl

    def test_add_fk_ddl_targets_table_and_ref(self):
        col = self._col("id")
        ddl = col.add_foreign_key_ddl(
            ref_table=self.ref_table,
            ref_column="id",
            constraint_name="it_fk_check",
        )
        assert self.table.full_name(safe=True) in ddl
        assert self.ref_table.full_name(safe=True) in ddl

    # ── set_tags ──────────────────────────────────────────────────────────────

    def test_set_tags_executes_without_error(self):
        col = self._col("name")
        col.set_tags({"test_tag": "yggdrasil_ci"})

    # ── primary key lifecycle ─────────────────────────────────────────────────

    def test_primary_key_add_and_drop(self):
        col = self._col("id")
        col.set_primary_key(constraint_name="it_main_pk", rely=False)
        col.unset_primary_key(if_exists=True, cascade=True)

    # ── foreign key lifecycle ─────────────────────────────────────────────────

    def test_foreign_key_lifecycle(self):
        ref_col = self._ref_col("id")
        id_col = self._col("id")

        # PK must exist on ref table before FK can be added
        ref_col.set_primary_key(constraint_name="it_ref_pk")
        try:
            id_col.set_foreign_key(
                ref_table=self.ref_table,
                ref_column=ref_col,
                constraint_name="it_main_fk",
            )
            id_col.unset_foreign_key(if_exists=True)
        finally:
            ref_col.unset_primary_key(if_exists=True, cascade=True)

    def test_foreign_key_via_string_ref_table(self):
        """set_foreign_key(ref_table=<str>) resolves the table via Tables.find_table."""
        ref_col = self._ref_col("id")
        id_col = self._col("id")

        ref_col.set_primary_key(constraint_name="it_ref_pk_str")
        try:
            id_col.set_foreign_key(
                ref_table=self.ref_table.full_name(),  # plain string
                ref_column="id",
                constraint_name="it_fk_via_str_table",
            )
            id_col.unset_foreign_key(if_exists=True)
        finally:
            ref_col.unset_primary_key(if_exists=True, cascade=True)

    def test_foreign_key_via_column_string_fourpart(self):
        """set_foreign_key('cat.sch.tbl.col') resolves both ref_table and ref_column."""
        ref_col = self._ref_col("id")
        id_col = self._col("id")

        four_part = (
            f"{self.ref_table.catalog_name}"
            f".{self.ref_table.schema_name}"
            f".{self.ref_table.table_name}"
            f".id"
        )

        ref_col.set_primary_key(constraint_name="it_ref_pk_4p")
        try:
            id_col.set_foreign_key(four_part, constraint_name="it_fk_via_4part")
            id_col.unset_foreign_key(if_exists=True)
        finally:
            ref_col.unset_primary_key(if_exists=True, cascade=True)

    # ── columns_service ───────────────────────────────────────────────────────

    def test_table_columns_service_returns_columns_instance(self):
        svc = self.table.columns_service
        assert isinstance(svc, Columns)
        assert svc.catalog_name == self.table.catalog_name
        assert svc.schema_name == self.table.schema_name
        assert svc.table_name == self.table.table_name

    def test_columns_service_list_columns(self):
        cols = self.table.columns_service.list_columns()
        names = [c.name for c in cols]
        assert "id" in names
        assert "name" in names

    def test_columns_service_find_column_by_name(self):
        col = self.table.columns_service.column("id")
        assert isinstance(col, Column)
        assert col.name == "id"

    def test_client_columns_fully_qualified(self):
        four_part = (
            f"{self.table.catalog_name}"
            f".{self.table.schema_name}"
            f".{self.table.table_name}"
            f".id"
        )
        col = self.workspace.columns.column(four_part)
        assert isinstance(col, Column)
        assert col.name == "id"

