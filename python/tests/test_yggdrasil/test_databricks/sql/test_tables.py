from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from databricks.sdk.service.catalog import CatalogInfo, SchemaInfo, TableInfo

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.sql import Tables
from yggdrasil.databricks.sql.table import Table


def _tbl_info(name: str, *, catalog: str = "main", schema: str = "sales") -> TableInfo:
    return TableInfo(catalog_name=catalog, schema_name=schema, name=name, table_id=f"id-{name}")


def _cat_info(name: str) -> CatalogInfo:
    return CatalogInfo(name=name)


def _sch_info(name: str, *, catalog: str = "main") -> SchemaInfo:
    return SchemaInfo(catalog_name=catalog, name=name)


@pytest.fixture()
def mock_client():
    client = MagicMock(spec=DatabricksClient)
    client.base_url.to_string.return_value = "https://adb-123.azuredatabricks.net"
    client.base_url.with_path.side_effect = lambda p: MagicMock(
        to_string=lambda: f"https://adb-123.azuredatabricks.net{p}"
    )
    return client


@pytest.fixture()
def mock_ws(mock_client):
    ws = MagicMock()
    mock_client.workspace_client.return_value = ws
    return ws


@pytest.fixture()
def tables(mock_client):
    return Tables(client=mock_client, catalog_name="main", schema_name="sales")


class TestTablesList:
    def test_list_tables_without_name_filter_yields_all(self, tables, mock_ws):
        mock_ws.tables.list.return_value = [
            _tbl_info("orders"),
            _tbl_info("customers"),
        ]

        result = list(tables.list_tables())

        assert [table.table_name for table in result] == ["orders", "customers"]

    def test_list_tables_exact_name_filter(self, tables, mock_ws):
        mock_ws.tables.list.return_value = [
            _tbl_info("orders"),
            _tbl_info("OrdersArchive"),
        ]

        result = list(tables.list_tables(name="orders"))

        assert [table.table_name for table in result] == ["orders"]

    def test_list_tables_glob_filter_is_case_insensitive(self, tables, mock_ws):
        mock_ws.tables.list.return_value = [
            _tbl_info("Orders"),
            _tbl_info("orders_archive"),
            _tbl_info("customers"),
        ]

        result = list(tables.list_tables(name="ord*"))

        assert [table.table_name for table in result] == ["Orders", "orders_archive"]

    def test_list_tables_without_schema_lists_all_schemas_in_catalog(self, mock_client, mock_ws):
        tables = Tables(client=mock_client, catalog_name="main")
        mock_ws.schemas.list.return_value = [
            _sch_info("sales", catalog="main"),
            _sch_info("analytics", catalog="main"),
        ]
        mock_ws.tables.list.side_effect = [
            [_tbl_info("orders", catalog="main", schema="sales")],
            [_tbl_info("daily_rollup", catalog="main", schema="analytics")],
        ]

        result = list(tables.list_tables())

        assert [(table.schema_name, table.table_name) for table in result] == [
            ("sales", "orders"),
            ("analytics", "daily_rollup"),
        ]

    def test_list_tables_without_catalog_or_schema_lists_all_visible_tables(self, mock_client, mock_ws):
        tables = Tables(client=mock_client)
        mock_ws.catalogs.list.return_value = [
            _cat_info("main"),
            _cat_info("staging"),
        ]
        mock_ws.schemas.list.side_effect = [
            [_sch_info("sales", catalog="main")],
            [_sch_info("raw", catalog="staging")],
        ]
        mock_ws.tables.list.side_effect = [
            [_tbl_info("orders", catalog="main", schema="sales")],
            [_tbl_info("events", catalog="staging", schema="raw")],
        ]

        result = list(tables.list_tables())

        assert [
            (table.catalog_name, table.schema_name, table.table_name)
            for table in result
        ] == [
            ("main", "sales", "orders"),
            ("staging", "raw", "events"),
        ]

    def test_list_tables_glob_catalog_fans_out_across_matching_catalogs(self, mock_client, mock_ws):
        tables = Tables(client=mock_client, schema_name="sales")
        mock_ws.catalogs.list.return_value = [
            _cat_info("prod_main"),
            _cat_info("prod_staging"),
            _cat_info("dev_main"),
        ]
        mock_ws.tables.list.side_effect = [
            [_tbl_info("orders", catalog="prod_main", schema="sales")],
            [_tbl_info("events", catalog="prod_staging", schema="sales")],
        ]

        result = list(tables.list_tables(catalog_name="prod_*"))

        assert [(t.catalog_name, t.table_name) for t in result] == [
            ("prod_main", "orders"),
            ("prod_staging", "events"),
        ]

    def test_list_tables_glob_schema_fans_out_across_matching_schemas(self, mock_client, mock_ws):
        tables = Tables(client=mock_client, catalog_name="main")
        mock_ws.schemas.list.return_value = [
            _sch_info("sales_us", catalog="main"),
            _sch_info("sales_eu", catalog="main"),
            _sch_info("analytics", catalog="main"),
        ]
        mock_ws.tables.list.side_effect = [
            [_tbl_info("orders", catalog="main", schema="sales_us")],
            [_tbl_info("orders", catalog="main", schema="sales_eu")],
        ]

        result = list(tables.list_tables(schema_name="sales_*"))

        assert [(t.schema_name, t.table_name) for t in result] == [
            ("sales_us", "orders"),
            ("sales_eu", "orders"),
        ]

    def test_list_tables_glob_name_with_middle_wildcard(self, tables, mock_ws):
        mock_ws.tables.list.return_value = [
            _tbl_info("prefix_a_table"),
            _tbl_info("prefix_b_table"),
            _tbl_info("other"),
        ]

        result = list(tables.list_tables(name="prefix_*_table"))

        assert [t.table_name for t in result] == ["prefix_a_table", "prefix_b_table"]

    def test_list_tables_star_name_matches_all(self, tables, mock_ws):
        mock_ws.tables.list.return_value = [
            _tbl_info("a"),
            _tbl_info("b"),
        ]

        result = list(tables.list_tables(name="*"))

        assert [t.table_name for t in result] == ["a", "b"]


class TestTablesGetitem:
    def test_1part_uses_service_defaults(self, tables):
        result = tables["orders"]
        assert isinstance(result, Table)
        assert result.catalog_name == "main"
        assert result.schema_name == "sales"
        assert result.table_name == "orders"

    def test_1part_without_defaults_raises(self, mock_client):
        svc = Tables(client=mock_client)
        with pytest.raises(ValueError, match="default catalog_name"):
            _ = svc["orders"]

    def test_2part_uses_catalog_default(self, tables):
        result = tables["analytics.events"]
        assert isinstance(result, Table)
        assert result.catalog_name == "main"
        assert result.schema_name == "analytics"
        assert result.table_name == "events"

    def test_2part_without_catalog_default_raises(self, mock_client):
        svc = Tables(client=mock_client)
        with pytest.raises(ValueError, match="default catalog_name"):
            _ = svc["analytics.events"]

    def test_3part_fully_qualified(self, mock_client):
        svc = Tables(client=mock_client)
        result = svc["main.sales.orders"]
        assert isinstance(result, Table)
        assert (result.catalog_name, result.schema_name, result.table_name) == (
            "main", "sales", "orders",
        )

    def test_4part_delegates_to_columns_service(self, tables, mock_client):
        mock_columns = MagicMock()
        mock_client.columns = mock_columns
        result = tables["main.sales.orders.price"]
        mock_columns.column.assert_called_once_with("main.sales.orders.price")
        assert result is mock_columns.column.return_value

    def test_backticks_stripped(self, tables):
        result = tables["`analytics`.`events`"]
        assert result.schema_name == "analytics"
        assert result.table_name == "events"

    def test_too_many_parts_raises(self, tables):
        with pytest.raises(KeyError, match="1- to 4-part"):
            _ = tables["a.b.c.d.e"]


class TestTablesIter:
    def test_iter_delegates_to_list_tables(self, tables, mock_ws):
        mock_ws.tables.list.return_value = [
            _tbl_info("orders"),
            _tbl_info("customers"),
        ]
        result = list(iter(tables))
        assert [t.table_name for t in result] == ["orders", "customers"]


class TestTablesSetitem:
    def test_setitem_renames_table_via_defaults(self, tables, mock_client):
        mock_engine = MagicMock()
        mock_client.sql.return_value = mock_engine

        tables["orders"] = "orders_v2"

        mock_engine.execute.assert_called_once()
        stmt = mock_engine.execute.call_args[0][0]
        assert "ALTER TABLE" in stmt
        assert "`main`.`sales`.`orders`" in stmt
        assert "RENAME TO `orders_v2`" in stmt

