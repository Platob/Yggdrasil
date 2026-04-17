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

