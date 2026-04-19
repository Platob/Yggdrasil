from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.sql import Columns


@pytest.fixture()
def mock_client():
    client = MagicMock(spec=DatabricksClient)
    client.base_url.to_string.return_value = "https://adb-123.azuredatabricks.net"
    return client


class TestColumnsGetitem:
    def test_getitem_delegates_to_column(self, mock_client):
        svc = Columns(
            client=mock_client,
            catalog_name="main",
            schema_name="sales",
            table_name="orders",
        )
        mock_table = MagicMock()
        mock_column = MagicMock()
        mock_table.column.return_value = mock_column
        mock_client.tables.find_table.return_value = mock_table

        result = svc["price"]

        mock_client.tables.find_table.assert_called_once_with(
            catalog_name="main", schema_name="sales", table_name="orders",
        )
        mock_table.column.assert_called_once_with("price")
        assert result is mock_column

    def test_getitem_2part_uses_catalog_schema_defaults(self, mock_client):
        svc = Columns(
            client=mock_client,
            catalog_name="main",
            schema_name="sales",
        )
        mock_table = MagicMock()
        mock_client.tables.find_table.return_value = mock_table

        svc["orders.price"]

        mock_client.tables.find_table.assert_called_once_with(
            catalog_name="main", schema_name="sales", table_name="orders",
        )
        mock_table.column.assert_called_once_with("price")

    def test_getitem_4part_fully_qualified(self, mock_client):
        svc = Columns(client=mock_client)
        mock_table = MagicMock()
        mock_client.tables.find_table.return_value = mock_table

        svc["main.sales.orders.price"]

        mock_client.tables.find_table.assert_called_once_with(
            catalog_name="main", schema_name="sales", table_name="orders",
        )
        mock_table.column.assert_called_once_with("price")

    def test_getitem_1part_without_defaults_asserts(self, mock_client):
        svc = Columns(client=mock_client)
        with pytest.raises(AssertionError):
            _ = svc["price"]


class TestColumnsIter:
    def test_iter_delegates_to_list_columns(self, mock_client):
        svc = Columns(
            client=mock_client,
            catalog_name="main",
            schema_name="sales",
            table_name="orders",
        )
        mock_table = MagicMock()
        mock_col_a, mock_col_b = MagicMock(), MagicMock()
        mock_table.columns = [mock_col_a, mock_col_b]
        mock_client.tables.find_table.return_value = mock_table

        result = list(iter(svc))
        assert result == [mock_col_a, mock_col_b]


class TestColumnsSetitem:
    def test_setitem_delegates_to_column_rename(self, mock_client):
        svc = Columns(
            client=mock_client,
            catalog_name="main",
            schema_name="sales",
            table_name="orders",
        )
        mock_table = MagicMock()
        mock_column = MagicMock()
        mock_table.column.return_value = mock_column
        mock_client.tables.find_table.return_value = mock_table

        svc["price"] = "unit_price"

        mock_column.rename.assert_called_once_with("unit_price")
