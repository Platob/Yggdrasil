from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.sql import Views
from yggdrasil.databricks.sql.view import View


@pytest.fixture()
def mock_client():
    client = MagicMock(spec=DatabricksClient)
    client.base_url.to_string.return_value = "https://adb-123.azuredatabricks.net"
    client.base_url.with_path.side_effect = lambda p: MagicMock(
        to_string=lambda: f"https://adb-123.azuredatabricks.net{p}"
    )
    return client


@pytest.fixture()
def views(mock_client):
    return Views(client=mock_client, catalog_name="main", schema_name="sales")


class TestViewsGetitem:
    def test_1part_uses_service_defaults(self, views):
        result = views["orders_summary"]
        assert isinstance(result, View)
        assert result.catalog_name == "main"
        assert result.schema_name == "sales"
        assert result.view_name == "orders_summary"

    def test_1part_without_defaults_raises(self, mock_client):
        svc = Views(client=mock_client)
        with pytest.raises(ValueError, match="default catalog_name"):
            _ = svc["orders_summary"]

    def test_2part_uses_catalog_default(self, views):
        result = views["analytics.daily_rollup"]
        assert isinstance(result, View)
        assert result.catalog_name == "main"
        assert result.schema_name == "analytics"
        assert result.view_name == "daily_rollup"

    def test_2part_without_catalog_default_raises(self, mock_client):
        svc = Views(client=mock_client)
        with pytest.raises(ValueError, match="default catalog_name"):
            _ = svc["analytics.daily_rollup"]

    def test_3part_fully_qualified(self, mock_client):
        svc = Views(client=mock_client)
        result = svc["main.sales.orders_summary"]
        assert isinstance(result, View)
        assert (result.catalog_name, result.schema_name, result.view_name) == (
            "main", "sales", "orders_summary",
        )

    def test_4part_delegates_to_columns_service(self, views, mock_client):
        mock_columns = MagicMock()
        mock_client.columns = mock_columns
        result = views["main.sales.orders_summary.price"]
        mock_columns.column.assert_called_once_with("main.sales.orders_summary.price")
        assert result is mock_columns.column.return_value

    def test_backticks_stripped(self, views):
        result = views["`analytics`.`daily_rollup`"]
        assert result.schema_name == "analytics"
        assert result.view_name == "daily_rollup"

    def test_too_many_parts_raises(self, views):
        with pytest.raises(KeyError, match="1- to 4-part"):
            _ = views["a.b.c.d.e"]
