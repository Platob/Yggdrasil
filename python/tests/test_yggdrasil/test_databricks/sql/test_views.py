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


@pytest.fixture()
def mock_ws(mock_client):
    ws = MagicMock()
    mock_client.workspace_client.return_value = ws
    return ws


class TestViewsIter:
    def test_iter_delegates_to_list_views(self, views, mock_ws):
        from databricks.sdk.service.catalog import TableInfo, TableType

        def _view(name):
            return TableInfo(
                catalog_name="main", schema_name="sales", name=name,
                table_type=TableType.VIEW,
            )

        mock_ws.tables.list.return_value = [_view("v1"), _view("v2")]

        result = list(iter(views))
        assert [v.view_name for v in result] == ["v1", "v2"]


class TestViewsSetitem:
    def test_setitem_renames_view_via_defaults(self, views, mock_client):
        mock_engine = MagicMock()
        mock_client.sql.return_value = mock_engine

        views["orders_summary"] = "orders_summary_v2"

        mock_engine.execute.assert_called_once()
        stmt = mock_engine.execute.call_args[0][0]
        assert "ALTER VIEW" in stmt
        assert "`main`.`sales`.`orders_summary`" in stmt
        assert "RENAME TO `orders_summary_v2`" in stmt


# ---------------------------------------------------------------------------
# concat_tables
# ---------------------------------------------------------------------------


def _fake_table(
    table_name: str,
    catalog_name: str = "main",
    schema_name: str = "sales",
):
    """Minimal stand-in for :class:`Table` — only ``full_name`` + fields used."""
    tbl = MagicMock()
    tbl.catalog_name = catalog_name
    tbl.schema_name = schema_name
    tbl.table_name = table_name
    tbl.full_name.side_effect = lambda safe=False: (
        f"`{catalog_name}`.`{schema_name}`.`{table_name}`"
        if safe
        else f"{catalog_name}.{schema_name}.{table_name}"
    )
    return tbl


class TestCommonTableNameRoot:
    def test_shared_prefix_strips_trailing_underscore(self):
        assert Views._common_table_name_root(
            ["sales_jan", "sales_feb", "sales_mar"]
        ) == "sales"

    def test_single_name_returns_name_trimmed(self):
        assert Views._common_table_name_root(["orders_"]) == "orders"

    def test_no_common_prefix_returns_empty(self):
        assert Views._common_table_name_root(["alpha", "beta"]) == ""

    def test_empty_inputs_returns_empty(self):
        assert Views._common_table_name_root([]) == ""

    def test_filters_out_empty_entries(self):
        assert Views._common_table_name_root(
            ["sales_jan", "", "sales_feb"]
        ) == "sales"


class TestConcatTables:
    def test_builds_union_all_by_name_view_using_common_prefix(
        self, views, mock_client
    ):
        mock_engine = MagicMock()
        mock_client.sql.return_value = mock_engine

        result = views.concat_tables(
            [
                _fake_table("sales_jan"),
                _fake_table("sales_feb"),
                _fake_table("sales_mar"),
            ]
        )

        assert isinstance(result, View)
        assert result.view_name == "sales"
        assert result.catalog_name == "main"
        assert result.schema_name == "sales"

        mock_engine.execute.assert_called_once()
        stmt = mock_engine.execute.call_args[0][0]
        assert "CREATE OR REPLACE VIEW" in stmt
        assert "`main`.`sales`.`sales`" in stmt
        assert "UNION ALL BY NAME" in stmt
        for name in ("sales_jan", "sales_feb", "sales_mar"):
            assert f"SELECT * FROM `main`.`sales`.`{name}`" in stmt

    def test_by_name_false_uses_plain_union_all(self, views, mock_client):
        mock_engine = MagicMock()
        mock_client.sql.return_value = mock_engine

        views.concat_tables(
            [_fake_table("sales_jan"), _fake_table("sales_feb")],
            by_name=False,
        )

        stmt = mock_engine.execute.call_args[0][0]
        assert "UNION ALL BY NAME" not in stmt
        assert "UNION ALL" in stmt

    def test_explicit_view_name_overrides_prefix_detection(
        self, views, mock_client
    ):
        mock_engine = MagicMock()
        mock_client.sql.return_value = mock_engine

        result = views.concat_tables(
            [_fake_table("alpha"), _fake_table("beta")],
            view_name="combined",
        )

        assert result.view_name == "combined"
        stmt = mock_engine.execute.call_args[0][0]
        assert "`main`.`sales`.`combined`" in stmt

    def test_no_common_prefix_raises_when_view_name_omitted(
        self, views, mock_client
    ):
        with pytest.raises(ValueError, match="no common prefix"):
            views.concat_tables(
                [_fake_table("alpha"), _fake_table("beta")],
            )

    def test_empty_tables_raises(self, views):
        with pytest.raises(ValueError, match="at least one Table"):
            views.concat_tables([])

    def test_falls_back_to_first_tables_catalog_and_schema(self, mock_client):
        svc = Views(client=mock_client)  # no service defaults
        mock_engine = MagicMock()
        mock_client.sql.return_value = mock_engine

        result = svc.concat_tables(
            [
                _fake_table("sales_jan", catalog_name="prod", schema_name="fin"),
                _fake_table("sales_feb", catalog_name="prod", schema_name="fin"),
            ]
        )

        assert result.catalog_name == "prod"
        assert result.schema_name == "fin"
        stmt = mock_engine.execute.call_args[0][0]
        assert "`prod`.`fin`.`sales`" in stmt

    def test_custom_mode_maps_to_view_create(self, views, mock_client):
        from yggdrasil.io.enums.save_mode import SaveMode

        mock_engine = MagicMock()
        mock_client.sql.return_value = mock_engine

        views.concat_tables(
            [_fake_table("sales_jan"), _fake_table("sales_feb")],
            mode=SaveMode.AUTO,
        )

        stmt = mock_engine.execute.call_args[0][0]
        # AUTO in View.create maps to CREATE VIEW IF NOT EXISTS.
        assert "CREATE VIEW IF NOT EXISTS" in stmt

    def test_comment_is_emitted_in_ddl(self, views, mock_client):
        mock_engine = MagicMock()
        mock_client.sql.return_value = mock_engine

        views.concat_tables(
            [_fake_table("sales_jan"), _fake_table("sales_feb")],
            comment="Monthly sales roll-up",
        )

        stmt = mock_engine.execute.call_args[0][0]
        assert "COMMENT 'Monthly sales roll-up'" in stmt
