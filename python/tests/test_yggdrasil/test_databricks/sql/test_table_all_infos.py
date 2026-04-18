"""
Unit tests for :meth:`Table.fetch_all_infos` and :class:`TableAllInfos`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from databricks.sdk.service.catalog import (
    ColumnInfo,
    EntityTagAssignment,
    GetPermissionsResponse,
    Privilege,
    PrivilegeAssignment,
    TableInfo,
    TableRowFilter,
)

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.sql import Table, Tables
from yggdrasil.databricks.sql.table import TableAllInfos


# ── fixtures ──────────────────────────────────────────────────────────────────


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
def table_infos():
    return TableInfo(
        catalog_name="main",
        schema_name="sales",
        name="orders",
        full_name="main.sales.orders",
        table_id="tbl-123",
        owner="alice@example.com",
        comment="Orders fact table",
        properties={"delta.enableChangeDataFeed": "true"},
        columns=[
            ColumnInfo(name="id", position=0),
            ColumnInfo(name="amount", position=1),
        ],
        row_filter=TableRowFilter(
            function_name="main.sec.row_filter",
            input_column_names=["id"],
        ),
    )


@pytest.fixture()
def table(mock_client, table_infos):
    service = Tables(client=mock_client, catalog_name="main", schema_name="sales")
    tb = Table(
        service=service,
        catalog_name="main",
        schema_name="sales",
        table_name="orders",
    )
    # Prime the cache so .infos does not hit the network.
    object.__setattr__(tb, "_infos", table_infos)
    object.__setattr__(tb, "_infos_fetched_at", 9e18)
    return tb


def _assignment(principal: str, privileges: list[Privilege]) -> PrivilegeAssignment:
    return PrivilegeAssignment(principal=principal, privileges=privileges)


def _tag(entity_type: str, entity_name: str, key: str, value: str) -> EntityTagAssignment:
    return EntityTagAssignment(
        entity_type=entity_type,
        entity_name=entity_name,
        tag_key=key,
        tag_value=value,
    )


# ── fetch_all_infos ───────────────────────────────────────────────────────────


class TestFetchAllInfos:
    def test_gathers_tags_column_tags_and_grants(self, table, mock_ws, table_infos):
        def fake_tag_list(*, entity_type, entity_name, **_):
            if entity_type == "tables":
                return iter([_tag("tables", entity_name, "pii", "true")])
            if entity_type == "columns" and entity_name.endswith(".id"):
                return iter([_tag("columns", entity_name, "primary_key", "true")])
            return iter([])

        mock_ws.entity_tag_assignments.list.side_effect = fake_tag_list
        mock_ws.grants.get.return_value = GetPermissionsResponse(
            privilege_assignments=[_assignment("alice", [Privilege.SELECT])]
        )

        result = table.fetch_all_infos()

        assert isinstance(result, TableAllInfos)
        assert result.infos is table_infos
        assert result.row_filter is table_infos.row_filter
        assert result.properties == {"delta.enableChangeDataFeed": "true"}
        assert result.owner == "alice@example.com"

        # Table-level tag
        assert len(result.tags) == 1
        assert result.tags[0].tag_key == "pii"

        # Only the "id" column had a tag — "amount" is absent.
        assert set(result.column_tags) == {"id"}
        assert result.column_tags["id"][0].tag_key == "primary_key"

        # Grants arrived, effective grants did not since include_effective_grants=False
        assert [g.principal for g in result.grants] == ["alice"]
        assert result.effective_grants == ()
        mock_ws.grants.get_effective.assert_not_called()

    def test_include_effective_grants_calls_get_effective(self, table, mock_ws):
        mock_ws.entity_tag_assignments.list.return_value = iter([])
        mock_ws.grants.get.return_value = GetPermissionsResponse(privilege_assignments=[])
        mock_ws.grants.get_effective.return_value = GetPermissionsResponse(
            privilege_assignments=[_assignment("bob", [Privilege.MODIFY])]
        )

        result = table.fetch_all_infos(
            include_tags=False,
            include_column_tags=False,
            include_grants=False,
            include_effective_grants=True,
        )

        assert [g.principal for g in result.effective_grants] == ["bob"]
        mock_ws.grants.get_effective.assert_called_once()

    def test_disable_flags_skip_fetches(self, table, mock_ws):
        mock_ws.grants.get.return_value = GetPermissionsResponse(privilege_assignments=[])

        result = table.fetch_all_infos(
            include_tags=False,
            include_column_tags=False,
            include_grants=False,
            include_effective_grants=False,
        )

        mock_ws.entity_tag_assignments.list.assert_not_called()
        mock_ws.grants.get.assert_not_called()
        mock_ws.grants.get_effective.assert_not_called()
        assert result.tags == ()
        assert result.column_tags == {}
        assert result.grants == ()
        assert result.effective_grants == ()

    def test_swallows_tag_errors_by_default(self, table, mock_ws):
        mock_ws.entity_tag_assignments.list.side_effect = RuntimeError("tags API off")
        mock_ws.grants.get.return_value = GetPermissionsResponse(
            privilege_assignments=[_assignment("alice", [Privilege.SELECT])]
        )

        result = table.fetch_all_infos()

        # Tag fetches failed silently; grants still populated.
        assert result.tags == ()
        assert result.column_tags == {}
        assert [g.principal for g in result.grants] == ["alice"]

    def test_raise_error_propagates_tag_failure(self, table, mock_ws):
        mock_ws.entity_tag_assignments.list.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            table.fetch_all_infos(raise_error=True)

    def test_missing_tag_api_returns_empty(self, table, mock_ws):
        # Some older workspaces / SDKs do not expose entity_tag_assignments.
        del mock_ws.entity_tag_assignments
        mock_ws.grants.get.return_value = GetPermissionsResponse(privilege_assignments=[])

        result = table.fetch_all_infos()

        assert result.tags == ()
        assert result.column_tags == {}

    def test_refresh_invalidates_cache(self, table, mock_ws, mock_client, table_infos):
        mock_ws.entity_tag_assignments.list.return_value = iter([])
        mock_ws.grants.get.return_value = GetPermissionsResponse(privilege_assignments=[])

        refreshed = TableInfo(
            catalog_name="main",
            schema_name="sales",
            name="orders",
            table_id="tbl-123",
            owner="carol@example.com",
            columns=[ColumnInfo(name="id", position=0)],
        )
        mock_client.tables.find_table_remote.return_value = refreshed

        result = table.fetch_all_infos(refresh=True)

        mock_client.tables.find_table_remote.assert_called_once()
        assert result.infos is refreshed
        assert result.owner == "carol@example.com"


# ── TableAllInfos helpers ─────────────────────────────────────────────────────


class TestTableAllInfosHelpers:
    def test_convenience_properties(self, table, table_infos):
        snapshot = TableAllInfos(table=table, infos=table_infos)

        assert snapshot.full_name == "main.sales.orders"
        assert snapshot.owner == "alice@example.com"
        assert snapshot.comment == "Orders fact table"
        assert snapshot.row_filter is table_infos.row_filter
        assert snapshot.properties == {"delta.enableChangeDataFeed": "true"}
        assert [c.name for c in snapshot.columns] == ["id", "amount"]

    def test_to_dict_is_json_friendly(self, table, table_infos):
        from yggdrasil.databricks.sql.grants import Grant

        grant = Grant(
            service=MagicMock(),
            securable_type="TABLE",
            full_name="main.sales.orders",
            principal="alice",
            privileges=("SELECT",),
        )
        tag = _tag("tables", "main.sales.orders", "pii", "true")
        col_tag = _tag("columns", "main.sales.orders.id", "pk", "true")

        snapshot = TableAllInfos(
            table=table,
            infos=table_infos,
            tags=(tag,),
            column_tags={"id": (col_tag,)},
            grants=(grant,),
        )

        out = snapshot.to_dict()
        assert out["full_name"] == "main.sales.orders"
        assert out["infos"]["name"] == "orders"
        assert out["tags"][0]["tag_key"] == "pii"
        assert out["column_tags"]["id"][0]["tag_key"] == "pk"
        assert out["grants"] == [
            {"principal": "alice", "privileges": ["SELECT"]}
        ]
        assert out["effective_grants"] == []
