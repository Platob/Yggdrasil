"""
Unit tests for :meth:`Tables.fetch_all_infos`, :attr:`Table.all_infos`, and
the shared caching with :attr:`Table.infos`.
"""

from __future__ import annotations

import time
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
def tables(mock_client):
    return Tables(client=mock_client, catalog_name="main", schema_name="sales")


@pytest.fixture()
def primed_table(mock_client, tables, table_infos):
    """Table with a fresh basic ``_infos`` cache — no network access required."""
    tb = Table(
        service=tables,
        catalog_name="main",
        schema_name="sales",
        table_name="orders",
    )
    tb._store_infos(table_infos)
    # Route Table.sql.tables back to the same `tables` service instance so
    # ``Table.all_infos`` delegates through our fixture.
    mock_client.sql.return_value.tables = tables
    mock_client.tables = tables
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


# ── Tables.fetch_all_infos ────────────────────────────────────────────────────


class TestTablesFetchAllInfos:
    def test_gathers_tags_column_tags_and_grants(self, tables, mock_ws, table_infos):
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

        result = tables.fetch_all_infos(
            catalog_name="main",
            schema_name="sales",
            table_name="orders",
            infos=table_infos,
        )

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

    def test_include_effective_grants_calls_get_effective(self, tables, mock_ws, table_infos):
        mock_ws.entity_tag_assignments.list.return_value = iter([])
        mock_ws.grants.get.return_value = GetPermissionsResponse(privilege_assignments=[])
        mock_ws.grants.get_effective.return_value = GetPermissionsResponse(
            privilege_assignments=[_assignment("bob", [Privilege.MODIFY])]
        )

        result = tables.fetch_all_infos(
            catalog_name="main",
            schema_name="sales",
            table_name="orders",
            infos=table_infos,
            include_tags=False,
            include_column_tags=False,
            include_grants=False,
            include_effective_grants=True,
        )

        assert [g.principal for g in result.effective_grants] == ["bob"]
        mock_ws.grants.get_effective.assert_called_once()

    def test_disable_flags_skip_fetches(self, tables, mock_ws, table_infos):
        mock_ws.grants.get.return_value = GetPermissionsResponse(privilege_assignments=[])

        result = tables.fetch_all_infos(
            catalog_name="main",
            schema_name="sales",
            table_name="orders",
            infos=table_infos,
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

    def test_swallows_tag_errors_by_default(self, tables, mock_ws, table_infos):
        mock_ws.entity_tag_assignments.list.side_effect = RuntimeError("tags API off")
        mock_ws.grants.get.return_value = GetPermissionsResponse(
            privilege_assignments=[_assignment("alice", [Privilege.SELECT])]
        )

        result = tables.fetch_all_infos(
            catalog_name="main",
            schema_name="sales",
            table_name="orders",
            infos=table_infos,
        )

        # Tag fetches failed silently; grants still populated.
        assert result.tags == ()
        assert result.column_tags == {}
        assert [g.principal for g in result.grants] == ["alice"]

    def test_raise_error_propagates_tag_failure(self, tables, mock_ws, table_infos):
        mock_ws.entity_tag_assignments.list.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            tables.fetch_all_infos(
                catalog_name="main",
                schema_name="sales",
                table_name="orders",
                infos=table_infos,
                raise_error=True,
            )

    def test_missing_tag_api_returns_empty(self, tables, mock_ws, table_infos):
        # Some older workspaces / SDKs do not expose entity_tag_assignments.
        del mock_ws.entity_tag_assignments
        mock_ws.grants.get.return_value = GetPermissionsResponse(privilege_assignments=[])

        result = tables.fetch_all_infos(
            catalog_name="main",
            schema_name="sales",
            table_name="orders",
            infos=table_infos,
        )

        assert result.tags == ()
        assert result.column_tags == {}

    def test_fetches_basic_infos_when_none_provided(self, tables, mock_ws, mock_client, table_infos):
        mock_ws.tables.get.return_value = table_infos
        mock_ws.entity_tag_assignments.list.return_value = iter([])
        mock_ws.grants.get.return_value = GetPermissionsResponse(privilege_assignments=[])

        result = tables.fetch_all_infos(
            catalog_name="main",
            schema_name="sales",
            table_name="orders",
        )

        mock_ws.tables.get.assert_called_once_with(full_name="main.sales.orders")
        assert result.infos is table_infos


# ── Table.all_infos cached property + mutualization with infos ────────────────


class TestTableAllInfosProperty:
    def test_all_infos_delegates_to_service_and_caches(
        self, primed_table, tables, mock_ws, table_infos
    ):
        mock_ws.entity_tag_assignments.list.return_value = iter([])
        mock_ws.grants.get.return_value = GetPermissionsResponse(
            privilege_assignments=[_assignment("alice", [Privilege.SELECT])]
        )
        # Spy on the service method to count calls.
        original = tables.fetch_all_infos
        call_count = {"n": 0}

        def spy(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        object.__setattr__(tables, "fetch_all_infos", spy)

        snap1 = primed_table.all_infos
        snap2 = primed_table.all_infos

        assert snap1 is snap2  # cached on the Table
        assert call_count["n"] == 1
        assert isinstance(snap1, TableAllInfos)
        assert snap1.infos is table_infos
        assert [g.principal for g in snap1.grants] == ["alice"]

    def test_all_infos_reuses_primed_infos_as_seed(
        self, primed_table, tables, mock_ws, mock_client, table_infos
    ):
        mock_ws.entity_tag_assignments.list.return_value = iter([])
        mock_ws.grants.get.return_value = GetPermissionsResponse(privilege_assignments=[])

        _ = primed_table.all_infos

        # The seed was supplied, so no basic lookup hit the SDK tables API.
        mock_ws.tables.get.assert_not_called()

    def test_all_infos_seeds_basic_infos_cache(
        self, primed_table, tables, mock_ws, table_infos
    ):
        # Wipe the primed caches so we're sure the seeding path is what populates _infos.
        primed_table._reset_cache()

        refreshed = TableInfo(
            catalog_name="main",
            schema_name="sales",
            name="orders",
            table_id="tbl-123",
            owner="carol@example.com",
            columns=[ColumnInfo(name="id", position=0)],
        )
        mock_ws.tables.get.return_value = refreshed
        mock_ws.entity_tag_assignments.list.return_value = iter([])
        mock_ws.grants.get.return_value = GetPermissionsResponse(privilege_assignments=[])

        snap = primed_table.all_infos
        assert snap.infos is refreshed

        # A follow-up `infos` read should not re-fetch — it lives in _infos now.
        mock_ws.tables.get.reset_mock()
        assert primed_table.infos is refreshed
        mock_ws.tables.get.assert_not_called()

    def test_infos_expired_cache_triggers_refetch(
        self, primed_table, mock_ws, table_infos
    ):
        # Age the basic-infos cache past the TTL horizon.
        object.__setattr__(primed_table, "_infos_fetched_at", time.time() - 10_000)

        refreshed = TableInfo(
            catalog_name="main",
            schema_name="sales",
            name="orders",
            table_id="tbl-123",
            owner="carol@example.com",
            columns=[ColumnInfo(name="id", position=0)],
        )
        mock_ws.tables.get.return_value = refreshed

        result = primed_table.infos
        assert result is refreshed

    def test_reset_cache_clears_both_caches(self, primed_table, table_infos):
        # Populate _all_infos manually.
        snapshot = TableAllInfos(table=primed_table, infos=table_infos)
        primed_table._store_all_infos(snapshot)

        primed_table._reset_cache()

        assert primed_table._infos is None
        assert primed_table._all_infos is None
        assert primed_table._columns is None


# ── TableAllInfos helpers ─────────────────────────────────────────────────────


class TestTableAllInfosHelpers:
    def test_convenience_properties(self, primed_table, table_infos):
        snapshot = TableAllInfos(table=primed_table, infos=table_infos)

        assert snapshot.full_name == "main.sales.orders"
        assert snapshot.owner == "alice@example.com"
        assert snapshot.comment == "Orders fact table"
        assert snapshot.row_filter is table_infos.row_filter
        assert snapshot.properties == {"delta.enableChangeDataFeed": "true"}
        assert [c.name for c in snapshot.columns] == ["id", "amount"]

    def test_to_dict_is_json_friendly(self, primed_table, table_infos):
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
            table=primed_table,
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
