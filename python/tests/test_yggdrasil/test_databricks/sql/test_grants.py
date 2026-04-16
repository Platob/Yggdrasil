"""
Unit tests for the :class:`~yggdrasil.databricks.sql.grants.GrantsMixin`
helpers exposed on :class:`Catalog`, :class:`Schema`, :class:`Table`, and
on :class:`~yggdrasil.databricks.fs.path.VolumePath`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from databricks.sdk.service.catalog import (
    GetPermissionsResponse,
    PermissionsChange,
    Privilege,
    PrivilegeAssignment,
    SecurableType,
)

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.fs.path import VolumePath
from yggdrasil.databricks.sql import Catalog, Catalogs, Schema, Table, Tables
from yggdrasil.databricks.sql.grants import Grant, Grants


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
def grants_api(mock_ws):
    """Return a stub for ``workspace_client().grants``."""
    return mock_ws.grants


@pytest.fixture()
def catalog(mock_client):
    return Catalog(service=Catalogs(client=mock_client), catalog_name="main")


@pytest.fixture()
def schema(mock_client):
    return Schema(
        service=Catalogs(client=mock_client),
        catalog_name="main",
        schema_name="sales",
    )


@pytest.fixture()
def table(mock_client):
    return Table(
        service=Tables(client=mock_client, catalog_name="main", schema_name="sales"),
        catalog_name="main",
        schema_name="sales",
        table_name="orders",
    )


@pytest.fixture()
def volume_path(mock_client):
    p = VolumePath(parts=["main", "sales", "raw"])
    p._client = mock_client
    return p


def _assignment(principal: str, privileges: list[Privilege]) -> PrivilegeAssignment:
    return PrivilegeAssignment(principal=principal, privileges=privileges)


# ── GrantsMixin: securable type + full name ──────────────────────────────────


class TestSecurableTypeAndFullName:
    def test_catalog(self, catalog):
        assert catalog._grants_securable_type() == SecurableType.CATALOG
        assert catalog._grants_full_name() == "main"

    def test_schema(self, schema):
        assert schema._grants_securable_type() == SecurableType.SCHEMA
        assert schema._grants_full_name() == "main.sales"

    def test_table(self, table):
        assert table._grants_securable_type() == SecurableType.TABLE
        assert table._grants_full_name() == "main.sales.orders"

    def test_volume_path(self, volume_path):
        assert volume_path._grants_full_name() == "main.sales.raw"

    def test_volume_path_partial_raises(self, mock_client):
        partial = VolumePath(parts=["main", "sales"])
        partial._client = mock_client
        with pytest.raises(ValueError, match="Cannot manage grants"):
            partial._grants_full_name()


# ── grants_service property ──────────────────────────────────────────────────


class TestGrantsService:
    @pytest.mark.parametrize("name", ["catalog", "schema", "table", "volume_path"])
    def test_returns_grants_service_bound_to_client(self, request, name, mock_client):
        target = request.getfixturevalue(name)
        svc = target.grants_service
        assert isinstance(svc, Grants)
        assert svc.client is mock_client


# ── grant() ──────────────────────────────────────────────────────────────────


class TestGrant:
    def test_grant_on_catalog_calls_update_then_get(self, catalog, grants_api):
        grants_api.get.return_value = GetPermissionsResponse(
            privilege_assignments=[_assignment("alice", [Privilege.USE_CATALOG])]
        )

        result = catalog.grant("alice", ["USE_CATALOG"])

        grants_api.update.assert_called_once()
        update_kwargs = grants_api.update.call_args.kwargs
        assert update_kwargs["securable_type"] == SecurableType.CATALOG.value
        assert update_kwargs["full_name"] == "main"
        change = update_kwargs["changes"][0]
        assert change.principal == "alice"
        assert change.add == [Privilege.USE_CATALOG]
        assert isinstance(result, Grant)
        assert result.principal == "alice"

    def test_grant_on_schema(self, schema, grants_api):
        grants_api.get.return_value = GetPermissionsResponse(
            privilege_assignments=[_assignment("bob", [Privilege.USE_SCHEMA])]
        )

        schema.grant("bob", [Privilege.USE_SCHEMA])

        update_kwargs = grants_api.update.call_args.kwargs
        assert update_kwargs["securable_type"] == SecurableType.SCHEMA.value
        assert update_kwargs["full_name"] == "main.sales"

    def test_grant_on_table(self, table, grants_api):
        grants_api.get.return_value = GetPermissionsResponse(
            privilege_assignments=[_assignment("carol", [Privilege.SELECT])]
        )

        table.grant("carol", ["select"])

        update_kwargs = grants_api.update.call_args.kwargs
        assert update_kwargs["securable_type"] == SecurableType.TABLE.value
        assert update_kwargs["full_name"] == "main.sales.orders"
        assert update_kwargs["changes"][0].add == [Privilege.SELECT]

    def test_grant_on_volume_path(self, volume_path, grants_api):
        grants_api.get.return_value = GetPermissionsResponse(
            privilege_assignments=[_assignment("dave", [Privilege.READ_VOLUME])]
        )

        volume_path.grant("dave", ["READ_VOLUME"])

        update_kwargs = grants_api.update.call_args.kwargs
        assert update_kwargs["securable_type"] == SecurableType.VOLUME.value
        assert update_kwargs["full_name"] == "main.sales.raw"


# ── revoke() ─────────────────────────────────────────────────────────────────


class TestRevoke:
    def test_revoke_on_table_sends_remove_change(self, table, grants_api):
        grants_api.get.return_value = GetPermissionsResponse(privilege_assignments=[])

        result = table.revoke("carol", ["SELECT"])

        update_kwargs = grants_api.update.call_args.kwargs
        change = update_kwargs["changes"][0]
        assert change.principal == "carol"
        assert change.remove == [Privilege.SELECT]
        assert result is None

    def test_revoke_returns_remaining_grant(self, schema, grants_api):
        grants_api.get.return_value = GetPermissionsResponse(
            privilege_assignments=[_assignment("bob", [Privilege.USE_SCHEMA])]
        )

        result = schema.revoke("bob", ["SELECT"])
        assert result is not None
        assert result.principal == "bob"
        assert result.privileges == ("USE_SCHEMA",)


# ── set_grants() ─────────────────────────────────────────────────────────────


class TestSetGrants:
    def test_set_grants_creates_when_missing(self, catalog, grants_api):
        grants_api.get.side_effect = [
            GetPermissionsResponse(privilege_assignments=[]),
            GetPermissionsResponse(
                privilege_assignments=[_assignment("alice", [Privilege.USE_CATALOG])]
            ),
        ]

        result = catalog.set_grants("alice", ["USE_CATALOG"])

        assert result.principal == "alice"
        update_kwargs = grants_api.update.call_args.kwargs
        change = update_kwargs["changes"][0]
        assert change.add == [Privilege.USE_CATALOG]

    def test_set_grants_replaces_existing(self, table, grants_api):
        grants_api.get.side_effect = [
            GetPermissionsResponse(
                privilege_assignments=[
                    _assignment("carol", [Privilege.SELECT, Privilege.MODIFY])
                ]
            ),
            GetPermissionsResponse(
                privilege_assignments=[_assignment("carol", [Privilege.SELECT])]
            ),
        ]

        table.set_grants("carol", ["SELECT"])

        update_kwargs = grants_api.update.call_args.kwargs
        change = update_kwargs["changes"][0]
        assert change.principal == "carol"
        assert change.remove == [Privilege.MODIFY]
        assert change.add is None


# ── grants() iteration ───────────────────────────────────────────────────────


class TestGrantsListing:
    def test_grants_lists_all_assignments(self, table, grants_api):
        grants_api.get.return_value = GetPermissionsResponse(
            privilege_assignments=[
                _assignment("alice", [Privilege.SELECT]),
                _assignment("bob", [Privilege.MODIFY]),
            ],
            next_page_token=None,
        )

        results = list(table.grants())

        get_kwargs = grants_api.get.call_args.kwargs
        assert get_kwargs["securable_type"] == SecurableType.TABLE.value
        assert get_kwargs["full_name"] == "main.sales.orders"
        assert {(g.principal, g.privileges) for g in results} == {
            ("alice", ("SELECT",)),
            ("bob", ("MODIFY",)),
        }

    def test_grants_filter_by_principal_passes_through(self, schema, grants_api):
        grants_api.get.return_value = GetPermissionsResponse(privilege_assignments=[])

        list(schema.grants("alice"))

        get_kwargs = grants_api.get.call_args.kwargs
        assert get_kwargs["principal"] == "alice"

    def test_grants_effective_uses_get_effective_endpoint(self, catalog, grants_api):
        grants_api.get_effective.return_value = GetPermissionsResponse(
            privilege_assignments=[]
        )

        list(catalog.grants(effective=True))

        grants_api.get.assert_not_called()
        grants_api.get_effective.assert_called_once()


# ── client.grants shortcut ───────────────────────────────────────────────────


class TestClientGrantsShortcut:
    def test_databricks_service_exposes_grants(self, mock_client):
        catalogs = Catalogs(client=mock_client)
        # mock_client.grants is a MagicMock attribute on the spec
        assert catalogs.grants is mock_client.grants
