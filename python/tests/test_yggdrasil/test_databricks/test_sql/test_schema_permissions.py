"""Mock-driven tests for :class:`Schema` permissions CRUD."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from databricks.sdk.service.catalog import (
    PermissionsChange,
    Privilege,
    PrivilegeAssignment,
    SecurableType,
)

from yggdrasil.databricks.schema.schema import Schema, _normalize_privileges


@pytest.fixture
def service():
    """Mock :class:`Schemas` service shaped so :class:`Schema` can talk
    to ``service.client.workspace_client().grants``."""
    svc = MagicMock()
    svc.catalog_name = None
    svc.schema_name = None
    return svc


@pytest.fixture
def workspace(service):
    return service.client.workspace_client.return_value


@pytest.fixture
def schema(service):
    return Schema(service=service, catalog_name="trading_tgp_dev", schema_name="unittest")


class TestNormalizePrivileges:

    def test_enum_passthrough(self) -> None:
        assert list(_normalize_privileges(Privilege.SELECT)) == [Privilege.SELECT]

    def test_string_token_underscored(self) -> None:
        assert list(_normalize_privileges("EXTERNAL_USE_SCHEMA")) == [
            Privilege.EXTERNAL_USE_SCHEMA,
        ]

    def test_string_token_with_spaces(self) -> None:
        assert list(_normalize_privileges("external use schema")) == [
            Privilege.EXTERNAL_USE_SCHEMA,
        ]

    def test_string_token_with_dashes_mixed_case(self) -> None:
        assert list(_normalize_privileges("Use-Schema")) == [Privilege.USE_SCHEMA]

    def test_iterable_dedupes_preserving_order(self) -> None:
        out = list(_normalize_privileges(
            ["SELECT", Privilege.SELECT, "use schema", "USE_SCHEMA"]
        ))
        assert out == [Privilege.SELECT, Privilege.USE_SCHEMA]

    def test_skips_none_and_blank(self) -> None:
        out = list(_normalize_privileges([None, "", "   ", "SELECT"]))
        assert out == [Privilege.SELECT]

    def test_none_input(self) -> None:
        assert list(_normalize_privileges(None)) == []

    def test_unknown_privilege_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Unity Catalog privilege"):
            list(_normalize_privileges("not_a_real_privilege"))


class TestGrant:

    def test_grant_single_string(self, schema, workspace) -> None:
        schema.grant("alice@example.com", "EXTERNAL USE SCHEMA")
        workspace.grants.update.assert_called_once()
        kwargs = workspace.grants.update.call_args.kwargs
        assert kwargs["securable_type"] == SecurableType.SCHEMA.value
        assert kwargs["full_name"] == "trading_tgp_dev.unittest"
        (change,) = kwargs["changes"]
        assert isinstance(change, PermissionsChange)
        assert change.principal == "alice@example.com"
        assert change.add == [Privilege.EXTERNAL_USE_SCHEMA]
        assert not change.remove

    def test_grant_multiple_privileges(self, schema, workspace) -> None:
        schema.grant("data-engs", [Privilege.USE_SCHEMA, "SELECT"])
        (change,) = workspace.grants.update.call_args.kwargs["changes"]
        assert change.principal == "data-engs"
        assert change.add == [Privilege.USE_SCHEMA, Privilege.SELECT]

    def test_grant_returns_self(self, schema, workspace) -> None:
        assert schema.grant("alice", "SELECT") is schema


class TestRevoke:

    def test_revoke(self, schema, workspace) -> None:
        schema.revoke("alice", "external_use_schema")
        (change,) = workspace.grants.update.call_args.kwargs["changes"]
        assert change.principal == "alice"
        assert change.remove == [Privilege.EXTERNAL_USE_SCHEMA]
        assert not change.add


class TestSetPermissions:

    def test_diffs_against_current_grants(self, schema, workspace) -> None:
        # Current direct grants: SELECT only. Desired: USE_SCHEMA + SELECT.
        workspace.grants.get.return_value = SimpleNamespace(
            privilege_assignments=[
                PrivilegeAssignment(
                    principal="alice",
                    privileges=[Privilege.SELECT],
                ),
            ],
        )
        schema.set_permissions("alice", ["SELECT", "USE_SCHEMA"])
        workspace.grants.get.assert_called_once_with(
            securable_type=SecurableType.SCHEMA.value,
            full_name="trading_tgp_dev.unittest",
            principal="alice",
        )
        (change,) = workspace.grants.update.call_args.kwargs["changes"]
        assert change.add == [Privilege.USE_SCHEMA]
        assert not change.remove

    def test_no_op_when_already_matches(self, schema, workspace) -> None:
        workspace.grants.get.return_value = SimpleNamespace(
            privilege_assignments=[
                PrivilegeAssignment(
                    principal="alice",
                    privileges=[Privilege.SELECT, Privilege.USE_SCHEMA],
                ),
            ],
        )
        schema.set_permissions("alice", [Privilege.SELECT, Privilege.USE_SCHEMA])
        workspace.grants.update.assert_not_called()

    def test_removes_extras(self, schema, workspace) -> None:
        # Currently: SELECT + USE_SCHEMA; desired: just SELECT — USE_SCHEMA goes.
        workspace.grants.get.return_value = SimpleNamespace(
            privilege_assignments=[
                PrivilegeAssignment(
                    principal="alice",
                    privileges=[Privilege.SELECT, Privilege.USE_SCHEMA],
                ),
            ],
        )
        schema.set_permissions("alice", ["SELECT"])
        (change,) = workspace.grants.update.call_args.kwargs["changes"]
        assert not change.add
        assert change.remove == [Privilege.USE_SCHEMA]


class TestPermissionsRead:

    def test_permissions(self, schema, workspace) -> None:
        pa = PrivilegeAssignment(principal="alice", privileges=[Privilege.SELECT])
        workspace.grants.get.return_value = SimpleNamespace(
            privilege_assignments=[pa],
        )
        out = schema.permissions()
        assert out == (pa,)
        workspace.grants.get.assert_called_once_with(
            securable_type=SecurableType.SCHEMA.value,
            full_name="trading_tgp_dev.unittest",
        )

    def test_permissions_with_principal_filter(self, schema, workspace) -> None:
        workspace.grants.get.return_value = SimpleNamespace(privilege_assignments=None)
        schema.permissions(principal="alice")
        workspace.grants.get.assert_called_once_with(
            securable_type=SecurableType.SCHEMA.value,
            full_name="trading_tgp_dev.unittest",
            principal="alice",
        )

    def test_effective_permissions(self, schema, workspace) -> None:
        workspace.grants.get_effective.return_value = SimpleNamespace(
            privilege_assignments=None,
        )
        out = schema.effective_permissions(principal="alice")
        assert out == ()
        workspace.grants.get_effective.assert_called_once_with(
            securable_type=SecurableType.SCHEMA.value,
            full_name="trading_tgp_dev.unittest",
            principal="alice",
        )


class TestUpdatePermissions:

    def test_filters_no_op_changes(self, schema, workspace) -> None:
        change_real = PermissionsChange(
            principal="alice", add=[Privilege.SELECT],
        )
        change_noop = PermissionsChange(principal="bob")
        schema.update_permissions([change_real, change_noop])
        (sent,) = workspace.grants.update.call_args.kwargs["changes"]
        assert sent is change_real

    def test_accepts_mapping(self, schema, workspace) -> None:
        schema.update_permissions([
            {"principal": "alice", "add": ["SELECT", "use schema"]},
            {"principal": "bob", "remove": [Privilege.MODIFY]},
        ])
        sent = workspace.grants.update.call_args.kwargs["changes"]
        assert sent[0].principal == "alice"
        assert sent[0].add == [Privilege.SELECT, Privilege.USE_SCHEMA]
        assert sent[1].principal == "bob"
        assert sent[1].remove == [Privilege.MODIFY]

    def test_empty_changes_skip_api(self, schema, workspace) -> None:
        schema.update_permissions([])
        workspace.grants.update.assert_not_called()

    def test_missing_principal_raises(self, schema, workspace) -> None:
        with pytest.raises(ValueError, match="missing 'principal'"):
            schema.update_permissions([{"add": ["SELECT"]}])
