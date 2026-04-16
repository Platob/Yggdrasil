from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.iam import IAMGroup, IAMUser
from yggdrasil.databricks.sql.table import Table
from yggdrasil.databricks.sql.tables import Tables


@pytest.fixture()
def mock_client():
    client = MagicMock(spec=DatabricksClient)
    client.base_url.to_string.return_value = "https://adb-123.azuredatabricks.net"
    client.base_url.with_path.side_effect = lambda p: MagicMock(
        to_string=lambda: f"https://adb-123.azuredatabricks.net{p}"
    )
    client.sql = MagicMock()
    return client


@pytest.fixture()
def table(mock_client):
    service = Tables(client=mock_client, catalog_name="main", schema_name="sales")
    return Table(service=service, catalog_name="main", schema_name="sales", table_name="orders")


class TestTablePermissions:
    def test_grant_permissions_ddl_quotes_principal_and_normalizes_privileges(self, table):
        ddl = table.grant_permissions_ddl("User.Name@example.com", ["select", "apply_tag"])

        assert ddl == (
            "GRANT SELECT, APPLY TAG "
            "ON TABLE `main`.`sales`.`orders` "
            "TO `User.Name@example.com`"
        )

    def test_add_permissions_executes_one_grant_per_principal(self, table, mock_client):
        table.add_permissions(
            users=[IAMUser(username="alice@example.com")],
            groups=[IAMGroup(name="Data Scientists")],
            privileges=["select", "modify"],
        )

        statements = [call.args[0] for call in mock_client.sql.execute.call_args_list]
        assert statements == [
            "GRANT SELECT, MODIFY ON TABLE `main`.`sales`.`orders` TO `alice@example.com`",
            "GRANT SELECT, MODIFY ON TABLE `main`.`sales`.`orders` TO `Data Scientists`",
        ]

    def test_add_permissions_merges_legacy_group_arg_and_deduplicates(self, table, mock_client):
        table.add_permissions(
            iam_id=["analysts", "analysts"],
            group=[IAMGroup(name="analysts")],
            privileges="all_privileges",
        )

        mock_client.sql.execute.assert_called_once_with(
            "GRANT ALL PRIVILEGES ON TABLE `main`.`sales`.`orders` TO `analysts`"
        )

    def test_add_permissions_requires_a_principal(self, table):
        with pytest.raises(ValueError, match="at least one user, group, or iam_id"):
            table.add_permissions(privileges="select")
