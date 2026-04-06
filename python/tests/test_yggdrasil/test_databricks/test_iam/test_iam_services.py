import pytest
from databricks.sdk.client_types import ClientType

from ..conftest import requires_databricks, DatabricksCase

pytestmark = [requires_databricks, pytest.mark.integration]


class TestIAMServices(DatabricksCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.groups = cls.workspace.iam.groups
        cls.group = cls.groups.create(name="ygg-test-group", client_type=ClientType.ACCOUNT)
        cls.users = cls.workspace.iam.users

    @classmethod
    def tearDownClass(cls) -> None:
        cls.groups.delete_group(group=cls.group)
        super().tearDownClass()

    def test_group_properties(self):
        assert self.group.name == "ygg-test-group"
        assert self.group.id is not None
        assert self.group.external_id is not None

    def test_groups(self):
        existing = list(self.groups.list())
        assert isinstance(existing, list)

    def test_add_members(self):
        pass

    def test_users(self):
        existing = list(self.users.list())
        assert isinstance(existing, list)