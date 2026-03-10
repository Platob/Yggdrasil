from unittest import TestCase

from databricks.sdk.client_types import ClientType

from yggdrasil.databricks import DatabricksClient


class TestDecorator(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = DatabricksClient.current()
        cls.groups = cls.client.iam.groups
        cls.group = cls.groups.create(name="ygg-test-group", client_type=ClientType.ACCOUNT)

        cls.users = cls.client.iam.users
        # cls.user = cls.users.create(name="ygg-test-user")

    @classmethod
    def tearDownClass(cls):
        cls.groups.delete_group(group=cls.group)
        # cls.users.delete(cls.user.id)

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