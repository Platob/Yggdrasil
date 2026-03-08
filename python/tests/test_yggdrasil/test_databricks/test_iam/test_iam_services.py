from unittest import TestCase

from yggdrasil.databricks import DatabricksClient


class TestDecorator(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = DatabricksClient.current(
            host="dbc-0150e9a2-ae64.cloud.databricks.com"
        )
        cls.groups = cls.client.iam.groups
        cls.users = cls.client.iam.users

    def test_groups(self):
        existing = list(self.groups.list())
        assert isinstance(existing, list)

    def test_users(self):
        existing = list(self.users.list())
        assert isinstance(existing, list)