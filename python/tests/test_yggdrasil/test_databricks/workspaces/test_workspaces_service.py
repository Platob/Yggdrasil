from unittest import TestCase

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.workspaces.service import Workspaces


class TestWorkspaces(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = DatabricksClient().connect()
        cls.service = cls.client.workspaces

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_ok(self):
        assert self.service is not None

    def test_list(self):
        workspaces = list(self.service.list())
        assert len(workspaces) > 0
        for workspace in workspaces:
            assert workspace.id != ""
            assert workspace.name != ""
            assert workspace.url != ""

    def test_url(self):
        url = self.service.to_url()
        built = Workspaces.from_parsed_url(url)
        assert built.to_url() == url
