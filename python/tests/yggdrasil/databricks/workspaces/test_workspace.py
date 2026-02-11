import unittest

from yggdrasil.databricks import Workspace


class TestCluster(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.workspace = Workspace().connect()

    def test_get_current_token(self):
        self.assertTrue(self.workspace.connected)

        assert Workspace(
            host=self.workspace.host,
            token=self.workspace.current_token()
        ).current_user.user_name == self.workspace.current_user.user_name
