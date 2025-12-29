import os
import unittest

from yggdrasil.databricks import Workspace


class TestCluster(unittest.TestCase):

    def setUp(self):
        self.workspace = Workspace(host=os.environ["DATABRICKS_HOST"]).connect()

    def test_get_current_token(self):
        assert Workspace(
            host=self.workspace.host,
            token=self.workspace.current_token()
        ).current_user.user_name == self.workspace.current_user.user_name
