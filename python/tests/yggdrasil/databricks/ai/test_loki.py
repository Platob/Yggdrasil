import unittest

from yggdrasil.databricks import Workspace


class TestCluster(unittest.TestCase):

    def setUp(self):
        self.workspace = Workspace().connect()
        self.loki = self.workspace.loki()
