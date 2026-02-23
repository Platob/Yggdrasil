import unittest

from yggdrasil.databricks import Workspace


class TestAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.workspace = Workspace().connect()
        cls.loki = cls.workspace.sql().ai()

    def test_trading_chat_structured(self):
        chat = self.loki.generate("test")

        assert chat
