import unittest

import pytest
from databricks.sdk.errors import ResourceDoesNotExist

from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.secrets.resource import Scope, Secret


class TestSecrets(unittest.TestCase):
    """
    Real-world integration tests (no mocking).

    Required env vars:
      - DATABRICKS_HOST
      - DATABRICKS_TOKEN
    Optional env vars:
      - DATABRICKS_TEST_SCOPE: scope name to use (default: yggdrasil_it_<uuid8>)
      - DATABRICKS_TEST_PRINCIPAL: principal for ACL tests (default: skip ACL tests)
      - DATABRICKS_TEST_NO_CLEANUP=1 to keep created resources for debugging

    Notes:
      - These tests create a secrets scope + secret key in your workspace.
      - get_secret may be restricted depending on environment; tests handle that gracefully.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = DatabricksClient.current()
        cls.secrets = cls.client.secrets

    def test_crud(self):
        self.secrets["ygg-test-scope/ygg-test-key"] = "test-value"

        created = self.secrets["ygg-test-scope/ygg-test-key"]

        self.assertEqual(created.object, b"test-value")

        # Cleanup
        del self.secrets["ygg-test-scope/ygg-test-key"]

        with pytest.raises(ResourceDoesNotExist):
            _ = self.secrets["ygg-test-scope/ygg-test-key"]

    def test_json(self):
        self.secrets["ygg-test-scope/ygg-test-json"] = {
            "a": 1,
            "b": [1, 2, 3],
            "c": {"nested": "value"},
        }

        created = self.secrets["ygg-test-scope/ygg-test-json"]

        self.assertEqual(created.object, {
            "a": 1,
            "b": [1, 2, 3],
            "c": {"nested": "value"},
        })

        # Cleanup
        del self.secrets["ygg-test-scope/ygg-test-json"]

        with pytest.raises(ResourceDoesNotExist):
            _ = self.secrets["ygg-test-scope/ygg-test-json"]

    def test_scope_mapping(self):
        scope = self.secrets["ygg-test-scope"]

        self.assertIsInstance(scope, Scope)

        scope["ygg-test-mapping"] = "mapped-value"
        secret = scope["ygg-test-mapping"]

        self.assertIsInstance(secret, Secret)
        self.assertEqual(secret.object, b"mapped-value")

        # Cleanup
        del scope["ygg-test-mapping"]

        with pytest.raises(ResourceDoesNotExist):
            _ = scope["ygg-test-mapping"]