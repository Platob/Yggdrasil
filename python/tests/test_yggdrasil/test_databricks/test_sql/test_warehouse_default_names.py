"""``DatabricksClient.project`` is an alias of ``product`` (always lowercased) and
persists with the client; ``product_name`` / ``project_name`` are its nice
display. ``yggdrasil`` and ``ygg`` are the same project — ``yggdrasil`` is the
name, ``ygg`` only its PyPI distribution alias (mapped in the wheel/environment
layer, not on the client). The default SQL warehouse is named for the project,
falling back to the workspace ygg defaults for the default ``ygg`` / ``yggdrasil``
product (or no project)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.warehouse.service import Warehouses
from yggdrasil.databricks.wheels.service import project_display_name
from yggdrasil.databricks.warehouse.wh_utils import (
    DEFAULT_ALL_PURPOSE_CLASSIC_NAME,
    DEFAULT_ALL_PURPOSE_SERVERLESS_NAME,
)

_DEFAULTS = (DEFAULT_ALL_PURPOSE_CLASSIC_NAME, DEFAULT_ALL_PURPOSE_SERVERLESS_NAME)


class TestClientProjectAliasesProduct(unittest.TestCase):
    @staticmethod
    def _client(product):
        client = DatabricksClient.__new__(DatabricksClient)
        client.product = product
        return client

    def test_project_is_lowercased_product(self):
        self.assertEqual(self._client("My-App").project, "my-app")
        # ``yggdrasil`` stays the project name; ``ygg`` is only the PyPI alias.
        self.assertEqual(self._client("yggdrasil").project, "yggdrasil")
        self.assertEqual(self._client("ygg").project, "ygg")
        self.assertIsNone(self._client(None).project)

    def test_setting_project_writes_product(self):
        client = self._client("yggdrasil")
        client.project = "My-App"
        self.assertEqual(client.product, "my-app")       # persisted on product
        self.assertEqual(client.project, "my-app")

    def test_product_name_and_project_name_are_nice_and_identical(self):
        self.assertEqual(self._client("my-app").product_name, "My App")
        self.assertEqual(self._client("my-app").project_name, "My App")
        self.assertIsNone(self._client(None).product_name)
        self.assertIsNone(self._client(None).project_name)

    def test_product_name_canonicalizes_through_pypi_alias(self):
        # The display name a project's default warehouse / cluster are named for
        # is canonical: ``yggdrasil`` (the name) and ``ygg`` (its PyPI alias) both
        # render ``Ygg``, so a deploy and a later default() resolve one resource.
        self.assertEqual(self._client("yggdrasil").product_name, "Ygg")
        self.assertEqual(self._client("ygg").product_name, "Ygg")

    def test_display_name_helper(self):
        self.assertEqual(project_display_name("my-app"), "My App")
        self.assertEqual(project_display_name("my_cool_project"), "My Cool Project")


class TestDefaultNames(unittest.TestCase):
    def _names(self, project):
        client = MagicMock()
        client.project = project
        client.product_name = project_display_name(project) if project else None
        return Warehouses(client=client).default_names()

    def test_project_nice_name_with_serverless_sibling(self):
        self.assertEqual(self._names("my-app"), ("My App", "My App Serverless"))
        self.assertEqual(self._names("meteologica"), ("Meteologica", "Meteologica Serverless"))

    def test_default_ygg_product_falls_back(self):
        self.assertEqual(self._names("ygg"), _DEFAULTS)
        self.assertEqual(self._names("yggdrasil"), _DEFAULTS)

    def test_no_project_falls_back(self):
        self.assertEqual(self._names(None), _DEFAULTS)


if __name__ == "__main__":
    unittest.main()
