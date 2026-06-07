"""The default SQL warehouse is named for the running client project: the
client's nice :attr:`project_name` (capitalized) with its serverless sibling,
falling back to the workspace ygg defaults for ``ygg`` / no project. Also covers
the ``DatabricksClient.project`` (lowercased) / ``project_name`` (nice) props."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.warehouse.service import Warehouses
from yggdrasil.databricks.wheels.service import project_display_name
from yggdrasil.databricks.warehouse.wh_utils import (
    DEFAULT_ALL_PURPOSE_CLASSIC_NAME,
    DEFAULT_ALL_PURPOSE_SERVERLESS_NAME,
)

_DEFAULTS = (DEFAULT_ALL_PURPOSE_CLASSIC_NAME, DEFAULT_ALL_PURPOSE_SERVERLESS_NAME)


class TestDefaultNames(unittest.TestCase):
    def _names(self, project):
        client = MagicMock()
        client.project = project                       # always lowercased
        client.project_name = project_display_name(project) if project else None
        return Warehouses(client=client).default_names()

    def test_project_nice_name_with_serverless_sibling(self):
        self.assertEqual(self._names("my-app"), ("My App", "My App Serverless"))
        self.assertEqual(self._names("meteologica"), ("Meteologica", "Meteologica Serverless"))

    def test_ygg_falls_back_to_workspace_defaults(self):
        self.assertEqual(self._names("ygg"), _DEFAULTS)

    def test_no_project_falls_back_to_workspace_defaults(self):
        self.assertEqual(self._names(None), _DEFAULTS)


class TestClientProjectProps(unittest.TestCase):
    def test_display_name_capitalizes_each_word(self):
        self.assertEqual(project_display_name("my-app"), "My App")
        self.assertEqual(project_display_name("my_cool_project"), "My Cool Project")
        self.assertEqual(project_display_name("ygg"), "Ygg")

    def test_project_is_lowercased_and_name_is_nice(self):
        client = DatabricksClient.__new__(DatabricksClient)
        with patch("yggdrasil.databricks.wheels.service.find_pyproject",
                   return_value="/proj/pyproject.toml"), \
             patch("yggdrasil.databricks.wheels.service.read_pyproject",
                   return_value={"name": "My-App"}):
            self.assertEqual(client.project, "my-app")          # always lowered
            self.assertEqual(client.project_name, "My App")     # nice, capitalized

    def test_no_project_is_none(self):
        client = DatabricksClient.__new__(DatabricksClient)
        with patch("yggdrasil.databricks.wheels.service.find_pyproject", return_value=None):
            self.assertIsNone(client.project)
            self.assertIsNone(client.project_name)


if __name__ == "__main__":
    unittest.main()
