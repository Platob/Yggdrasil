"""``Warehouses.default_names`` — the default SQL warehouse is named for the
running client project (capitalized) with its serverless sibling, falling back
to the workspace ygg defaults for ``ygg`` / no project."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.warehouse.service import Warehouses
from yggdrasil.databricks.warehouse.wh_utils import (
    DEFAULT_ALL_PURPOSE_CLASSIC_NAME,
    DEFAULT_ALL_PURPOSE_SERVERLESS_NAME,
)

_DEFAULTS = (DEFAULT_ALL_PURPOSE_CLASSIC_NAME, DEFAULT_ALL_PURPOSE_SERVERLESS_NAME)


def _svc():
    return Warehouses(client=MagicMock())


class TestDefaultNames(unittest.TestCase):
    def _names(self, project):
        with patch("yggdrasil.databricks.warehouse.service._client_project_name",
                   return_value=project):
            return _svc().default_names()

    def test_project_capitalized_with_serverless_sibling(self):
        self.assertEqual(self._names("my-app"), ("My-app", "My-app Serverless"))
        self.assertEqual(self._names("meteologica"), ("Meteologica", "Meteologica Serverless"))

    def test_ygg_falls_back_to_workspace_defaults(self):
        self.assertEqual(self._names("ygg"), _DEFAULTS)
        self.assertEqual(self._names("YGG"), _DEFAULTS)      # case-insensitive

    def test_no_project_falls_back_to_workspace_defaults(self):
        self.assertEqual(self._names(None), _DEFAULTS)


if __name__ == "__main__":
    unittest.main()
