"""Shared base class for Databricks workspaces integration tests.

Requires ``DATABRICKS_HOST`` to be set — skipped automatically otherwise.

Optional env vars:
  DATABRICKS_TEST_DBFS_BASE:      default "/tmp/yggdrasil_databricks_path_it"
  DATABRICKS_TEST_WORKSPACE_BASE: default "/Users/<me>/yggdrasil_databricks_path_it"
  DATABRICKS_TEST_VOLUME_BASE:    e.g. "/Volumes/<catalog>/<schema>/<volume>/yggdrasil_databricks_path_it"
"""
from __future__ import annotations

import os

from yggdrasil.databricks.workspaces import DatabricksPath
from .._base import DatabricksCase


class DatabricksIntegrationBase(DatabricksCase):
    """
    Real integration tests — no fakes, no mocks.

    Provides ``self.dbfs_base``, ``self.ws_base``, and ``self.vol_base``
    as :class:`~yggdrasil.databricks.workspaces.DatabricksPath` instances.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()  # env-var guard + workspace connection

        cls.dbfs_root = os.getenv("DATABRICKS_TEST_DBFS_BASE", "/tmp/yggdrasil_databricks_path_it")
        cls.workspace_root = os.getenv(
            "DATABRICKS_TEST_WORKSPACE_BASE",
            f"/Users/{cls.workspace.iam.users.current_user.username}/yggdrasil_databricks_path_it",
        )
        cls.schema_root = os.getenv("DATABRICKS_TEST_VOLUME_BASE", "/Volumes/trading/unittest")

    def setUp(self) -> None:
        self.test_id = "unittest"
        self.dbfs_base = DatabricksPath(
            f"{self.dbfs_root}/{self.test_id}", _client=self.workspace
        )
        self.ws_base = DatabricksPath(
            f"{self.workspace_root}/{self.test_id}", _client=self.workspace
        )
        self.vol_base = DatabricksPath(
            f"{self.schema_root}/{self.test_id}", _client=self.workspace
        )

    def tearDown(self) -> None:
        for p in (self.vol_base, self.ws_base, self.dbfs_base):
            if p is None:
                continue
            try:
                p.rmdir(recursive=True)
            except Exception:
                pass

