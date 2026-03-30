"""Shared base class for Databricks FS integration tests.

Requirements:
  - Databricks auth configured (env vars / config file)
  - The cluster / workspace must allow DBFS + Workspace API
  - Volume tests require an existing UC volume (or auto-create perms)
"""
from __future__ import annotations

import os
import unittest

from yggdrasil.databricks.fs.path import DatabricksPath


class DatabricksIntegrationBase(unittest.TestCase):
    """
    Real integration tests — no fakes, no mocks.

    Override ``setUpClass`` env-var defaults via:
      DATABRICKS_TEST_DBFS_BASE       default "/dbfs/tmp/yggdrasil_fs_it"
      DATABRICKS_TEST_WORKSPACE_BASE  default "/Workspace/Users/<me>/yggdrasil_fs_it"
      DATABRICKS_TEST_VOLUME_BASE     e.g. "/Volumes/trading/unittest"
    """

    @classmethod
    def setUpClass(cls):
        from yggdrasil.databricks.workspaces.workspace import Workspace

        cls.workspace = Workspace()

        # Hard gate: skip all tests if auth/network is broken
        try:
            cls.workspace.workspace_client().current_user.me()
        except Exception as e:
            raise unittest.SkipTest(
                f"Databricks auth not configured or API not reachable: {e}"
            )

        cls.dbfs_root = os.getenv(
            "DATABRICKS_TEST_DBFS_BASE", "/dbfs/tmp/yggdrasil_fs_it"
        )
        cls.workspace_root = os.getenv(
            "DATABRICKS_TEST_WORKSPACE_BASE",
            f"/Workspace/Users/{cls.workspace.iam.users.current_user.username}/yggdrasil_fs_it",
        )
        cls.schema_root = os.getenv(
            "DATABRICKS_TEST_VOLUME_BASE",
            "/Volumes/trading/unittest",
        )

    def setUp(self):
        self.test_id = "unittest"
        self.dbfs_base = DatabricksPath.parse(
            f"{self.dbfs_root}/{self.test_id}", client=self.workspace
        )
        self.ws_base = DatabricksPath.parse(
            f"{self.workspace_root}/{self.test_id}", client=self.workspace
        )
        self.vol_base = DatabricksPath.parse(
            f"{self.schema_root}/{self.test_id}", client=self.workspace
        )

    def tearDown(self):
        for p in (self.vol_base, self.ws_base, self.dbfs_base):
            if p is None:
                continue
            try:
                p.rmdir(recursive=True)
            except Exception:
                pass

