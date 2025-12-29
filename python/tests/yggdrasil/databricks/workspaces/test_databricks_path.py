# tests/integration/test_databricks_path_integration.py
from __future__ import annotations

import os
import unittest

from yggdrasil.databricks.workspaces.databricks_path import DatabricksPath


class DatabricksIntegrationBase(unittest.TestCase):
    """
    Real integration tests. No fakes, no mocks.

    Requirements:
      - Databricks auth configured for databricks-sdk (env vars / config file)
      - The cluster / workspace must allow DBFS + Workspace API.
      - Volume tests require an existing UC volume base path.

    Optional env vars:
      - DATABRICKS_TEST_DBFS_BASE:    default "/tmp/yggdrasil_databricks_path_it"
      - DATABRICKS_TEST_WORKSPACE_BASE: default "/Users/<me>/yggdrasil_databricks_path_it"
      - DATABRICKS_TEST_VOLUME_BASE:  e.g. "/Volumes/<catalog>/<schema>/<volume>/yggdrasil_databricks_path_it"
    """

    @classmethod
    def setUpClass(cls):
        from yggdrasil.databricks.workspaces.workspace import Workspace

        cls.workspace = Workspace()

        # hard gate: if auth/network is broken, skip all tests in this file
        try:
            cls.workspace.sdk().current_user.me()
        except Exception as e:
            raise unittest.SkipTest(f"Databricks auth not configured or API not reachable: {e}")

        cls.dbfs_root = os.getenv("DATABRICKS_TEST_DBFS_BASE", "/tmp/yggdrasil_databricks_path_it")
        cls.workspace_root = os.getenv(
            "DATABRICKS_TEST_WORKSPACE_BASE",
            f"/Users/{cls.workspace.current_user.user_name}/yggdrasil_databricks_path_it",
        )
        cls.schema_root = os.getenv(
            "DATABRICKS_TEST_VOLUME_BASE",
            "/Volumes/trading/unittest"
        )  # may be None

    def setUp(self):
        # Unique per test so parallel runs don’t punch each other
        self.test_id = "unittest"

        self.dbfs_base = DatabricksPath(f"{self.dbfs_root}/{self.test_id}", workspace=self.workspace)
        self.ws_base = DatabricksPath(f"{self.workspace_root}/{self.test_id}", workspace=self.workspace)
        self.vol_base = DatabricksPath(f"{self.schema_root}/{self.test_id}", workspace=self.workspace)

    def tearDown(self):
        # Best-effort cleanup; don’t fail teardown
        for p in (self.vol_base, self.ws_base, self.dbfs_base):
            if p is None:
                continue
            try:
                p.rmdir(recursive=True)
            except Exception:
                pass


class TestDatabricksPathIntegrationDBFS(DatabricksIntegrationBase):
    def test_dbfs_roundtrip_text(self):
        d = self.dbfs_base / "dir"
        f = d / "hello.txt"

        from yggdrasil.databricks import Workspace

        workspace = Workspace()

        filepath = workspace.path("/Volumes/trading/schema/volume/data.bin")
        with filepath.open("wb") as out:
            out.write(b"hello from DBFS integration")

        self.assertTrue(filepath.exists())

        with filepath.open("r") as inp:
            got = inp.read()

        filepath.rmfile()
        filepath.parent.rmdir()
        filepath.remove()

        self.assertEqual(got, "hello from DBFS integration")

        files = list(d.ls(raise_error=False))

        assert f not in files

    def test_dbfs_roundtrip_binary(self):
        d = self.dbfs_base / "dirbin"
        f = d / "data.bin"
        payload = b"\x00\xff\x10\x20"

        with f.open("wb") as out:
            out.write(payload)

        with f.open("rb") as inp:
            got = inp.read()

        self.assertEqual(got, payload)

        files = list(d.ls())

        assert f in files

    def test_dbfs_mkdir_rmdir(self):
        d = self.dbfs_base / "subdir"
        d.mkdir()
        self.assertTrue(d.exists())

        # Not raise error if not exists
        d.mkdir()

        d.rmdir()
        self.assertFalse(d.exists())

        # Not raise error if not exists
        d.rmdir()

        files = list(d.ls(raise_error=False))

        assert len(files) == 0


class TestDatabricksPathIntegrationWorkspace(DatabricksIntegrationBase):
    def test_workspace_roundtrip_text(self):
        # nested path tests the "parent folder does not exist" retry logic
        d = self.ws_base / "a" / "b"
        f = d / "c.py"

        with f.open("w") as out:
            out.write("print('hello from workspace integration')\n")

        self.assertTrue(f.exists())

        with f.open("r") as inp:
            got = inp.read()

        self.assertIn("hello from workspace integration", got)

        files = list(d.ls())

        assert f in files

    def test_workspace_roundtrip_binary(self):
        d = self.ws_base / "bin"
        f = d / "data.bin"
        payload = b"\x01\x02\x03\x04\x05"

        with f.open("wb") as out:
            out.write(payload)

        with f.open("rb") as inp:
            got = inp.read()

        self.assertEqual(got, payload)

        files = list(d.ls())

        assert f in files

    def test_workspace_mkdir_rmdir(self):
        d = self.ws_base / "dir1" / "dir2"
        d.mkdir()
        self.assertTrue(d.exists())

        # Not raise error if not exists
        d.mkdir()

        d.rmdir()
        self.assertFalse(d.exists())

        # Not raise error if not exists
        d.rmdir()


class TestDatabricksPathIntegrationVolumes(DatabricksIntegrationBase):
    def setUp(self):
        super().setUp()
        if not self.schema_root:
            self.skipTest(
                "Set DATABRICKS_TEST_VOLUME_BASE to an existing volume path, "
                "e.g. /Volumes/<catalog>/<schema>/<volume>/yggdrasil_databricks_path_it"
            )

    def test_volume_roundtrip_text(self):
        # Ensure parent directory exists (Files API won’t auto-create parents on upload)
        d = self.vol_base / "nested"

        f = d / "hello.txt"
        with f.open("w") as out:
            out.write("hello from volumes integration")

        self.assertTrue(f.exists())

        with f.open("r") as inp:
            got = inp.read()

        self.assertEqual(got, "hello from volumes integration")

        files = list(d.ls())

        assert f in files

    def test_volume_roundtrip_binary(self):
        d = self.vol_base / "nestedbin"

        f = d / "data.bin"
        payload = b"\xde\xad\xbe\xef"

        with f.open("wb") as out:
            out.write(payload)

        with f.open("rb") as inp:
            got = inp.read()

        self.assertEqual(got, payload)

        files = list(d.ls())

        assert f in files

    def test_volume_mkdir_rmdir(self):
        d = self.vol_base / "dirA" / "dirB"
        d.mkdir()
        self.assertTrue(d.exists())

        # Not raise error if not exists
        d.mkdir()

        d.rmdir()
        self.assertFalse(d.exists())

        # Not raise error if not exists
        d.rmdir()