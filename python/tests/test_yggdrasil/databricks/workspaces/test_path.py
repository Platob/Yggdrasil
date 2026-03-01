# tests/integration/test_databricks_path_integration.py
from __future__ import annotations

import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.databricks.workspaces import Workspace, DatabricksPath


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
        cls.workspace = Workspace().connect()

        # hard gate: if auth/network is broken, skip all tests in this file
        try:
            _ = cls.workspace.current_user
        except Exception as e:
            raise unittest.SkipTest(f"Databricks auth not configured or API not reachable: {e}")

        # Unique per test so parallel runs don’t punch each other
        cls.test_id = "unittest"

        cls.dbfs_base = DatabricksPath.parse(f"/dbfs/tmp/unittest/{cls.test_id}", workspace=cls.workspace)
        cls.ws_base = DatabricksPath.parse(f"/Workspace/Users/{cls.workspace.current_user.user_name}/unittest/{cls.test_id}", workspace=cls.workspace)
        cls.vol_base = DatabricksPath.parse(f"/Volumes/trading/unittest/{cls.test_id}", workspace=cls.workspace)

    @classmethod
    def tearDownClass(cls):
        # Best-effort cleanup; don’t fail teardown
        for p in (cls.vol_base, cls.ws_base, cls.dbfs_base):
            if p is None:
                continue
            try:
                p.rmdir(recursive=True)
            except Exception as e:
                print(e)


class TestDatabricksPathIntegrationDBFS(DatabricksIntegrationBase):
    def test_dbfs_roundtrip_text(self):
        d = self.dbfs_base / "dir"
        f = d / "hello.txt"

        with f.open("wb") as out:
            out.write(b"hello from DBFS integration")

        self.assertTrue(f.exists())

        with f.open("r") as inp:
            got = inp.read()

        self.assertEqual(got, "hello from DBFS integration")

        files = list(d.ls())

        assert f in files

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

        files = list(d.ls())

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

        d.rmdir()

        self.assertFalse(d.exists())

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

    def test_read_write_data(self):
        d = self.vol_base / "datafolder"
        f = d / "data.parquet"

        table = pa.table({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })

        with f.open("wb") as out:
            pq.write_table(table, out)

        with f.open("rb") as inp:
            read_table = pq.read_table(inp)

        self.assertTrue(table.equals(read_table))

        d.rmdir()

    def test_path_data_io(self):
        filepath = self.vol_base / "file.parquet"
        folder_path = self.vol_base / "folder.parquet/"

        my_arrow_table = pa.table({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })

        filepath.write_table(my_arrow_table)
        folder_path.write_table(my_arrow_table)
        arrow_dataset = folder_path.read_arrow_dataset()

        data = filepath.read_arrow_table()
        self.assertTrue(my_arrow_table.equals(data))

        data = folder_path.read_arrow_table()
        self.assertTrue(my_arrow_table.equals(data))

        data = arrow_dataset.to_table()
        self.assertTrue(my_arrow_table.equals(data))

        data = filepath.sql(f"SELECT * from dbfs.`{filepath}`")
        self.assertTrue(my_arrow_table.equals(data))

    def test_pyarrow_filesystem(self):
        folder_path = self.vol_base / "arrow_dataset"

        my_arrow_table = pa.table({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })

        folder_path.mkdir()

        folder_path.write_table(my_arrow_table)

        self.assertEqual(my_arrow_table, folder_path.read_arrow_table())

    def test_temporary_credentials(self):
        folder_path = self.vol_base / "tmp_path"

        folder_path.mkdir()

        credentials = folder_path.temporary_credentials()

        self.assertTrue(credentials)