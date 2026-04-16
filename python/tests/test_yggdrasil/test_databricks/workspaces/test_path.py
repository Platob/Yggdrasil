# tests/integration/test_databricks_path_integration.py
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yggdrasil.databricks.workspaces import DatabricksPath
from ..conftest import requires_databricks, DatabricksCase

pytestmark = [requires_databricks, pytest.mark.integration]


class _PathBase(DatabricksCase):
    """Path-test base: adds DBFS / Workspace / Volume path fixtures."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.test_id = "unittest"
        cls.dbfs_base = DatabricksPath.parse(
            f"/dbfs/tmp/unittest/{cls.test_id}", client=cls.workspace
        )
        cls.ws_base = DatabricksPath.parse(
            f"/Workspace/Users/{cls.workspace.iam.users.current_user.email}/unittest/{cls.test_id}",
            client=cls.workspace,
        )
        cls.vol_base = DatabricksPath.parse(
            f"/Volumes/trading/unittest/{cls.test_id}", client=cls.workspace
        )

    @classmethod
    def tearDownClass(cls) -> None:
        for p in (cls.vol_base, cls.ws_base, cls.dbfs_base):
            if p is None:
                continue
            try:
                p.rmdir(recursive=True)
            except Exception as e:
                print(e)
        super().tearDownClass()


class TestDatabricksPathIntegrationDBFS(_PathBase):
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


class TestDatabricksPathIntegrationWorkspace(_PathBase):
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


class TestDatabricksPathIntegrationVolumes(_PathBase):
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

        table = pa.table(
            {
                "col1": [1, 2, 3],
                "col2": ["a", "b", "c"],
            }
        )

        with f.open("wb") as out:
            pq.write_table(table, out)

        with f.open("rb") as inp:
            read_table = pq.read_table(inp)

        self.assertTrue(table.equals(read_table))

        d.rmdir()
