import unittest
import pyarrow as pa

from yggdrasil.databricks.workspaces import Workspace


class TestSQLEngine(unittest.TestCase):

    def setUp(self):
        self.workspace = Workspace().connect()
        self.engine = self.workspace.sql(catalog_name="trading", schema_name="unittest")

    def test_insert_read_same(self):
        data = pa.table([
            pa.array(["a", None, "c"]),
            pa.array([1, 2, 4])
        ], names=["c0", "c1"])

        self.engine.insert_into(data, table_name="test_insert", mode="overwrite")

        n = self.engine.table_full_name(table_name="test_insert")

        read = self.engine.execute(
            f"SELECT * from {n}"
        ).to_arrow_table()

        assert data == read

        self.engine.drop_table(table_name="test_insert")
