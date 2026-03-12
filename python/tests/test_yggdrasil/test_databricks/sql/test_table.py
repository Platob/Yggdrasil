import datetime as dt
import unittest

import pyarrow as pa

from yggdrasil.databricks.workspaces import Workspace


class TestSQLEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.arrow_data = pa.table([
            pa.array(["a", None, "c"]),
            pa.array([1, 2, 4]),
            pa.array([{"q": dt.datetime.now(dt.timezone.utc), "v": 1.0}, None, None]),
            pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
            pa.array([{"k": "v"}, None, None], type=pa.map_(pa.string(), pa.string()))
        ], names=["c0", "c1", "c2", "c3", "map column"])

        cls.workspace = Workspace().connect()
        cls.engine = cls.workspace.sql(catalog_name="trading", schema_name="unittest")

        cls.table = cls.engine.table("test_table_crud").create(cls.arrow_data, if_not_exists=True)
        cls.table.insert(cls.arrow_data)

    @classmethod
    def tearDownClass(cls):
        cls.table.delete()

    def test_arrow_schema(self):
        assert isinstance(self.table.arrow_schema, pa.Schema)

    def test_arrow_dataset(self):
        arrow_table = self.table.to_arrow_dataset().to_table()

        assert arrow_table.num_rows == 3
