import datetime as dt
import unittest

import pyarrow as pa
from databricks.sdk.service.catalog import TableType

from yggdrasil.databricks.workspaces import Workspace


class TestSQLEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data = pa.table([
            pa.array(["a", None, "c"]),
            pa.array([1, 2, 4]),
            pa.array([{"q": dt.datetime.now(dt.timezone.utc), "v": 1.0}, None, None]),
            pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
            pa.array([{"k": "v"}, None, None], type=pa.map_(pa.string(), pa.string()))
        ], names=["c0", "c1", "c2", "c3", "map column"])

        cls.workspace = Workspace().connect()
        cls.engine = cls.workspace.sql(catalog_name="trading", schema_name="unittest")

        cls.table = cls.engine.table("test_table_crud").create(
            cls.test_data,
        )

    @classmethod
    def tearDownClass(cls):
        cls.table.delete()

    def test_credentials(self):
        credentials = self.table.credentials()

        assert credentials is not None
