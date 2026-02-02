import datetime as dt
import unittest

import pyarrow as pa
import pytest

from yggdrasil.databricks.sql.exceptions import SqlStatementError
from yggdrasil.databricks.workspaces import Workspace


class TestSQLEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.workspace = Workspace().connect()
        cls.engine = cls.workspace.sql(catalog_name="trading", schema_name="unittest")

    def test_insert_read_same(self):
        data = pa.table([
            pa.array(["a", None, "c"]),
            pa.array([1, 2, 4]),
            pa.array([{"q": dt.datetime.now()}, None, None]),
            pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
            pa.array([{"k": "v"}, None, None], type=pa.map_(pa.string(), pa.string()))
        ], names=["c0", "c1", "c2", "c3", "map column"])

        self.engine.insert_into(data, table_name="test_insert", mode="overwrite")

        n = self.engine.table_full_name(table_name="test_insert")

        read = self.engine.execute(f"SELECT * from {n}").to_arrow_table()

        self.assertEqual(data, read)

        with pytest.raises(SqlStatementError):
            self.engine.execute(f"SELECT * from unknown_table")

        self.engine.drop_table(table_name="test_insert")

    def test_warehouse_api_sql(self):
        data = pa.table([
            pa.array(["a", None, "c"]),
            pa.array([1, 2, 4]),
            pa.array([{"q": dt.datetime.now()}, None, None]),
            pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
            pa.array([{"k": "v"}, None, None], type=pa.map_(pa.string(), pa.string()))
        ], names=["c0", "c1", "c2", "c3", "map column"])

        query = self.engine.create_table(
            data,
            table_name="test_warehouse_api",
            execute=False
        )

        self.engine.execute(query)

        self.engine.drop_table(table_name="test_warehouse_api")