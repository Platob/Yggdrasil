import os
import unittest
import pyarrow as pa
import datetime as dt

import pytest

from yggdrasil.databricks.sql.exceptions import SqlStatementError
from yggdrasil.databricks.workspaces import Workspace


class TestSQLEngine(unittest.TestCase):

    def setUp(self):
        self.workspace = Workspace().connect()
        self.engine = self.workspace.sql(catalog_name="trading", schema_name="unittest")
        self.warehouse = self.workspace.warehouses().default()

    def test_warehouse(self):
        self.assertEqual("YGG-DEFAULT", self.warehouse.warehouse_name)

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

        read = self.engine.execute(
            f"SELECT * from {n}"
        ).to_arrow_table()

        self.assertEqual(data, read)

        with pytest.raises(SqlStatementError):
            self.engine.execute(f"SELECT t from {n}")

        self.engine.drop_table(table_name="test_insert")
