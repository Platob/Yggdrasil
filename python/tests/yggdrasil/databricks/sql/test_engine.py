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
            pa.array([{"q": dt.datetime.now(dt.timezone.utc), "v": 1.0}, None, None]),
            pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
            pa.array([{"k": "v"}, None, None], type=pa.map_(pa.string(), pa.string()))
        ], names=["c0", "c1", "c2", "c3", "map column"])

        self.engine.insert_into(data, table_name="test_insert", mode="overwrite")

        other_data = pa.table([
            pa.array(["1", "2", "4"]),
            pa.array([{"q": dt.datetime.now(dt.timezone.utc)}, None, None]),
            pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
            pa.array([{"k": "v"}, None, None], type=pa.map_(pa.string(), pa.string()))
        ], names=["c1", "c2", "c3", "map column"])

        self.engine.insert_into(data, table_name="test_insert", mode="append")

        n = self.engine.table_full_name(table_name="test_insert")

        result = self.engine.execute(f"SELECT * from {n}")

        read = result.to_arrow_dataset()

        ds_table = read.to_table()

        self.assertEqual(ds_table.num_rows, data.num_rows + other_data.num_rows)

        read = result.to_arrow_table()

        self.assertEqual(read.num_rows, data.num_rows + other_data.num_rows)

        read = result.to_polars(stream=False)

        self.assertEqual(read.shape[0], data.num_rows + other_data.num_rows)

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

        self.engine.execute(query.sql)

        self.engine.drop_table(table_name="test_warehouse_api")

    def test_warehouse_crud(self):
        warehouses = self.workspace.warehouses()
        warehouse = warehouses.create(name="tmp warehouse")

        self.assertEqual(warehouse.warehouse_name, "tmp warehouse")

        warehouse.delete()
