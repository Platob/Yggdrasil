import datetime as dt

import pyarrow as pa
import pytest

from yggdrasil.databricks.sql import SQLEngine
from yggdrasil.databricks.sql.exceptions import SQLError
from ..conftest import requires_databricks, DatabricksCase

pytestmark = [requires_databricks, pytest.mark.integration]


class TestSQLEngine(DatabricksCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.engine: SQLEngine = cls.workspace.sql(catalog_name="trading", schema_name="unittest")

    def test_insert_read_same(self):
        data = pa.table([
            pa.array(["a", None, "c"]),
            pa.array([1, 2, 4]),
            pa.array([{"q": dt.datetime.now(dt.timezone.utc), "v": 1.0}, None, None]),
            pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
            pa.array([{"k": "v"}, None, None], type=pa.map_(pa.string(), pa.string()))
        ], names=["c0", "id", "c2", "c3", "map column"])

        self.engine.insert_into(data, table_name="test_insert", mode="overwrite")

        other_data = pa.table([
            pa.array(["1", "2", "4"]),
            pa.array([{"q": dt.datetime.now(dt.timezone.utc)}, None, None]),
            pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
            pa.array([{"k": "v"}, None, None], type=pa.map_(pa.string(), pa.string()))
        ], names=["id", "new column", "c3", "map column"])

        self.engine.insert_into(
            other_data,
            table_name="test_insert",
            mode="append",
            schema_mode="append",
        )

        n = self.engine.table(table_name="test_insert")

        result = self.engine.execute(f"SELECT * from {n.full_name(safe=True)}")

        read = result.to_arrow_dataset()

        ds_table = read.to_table()

        self.assertEqual(ds_table.num_rows, data.num_rows + other_data.num_rows)

        read = result.to_arrow_table()

        self.assertEqual(read.num_rows, data.num_rows + other_data.num_rows)

        read = result.to_polars(stream=False)

        self.assertEqual(read.shape[0], data.num_rows + other_data.num_rows)

        with pytest.raises(SQLError):
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

        self.engine.create_table(
            data,
            table_name="test_warehouse_api",
        )

        self.engine.drop_table(table_name="test_warehouse_api")

    def test_warehouse_crud(self):
        warehouses = self.workspace.warehouses()

        warehouse = None

        try:
            warehouse = warehouses.create(name="tmp warehouse", wait=False)

            self.assertEqual(warehouse.warehouse_name, "tmp warehouse")
        finally:
            if warehouse:
                warehouse.delete()
