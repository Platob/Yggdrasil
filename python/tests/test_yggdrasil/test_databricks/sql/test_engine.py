import datetime as dt

import pyarrow as pa
import pytest

from yggdrasil.databricks.sql import SQLEngine
from yggdrasil.databricks.sql.exceptions import SQLError
from ..conftest import DatabricksCase, requires_databricks

integration = pytest.mark.integration


# ---------------------------------------------------------------------------
# Integration tests — require a live Databricks workspace
# ---------------------------------------------------------------------------


@requires_databricks
@integration
class TestSQLEngine(DatabricksCase):
    """End-to-end tests that hit a real Databricks warehouse."""

    engine: SQLEngine

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.engine = cls.client.sql(catalog_name="trading", schema_name="unittest")

    @staticmethod
    def _sample_table() -> pa.Table:
        return pa.table(
            [
                pa.array(["a", None, "c"]),
                pa.array([1, 2, 4]),
                pa.array(
                    [{"q": dt.datetime.now(dt.timezone.utc), "v": 1.0}, None, None]
                ),
                pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
                pa.array(
                    [{"k": "v"}, None, None],
                    type=pa.map_(pa.string(), pa.string()),
                ),
            ],
            names=["c0", "id", "c2", "c3", "map column"],
            metadata={
                "partition_by": "id",
                "primary_key": "id",
            }
        )

    def test_insert_then_read_roundtrip(self):
        data = self._sample_table()
        batch = self.engine.insert_into(data, table_name="test_insert", mode="overwrite")

        try:
            handle = self.engine.table(table_name="test_insert")
            result = self.engine.execute(f"SELECT * FROM {handle.full_name(safe=True)}")

            self.assertEqual(result.to_arrow_table().num_rows, data.num_rows)
            self.assertEqual(result.to_arrow_dataset().to_table().num_rows, data.num_rows)
            self.assertEqual(result.to_polars().shape[0], data.num_rows)
        finally:
            self.engine.drop_table(table_name="test_insert")

    def test_insert_schema_evolution_append(self):
        data = self._sample_table()
        self.engine.insert_into(data, table_name="test_insert", mode="overwrite")

        try:
            extra = pa.table(
                [
                    pa.array(["1", "2", "4"]),
                    pa.array([{"q": dt.datetime.now(dt.timezone.utc)}, None, None]),
                    pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
                    pa.array(
                        [{"k": "v"}, None, None],
                        type=pa.map_(pa.string(), pa.string()),
                    ),
                ],
                names=["id", "new column", "c3", "map column"],
            )

            self.engine.insert_into(
                extra,
                table_name="test_insert",
                mode="append",
                schema_mode="append",
            )

            handle = self.engine.table(table_name="test_insert")
            result = self.engine.execute(f"SELECT * FROM {handle.full_name(safe=True)}")
            self.assertEqual(
                result.to_arrow_table().num_rows,
                data.num_rows + extra.num_rows,
            )
        finally:
            self.engine.drop_table(table_name="test_insert")

    def test_execute_unknown_table_raises(self):
        with pytest.raises(SQLError):
            self.engine.execute("SELECT * FROM unknown_table")

    def test_warehouse_api_create_drop(self):
        data = pa.table(
            [
                pa.array(["a", None, "c"]),
                pa.array([1, 2, 4]),
                pa.array([{"q": dt.datetime.now()}, None, None]),
                pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
                pa.array(
                    [{"k": "v"}, None, None],
                    type=pa.map_(pa.string(), pa.string()),
                ),
            ],
            names=["c0", "c1", "c2", "c3", "map column"],
        )
        self.engine.create_table(data, table_name="test_warehouse_api")
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
