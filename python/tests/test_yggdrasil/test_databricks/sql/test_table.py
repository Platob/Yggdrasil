import datetime as dt

import pyarrow as pa
import pytest

from ..conftest import requires_databricks, DatabricksCase

pytestmark = [requires_databricks, pytest.mark.integration]


class TestSQLTable(DatabricksCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.arrow_data = pa.table([
            pa.array(["a", None, "c"]),
            pa.array([1, 2, 4]),
            pa.array([{"q": dt.datetime.now(dt.timezone.utc), "v": 1.0}, None, None]),
            pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
            pa.array([{"k": "v"}, None, None], type=pa.map_(pa.string(), pa.string()))
        ], names=["c0", "id", "c2", "c3", "map column"], metadata={
            "foo": "bar",
            "primary_key": "id"
        })

        cls.engine = cls.workspace.sql(catalog_name="trading", schema_name="unittest")
        cls.table = (
            cls.engine.table("test_table_crud")
            .create(cls.arrow_data)
        )
        cls.table.insert(cls.arrow_data, mode="overwrite")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.table.delete()
        super().tearDownClass()

    def test_permissions(self):
        permissions = list(self.table.sql.grants.list(self.table.schema))
        assert permissions == {"SELECT", "INSERT", "UPDATE", "DELETE"}

    def test_arrow_schema(self):
        assert isinstance(self.table.arrow_schema, pa.Schema)

    def test_sql(self):
        result = self.table.sql.execute("SELECT * FROM test_table_crud")

        assert isinstance(result.to_arrow_table(), pa.Table)

    def test_arrow_dataset(self):
        arrow_table = self.table.to_arrow_dataset().to_table()
        assert arrow_table.num_rows == 3
