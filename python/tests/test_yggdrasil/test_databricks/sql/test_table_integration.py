import time

import pyarrow as pa
import pytest

from yggdrasil.databricks.sql import Table

from ..conftest import requires_databricks, DatabricksCase

pytestmark = [requires_databricks, pytest.mark.integration]


class TestSQLTableIntegration(DatabricksCase):

    _CATALOG = "trading"
    _SCHEMA_NAME = "unittest"

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.engine = cls.workspace.sql(
            catalog_name=cls._CATALOG,
            schema_name=cls._SCHEMA_NAME,
        )

    def _scratch_name(self, prefix: str) -> str:
        return f"{prefix}_{time.time_ns()}"

    def _table(self, name: str) -> Table:
        return self.engine.table(name)

    def _delete_table(self, table: Table) -> None:
        try:
            if table.exists:
                table.delete()
        except Exception:
            pass

    def test_create_insert_read_with_primary_key(self):
        table = self._table(self._scratch_name("test_table_pk"))
        self.addCleanup(self._delete_table, table)

        initial = pa.table(
            [
                pa.array([1, 2], type=pa.int64()),
                pa.array(["alpha", "beta"]),
            ],
            names=["id", "name"],
        )
        appended = pa.table(
            [
                pa.array([3], type=pa.int64()),
                pa.array(["gamma"]),
            ],
            names=["id", "name"],
        )

        table.create(initial, if_not_exists=False, primary_keys=["id"])
        table.insert(initial, mode="append")
        table.insert(appended, mode="append")

        read_back = table.to_arrow_dataset().to_table()
        rows = sorted(
            zip(
                read_back.column("id").to_pylist(),
                read_back.column("name").to_pylist(),
            )
        )

        assert read_back.num_rows == 3
        assert rows == [(1, "alpha"), (2, "beta"), (3, "gamma")]
        assert table.arrow_schema.field("id").nullable is False

    def test_create_insert_read_with_primary_and_foreign_keys(self):
        parent = self._table(self._scratch_name("test_table_fk_parent"))
        child = self._table(self._scratch_name("test_table_fk_child"))
        self.addCleanup(self._delete_table, child)
        self.addCleanup(self._delete_table, parent)

        parent_data = pa.table(
            [
                pa.array([10, 20], type=pa.int64()),
                pa.array(["north", "south"]),
            ],
            names=["id", "label"],
        )
        child_initial = pa.table(
            [
                pa.array([1, 2], type=pa.int64()),
                pa.array([10, 20], type=pa.int64()),
                pa.array(["open", "closed"]),
            ],
            names=["id", "parent_id", "status"],
        )
        child_appended = pa.table(
            [
                pa.array([3], type=pa.int64()),
                pa.array([10], type=pa.int64()),
                pa.array(["pending"]),
            ],
            names=["id", "parent_id", "status"],
        )

        parent.create(parent_data, if_not_exists=False, primary_keys=["id"])
        parent.insert(parent_data, mode="append")
        child.create(
            child_initial,
            if_not_exists=False,
            primary_keys=["id"],
            foreign_keys={
                "parent_id": (
                    f"{self._CATALOG}.{self._SCHEMA_NAME}.{parent.table_name}.id"
                )
            },
        )
        child.insert(child_initial, mode="append")
        child.insert(child_appended, mode="append")

        child_read = child.to_arrow_dataset().to_table()
        child_rows = sorted(
            zip(
                child_read.column("id").to_pylist(),
                child_read.column("parent_id").to_pylist(),
                child_read.column("status").to_pylist(),
            )
        )
        join_result = self.engine.execute(
            "SELECT c.id, c.parent_id, p.label "
            f"FROM {child.full_name(safe=True)} c "
            f"JOIN {parent.full_name(safe=True)} p ON c.parent_id = p.id "
            "ORDER BY c.id"
        ).to_arrow_table()

        assert child_read.num_rows == 3
        assert child_rows == [
            (1, 10, "open"),
            (2, 20, "closed"),
            (3, 10, "pending"),
        ]
        assert child.arrow_schema.field("id").nullable is False

        assert join_result.column("id").to_pylist() == [1, 2, 3]
        assert join_result.column("parent_id").to_pylist() == [10, 20, 10]
        assert join_result.column("label").to_pylist() == ["north", "south", "north"]
