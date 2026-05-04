"""Live-Postgres integration tests for :class:`Table` — DDL + Arrow IO."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from yggdrasil.io.enums import Mode
from yggdrasil.postgres.tests import PostgresTestCase

pytestmark = pytest.mark.postgres_integration


class TestTableDDL(PostgresTestCase):
    """``CREATE`` / ``DROP`` / ``RENAME`` / ``COMMENT`` / ``TRUNCATE``."""

    def test_create_from_arrow_schema(self) -> None:
        schema = self.pa.schema([
            self.pa.field("id", self.pa.int64(), nullable=False),
            self.pa.field("name", self.pa.string()),
        ])
        tbl = self.table("create_smoke").create(schema, primary_key=["id"])
        try:
            self.assertTrue(tbl.exists)
            cols = {c.name: c for c in tbl.columns()}
            self.assertEqual(set(cols), {"id", "name"})
            self.assertFalse(cols["id"].nullable)
            self.assertTrue(cols["name"].nullable)
        finally:
            tbl.delete(if_exists=True)

    def test_rename_then_drop(self) -> None:
        schema = self.pa.schema([self.pa.field("v", self.pa.int32())])
        tbl = self.table("rename_src").create(schema)
        try:
            tbl.rename("rename_dst")
            self.assertEqual(tbl.table_name, "rename_dst")
            self.assertTrue(self.table("rename_dst").exists)
            self.assertFalse(self.table("rename_src").exists)
        finally:
            self.table("rename_src").delete(if_exists=True)
            self.table("rename_dst").delete(if_exists=True)

    def test_truncate_clears_rows(self) -> None:
        schema = self.pa.schema([self.pa.field("id", self.pa.int64())])
        tbl = self.table("trunc").create(schema)
        try:
            tbl.write_arrow_table(self.pa.table({"id": [1, 2, 3]}))
            self.assertEqual(tbl.read_arrow_table().num_rows, 3)
            tbl.truncate()
            self.assertEqual(tbl.read_arrow_table().num_rows, 0)
        finally:
            tbl.delete(if_exists=True)

    def test_set_comment_persists(self) -> None:
        schema = self.pa.schema([self.pa.field("id", self.pa.int64())])
        tbl = self.table("commented").create(schema)
        try:
            tbl.set_comment("regression-fence")
            result = self.engine.execute(
                "SELECT obj_description(c.oid) "
                "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = %s AND c.relname = %s",
                parameters=(self.test_schema_name, "commented"),
            )
            comment = result.read_arrow_table().column(0).to_pylist()[0]
            self.assertEqual(comment, "regression-fence")
        finally:
            tbl.delete(if_exists=True)


class TestTableArrowIO(PostgresTestCase):
    """Round-trip Arrow tables through the ADBC fast-path."""

    def _sample_table(self):
        return self.pa.table({
            "id": self.pa.array([1, 2, 3], type=self.pa.int64()),
            "name": self.pa.array(["a", "b", "c"], type=self.pa.string()),
            "amount": self.pa.array(
                [Decimal("1.50"), Decimal("2.75"), Decimal("0.00")],
                type=self.pa.decimal128(10, 2),
            ),
            "ts": self.pa.array(
                [
                    dt.datetime(2026, 1, 1, 0, 0, 0),
                    dt.datetime(2026, 1, 2, 0, 0, 0),
                    dt.datetime(2026, 1, 3, 0, 0, 0),
                ],
                type=self.pa.timestamp("us", tz="UTC"),
            ),
        })

    def test_write_then_read_roundtrip(self) -> None:
        src = self._sample_table()
        tbl = self.table("io_roundtrip").create(src.schema)
        try:
            tbl.write_arrow_table(src)
            out = tbl.read_arrow_table()
            self.assertEqual(out.num_rows, src.num_rows)
            self.assertEqual(set(out.column_names), set(src.column_names))
            # Compare value-by-value — the order is not guaranteed
            # without an ORDER BY, so sort by id first.
            sorted_out = out.sort_by("id")
            self.assertEqual(
                sorted_out.column("id").to_pylist(),
                [1, 2, 3],
            )
            self.assertEqual(
                sorted_out.column("name").to_pylist(),
                ["a", "b", "c"],
            )
        finally:
            tbl.delete(if_exists=True)

    def test_append_extends_existing_rows(self) -> None:
        src = self.pa.table({"id": self.pa.array([1, 2], type=self.pa.int64())})
        tbl = self.table("io_append").create(src.schema)
        try:
            tbl.write_arrow_table(src, mode=Mode.APPEND)
            tbl.write_arrow_table(
                self.pa.table({"id": self.pa.array([3, 4], type=self.pa.int64())}),
                mode=Mode.APPEND,
            )
            ids = sorted(tbl.read_arrow_table().column("id").to_pylist())
            self.assertEqual(ids, [1, 2, 3, 4])
        finally:
            tbl.delete(if_exists=True)

    def test_truncate_mode_replaces_rows(self) -> None:
        schema = self.pa.schema([self.pa.field("id", self.pa.int64())])
        tbl = self.table("io_truncate").create(schema)
        try:
            tbl.write_arrow_table(
                self.pa.table({"id": [1, 2, 3]}),
                mode=Mode.APPEND,
            )
            tbl.write_arrow_table(
                self.pa.table({"id": [9]}),
                mode=Mode.TRUNCATE,
            )
            self.assertEqual(
                tbl.read_arrow_table().column("id").to_pylist(),
                [9],
            )
        finally:
            tbl.delete(if_exists=True)

    def test_select_subset_of_columns(self) -> None:
        src = self.pa.table({
            "id": [1, 2],
            "name": ["a", "b"],
            "extra": [10, 20],
        })
        tbl = self.table("io_subset").create(src.schema)
        try:
            tbl.write_arrow_table(src)
            from yggdrasil.data.options import CastOptions
            options = CastOptions(column_names=["id", "name"])
            out = tbl.read_arrow_table(options=options).sort_by("id")
            self.assertEqual(out.column_names, ["id", "name"])
            self.assertEqual(out.column("id").to_pylist(), [1, 2])
        finally:
            tbl.delete(if_exists=True)


class TestTableUpsert(PostgresTestCase):
    """``Mode.UPSERT`` via the temp-table + ``ON CONFLICT`` path."""

    def test_upsert_inserts_new_and_updates_existing(self) -> None:
        schema = self.pa.schema([
            self.pa.field("id", self.pa.int64(), nullable=False),
            self.pa.field("v", self.pa.int64()),
        ])
        tbl = self.table("upsert_tbl").create(schema, primary_key=["id"])
        try:
            tbl.write_arrow_table(
                self.pa.table({"id": [1, 2], "v": [10, 20]}),
                mode=Mode.APPEND,
            )
            # Mix of overlap (id=2 → updates v) and new (id=3).
            tbl.insert_into(
                self.pa.table({"id": [2, 3], "v": [200, 30]}),
                mode=Mode.UPSERT,
                match_by=["id"],
            )
            out = tbl.read_arrow_table().sort_by("id").to_pylist()
            self.assertEqual(
                out,
                [
                    {"id": 1, "v": 10},
                    {"id": 2, "v": 200},
                    {"id": 3, "v": 30},
                ],
            )
        finally:
            tbl.delete(if_exists=True)

    def test_upsert_requires_match_by_or_pk(self) -> None:
        schema = self.pa.schema([self.pa.field("id", self.pa.int64())])
        tbl = self.table("upsert_no_pk").create(schema)
        try:
            with self.assertRaises(ValueError):
                tbl.insert_into(
                    self.pa.table({"id": [1]}),
                    mode=Mode.UPSERT,
                )
        finally:
            tbl.delete(if_exists=True)
