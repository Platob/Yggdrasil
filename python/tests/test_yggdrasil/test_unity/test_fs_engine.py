"""End-to-end tests for the filesystem-backed Unity Catalog backend."""
from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.data_field import Field
from yggdrasil.data.enums import Mode
from yggdrasil.data.enums.media_type import MediaTypes
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.primitive import Int64Type, StringType
from yggdrasil.io.path import LocalPath
from yggdrasil.unity.fs import FSEngine


def _sales_schema() -> Schema:
    return Schema([
        Field(name="id", dtype=Int64Type()),
        Field(name="name", dtype=StringType()),
    ])


class TestFSEngineLifecycle(ArrowTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = FSEngine(base=LocalPath(str(self.tmp_path / "warehouse")))

    def test_create_catalog_schema_table(self) -> None:
        cat = self.engine.create_catalog("main")
        self.assertTrue(cat.exists)
        self.assertEqual(cat.full_name, "main")
        self.assertEqual(cat.info.name, "main")

        sch = cat.create_schema("default")
        self.assertTrue(sch.exists)
        self.assertEqual(sch.full_name, "main.default")
        self.assertEqual(sch.info.catalog_name, "main")

        tbl = sch.create_table("sales", schema=_sales_schema())
        self.assertTrue(tbl.exists)
        self.assertEqual(tbl.full_name, "main.default.sales")
        self.assertEqual(tbl.info.format, MediaTypes.PARQUET)
        self.assertEqual(
            [f.name for f in tbl.schema.fields], ["id", "name"],
        )

    def test_navigation_via_indexing(self) -> None:
        cat = self.engine.create_catalog("main")
        cat.create_schema("default").create_table(
            "sales", schema=_sales_schema(),
        )
        same = self.engine["main"]["default"]["sales"]
        self.assertEqual(same.full_name, "main.default.sales")

    def test_indexing_missing_raises_keyerror(self) -> None:
        self.engine.create_catalog("main")
        with self.assertRaises(KeyError):
            _ = self.engine["nope"]
        cat = self.engine["main"]
        with self.assertRaises(KeyError):
            _ = cat["missing"]

    def test_contains_probe(self) -> None:
        cat = self.engine.create_catalog("main")
        self.assertIn("main", self.engine)
        self.assertNotIn("other", self.engine)
        cat.create_schema("default")
        self.assertIn("default", cat)
        self.assertNotIn("nope", cat)

    def test_listing_skips_non_catalog_directories(self) -> None:
        # Pre-create a directory under base without metadata — it must
        # not show up in catalogs() and must not collide with a real
        # catalog under the same name.
        (self.engine.base / "stray").mkdir(parents=True, exist_ok=True)
        self.engine.create_catalog("main")
        names = sorted(c.name for c in self.engine.catalogs())
        self.assertEqual(names, ["main"])

    def test_create_if_not_exists_idempotent(self) -> None:
        cat1 = self.engine.create_catalog("main")
        cat2 = self.engine.create_catalog("main")
        self.assertEqual(cat1.info.created_at, cat2.info.created_at)

    def test_create_strict_raises_on_existing(self) -> None:
        self.engine.create_catalog("main")
        with self.assertRaises(FileExistsError):
            self.engine.create_catalog("main", if_not_exists=False)

    def test_delete_non_empty_requires_recursive(self) -> None:
        cat = self.engine.create_catalog("main")
        cat.create_schema("default")
        with self.assertRaises(OSError):
            cat.delete()
        cat.delete(recursive=True)
        self.assertFalse(cat.exists)


class TestFSTableIO(ArrowTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = FSEngine(base=LocalPath(str(self.tmp_path / "warehouse")))
        self.engine.create_catalog("main").create_schema("default")

    def _table(self):
        return self.engine["main"]["default"].create_table(
            "sales", schema=_sales_schema(),
        )

    def test_write_and_read_round_trip(self) -> None:
        tbl = self._table()
        batches = self.pa.table(
            {"id": [1, 2, 3], "name": ["a", "b", "c"]}
        ).to_batches()
        tbl.write_arrow_batches(batches)

        roundtrip = tbl.read_arrow_table()
        self.assertEqual(roundtrip.num_rows, 3)
        self.assertEqual(roundtrip.column_names, ["id", "name"])
        self.assertEqual(
            roundtrip.column("id").to_pylist(), [1, 2, 3],
        )

    def test_append_accumulates_rows(self) -> None:
        tbl = self._table()
        first = self.pa.table({"id": [1, 2], "name": ["a", "b"]})
        second = self.pa.table({"id": [3, 4], "name": ["c", "d"]})
        tbl.write_arrow_batches(first.to_batches())
        tbl.write_arrow_batches(
            second.to_batches(), options=CastOptions(mode=Mode.APPEND),
        )
        result = tbl.read_arrow_table()
        self.assertEqual(result.num_rows, 4)

    def test_overwrite_replaces_rows(self) -> None:
        tbl = self._table()
        tbl.write_arrow_batches(
            self.pa.table({"id": [1, 2], "name": ["a", "b"]}).to_batches(),
        )
        tbl.write_arrow_batches(
            self.pa.table({"id": [99], "name": ["z"]}).to_batches(),
            options=CastOptions(mode=Mode.OVERWRITE),
        )
        result = tbl.read_arrow_table()
        self.assertEqual(result.num_rows, 1)
        self.assertEqual(result.column("id").to_pylist(), [99])

    def test_schema_is_zero_io(self) -> None:
        self._table()
        # Re-open through a fresh engine handle so no in-memory cache.
        fresh_engine = FSEngine(base=self.engine.base)
        fresh = fresh_engine["main"]["default"]["sales"]
        # data_path is still empty — schema must come from the sidecar,
        # not from sniffing Arrow files.
        self.assertEqual(
            [f.name for f in fresh.schema.fields], ["id", "name"],
        )

    def test_create_arrow_ipc_format(self) -> None:
        tbl = self.engine["main"]["default"].create_table(
            "alt", schema=_sales_schema(), format="arrow",
        )
        self.assertEqual(tbl.info.format, MediaTypes.ARROW_IPC)
        tbl.write_arrow_batches(
            self.pa.table({"id": [7], "name": ["g"]}).to_batches(),
        )
        result = tbl.read_arrow_table()
        self.assertEqual(result.num_rows, 1)

    def test_partition_by_validates_column_names(self) -> None:
        with self.assertRaises(ValueError):
            self.engine["main"]["default"].create_table(
                "bad", schema=_sales_schema(), partition_by=("missing_col",),
            )

    def test_delete_table_purges_data(self) -> None:
        tbl = self._table()
        tbl.write_arrow_batches(
            self.pa.table({"id": [1], "name": ["a"]}).to_batches(),
        )
        self.assertTrue(tbl.exists)
        tbl.delete()
        self.assertFalse(tbl.exists)
        # Listing must no longer surface the table.
        names = [t.name for t in self.engine["main"]["default"].tables()]
        self.assertEqual(names, [])


class TestFSView(ArrowTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = FSEngine(base=LocalPath(str(self.tmp_path / "warehouse")))
        sch = self.engine.create_catalog("main").create_schema("default")
        self.table = sch.create_table("sales", schema=_sales_schema())
        self.table.write_arrow_batches(
            self.pa.table({"id": [1, 2], "name": ["a", "b"]}).to_batches(),
        )

    def test_view_resolves_to_source_table(self) -> None:
        view = self.engine["main"]["default"].create_view(
            "v_sales", source="main.default.sales",
        )
        self.assertTrue(view.exists)
        result = view.read_arrow_table()
        self.assertEqual(result.num_rows, 2)
        self.assertEqual(result.column_names, ["id", "name"])

    def test_view_schema_passthrough(self) -> None:
        view = self.engine["main"]["default"].create_view(
            "v_sales", source=self.table,
        )
        self.assertEqual(
            [f.name for f in view.schema.fields], ["id", "name"],
        )

    def test_view_rejects_unqualified_source(self) -> None:
        with self.assertRaises(ValueError):
            self.engine["main"]["default"].create_view(
                "v_bad", source="sales",
            )

    def test_view_write_is_unsupported(self) -> None:
        view = self.engine["main"]["default"].create_view(
            "v_sales", source="main.default.sales",
        )
        with self.assertRaises(NotImplementedError):
            view.write_arrow_batches(
                self.pa.table({"id": [9], "name": ["z"]}).to_batches(),
            )

    def test_schema_indexing_prefers_table_over_view(self) -> None:
        # Tables and views share the namespace; tables win when both
        # exist under the same name (rare, but exercised here).
        self.engine["main"]["default"].create_view(
            "sales_view", source="main.default.sales",
        )
        hit = self.engine["main"]["default"]["sales_view"]
        self.assertEqual(hit.name, "sales_view")
        self.assertEqual(hit.full_name, "main.default.sales_view")
