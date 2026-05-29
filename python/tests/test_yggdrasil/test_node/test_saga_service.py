"""Saga catalog — CRUD hierarchy, schema/stat inference, SQL over the plan
engine, Arrow streaming, persistence, and the tmp janitor."""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from yggdrasil.exceptions.api import BadRequestError, ConflictError, NotFoundError
from yggdrasil.node.api.schemas.saga import (
    CatalogCreate, DiscoverRequest, SchemaCreate, SqlRequest, TableCreate,
)
from yggdrasil.node.api.services.saga import SagaService
from yggdrasil.node.config import Settings
from yggdrasil.node.daemon import cleanup_tmp


def _settings(home: Path, **kw) -> Settings:
    return Settings(node_id="t", node_home=home, front_home=home, **kw)


def _svc(home: Path, **kw) -> SagaService:
    return SagaService(_settings(home, **kw))


def _trades(s: Settings, name: str = "trades.parquet") -> str:
    d = s.files_root / "data"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "sym": ["A", "B", "A", "B", "A"],
        "px": [10.0, 20.0, 11.0, 19.0, 12.0],
        "qty": [1, 2, 3, 4, 5],
    }), str(d / name))
    return f"data/{name}"


def _seed(svc: SagaService, src: str) -> None:
    run = asyncio.run
    run(svc.create_catalog(CatalogCreate(name="main")))
    run(svc.create_schema("main", SchemaCreate(name="market")))
    run(svc.create_table("main", "market", TableCreate(name="trades", source_url=src)))


class TestCatalogCrud(unittest.TestCase):
    def test_hierarchy_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home); src = _trades(svc.settings)
            _seed(svc, src)
            cats = asyncio.run(svc.list_catalogs())
            self.assertEqual([c.name for c in cats.catalogs], ["main"])
            self.assertEqual(cats.catalogs[0].schema_count, 1)
            schemas = asyncio.run(svc.list_schemas("main"))
            self.assertEqual(schemas.schemas[0].table_count, 1)
            tables = asyncio.run(svc.list_tables("main", "market"))
            self.assertEqual(tables.tables[0].full_name, "main.market.trades")

    def test_upsert_catalog_keeps_id(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home)
            a = asyncio.run(svc.create_catalog(CatalogCreate(name="c", comment="one")))
            b = asyncio.run(svc.create_catalog(CatalogCreate(name="c", comment="two")))
            self.assertEqual(a.catalog.id, b.catalog.id)
            self.assertEqual(b.catalog.comment, "two")

    def test_delete_non_empty_requires_cascade(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home); src = _trades(svc.settings)
            _seed(svc, src)
            with self.assertRaises(ConflictError):
                asyncio.run(svc.delete_catalog("main"))
            asyncio.run(svc.delete_catalog("main", cascade=True))
            self.assertEqual(asyncio.run(svc.list_catalogs()).catalogs, [])

    def test_missing_raises_not_found(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(Path(d))
            with self.assertRaises(NotFoundError):
                asyncio.run(svc.get_catalog("nope"))


class TestInference(unittest.TestCase):
    def test_columns_and_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home); src = _trades(svc.settings)
            asyncio.run(svc.create_catalog(CatalogCreate(name="main")))
            asyncio.run(svc.create_schema("main", SchemaCreate(name="market")))
            resp = asyncio.run(svc.create_table(
                "main", "market", TableCreate(name="trades", source_url=src)))
            t = resp.table
            self.assertEqual({c.name for c in t.columns}, {"sym", "px", "qty"})
            self.assertEqual(t.statistics.row_count, 5)
            self.assertTrue(t.statistics.size_bytes and t.statistics.size_bytes > 0)
            qty = next(c for c in t.statistics.columns if c.column == "qty")
            self.assertEqual(qty.min, 1)
            self.assertEqual(qty.max, 5)
            self.assertEqual(qty.distinct_count, 5)


class TestSql(unittest.TestCase):
    def test_group_by_over_registered_table(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home); src = _trades(svc.settings)
            _seed(svc, src)
            res = asyncio.run(svc.execute_sql(SqlRequest(
                sql="SELECT sym, sum(qty) AS total FROM main.market.trades GROUP BY sym ORDER BY sym")))
            self.assertEqual([c.name for c in res.columns], ["sym", "total"])
            self.assertEqual(res.rows, [["A", 9], ["B", 6]])
            self.assertIn("main", res.referenced_tables[0])
            self.assertTrue(res.plan_sql)

    def test_unqualified_uses_context(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home); src = _trades(svc.settings)
            _seed(svc, src)
            res = asyncio.run(svc.execute_sql(SqlRequest(
                sql="SELECT count(*) AS n FROM trades", catalog="main", schema="market")))
            self.assertEqual(res.rows, [[5]])

    def test_raw_url_rooted_at_files_root(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home); src = _trades(svc.settings)
            _seed(svc, src)
            res = asyncio.run(svc.execute_sql(SqlRequest(
                sql=f"SELECT count(*) AS n FROM '{src}'")))
            self.assertEqual(res.rows, [[5]])

    def test_preview_limit_truncates(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home, saga_sql_preview_rows=2); src = _trades(svc.settings)
            _seed(svc, src)
            res = asyncio.run(svc.execute_sql(SqlRequest(sql="SELECT * FROM main.market.trades")))
            self.assertEqual(res.row_count, 2)
            self.assertTrue(res.truncated)

    def test_parse_error_is_bad_request(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(Path(d))
            with self.assertRaises(BadRequestError):
                asyncio.run(svc.execute_sql(SqlRequest(sql="SELEKT 1")))

    def test_explain_lists_refs(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home); src = _trades(svc.settings)
            _seed(svc, src)
            ex = svc.explain(SqlRequest(sql="SELECT * FROM main.market.trades WHERE px > 11"))
            self.assertEqual(ex.referenced_tables, ["main.market.trades"])
            self.assertEqual(ex.dialect, "postgres")

    def test_arrow_stream_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home); src = _trades(svc.settings)
            _seed(svc, src)
            stream, cleanup = svc.execute_sql_arrow(SqlRequest(
                sql="SELECT sym, px FROM main.market.trades WHERE px > 11"))
            blob = b"".join(stream)
            if cleanup:
                cleanup()
            table = ipc.open_stream(blob).read_all()
            self.assertEqual(table.column_names, ["sym", "px"])
            self.assertEqual(table.num_rows, 3)

    def test_arrow_stream_spills_when_heavy(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home, saga_result_spill_rows=2); src = _trades(svc.settings)
            _seed(svc, src)
            stream, cleanup = svc.execute_sql_arrow(SqlRequest(sql="SELECT * FROM main.market.trades"))
            blob = b"".join(stream)
            if cleanup:
                cleanup()
            self.assertEqual(ipc.open_stream(blob).read_all().num_rows, 5)


class TestDiscover(unittest.TestCase):
    def test_discover_registers_files(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home)
            _trades(svc.settings, "a.parquet")
            _trades(svc.settings, "b.parquet")
            asyncio.run(svc.create_catalog(CatalogCreate(name="main")))
            asyncio.run(svc.create_schema("main", SchemaCreate(name="market")))
            res = asyncio.run(svc.discover(DiscoverRequest(catalog="main", schema="market", path="data")))
            self.assertEqual({t.name for t in res.tables}, {"a", "b"})


class TestPersistence(unittest.TestCase):
    def test_survives_reload(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home); src = _trades(svc.settings)
            _seed(svc, src)
            reloaded = SagaService(svc.settings)
            cats = asyncio.run(reloaded.list_catalogs())
            self.assertEqual([c.name for c in cats.catalogs], ["main"])
            tables = asyncio.run(reloaded.list_tables("main", "market"))
            self.assertEqual(tables.tables[0].name, "trades")


class TestTmpJanitor(unittest.TestCase):
    def test_reclaims_stale_files(self):
        with tempfile.TemporaryDirectory() as d:
            s = _settings(Path(d), tmp_ttl=100)
            s.tmp_root.mkdir(parents=True, exist_ok=True)
            s.spill_root.mkdir(parents=True, exist_ok=True)
            stale = s.tmp_root / "old.tmp"; stale.write_text("x")
            os.utime(stale, (time.time() - 99999, time.time() - 99999))
            fresh = s.tmp_root / "new.tmp"; fresh.write_text("y")
            removed = cleanup_tmp(s)
            self.assertGreaterEqual(removed, 1)
            self.assertFalse(stale.exists())
            self.assertTrue(fresh.exists())


if __name__ == "__main__":
    unittest.main()
