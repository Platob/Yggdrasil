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
    return Settings(node_id="t", node_home=home, saga_home=home / ".saga",
                    front_home=home, **kw)


def _svc(home: Path, **kw) -> SagaService:
    return SagaService(_settings(home, **kw))


def _trades(s: Settings, name: str = "trades.parquet") -> str:
    # source_url is node-home-relative (same rooting as /fs and /tabular).
    d = s.node_home / "data"
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


class TestRegister(unittest.TestCase):
    def test_one_shot_register_creates_catalog_schema_and_infers_name(self):
        from yggdrasil.node.api.schemas.saga import RegisterRequest
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home); src = _trades(svc.settings)
            resp = asyncio.run(svc.register(RegisterRequest(source_url=src)))
            t = resp.table
            self.assertEqual(t.full_name, "main.default.trades")  # name from filename
            self.assertEqual(t.statistics.row_count, 5)
            self.assertEqual(asyncio.run(svc.list_catalogs()).catalogs[0].name, "main")

    def test_dialect_inferred_from_catalog(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home); src = _trades(svc.settings)
            asyncio.run(svc.create_catalog(CatalogCreate(name="dbx", dialect="databricks")))
            asyncio.run(svc.create_schema("dbx", SchemaCreate(name="s")))
            asyncio.run(svc.create_table("dbx", "s", TableCreate(name="trades", source_url=src)))
            # No dialect in the request — should pick up the catalog's.
            _, dialect, _ = svc.plan_for(SqlRequest(sql="SELECT * FROM dbx.s.trades", catalog="dbx"))
            self.assertEqual(dialect.value, "databricks")


class TestPlanGraph(unittest.TestCase):
    def _svc_seeded(self, home):
        svc = _svc(home); _seed(svc, _trades(svc.settings)); return svc

    def test_logical_plan_dag(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc_seeded(Path(d))
            g = svc.build_plan(SqlRequest(
                sql="SELECT sym, sum(qty) AS q FROM main.market.trades WHERE px > 11 GROUP BY sym ORDER BY q DESC LIMIT 5"))
            ops = [(o.op, o.inputs) for o in g.ops]
            kinds = [o for o, _ in ops]
            self.assertEqual(kinds, ["scan", "filter", "aggregate", "sort", "limit"])
            # edges chain forward
            self.assertEqual(g.ops[1].inputs, [g.ops[0].id])
            self.assertEqual(g.ops[-1].inputs, [g.ops[-2].id])

    def test_analyze_fills_rows_and_times(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc_seeded(Path(d))
            g = asyncio.run(svc.analyze_plan(SqlRequest(
                sql="SELECT sym, sum(qty) AS q FROM main.market.trades WHERE px > 11 GROUP BY sym")))
            self.assertTrue(g.analyzed)
            scan = next(o for o in g.ops if o.op == "scan")
            self.assertEqual(scan.rows, 5)               # full table
            filt = next(o for o in g.ops if o.op == "filter")
            self.assertEqual(filt.rows, 3)               # px>11 keeps 3
            self.assertIsNotNone(g.total_ms)

    def test_edit_plan_set_limit_and_drop_order(self):
        from yggdrasil.node.api.schemas.saga import PlanEdit, PlanEditRequest
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc_seeded(Path(d))
            r = svc.edit_plan(PlanEditRequest(
                sql="SELECT * FROM main.market.trades ORDER BY px LIMIT 100",
                edits=[PlanEdit(op="set_limit", value=7), PlanEdit(op="drop_order")]))
            self.assertIn("LIMIT 7", r.sql.upper())
            self.assertNotIn("ORDER BY", r.sql.upper())


class TestObjectTypes(unittest.TestCase):
    def test_view_resolves_and_queries(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(Path(d)); _seed(svc, _trades(svc.settings))
            v = asyncio.run(svc.create_table("main", "market", TableCreate(
                name="big", object_type="VIEW",
                definition="SELECT sym, qty FROM main.market.trades WHERE qty >= 3")))
            self.assertEqual(v.table.object_type, "VIEW")
            self.assertEqual(v.table.statistics.row_count, 3)
            r = asyncio.run(svc.execute_sql(SqlRequest(
                sql="SELECT count(*) AS n FROM main.market.big")))
            self.assertEqual(r.rows, [[3]])

    def test_function_not_queryable(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(Path(d)); _seed(svc, _trades(svc.settings))
            asyncio.run(svc.create_table("main", "market", TableCreate(
                name="fn", object_type="FUNCTION", definition="x=1", infer=False)))
            with self.assertRaises(BadRequestError):
                asyncio.run(svc.execute_sql(SqlRequest(sql="SELECT * FROM main.market.fn")))

    def test_recursive_view_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(Path(d)); _seed(svc, _trades(svc.settings))
            asyncio.run(svc.create_table("main", "market", TableCreate(
                name="loop", object_type="VIEW",
                definition="SELECT * FROM main.market.loop", infer=False)))
            with self.assertRaises(BadRequestError):
                asyncio.run(svc.execute_sql(SqlRequest(sql="SELECT * FROM main.market.loop")))


class TestSearchActivity(unittest.TestCase):
    def test_search_and_limit(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(Path(d)); _seed(svc, _trades(svc.settings))
            r = asyncio.run(svc.search("trad", limit=10))
            names = {h.full_name for h in r.hits}
            self.assertIn("main.market.trades", names)
            r2 = asyncio.run(svc.search("", limit=1))
            self.assertTrue(r2.truncated)
            self.assertEqual(len(r2.hits), 1)

    def test_activity_rollup(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(Path(d)); _seed(svc, _trades(svc.settings))
            asyncio.run(svc.execute_sql(SqlRequest(sql="SELECT * FROM main.market.trades")))
            a = asyncio.run(svc.activity("main", "market", "trades"))
            self.assertIn("register", a.op_counts)
            self.assertIn("query", a.op_counts)
            self.assertGreaterEqual(a.total_ops, 2)
            self.assertTrue(a.daily)


class TestExport(unittest.TestCase):
    def test_export_all_media_types(self):
        from yggdrasil.node.api.schemas.saga import SqlExportRequest
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(Path(d)); _seed(svc, _trades(svc.settings))
            for fmt in ["csv", "parquet", "json", "ndjson", "arrow", "xlsx"]:
                p, name = asyncio.run(svc.export_sql(SqlExportRequest(
                    sql="SELECT sym, sum(qty) AS q FROM main.market.trades GROUP BY sym", fmt=fmt)))
                self.assertTrue(p.exists() and p.stat().st_size > 0, fmt)
                self.assertTrue(name.endswith(f".{fmt}"))
                p.unlink(missing_ok=True)

    def test_export_max_rows_and_bad_fmt(self):
        from yggdrasil.node.api.schemas.saga import SqlExportRequest
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(Path(d)); _seed(svc, _trades(svc.settings))
            p, _ = asyncio.run(svc.export_sql(SqlExportRequest(
                sql="SELECT * FROM main.market.trades", fmt="csv", max_rows=2)))
            self.assertEqual(len(p.read_text().strip().splitlines()), 3)  # header + 2
            p.unlink(missing_ok=True)
            with self.assertRaises(BadRequestError):
                asyncio.run(svc.export_sql(SqlExportRequest(
                    sql="SELECT * FROM main.market.trades", fmt="tsv")))


class TestOpLog(unittest.TestCase):
    def test_register_query_and_drop_logging(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home); src = _trades(svc.settings)
            _seed(svc, src)
            asyncio.run(svc.execute_sql(SqlRequest(sql="SELECT * FROM main.market.trades")))
            log = asyncio.run(svc.read_log("main", "market", "trades"))
            ops = [e.op for e in log.entries]
            self.assertIn("register", ops)
            self.assertIn("query", ops)
            q = next(e for e in log.entries if e.op == "query")
            self.assertEqual(q.rows, 5)
            self.assertTrue(q.user is not None)
            # Drop purges the log.
            asyncio.run(svc.delete_table("main", "market", "trades"))
            after = asyncio.run(svc.read_log("main", "market", "trades"))
            self.assertEqual(after.entries, [])


class _LoopNetwork:
    """Minimal NetworkService stand-in that routes proxy_json to a peer service."""
    def __init__(self, self_id: str, peer_id: str, peer_svc):
        self._self_id = self_id
        self._peer_id = peer_id
        self._peer = peer_svc

    def peer_url(self, node_id):
        return None if node_id == self._self_id else f"http://{node_id}"

    async def proxy_json(self, node_id, method, api_path, *, params=None, json_body=None):
        from yggdrasil.node.api.schemas.saga import TablePayload
        if api_path.endswith("/import"):
            return (await self._peer.import_payload(TablePayload.model_validate(json_body))).model_dump(by_alias=True)
        raise AssertionError(f"unexpected proxy call {api_path}")


class TestReplication(unittest.TestCase):
    def test_metadata_replication_registers_on_peer(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            a = SagaService(_settings(Path(d1)))
            b = SagaService(_settings(Path(d2)))
            src = _trades(a.settings)
            _seed(a, src)
            a.bind_network(_LoopNetwork(a.settings.node_id, "peerB", b))
            from yggdrasil.node.api.schemas.saga import ReplicateRequest
            res = asyncio.run(a.replicate(ReplicateRequest(
                catalog="main", schema="market", table="trades", target="peerB", mode="metadata")))
            self.assertEqual(res.target_node, "peerB")
            # Peer now has the catalog/schema/table, pointing back at the source.
            tb = asyncio.run(b.get_table("main", "market", "trades"))
            self.assertEqual(tb.table.full_name, "main.market.trades")
            self.assertEqual(tb.table.node, a.settings.node_id)
            # Metadata replication does NOT make the peer a data holder, so it
            # is not listed as a replica (only data replicas are routable).
            ta = asyncio.run(a.get_table("main", "market", "trades"))
            self.assertNotIn("peerB", ta.table.replicas)

    def test_export_import_roundtrip(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            a = SagaService(_settings(Path(d1)))
            b = SagaService(_settings(Path(d2)))
            _seed(a, _trades(a.settings))
            payload = a.export_payload("main", "market", "trades")
            asyncio.run(b.import_payload(payload))
            self.assertEqual(asyncio.run(b.list_catalogs()).catalogs[0].name, "main")


class _LoadNetwork:
    """Network stub exposing least_loaded over a fixed load map."""
    def __init__(self, self_id, loads):
        self.settings = type("S", (), {"node_id": self_id})()
        self._loads = loads

    def least_loaded(self, candidates, *, offload_threshold=0.85):
        me = self.settings.node_id
        loads = {n: self._loads[n] for n in candidates if n in self._loads}
        if not loads:
            return next(iter(candidates))
        if me in loads and loads[me] < offload_threshold:
            return me
        return min(loads, key=lambda k: (loads[k], k != me))


class TestResourceRouting(unittest.TestCase):
    def _seeded(self, home, replicas):
        svc = _svc(home)
        src = _trades(svc.settings)
        asyncio.run(svc.create_catalog(CatalogCreate(name="main")))
        asyncio.run(svc.create_schema("main", SchemaCreate(name="market")))
        asyncio.run(svc.create_table("main", "market", TableCreate(name="trades", source_url=src)))
        # mark replicas directly (as a data replication would)
        with svc._lock:
            tid = svc._tbl_idx["main.market.trades"]
            svc._tables[tid] = svc._tables[tid].model_copy(update={"replicas": replicas})
            svc._save()
        return svc

    def test_local_when_not_busy(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._seeded(Path(d), ["peerB"])
            svc.bind_network(_LoadNetwork("t", {"t": 0.1, "peerB": 0.9}))
            self.assertIsNone(svc.compute_node(SqlRequest(sql="SELECT * FROM main.market.trades")))

    def test_offload_to_freer_replica_when_busy(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._seeded(Path(d), ["peerB"])
            svc.bind_network(_LoadNetwork("t", {"t": 0.95, "peerB": 0.2}))
            self.assertEqual(svc.compute_node(SqlRequest(sql="SELECT * FROM main.market.trades")), "peerB")

    def test_no_common_holder_raises(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(Path(d)); src = _trades(svc.settings)
            asyncio.run(svc.create_catalog(CatalogCreate(name="main")))
            asyncio.run(svc.create_schema("main", SchemaCreate(name="m")))
            asyncio.run(svc.create_table("main", "m", TableCreate(name="local", source_url=src)))
            # a table that only lives on a peer (no local data, no replicas)
            asyncio.run(svc.create_table("main", "m", TableCreate(
                name="remote", source_url="x", node="peerB", infer=False)))
            with self.assertRaises(BadRequestError):
                svc.compute_node(SqlRequest(
                    sql="SELECT * FROM main.m.local JOIN main.m.remote ON 1=1"))


class TestStaging(unittest.TestCase):
    def test_stage_result_to_local_nodepath(self):
        import pyarrow.ipc as _ipc
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); svc = _svc(home); src = _trades(svc.settings)
            _seed(svc, src)
            # A scheme-less staging path resolves under the node files root.
            res = asyncio.run(svc.stage_result(SqlRequest(
                sql="SELECT sym, px FROM main.market.trades WHERE px > 11",
                staging_path="stg-out.arrows")))
            self.assertEqual(res.row_count, 3)
            out = svc.settings.files_root / "stg-out.arrows"
            self.assertTrue(out.exists())
            table = _ipc.open_stream(out.read_bytes()).read_all()
            self.assertEqual(table.num_rows, 3)


class TestTmpJanitor(unittest.TestCase):
    def test_reclaims_by_name_encoded_expiry(self):
        from yggdrasil.node import scratch
        with tempfile.TemporaryDirectory() as d:
            s = _settings(Path(d), tmp_ttl=100)
            s.tmp_root.mkdir(parents=True, exist_ok=True)
            now = scratch.now_ms()
            # end_ms in the past → expired; in the future → kept.
            expired = s.tmp_root / f"tmp-{now - 200000}-{now - 100000}-spill.arrows"
            expired.write_text("x")
            fresh = s.tmp_root / f"tmp-{now}-{now + 100000}-spill.arrows"
            fresh.write_text("y")
            # Foreign file with no encoded expiry → mtime fallback (tmp_ttl=100s).
            foreign = s.tmp_root / "legacy.tmp"; foreign.write_text("z")
            os.utime(foreign, (time.time() - 99999, time.time() - 99999))
            removed = cleanup_tmp(s)
            self.assertGreaterEqual(removed, 2)
            self.assertFalse(expired.exists())
            self.assertTrue(fresh.exists())
            self.assertFalse(foreign.exists())

    def test_stg_is_name_only(self):
        with tempfile.TemporaryDirectory() as d:
            s = _settings(Path(d))
            s.stg_root.mkdir(parents=True, exist_ok=True)
            # A foreign file in stg is persistent — no fallback TTL there.
            keep = s.stg_root / "result.arrows"; keep.write_text("x")
            os.utime(keep, (time.time() - 9_999_999, time.time() - 9_999_999))
            cleanup_tmp(s)
            self.assertTrue(keep.exists())


if __name__ == "__main__":
    unittest.main()
