"""End-to-end coverage for the rebuilt :mod:`yggdrasil.fastapi` service.

Tests use FastAPI's :class:`TestClient` against a fresh
:class:`TabularEngine` per test so the process-wide ``SYSTEM_ENGINE``
stays clean. Data-shaped assertions go through :class:`ArrowTestCase`
so the suite skips cleanly when a downstream optional dep is missing.

Coverage:

- catalog navigation across catalogs / schemas / tables / per-table
  schema (including 404 paths)
- inline + binary upload + path registration
- Arrow IPC stream round-trip on ``GET /data/...``
- format negotiation through ``?format=`` and the ``Accept`` header
- SQL execution against the engine
- 404 / 415 / empty-body error shapes
"""

from __future__ import annotations

import io
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.arrow.tests import ArrowTestCase


def _need(*pkgs: str) -> None:
    for name in pkgs:
        try:
            __import__(name)
        except Exception as exc:  # pragma: no cover — surfaces a clear skip
            raise unittest.SkipTest(f"{name} is not installed: {exc}") from exc


class _BaseAPI(ArrowTestCase):
    """Shared fixture: a TestClient over a fresh engine with one table."""

    def setUp(self) -> None:
        super().setUp()
        _need("fastapi", "httpx")

        from fastapi.testclient import TestClient

        from yggdrasil.fastapi import Settings, create_app
        from yggdrasil.io.tabular import ArrowTabular, TabularEngine

        self.engine = TabularEngine()
        self.source_table = self.pa.table(
            {"id": [1, 2, 3], "name": ["a", "b", "c"]}
        )
        self.engine.register(
            "main", "core", "t1", ArrowTabular(self.source_table),
        )

        app = create_app(settings=Settings(allow_remote=True), engine=self.engine)
        self.client = TestClient(app)


class TestCatalogRouter(_BaseAPI):
    def test_engine_listing(self) -> None:
        r = self.client.get("/catalog")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["catalogs"], ["main"])
        self.assertEqual(body["qualified_names"], ["main.core.t1"])

    def test_catalog_listing(self) -> None:
        r = self.client.get("/catalog/main")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            r.json(), {"catalog": "main", "schemas": ["core"]},
        )

    def test_catalog_404(self) -> None:
        r = self.client.get("/catalog/missing")
        self.assertEqual(r.status_code, 404)
        self.assertIn("missing", r.json()["detail"])

    def test_schema_listing(self) -> None:
        r = self.client.get("/catalog/main/core")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            r.json(),
            {"catalog": "main", "schema": "core", "tables": ["t1"]},
        )

    def test_table_entry(self) -> None:
        r = self.client.get("/catalog/main/core/t1")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["qualified_name"], "main.core.t1")
        self.assertEqual(body["tabular_class"], "ArrowTabular")

    def test_table_schema(self) -> None:
        r = self.client.get("/catalog/main/core/t1/schema")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual([f["name"] for f in body["fields"]], ["id", "name"])
        self.assertTrue(all(f["nullable"] for f in body["fields"]))

    def test_table_404(self) -> None:
        r = self.client.get("/catalog/main/core/missing")
        self.assertEqual(r.status_code, 404)


class TestDataRouter(_BaseAPI):
    def test_arrow_ipc_stream_default(self) -> None:
        r = self.client.get("/data/main/core/t1")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(
            r.headers["content-type"].startswith(
                "application/vnd.apache.arrow.stream"
            )
        )
        reader = pa.ipc.open_stream(pa.py_buffer(r.content))
        self.assertFrameEqual(reader.read_all(), self.source_table)

    def test_format_query_overrides_accept(self) -> None:
        r = self.client.get(
            "/data/main/core/t1?format=application/vnd.apache.parquet",
            headers={"accept": "application/json"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertTrue(
            r.headers["content-type"].startswith("application/vnd.apache.parquet")
        )
        roundtrip = pq.read_table(io.BytesIO(r.content))
        self.assertFrameEqual(roundtrip, self.source_table)

    def test_accept_header_negotiation(self) -> None:
        r = self.client.get(
            "/data/main/core/t1", headers={"accept": "application/json"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("application/json"))
        self.assertEqual(
            r.json(),
            [
                {"id": 1, "name": "a"},
                {"id": 2, "name": "b"},
                {"id": 3, "name": "c"},
            ],
        )

    def test_csv(self) -> None:
        r = self.client.get("/data/main/core/t1?format=text/csv")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/csv"))
        body = r.text
        self.assertEqual(body.splitlines()[0], '"id","name"')
        self.assertIn('1,"a"', body)

    def test_unknown_format_falls_back_to_arrow_stream(self) -> None:
        r = self.client.get(
            "/data/main/core/t1?format=application/x-totally-bogus"
        )
        # The resolver always lands on Arrow stream when nothing else
        # matches — most useful behavior, no 415 surprise.
        self.assertEqual(r.status_code, 200)
        self.assertTrue(
            r.headers["content-type"].startswith(
                "application/vnd.apache.arrow.stream"
            )
        )

    def test_data_404(self) -> None:
        r = self.client.get("/data/main/core/missing")
        self.assertEqual(r.status_code, 404)


class TestSourcesRouter(_BaseAPI):
    def test_register_inline_rows(self) -> None:
        r = self.client.post(
            "/sources/main/core/inline_t/inline",
            json={"rows": [{"a": 1}, {"a": 2}]},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["rows"], 2)
        self.assertEqual(body["field_count"], 1)
        self.assertIsNotNone(self.engine.get("main", "core", "inline_t"))

    def test_register_inline_columns(self) -> None:
        r = self.client.post(
            "/sources/main/core/cols_t/inline",
            json={"columns": {"x": [10, 20, 30], "y": ["p", "q", "r"]}},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["rows"], 3)
        self.assertEqual(body["field_count"], 2)

    def test_register_inline_conflict_400(self) -> None:
        r = self.client.post(
            "/sources/main/core/conflict/inline",
            json={"rows": [{"a": 1}], "columns": {"a": [1]}},
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("exactly one", r.json()["detail"])

    def test_register_inline_empty_400(self) -> None:
        r = self.client.post("/sources/main/core/empty/inline", json={})
        self.assertEqual(r.status_code, 400)

    def test_register_upload_arrow_stream(self) -> None:
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, self.source_table.schema) as writer:
            for batch in self.source_table.to_batches():
                writer.write_batch(batch)

        r = self.client.post(
            "/sources/main/core/uploaded/upload",
            headers={"content-type": "application/vnd.apache.arrow.stream"},
            content=sink.getvalue().to_pybytes(),
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["rows"], self.source_table.num_rows)
        roundtrip = self.engine.get_tabular(
            "main", "core", "uploaded",
        ).read_arrow_table()
        self.assertFrameEqual(roundtrip, self.source_table)

    def test_register_upload_parquet(self) -> None:
        sink = pa.BufferOutputStream()
        pq.write_table(self.source_table, sink)

        r = self.client.post(
            "/sources/main/core/uploaded_pq/upload",
            headers={"content-type": "application/vnd.apache.parquet"},
            content=sink.getvalue().to_pybytes(),
        )
        self.assertEqual(r.status_code, 200)
        roundtrip = self.engine.get_tabular(
            "main", "core", "uploaded_pq",
        ).read_arrow_table()
        self.assertFrameEqual(roundtrip, self.source_table)

    def test_register_upload_unknown_415(self) -> None:
        r = self.client.post(
            "/sources/main/core/bogus/upload",
            headers={"content-type": "application/x-mystery-format"},
            content=b"not a real format",
        )
        self.assertEqual(r.status_code, 415)
        self.assertIn("Unsupported upload media type", r.json()["detail"])

    def test_register_upload_empty_400(self) -> None:
        r = self.client.post(
            "/sources/main/core/empty/upload",
            headers={"content-type": "application/vnd.apache.arrow.stream"},
            content=b"",
        )
        self.assertEqual(r.status_code, 400)

    def test_register_path_local_parquet(self) -> None:
        path = self.tmp_path / "t.parquet"
        pq.write_table(self.source_table, path)

        r = self.client.post(
            "/sources/main/core/from_path/path",
            json={"path": str(path)},
        )
        self.assertEqual(r.status_code, 200)
        roundtrip = self.engine.get_tabular(
            "main", "core", "from_path",
        ).read_arrow_table()
        self.assertFrameEqual(roundtrip, self.source_table)

    def test_deregister(self) -> None:
        r = self.client.delete("/sources/main/core/t1")
        self.assertEqual(r.status_code, 204)
        self.assertIsNone(self.engine.get("main", "core", "t1"))

    def test_deregister_404(self) -> None:
        r = self.client.delete("/sources/main/core/never_registered")
        self.assertEqual(r.status_code, 404)


class TestSqlEndpoint(_BaseAPI):
    def test_sql_select_all(self) -> None:
        r = self.client.post(
            "/data/sql", content="SELECT * FROM main.core.t1",
        )
        self.assertEqual(r.status_code, 200)
        self.assertTrue(
            r.headers["content-type"].startswith(
                "application/vnd.apache.arrow.stream"
            )
        )
        reader = pa.ipc.open_stream(pa.py_buffer(r.content))
        roundtrip = reader.read_all()
        self.assertFrameEqual(
            roundtrip.select(["id", "name"]),
            self.source_table.select(["id", "name"]),
        )

    def test_sql_json_response(self) -> None:
        r = self.client.post(
            "/data/sql?format=application/json",
            content="SELECT id FROM main.core.t1 WHERE id > 1",
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), [{"id": 2}, {"id": 3}])

    def test_sql_empty_body_400(self) -> None:
        r = self.client.post("/data/sql", content="")
        self.assertEqual(r.status_code, 400)


class TestLocalOnlyMiddleware(unittest.TestCase):
    def test_local_only_blocks_remote(self) -> None:
        _need("fastapi", "httpx")

        from fastapi.testclient import TestClient

        from yggdrasil.fastapi import Settings, create_app
        from yggdrasil.io.tabular import TabularEngine

        engine = TabularEngine()
        # ``allow_remote=False`` (default) and TestClient impersonates a
        # remote host (``testserver``), so every request 403s.
        app = create_app(settings=Settings(allow_remote=False), engine=engine)
        client = TestClient(app)
        r = client.get("/catalog")
        self.assertEqual(r.status_code, 403)
        self.assertIn("Remote access is disabled", r.json()["detail"])
