"""Node FastAPI app: route wiring and end-to-end behavior."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from yggdrasil.node.app import create_app
from yggdrasil.node.api.app import create_api
from yggdrasil.node.config import Settings
from yggdrasil.node.remote import remote
from yggdrasil.node.transport import (
    CONTENT_TYPE_PICKLE,
    deserialize_result,
    serialize_pickle,
)


class TestApp(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.home = Path(self.td.name)
        self.app = create_app(Settings(node_home=self.home, front_home=self.home, allow_remote=True))
        self.client = TestClient(self.app)

    def tearDown(self):
        self.td.cleanup()

    def test_discovery(self):
        self.assertEqual(self.client.get("/api/ping").json(), {"pong": True})
        self.assertEqual(self.client.get("/api/hello").json()["node_id"], "local")
        self.assertEqual(self.client.get("/api/hello/peers").json(), {"peers": []})

    def test_v2_core(self):
        self.assertEqual(self.client.get("/api/v2/health").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/api/v2/stats").status_code, 200)
        self.assertTrue(self.client.get("/api/v2/backend").json()["capabilities"]["fs"])
        self.assertEqual(self.client.get("/api/v2/backend/summary").json()["status"], "ok")
        self.assertIn("entries", self.client.get("/api/v2/audit?limit=5").json())
        self.assertIn("functions", self.client.get("/api/v2/pyfunc").json())
        self.assertIn("python", self.client.get("/api/v2/pyenv").json())

    def test_create_api_same_surface(self):
        api = create_api()
        self.assertEqual(len(api.routes), len(self.app.routes))

    def test_function_lifecycle_and_audit(self):
        r = self.client.post("/api/function", json={"name": "f", "code": "print(1)"})
        fid = r.json()["function"]["id"]
        self.assertEqual(self.client.get(f"/api/function/{fid}").status_code, 200)
        self.assertEqual(self.client.get("/api/function/9999").status_code, 404)
        run = self.client.post(f"/api/function/{fid}/run", json={})
        run_id = run.json()["run"]["id"]
        self.assertEqual(self.client.get(f"/api/run/{run_id}").json()["run"]["status"], "ok")
        # audit captured the create
        entries = self.client.get("/api/v2/audit").json()["entries"]
        self.assertTrue(any(e["action"] == "create" for e in entries))
        self.assertEqual(self.client.delete(f"/api/function/{fid}").json(), {"deleted": fid})

    def test_messenger(self):
        self.client.post("/api/messenger", json={"text": "hi", "sender": "u"})
        chans = self.client.get("/api/messenger/channels").json()["channels"]
        self.assertTrue(any(c["name"] == "general" for c in chans))
        msgs = self.client.get("/api/messenger/channels/general/messages?limit=10").json()["messages"]
        self.assertEqual(msgs[-1]["text"], "hi")

    def test_remote_call_pickle(self):
        @remote(name="test:mul")
        def mul(a, b):
            return a * b

        body = serialize_pickle({"func": "test:mul", "args": (6, 7), "kwargs": {}})
        r = self.client.post("/api/call", content=body, headers={"Content-Type": CONTENT_TYPE_PICKLE})
        result = deserialize_result(r.content, r.headers["content-type"])
        self.assertEqual(result, 42)

    def test_remote_call_tabular_return(self):
        @remote(name="test:tbl")
        def tbl():
            return pa.table({"x": [1, 2, 3]})

        body = serialize_pickle({"func": "test:tbl", "args": (), "kwargs": {}})
        r = self.client.post("/api/call", content=body, headers={"Content-Type": CONTENT_TYPE_PICKLE})
        result = deserialize_result(r.content, r.headers["content-type"])
        self.assertEqual(result.num_rows, 3)

    def test_remote_disabled(self):
        app = create_app(Settings(node_home=self.home, allow_remote=False))
        client = TestClient(app)
        body = serialize_pickle({"func": "x", "args": (), "kwargs": {}})
        r = client.post("/api/call", content=body, headers={"Content-Type": CONTENT_TYPE_PICKLE})
        self.assertEqual(r.status_code, 403)

    def test_saga_mounts(self):
        self.assertEqual(self.client.get("/api/v2/saga/mount").json(), {"mounts": []})
        self.client.post("/api/v2/saga/mount", json={"alias": "a", "target": "/tmp"})
        self.assertEqual(len(self.client.get("/api/v2/saga/mount").json()["mounts"]), 1)

    def test_fs_and_tabular_endpoints(self):
        pq.write_table(pa.table({"a": range(10)}), str(self.home / "t.parquet"))
        ls = self.client.get("/api/v2/fs/ls?path=").json()
        self.assertEqual(ls["total"], 1)
        insp = self.client.get("/api/v2/tabular/inspect?path=t.parquet").json()
        self.assertEqual(insp["row_count"], 10)
        self.assertIn("schema", insp)

    def test_analysis_finance_endpoint(self):
        import math
        price = [100.0 * (1.001 ** i) for i in range(300)]
        pq.write_table(pa.table({"price": price}), str(self.home / "p.parquet"))
        r = self.client.post("/api/v2/analysis/finance", json={"path": "p.parquet", "column": "price"})
        self.assertEqual(r.status_code, 200)
        self.assertGreater(r.json()["metrics"]["total_return"], 0)

    def test_market_symbols(self):
        syms = self.client.get("/api/v2/market/symbols").json()["symbols"]
        self.assertTrue(any(s["symbol"] == "BTCUSD" for s in syms))

    def test_market_tick_ws(self):
        with self.client.websocket_connect("/api/v2/market/tick") as ws:
            tick = ws.receive_json()
            self.assertEqual(tick["symbol"], "BTCUSD")
            self.assertIn("close", tick)


if __name__ == "__main__":
    unittest.main()
