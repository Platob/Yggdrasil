from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa
from fastapi.testclient import TestClient

from yggdrasil.node.app import create_app
from yggdrasil.node.config import Settings
from yggdrasil.node.remote import remote
from yggdrasil.node.transport import (
    CONTENT_TYPE_ARROW_STREAM,
    CONTENT_TYPE_PICKLE,
    deserialize_pickle,
    read_arrow_stream,
    serialize_pickle,
)


@remote(name="test:add")
def _add(x: int, y: int) -> int:
    return x + y


@remote(name="test:table")
def _table(n: int) -> pa.Table:
    return pa.table({"i": list(range(n))})


def _client() -> TestClient:
    d = tempfile.mkdtemp()
    app = create_app(Settings(node_home=Path(d), front_home=Path(d), allow_remote=True))
    return TestClient(app)


def test_ping_and_health():
    c = _client()
    assert c.get("/api/ping").json()["status"] == "ok"
    assert c.get("/api/v2/health").json()["status"] == "healthy"


def test_v2_backend_and_stats():
    c = _client()
    assert c.get("/api/v2/backend").json()["backend"] == "ygg-node"
    assert "messages" in c.get("/api/v2/stats").json()


def test_call_scalar():
    c = _client()
    body = serialize_pickle({"func": "test:add", "args": (2, 3), "kwargs": {}})
    r = c.post("/api/call", content=body, headers={"Content-Type": CONTENT_TYPE_PICKLE})
    assert r.headers["content-type"].startswith(CONTENT_TYPE_PICKLE)
    assert deserialize_pickle(r.content) == 5


def test_call_tabular_streams_arrow():
    c = _client()
    body = serialize_pickle({"func": "test:table", "args": (4,), "kwargs": {}})
    r = c.post("/api/call", content=body, headers={"Content-Type": CONTENT_TYPE_PICKLE})
    assert r.headers["content-type"].startswith(CONTENT_TYPE_ARROW_STREAM)
    assert read_arrow_stream(r.content).num_rows == 4


def test_call_unknown_func_404():
    c = _client()
    body = serialize_pickle({"func": "nope", "args": (), "kwargs": {}})
    r = c.post("/api/call", content=body, headers={"Content-Type": CONTENT_TYPE_PICKLE})
    assert r.status_code == 404


def test_messenger_flow():
    c = _client()
    c.post("/api/messenger", json={"text": "hello", "sender": "u"})
    chans = c.get("/api/messenger/channels").json()["channels"]
    assert any(ch["name"] == "general" for ch in chans)
    msgs = c.get("/api/messenger/channels/general/messages?limit=10").json()
    assert msgs["total"] == 1
