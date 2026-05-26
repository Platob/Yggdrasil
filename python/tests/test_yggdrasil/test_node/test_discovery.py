from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from yggdrasil.node.config import Settings
from yggdrasil.node.services.discovery import DiscoveryService, _PEER_TTL


PEER_PAYLOAD = {
    "node_id": "peer-1",
    "host": "10.0.0.2",
    "port": 8100,
    "version": "0.1.0",
}


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# -- Endpoint tests -----------------------------------------------------------


def test_hello_get_returns_node_info(client: TestClient):
    resp = client.get("/api/hello")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    assert isinstance(data["host"], str)
    assert isinstance(data["port"], int)
    assert isinstance(data["version"], str)
    assert data["uptime"] >= 0


def test_hello_get_includes_channels_and_functions(client: TestClient):
    resp = client.get("/api/hello")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["channels"], list)
    assert isinstance(data["functions"], list)


def test_hello_post_registers_peer(client: TestClient):
    resp = client.post("/api/hello", json=PEER_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    peer_ids = [p["node_id"] for p in data["peers"]]
    assert "peer-1" in peer_ids


def test_hello_post_returns_own_node_info(client: TestClient):
    resp = client.post("/api/hello", json=PEER_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"


def test_hello_post_updates_existing_peer(client: TestClient):
    client.post("/api/hello", json=PEER_PAYLOAD)
    updated = {**PEER_PAYLOAD, "port": 9999}
    resp = client.post("/api/hello", json=updated)
    assert resp.status_code == 200
    peers = resp.json()["peers"]
    matched = [p for p in peers if p["node_id"] == "peer-1"]
    assert len(matched) == 1
    assert matched[0]["port"] == 9999


def test_peers_list_returns_registered_peers(client: TestClient):
    client.post("/api/hello", json=PEER_PAYLOAD)
    from yggdrasil.node.middleware import invalidate_response_cache
    invalidate_response_cache()
    resp = client.get("/api/hello/peers")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    peer_ids = [p["node_id"] for p in data["peers"]]
    assert "peer-1" in peer_ids


def test_hello_get_includes_messenger_channels(client: TestClient):
    client.post("/api/messenger/channels", params={"name": "test-chan"})
    from yggdrasil.node.middleware import invalidate_response_cache
    invalidate_response_cache()
    resp = client.get("/api/hello")
    assert resp.status_code == 200
    channels = resp.json()["channels"]
    assert "test-chan" in channels


# -- Service unit tests --------------------------------------------------------


def test_service_stale_peers_purged(settings: Settings):
    svc = DiscoveryService(settings)
    _run(svc.hello(
        _make_hello_request("stale-peer", "10.0.0.5", 8100),
    ))
    with svc._lock:
        svc._peers["stale-peer"].last_seen = time.monotonic() - _PEER_TTL - 100
    result = _run(svc.get_peers())
    peer_ids = [p.node_id for p in result.peers]
    assert "stale-peer" not in peer_ids


def test_service_uptime_increases(settings: Settings):
    svc = DiscoveryService(settings)
    time.sleep(0.05)
    info = _run(svc.get_self_info())
    assert info.uptime > 0


def test_service_hello_returns_own_id(settings: Settings):
    svc = DiscoveryService(settings)
    resp = _run(svc.hello(
        _make_hello_request("other-node", "10.0.0.9", 8200),
    ))
    assert resp.node_id == settings.node_id


def _make_hello_request(node_id: str, host: str, port: int):
    from yggdrasil.node.schemas.discovery import HelloRequest
    return HelloRequest(node_id=node_id, host=host, port=port, version="0.1.0")
