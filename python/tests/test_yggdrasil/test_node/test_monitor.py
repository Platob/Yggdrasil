from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from yggdrasil.node.config import Settings
from yggdrasil.node.services.monitor import MonitorService


def test_get_monitor(client: TestClient):
    resp = client.get("/api/monitor")
    assert resp.status_code == 200
    data = resp.json()
    assert "snapshot" in data
    assert "history" in data
    assert data["node_id"] == "test-node"


def test_get_monitor_snapshot_fields(client: TestClient):
    resp = client.get("/api/monitor")
    assert resp.status_code == 200
    snap = resp.json()["snapshot"]
    assert snap["cpu_percent"] >= 0
    assert snap["memory_total_mb"] > 0
    assert snap["memory_used_mb"] >= 0
    assert snap["memory_percent"] >= 0
    assert snap["disk_percent"] >= 0
    assert snap["timestamp"] != ""


def test_get_monitor_network_fields(client: TestClient):
    resp = client.get("/api/monitor")
    assert resp.status_code == 200
    net = resp.json()["snapshot"]["network"]
    assert net["bytes_sent"] >= 0
    assert net["bytes_recv"] >= 0
    assert net["packets_sent"] >= 0
    assert net["packets_recv"] >= 0
    assert net["timestamp"] != ""


def test_get_monitor_with_limit(client: TestClient):
    resp = client.get("/api/monitor", params={"limit": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["history"]) <= 5


def test_service_snapshot_returns_data(settings: Settings):
    svc = MonitorService(settings, history_size=10)
    snap = svc.snapshot()
    assert snap.cpu_percent >= 0
    assert snap.memory_total_mb > 0
    assert snap.memory_used_mb >= 0
    assert snap.timestamp != ""


def test_service_snapshot_network(settings: Settings):
    svc = MonitorService(settings, history_size=10)
    snap = svc.snapshot()
    assert snap.network.bytes_sent >= 0
    assert snap.network.bytes_recv >= 0
    assert snap.network.packets_sent >= 0
    assert snap.network.packets_recv >= 0


def test_service_history_accumulates(settings: Settings):
    svc = MonitorService(settings, history_size=10)
    svc.snapshot()
    assert len(svc.history(10)) == 1

    svc._last_collect = 0
    svc.snapshot()
    assert len(svc.history(10)) == 2

    svc._last_collect = 0
    svc.snapshot()
    assert len(svc.history(10)) == 3


def test_service_history_limit(settings: Settings):
    svc = MonitorService(settings, history_size=3)
    for _ in range(5):
        svc._last_collect = 0
        svc.snapshot()
    assert len(svc.history(10)) == 3
