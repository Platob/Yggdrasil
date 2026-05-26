from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _mk_func(client: TestClient, code: str = "print('hi')") -> int:
    resp = client.post(
        "/api/function",
        json={"name": "test_fn", "code": code},
    )
    assert resp.status_code == 200
    return resp.json()["function"]["id"]


def test_list_runs_empty(client: TestClient):
    resp = client.get("/api/run")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    assert data["runs"] == []


def test_create_run(client: TestClient):
    fid = _mk_func(client, code="x = 1 + 1")
    resp = client.post("/api/run", json={"function_id": fid})
    assert resp.status_code == 200
    run = resp.json()["run"]
    assert run["function_id"] == fid
    assert run["status"] == "completed"
    assert run["returncode"] == 0
    assert run["node_id"] == "test-node"
    assert run["started_at"] is not None
    assert run["completed_at"] is not None
    assert run["duration"] is not None
    assert run["duration"] >= 0


def test_create_run_captures_stdout(client: TestClient):
    fid = _mk_func(client, code="print('hello from run')")
    resp = client.post("/api/run", json={"function_id": fid})
    assert resp.status_code == 200
    run = resp.json()["run"]
    assert run["status"] == "completed"
    assert "hello from run" in run["stdout"]


def test_create_run_with_timeout_success(client: TestClient):
    fid = _mk_func(client, code="x = 42")
    resp = client.post("/api/run", json={"function_id": fid, "timeout": 10.0})
    assert resp.status_code == 200
    run = resp.json()["run"]
    assert run["status"] == "completed"
    assert run["timeout"] == 10.0


def test_create_run_with_timeout_expiry(client: TestClient):
    fid = _mk_func(client, code="import time; time.sleep(10)")
    resp = client.post("/api/run", json={"function_id": fid, "timeout": 0.5})
    assert resp.status_code == 200
    run = resp.json()["run"]
    assert run["status"] == "failed"
    assert run["stderr"] is not None
    assert "imed out" in run["stderr"]


def test_create_run_with_resource_limits(client: TestClient):
    fid = _mk_func(client)
    resp = client.post("/api/run", json={
        "function_id": fid,
        "max_memory_mb": 512,
        "max_cpu_percent": 80.0,
        "timeout": 30.0,
    })
    assert resp.status_code == 200
    run = resp.json()["run"]
    assert run["max_memory_mb"] == 512
    assert run["max_cpu_percent"] == 80.0
    assert run["timeout"] == 30.0


def test_create_run_without_limits_defaults_none(client: TestClient):
    fid = _mk_func(client)
    resp = client.post("/api/run", json={"function_id": fid})
    assert resp.status_code == 200
    run = resp.json()["run"]
    assert run["max_memory_mb"] is None
    assert run["max_cpu_percent"] is None
    assert run["timeout"] is None


def test_get_run(client: TestClient):
    fid = _mk_func(client)
    create_resp = client.post("/api/run", json={"function_id": fid})
    run_id = create_resp.json()["run"]["id"]

    resp = client.get(f"/api/run/{run_id}")
    assert resp.status_code == 200
    run = resp.json()["run"]
    assert run["id"] == run_id
    assert run["function_id"] == fid


def test_get_nonexistent_404(client: TestClient):
    resp = client.get("/api/run/9999999999")
    assert resp.status_code == 404


def test_delete_run(client: TestClient):
    fid = _mk_func(client)
    create_resp = client.post("/api/run", json={"function_id": fid})
    run_id = create_resp.json()["run"]["id"]

    del_resp = client.delete(f"/api/run/{run_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["run"]["id"] == run_id

    get_resp = client.get(f"/api/run/{run_id}")
    assert get_resp.status_code == 404


def test_list_runs_after_creating(client: TestClient):
    fid = _mk_func(client)
    client.post("/api/run", json={"function_id": fid})
    client.post("/api/run", json={"function_id": fid})

    resp = client.get("/api/run")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    assert len(data["runs"]) >= 2
    for run in data["runs"]:
        assert run["function_id"] == fid


def test_stream_logs(client: TestClient):
    fid = _mk_func(client, code="print('streamed line')")
    create_resp = client.post("/api/run", json={"function_id": fid})
    run_id = create_resp.json()["run"]["id"]

    with client.stream("GET", f"/api/run/{run_id}/logs") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.read().decode()
    assert "streamed line" in body
    assert "complete" in body
