from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_execute_print(client: TestClient):
    resp = client.post("/api/python", json={"code": "print('hello world')"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["returncode"] == 0
    assert "hello world" in data["stdout"]


def test_execute_returns_result(client: TestClient):
    resp = client.post("/api/python", json={"code": "__result__ = [1, 2, 3]"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"] == [1, 2, 3]
    assert data["status"] == "completed"
    assert data["returncode"] == 0


def test_execute_dict_result(client: TestClient):
    resp = client.post("/api/python", json={"code": "__result__ = {'key': 'value'}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"] == {"key": "value"}


def test_execute_syntax_error(client: TestClient):
    resp = client.post("/api/python", json={"code": "def f(\n"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed" or data["returncode"] != 0


def test_execute_runtime_error(client: TestClient):
    resp = client.post("/api/python", json={"code": "raise ValueError('boom')"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["stderr"] is not None
    assert "ValueError" in data["stderr"]


def test_execute_with_env(client: TestClient):
    code = "import os\n__result__ = os.environ.get('MY_TEST_VAR')"
    resp = client.post("/api/python", json={"code": code, "env": {"MY_TEST_VAR": "hello_ygg"}})
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"] == "hello_ygg"


def test_arrow_ipc_tabular_response(client: TestClient):
    code = "__result__ = {'col': [1, 2, 3]}"
    resp = client.post(
        "/api/python",
        json={"code": code, "result_format": "arrow_ipc"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/vnd.apache.arrow.file"
    assert resp.content[:6] == b"ARROW1"
    assert "X-Bot-Exec-Id" in resp.headers


def test_arrow_ipc_scalar_response(client: TestClient):
    code = "__result__ = 42"
    resp = client.post(
        "/api/python",
        json={"code": code, "result_format": "arrow_ipc"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/vnd.apache.arrow.file"


def test_list_executions(client: TestClient):
    client.post("/api/python", json={"code": "x = 1"})
    client.post("/api/python", json={"code": "x = 2"})
    resp = client.get("/api/python")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) >= 2
    assert data["node_id"] == "test-node"


def test_get_execution_by_id(client: TestClient):
    post_resp = client.post("/api/python", json={"code": "__result__ = 99"})
    assert post_resp.status_code == 200
    exec_id = post_resp.json()["id"]

    get_resp = client.get(f"/api/python/{exec_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["id"] == exec_id
    assert data["result"] == 99


def test_delete_execution(client: TestClient):
    post_resp = client.post("/api/python", json={"code": "__result__ = 'bye'"})
    exec_id = post_resp.json()["id"]

    del_resp = client.delete(f"/api/python/{exec_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["id"] == exec_id

    get_resp = client.get(f"/api/python/{exec_id}")
    assert get_resp.status_code == 404


def test_get_nonexistent_returns_404(client: TestClient):
    resp = client.get("/api/python/does_not_exist")
    assert resp.status_code == 404
