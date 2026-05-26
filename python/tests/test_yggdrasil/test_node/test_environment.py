from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _mk_env(client: TestClient, name: str = "test_env", **kwargs) -> dict:
    body = {"name": name, **kwargs}
    resp = client.post("/api/environment", json=body)
    assert resp.status_code == 200
    return resp.json()["environment"]


def test_list_environments_empty(client: TestClient):
    resp = client.get("/api/environment")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    assert data["environments"] == []


def test_create_environment(client: TestClient):
    env = _mk_env(client, name="myenv", python_version="3.12", dependencies=["numpy"])
    assert env["name"] == "myenv"
    assert env["python_version"] == "3.12"
    assert env["dependencies"] == ["numpy"]
    assert env["status"] in ("pending", "ready")
    assert env["error"] is None
    assert env["deleted_at"] is None
    assert env["last_used_at"] is None
    assert "path" in env
    assert "created_at" in env
    assert "updated_at" in env


def test_create_environment_id_is_int(client: TestClient):
    env = _mk_env(client)
    assert isinstance(env["id"], int)


def test_upsert_same_name(client: TestClient):
    first = _mk_env(client, name="dup")
    second = _mk_env(client, name="dup", dependencies=["requests"])
    assert first["id"] == second["id"]
    assert second["dependencies"] == ["requests"]


def test_get_environment(client: TestClient):
    env = _mk_env(client, name="fetch_me")
    resp = client.get(f"/api/environment/{env['id']}")
    assert resp.status_code == 200
    fetched = resp.json()["environment"]
    assert fetched["id"] == env["id"]
    assert fetched["name"] == "fetch_me"


def test_get_nonexistent_404(client: TestClient):
    resp = client.get("/api/environment/999999999")
    assert resp.status_code == 404


def test_update_environment(client: TestClient):
    env = _mk_env(client, name="updatable")
    resp = client.put(
        f"/api/environment/{env['id']}",
        json={"dependencies": ["flask", "gunicorn"]},
    )
    assert resp.status_code == 200
    updated = resp.json()["environment"]
    assert updated["id"] == env["id"]
    assert updated["dependencies"] == ["flask", "gunicorn"]


def test_delete_environment(client: TestClient):
    env = _mk_env(client, name="deletable")
    del_resp = client.delete(f"/api/environment/{env['id']}")
    assert del_resp.status_code == 200
    assert del_resp.json()["environment"]["id"] == env["id"]

    get_resp = client.get(f"/api/environment/{env['id']}")
    assert get_resp.status_code == 404


def test_clone_default_name(client: TestClient):
    env = _mk_env(client, name="myenv")
    resp = client.post(f"/api/environment/{env['id']}/clone")
    assert resp.status_code == 200
    cloned = resp.json()["environment"]
    assert cloned["name"] == "myenv_clone"
    assert cloned["id"] != env["id"]


def test_clone_custom_name(client: TestClient):
    env = _mk_env(client, name="original")
    resp = client.post(
        f"/api/environment/{env['id']}/clone",
        json={"name": "custom_copy"},
    )
    assert resp.status_code == 200
    cloned = resp.json()["environment"]
    assert cloned["name"] == "custom_copy"


def test_clone_preserves_deps(client: TestClient):
    env = _mk_env(
        client,
        name="with_deps",
        python_version="3.12",
        dependencies=["pandas", "scikit-learn"],
    )
    resp = client.post(f"/api/environment/{env['id']}/clone")
    assert resp.status_code == 200
    cloned = resp.json()["environment"]
    assert cloned["python_version"] == "3.12"
    assert cloned["dependencies"] == ["pandas", "scikit-learn"]


def test_clone_nonexistent_404(client: TestClient):
    resp = client.post("/api/environment/999999999/clone")
    assert resp.status_code == 404
