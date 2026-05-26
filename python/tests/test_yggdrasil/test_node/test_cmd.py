from __future__ import annotations

import pytest


def test_execute_echo(client):
    resp = client.post("/api/cmd", json={"command": ["echo", "hello world"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["returncode"] == 0
    assert "hello world" in data["stdout"]
    assert data["status"] == "completed"
    assert data["node_id"] == "test-node"
    assert data["command"] == ["echo", "hello world"]


def test_execute_captures_stderr(client):
    resp = client.post("/api/cmd", json={
        "command": ["sh", "-c", "echo oops >&2"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["returncode"] == 0
    assert "oops" in data["stderr"]


def test_execute_failing_command(client):
    resp = client.post("/api/cmd", json={"command": ["false"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["returncode"] != 0
    assert data["status"] == "failed"


def test_execute_with_env(client):
    resp = client.post("/api/cmd", json={
        "command": ["sh", "-c", "echo $MY_TEST_VAR"],
        "env": {"MY_TEST_VAR": "yggdrasil_value"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["returncode"] == 0
    assert "yggdrasil_value" in data["stdout"]


def test_execute_with_cwd(client, tmp_path):
    target = tmp_path / "subdir"
    target.mkdir()
    resp = client.post("/api/cmd", json={
        "command": ["pwd"],
        "cwd": str(target),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["returncode"] == 0
    assert str(target) in data["stdout"]


def test_execute_with_timeout(client):
    resp = client.post("/api/cmd", json={
        "command": ["sleep", "30"],
        "timeout": 0.1,
    })
    assert resp.status_code == 408


def test_get_command_by_id(client):
    post_resp = client.post("/api/cmd", json={"command": ["echo", "lookup"]})
    assert post_resp.status_code == 200
    cmd_id = post_resp.json()["id"]

    get_resp = client.get(f"/api/cmd/{cmd_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["id"] == cmd_id
    assert data["command"] == ["echo", "lookup"]
    assert data["returncode"] == 0
    assert data["stdout"] == post_resp.json()["stdout"]


def test_list_commands(client):
    client.post("/api/cmd", json={"command": ["echo", "first"]})
    client.post("/api/cmd", json={"command": ["echo", "second"]})

    resp = client.get("/api/cmd")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    commands = [item["command"] for item in data["items"]]
    assert ["echo", "first"] in commands
    assert ["echo", "second"] in commands


def test_delete_command(client):
    post_resp = client.post("/api/cmd", json={"command": ["echo", "delete me"]})
    cmd_id = post_resp.json()["id"]

    del_resp = client.delete(f"/api/cmd/{cmd_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["id"] == cmd_id

    get_resp = client.get(f"/api/cmd/{cmd_id}")
    assert get_resp.status_code == 404


def test_get_nonexistent_returns_404(client):
    resp = client.get("/api/cmd/does_not_exist_999")
    assert resp.status_code == 404
