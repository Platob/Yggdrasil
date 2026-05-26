from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _create_job(client: TestClient, **overrides) -> dict:
    payload = {
        "name": overrides.pop("name", "test-job"),
        "tasks": overrides.pop("tasks", {
            "step1": {"type": "cmd", "command": ["echo", "hello"]},
        }),
        "schedule": overrides.pop("schedule", None),
    }
    payload.update(overrides)
    resp = client.post("/api/job", json=payload)
    assert resp.status_code == 200
    return resp.json()


def test_create_cmd_job(client: TestClient):
    data = _create_job(client, name="cmd-job", tasks={
        "greet": {"type": "cmd", "command": ["echo", "hi"]},
    })
    assert data["node_id"] == "test-node"
    job = data["job"]
    assert job["name"] == "cmd-job"
    assert job["task_keys"] == ["greet"]
    assert job["run_count"] == 0
    assert job["job_id"]
    assert job["created_at"]


def test_create_python_job(client: TestClient):
    data = _create_job(client, name="py-job", tasks={
        "calc": {"type": "python", "code": "print(1+1)"},
    })
    job = data["job"]
    assert job["name"] == "py-job"
    assert job["task_keys"] == ["calc"]
    assert job["schedule"] is None


def test_create_job_with_dependencies(client: TestClient):
    data = _create_job(client, name="dep-job", tasks={
        "first": {"type": "cmd", "command": ["echo", "a"]},
        "second": {"type": "cmd", "command": ["echo", "b"], "depends_on": ["first"]},
    })
    job = data["job"]
    assert "first" in job["task_keys"]
    assert "second" in job["task_keys"]
    assert len(job["task_keys"]) == 2


def test_get_job(client: TestClient):
    created = _create_job(client)
    job_id = created["job"]["job_id"]

    resp = client.get(f"/api/job/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    assert data["job"]["job_id"] == job_id
    assert data["job"]["name"] == "test-job"


def test_list_jobs(client: TestClient):
    _create_job(client, name="list-a")
    _create_job(client, name="list-b")
    _create_job(client, name="list-c")

    resp = client.get("/api/job")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    names = [j["name"] for j in data["items"]]
    assert "list-a" in names
    assert "list-b" in names
    assert "list-c" in names


def test_update_job(client: TestClient):
    created = _create_job(client, name="original")
    job_id = created["job"]["job_id"]

    resp = client.put(f"/api/job/{job_id}", json={
        "name": "updated",
        "tasks": {
            "new_task": {"type": "python", "code": "print('updated')"},
        },
        "schedule": "0 * * * *",
    })
    assert resp.status_code == 200
    job = resp.json()["job"]
    assert job["job_id"] == job_id
    assert job["name"] == "updated"
    assert job["task_keys"] == ["new_task"]
    assert job["schedule"] == "0 * * * *"


def test_delete_job(client: TestClient):
    created = _create_job(client)
    job_id = created["job"]["job_id"]

    resp = client.delete(f"/api/job/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["job"]["job_id"] == job_id

    resp = client.get(f"/api/job/{job_id}")
    assert resp.status_code == 404


def test_trigger_run_cmd(client: TestClient):
    created = _create_job(client, name="run-cmd", tasks={
        "echo_task": {"type": "cmd", "command": ["echo", "hello world"]},
    })
    job_id = created["job"]["job_id"]

    resp = client.post(f"/api/job/{job_id}/run")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    run = data["run"]
    assert run["job_id"] == job_id
    assert run["status"] == "completed"
    assert run["run_id"]
    assert run["started_at"]
    assert run["finished_at"]
    assert run["duration"] >= 0
    assert run["task_results"]["echo_task"]["status"] == "completed"
    assert "hello world" in run["task_results"]["echo_task"]["stdout"]


def test_trigger_run_python(client: TestClient):
    created = _create_job(client, name="run-py", tasks={
        "py_task": {"type": "python", "code": "print(2 ** 10)"},
    })
    job_id = created["job"]["job_id"]

    resp = client.post(f"/api/job/{job_id}/run")
    assert resp.status_code == 200
    run = resp.json()["run"]
    assert run["status"] == "completed"
    assert run["task_results"]["py_task"]["status"] == "completed"
    assert "1024" in run["task_results"]["py_task"]["stdout"]


def test_run_with_dependencies(client: TestClient):
    created = _create_job(client, name="dep-run", tasks={
        "step_a": {"type": "cmd", "command": ["echo", "a"]},
        "step_b": {"type": "cmd", "command": ["echo", "b"], "depends_on": ["step_a"]},
        "step_c": {"type": "python", "code": "print('c')", "depends_on": ["step_a", "step_b"]},
    })
    job_id = created["job"]["job_id"]

    resp = client.post(f"/api/job/{job_id}/run")
    assert resp.status_code == 200
    run = resp.json()["run"]
    assert run["status"] == "completed"
    results = run["task_results"]
    assert results["step_a"]["status"] == "completed"
    assert results["step_b"]["status"] == "completed"
    assert results["step_c"]["status"] == "completed"


def test_list_runs(client: TestClient):
    created = _create_job(client, name="multi-run")
    job_id = created["job"]["job_id"]

    client.post(f"/api/job/{job_id}/run")
    client.post(f"/api/job/{job_id}/run")

    resp = client.get(f"/api/job/{job_id}/run")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    assert len(data["items"]) == 2
    run_ids = {r["run_id"] for r in data["items"]}
    assert len(run_ids) == 2


def test_get_run(client: TestClient):
    created = _create_job(client)
    job_id = created["job"]["job_id"]

    run_resp = client.post(f"/api/job/{job_id}/run")
    run_id = run_resp.json()["run"]["run_id"]

    resp = client.get(f"/api/job/{job_id}/run/{run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    assert data["run"]["run_id"] == run_id
    assert data["run"]["job_id"] == job_id


def test_delete_run(client: TestClient):
    created = _create_job(client)
    job_id = created["job"]["job_id"]

    run_resp = client.post(f"/api/job/{job_id}/run")
    run_id = run_resp.json()["run"]["run_id"]

    resp = client.delete(f"/api/job/{job_id}/run/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["run"]["run_id"] == run_id

    resp = client.get(f"/api/job/{job_id}/run/{run_id}")
    assert resp.status_code == 404


def test_get_nonexistent_job_404(client: TestClient):
    resp = client.get("/api/job/does_not_exist")
    assert resp.status_code == 404


def test_get_nonexistent_run_404(client: TestClient):
    created = _create_job(client)
    job_id = created["job"]["job_id"]

    resp = client.get(f"/api/job/{job_id}/run/does_not_exist")
    assert resp.status_code == 404
