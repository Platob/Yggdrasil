from __future__ import annotations


def _mk_func(client, code="print('ok')"):
    resp = client.post("/api/function", json={"name": f"dag_fn_{id(code)}", "code": code})
    assert resp.status_code == 200
    return resp.json()["function"]


def test_list_dags_empty(client):
    resp = client.get("/api/dag")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    assert data["dags"] == []


def test_create_dag_single_step(client):
    fn = _mk_func(client)
    resp = client.post("/api/dag", json={
        "name": "single",
        "description": "one step",
        "steps": [
            {"id": "s1", "ref": {"function_id": fn["id"]}},
        ],
    })
    assert resp.status_code == 200
    dag = resp.json()["dag"]
    assert dag["name"] == "single"
    assert dag["description"] == "one step"
    assert len(dag["steps"]) == 1
    assert dag["steps"][0]["id"] == "s1"
    assert dag["steps"][0]["ref"]["function_id"] == fn["id"]
    assert dag["edges"] == []
    assert dag["run_count"] == 0
    assert dag["deleted_at"] is None
    assert dag["last_used_at"] is None
    assert "created_at" in dag
    assert "updated_at" in dag


def test_create_dag_multi_step(client):
    fn1 = _mk_func(client, code="print('step1')")
    fn2 = _mk_func(client, code="print('step2')")
    resp = client.post("/api/dag", json={
        "name": "multi",
        "steps": [
            {"id": "extract", "ref": {"function_id": fn1["id"]}},
            {"id": "transform", "ref": {"function_id": fn2["id"]}, "depends_on": ["extract"]},
        ],
    })
    assert resp.status_code == 200
    dag = resp.json()["dag"]
    assert len(dag["steps"]) == 2
    assert dag["steps"][1]["depends_on"] == ["extract"]


def test_create_dag_with_edges(client):
    fn1 = _mk_func(client, code="print('a')")
    fn2 = _mk_func(client, code="print('b')")
    resp = client.post("/api/dag", json={
        "name": "edged",
        "steps": [
            {"id": "src", "ref": {"function_id": fn1["id"]}},
            {"id": "dst", "ref": {"function_id": fn2["id"]}, "depends_on": ["src"]},
        ],
        "edges": [
            {"from_step": "src", "to_step": "dst", "output_key": "result", "input_key": "data"},
        ],
    })
    assert resp.status_code == 200
    dag = resp.json()["dag"]
    assert len(dag["edges"]) == 1
    edge = dag["edges"][0]
    assert edge["from_step"] == "src"
    assert edge["to_step"] == "dst"
    assert edge["output_key"] == "result"
    assert edge["input_key"] == "data"


def test_get_dag(client):
    fn = _mk_func(client)
    create_resp = client.post("/api/dag", json={
        "name": "fetchme",
        "steps": [{"id": "only", "ref": {"function_id": fn["id"]}}],
    })
    dag = create_resp.json()["dag"]

    resp = client.get(f"/api/dag/{dag['id']}")
    assert resp.status_code == 200
    fetched = resp.json()["dag"]
    assert fetched["id"] == dag["id"]
    assert fetched["name"] == "fetchme"
    assert fetched["steps"] == dag["steps"]


def test_get_nonexistent_404(client):
    resp = client.get("/api/dag/999999")
    assert resp.status_code == 404


def test_delete_dag(client):
    fn = _mk_func(client)
    dag = client.post("/api/dag", json={
        "name": "deleteme",
        "steps": [{"id": "x", "ref": {"function_id": fn["id"]}}],
    }).json()["dag"]

    del_resp = client.delete(f"/api/dag/{dag['id']}")
    assert del_resp.status_code == 200
    assert del_resp.json()["dag"]["id"] == dag["id"]

    get_resp = client.get(f"/api/dag/{dag['id']}")
    assert get_resp.status_code == 404


def test_trigger_dag_run(client):
    fn = _mk_func(client)
    dag = client.post("/api/dag", json={
        "name": "runnable",
        "steps": [{"id": "go", "ref": {"function_id": fn["id"]}}],
    }).json()["dag"]

    resp = client.post(f"/api/dag/{dag['id']}/run")
    assert resp.status_code == 200
    run = resp.json()["run"]
    assert run["dag_id"] == dag["id"]
    assert run["status"] == "completed"
    assert run["started_at"] is not None
    assert run["completed_at"] is not None
    assert run["duration"] is not None


def test_list_dag_runs(client):
    fn = _mk_func(client)
    dag = client.post("/api/dag", json={
        "name": "multi_run",
        "steps": [{"id": "r", "ref": {"function_id": fn["id"]}}],
    }).json()["dag"]

    client.post(f"/api/dag/{dag['id']}/run")
    client.post(f"/api/dag/{dag['id']}/run")

    resp = client.get(f"/api/dag/{dag['id']}/run")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    assert len(data["runs"]) >= 2


def test_get_dag_run(client):
    fn = _mk_func(client)
    dag = client.post("/api/dag", json={
        "name": "get_run",
        "steps": [{"id": "s", "ref": {"function_id": fn["id"]}}],
    }).json()["dag"]

    run = client.post(f"/api/dag/{dag['id']}/run").json()["run"]

    resp = client.get(f"/api/dag/{dag['id']}/run/{run['id']}")
    assert resp.status_code == 200
    fetched = resp.json()["run"]
    assert fetched["id"] == run["id"]
    assert fetched["dag_id"] == dag["id"]
    assert fetched["status"] == run["status"]


def test_dag_id_is_int(client):
    fn = _mk_func(client)
    dag = client.post("/api/dag", json={
        "name": "intid",
        "steps": [{"id": "z", "ref": {"function_id": fn["id"]}}],
    }).json()["dag"]
    assert isinstance(dag["id"], int)
