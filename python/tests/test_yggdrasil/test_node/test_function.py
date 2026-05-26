from __future__ import annotations


def _mk_func(client, name="test_fn", code="print('hi')"):
    resp = client.post("/api/function", json={"name": name, "code": code})
    assert resp.status_code == 200
    return resp.json()["function"]


def test_list_functions_empty(client):
    resp = client.get("/api/function")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    assert data["functions"] == []


def test_create_function(client):
    fn = _mk_func(client, name="greet", code="print('hello')")
    assert fn["name"] == "greet"
    assert fn["code"] == "print('hello')"
    assert fn["language"] == "python"
    assert fn["description"] == ""
    assert fn["dependencies"] == []
    assert fn["environment_id"] is None
    assert fn["python_version"] is not None or fn["python_version"] is None
    assert "creator" in fn
    assert "created_at" in fn
    assert "updated_at" in fn
    assert fn["run_count"] == 0
    assert fn["deleted_at"] is None
    assert fn["last_used_at"] is None


def test_create_function_id_is_int(client):
    fn = _mk_func(client)
    assert isinstance(fn["id"], int)


def test_create_function_default_language(client):
    resp = client.post("/api/function", json={"name": "lang_test", "code": "x = 1"})
    assert resp.status_code == 200
    assert resp.json()["function"]["language"] == "python"


def test_create_function_state_ready(client):
    fn = _mk_func(client)
    assert fn["state"] == "ready"


def test_upsert_same_name(client):
    fn1 = _mk_func(client, name="dup", code="v1")
    fn2 = _mk_func(client, name="dup", code="v2")
    assert fn2["id"] == fn1["id"]
    assert fn2["code"] == "v2"


def test_get_function(client):
    fn = _mk_func(client)
    resp = client.get(f"/api/function/{fn['id']}")
    assert resp.status_code == 200
    data = resp.json()["function"]
    assert data["id"] == fn["id"]
    assert data["name"] == fn["name"]
    assert data["code"] == fn["code"]


def test_get_nonexistent_404(client):
    resp = client.get("/api/function/999999")
    assert resp.status_code == 404


def test_update_function(client):
    fn = _mk_func(client, code="old")
    resp = client.put(f"/api/function/{fn['id']}", json={
        "code": "new",
        "description": "updated",
    })
    assert resp.status_code == 200
    updated = resp.json()["function"]
    assert updated["code"] == "new"
    assert updated["description"] == "updated"
    assert updated["id"] == fn["id"]


def test_delete_function(client):
    fn = _mk_func(client)
    del_resp = client.delete(f"/api/function/{fn['id']}")
    assert del_resp.status_code == 200
    assert del_resp.json()["function"]["id"] == fn["id"]

    get_resp = client.get(f"/api/function/{fn['id']}")
    assert get_resp.status_code == 404


def test_clone_default_name(client):
    fn = _mk_func(client, name="hello")
    resp = client.post(f"/api/function/{fn['id']}/clone")
    assert resp.status_code == 200
    clone = resp.json()["function"]
    assert clone["name"] == "hello_clone"
    assert clone["id"] != fn["id"]


def test_clone_custom_name(client):
    fn = _mk_func(client, name="origin")
    resp = client.post(f"/api/function/{fn['id']}/clone", json={"name": "custom"})
    assert resp.status_code == 200
    assert resp.json()["function"]["name"] == "custom"


def test_clone_preserves_code_and_deps(client):
    resp = client.post("/api/function", json={
        "name": "with_deps",
        "code": "import pandas",
        "dependencies": ["pandas", "numpy"],
    })
    assert resp.status_code == 200
    fn = resp.json()["function"]

    clone_resp = client.post(f"/api/function/{fn['id']}/clone")
    assert clone_resp.status_code == 200
    clone = clone_resp.json()["function"]
    assert clone["code"] == "import pandas"
    assert clone["dependencies"] == ["pandas", "numpy"]


def test_clone_nonexistent_404(client):
    resp = client.post("/api/function/999999/clone")
    assert resp.status_code == 404


def test_trigger_run(client):
    fn = _mk_func(client)
    resp = client.post(f"/api/function/{fn['id']}/run")
    assert resp.status_code == 200
    run = resp.json()["run"]
    assert run["function_id"] == fn["id"]
    assert "id" in run
    assert "status" in run


def test_list_function_runs(client):
    fn = _mk_func(client)
    client.post(f"/api/function/{fn['id']}/run")
    client.post(f"/api/function/{fn['id']}/run")

    resp = client.get(f"/api/function/{fn['id']}/run")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    assert len(data["runs"]) >= 2


def test_run_updates_last_used_at(client):
    fn = _mk_func(client)
    assert fn["last_used_at"] is None

    client.post(f"/api/function/{fn['id']}/run")

    resp = client.get(f"/api/function/{fn['id']}")
    updated = resp.json()["function"]
    assert updated["last_used_at"] is not None


def test_run_increments_run_count(client):
    fn = _mk_func(client)
    assert fn["run_count"] == 0

    client.post(f"/api/function/{fn['id']}/run")

    resp = client.get(f"/api/function/{fn['id']}")
    updated = resp.json()["function"]
    assert updated["run_count"] == 1
