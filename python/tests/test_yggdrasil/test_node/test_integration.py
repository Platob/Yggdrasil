from __future__ import annotations

from yggdrasil.node.middleware import invalidate_response_cache


def _mk_func(client, name="integ_fn", code="print('hi')"):
    resp = client.post("/api/function", json={"name": name, "code": code})
    assert resp.status_code == 200
    return resp.json()["function"]


def test_function_create_run_and_verify_via_run_api(client):
    fn = _mk_func(client, name="run_verify", code="print('run_verify')")
    run_resp = client.post(f"/api/function/{fn['id']}/run")
    assert run_resp.status_code == 200
    run = run_resp.json()["run"]
    assert run["function_id"] == fn["id"]

    all_runs = client.get("/api/run")
    assert all_runs.status_code == 200
    run_ids = [r["id"] for r in all_runs.json()["runs"]]
    assert run["id"] in run_ids


def test_function_upsert_preserves_runs(client):
    fn = _mk_func(client, name="upsert_fn", code="v1")
    client.post(f"/api/function/{fn['id']}/run")
    client.post(f"/api/function/{fn['id']}/run")

    runs_before = client.get(f"/api/function/{fn['id']}/run")
    assert runs_before.status_code == 200
    count_before = len(runs_before.json()["runs"])
    assert count_before >= 2

    fn2 = _mk_func(client, name="upsert_fn", code="v2")
    assert fn2["id"] == fn["id"]
    assert fn2["code"] == "v2"

    runs_after = client.get(f"/api/function/{fn['id']}/run")
    assert runs_after.status_code == 200
    count_after = len(runs_after.json()["runs"])
    assert count_after == count_before

    first_run_id = runs_before.json()["runs"][0]["id"]
    single_run = client.get(f"/api/run/{first_run_id}")
    assert single_run.status_code == 200
    assert single_run.json()["run"]["id"] == first_run_id


def test_environment_create_and_assign_to_function(client):
    env_resp = client.post("/api/environment", json={
        "name": "test-env",
        "python_version": "3.11",
    })
    assert env_resp.status_code == 200
    env = env_resp.json()["environment"]
    env_id = env["id"]

    fn = client.post("/api/function", json={
        "name": "env_bound_fn",
        "code": "print('env')",
        "environment_id": env_id,
    })
    assert fn.status_code == 200
    fn_data = fn.json()["function"]
    assert fn_data["environment_id"] == env_id

    fetched = client.get(f"/api/function/{fn_data['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["function"]["environment_id"] == env_id


def test_dag_with_multiple_functions(client):
    fn1 = _mk_func(client, name="dag_step_1", code="print('step1')")
    fn2 = _mk_func(client, name="dag_step_2", code="print('step2')")

    dag_resp = client.post("/api/dag", json={
        "name": "two_step_dag",
        "steps": [
            {"id": "first", "ref": {"function_id": fn1["id"]}},
            {"id": "second", "ref": {"function_id": fn2["id"]}, "depends_on": ["first"]},
        ],
    })
    assert dag_resp.status_code == 200
    dag = dag_resp.json()["dag"]
    assert len(dag["steps"]) == 2

    run_resp = client.post(f"/api/dag/{dag['id']}/run")
    assert run_resp.status_code == 200
    run = run_resp.json()["run"]
    assert run["status"] == "completed"
    assert run["dag_id"] == dag["id"]
    assert "first" in run["step_results"]
    assert "second" in run["step_results"]


def test_filesystem_write_read_delete_cycle(client):
    write_resp = client.post("/api/fs/write", json={
        "path": "integ_test/hello.txt",
        "content": "integration test content",
    })
    assert write_resp.status_code == 200
    assert write_resp.json()["name"] == "hello.txt"

    read_resp = client.get("/api/fs/read", params={"path": "integ_test/hello.txt"})
    assert read_resp.status_code == 200
    assert read_resp.json()["content"] == "integration test content"
    assert read_resp.json()["encoding"] == "utf-8"

    ls_resp = client.get("/api/fs/ls", params={"path": "integ_test"})
    assert ls_resp.status_code == 200
    names = [e["name"] for e in ls_resp.json()["entries"]]
    assert "hello.txt" in names

    del_resp = client.delete("/api/fs/delete", params={"path": "integ_test/hello.txt"})
    assert del_resp.status_code == 204

    gone_resp = client.get("/api/fs/read", params={"path": "integ_test/hello.txt"})
    assert gone_resp.status_code == 404


def test_messenger_full_lifecycle(client):
    ch_resp = client.post("/api/messenger/channels", params={"name": "integ-chan"})
    assert ch_resp.status_code == 200
    assert ch_resp.json()["channel"]["name"] == "integ-chan"
    assert ch_resp.json()["channel"]["message_count"] == 0
    assert ch_resp.json()["channel"]["members"] == []

    client.post("/api/messenger", json={
        "text": "hello from alice",
        "sender": "alice",
        "channel": "integ-chan",
    })
    client.post("/api/messenger", json={
        "text": "hello from bob",
        "sender": "bob",
        "channel": "integ-chan",
    })
    client.post("/api/messenger", json={
        "text": "alice again",
        "sender": "alice",
        "channel": "integ-chan",
    })

    msgs_resp = client.get(
        "/api/messenger/channels/integ-chan/messages",
        params={"limit": 2},
    )
    assert msgs_resp.status_code == 200
    assert len(msgs_resp.json()["messages"]) == 2

    ch_info = client.get("/api/messenger/channels/integ-chan")
    assert ch_info.status_code == 200
    members = ch_info.json()["channel"]["members"]
    assert "alice" in members
    assert "bob" in members
    assert ch_info.json()["channel"]["message_count"] == 3

    del_resp = client.delete("/api/messenger/channels/integ-chan")
    assert del_resp.status_code == 200
    assert del_resp.json()["channel"]["name"] == "integ-chan"

    gone = client.get("/api/messenger/channels/integ-chan")
    assert gone.status_code == 404


def test_discovery_registers_peer_and_lists(client):
    peer1 = {
        "node_id": "peer-alpha",
        "host": "10.0.0.10",
        "port": 8100,
        "version": "0.1.0",
    }
    resp1 = client.post("/api/hello", json=peer1)
    assert resp1.status_code == 200
    peer_ids = [p["node_id"] for p in resp1.json()["peers"]]
    assert "peer-alpha" in peer_ids

    invalidate_response_cache()
    peers_resp = client.get("/api/hello/peers")
    assert peers_resp.status_code == 200
    listed_ids = [p["node_id"] for p in peers_resp.json()["peers"]]
    assert "peer-alpha" in listed_ids

    peer2 = {
        "node_id": "peer-beta",
        "host": "10.0.0.11",
        "port": 8200,
        "version": "0.1.0",
    }
    client.post("/api/hello", json=peer2)

    invalidate_response_cache()
    peers_resp2 = client.get("/api/hello/peers")
    assert peers_resp2.status_code == 200
    listed_ids2 = [p["node_id"] for p in peers_resp2.json()["peers"]]
    assert "peer-alpha" in listed_ids2
    assert "peer-beta" in listed_ids2


def test_job_multi_task_pipeline(client):
    job_resp = client.post("/api/job", json={
        "name": "pipeline-job",
        "tasks": {
            "task_a": {"type": "cmd", "command": ["echo", "A"]},
            "task_b": {"type": "cmd", "command": ["echo", "B"], "depends_on": ["task_a"]},
            "task_c": {"type": "python", "code": "print('C')", "depends_on": ["task_b"]},
        },
    })
    assert job_resp.status_code == 200
    job = job_resp.json()["job"]
    job_id = job["job_id"]
    assert len(job["task_keys"]) == 3

    run_resp = client.post(f"/api/job/{job_id}/run")
    assert run_resp.status_code == 200
    run = run_resp.json()["run"]
    assert run["status"] == "completed"
    assert run["task_results"]["task_a"]["status"] == "completed"
    assert run["task_results"]["task_b"]["status"] == "completed"
    assert run["task_results"]["task_c"]["status"] == "completed"
    assert "A" in run["task_results"]["task_a"]["stdout"]
    assert "B" in run["task_results"]["task_b"]["stdout"]
    assert "C" in run["task_results"]["task_c"]["stdout"]


def test_cmd_and_python_exec_isolation(client):
    cmd_resp = client.post("/api/cmd", json={"command": ["echo", "cmd_marker"]})
    assert cmd_resp.status_code == 200
    cmd_id = cmd_resp.json()["id"]

    py_resp = client.post("/api/python", json={"code": "print('py_marker')"})
    assert py_resp.status_code == 200
    py_id = py_resp.json()["id"]

    cmd_list = client.get("/api/cmd")
    assert cmd_list.status_code == 200
    cmd_ids = [item["id"] for item in cmd_list.json()["items"]]
    assert cmd_id in cmd_ids
    assert py_id not in cmd_ids

    py_list = client.get("/api/python")
    assert py_list.status_code == 200
    py_ids = [item["id"] for item in py_list.json()["items"]]
    assert py_id in py_ids
    assert cmd_id not in py_ids


def test_function_clone_and_run_clone(client):
    original = _mk_func(client, name="clone_src", code="print('original')")
    client.post(f"/api/function/{original['id']}/run")

    clone_resp = client.post(
        f"/api/function/{original['id']}/clone",
        json={"name": "clone_dst"},
    )
    assert clone_resp.status_code == 200
    clone = clone_resp.json()["function"]
    assert clone["id"] != original["id"]
    assert clone["name"] == "clone_dst"
    assert clone["code"] == original["code"]

    clone_run = client.post(f"/api/function/{clone['id']}/run")
    assert clone_run.status_code == 200
    assert clone_run.json()["run"]["function_id"] == clone["id"]

    orig_runs = client.get(f"/api/function/{original['id']}/run")
    clone_runs = client.get(f"/api/function/{clone['id']}/run")
    assert orig_runs.status_code == 200
    assert clone_runs.status_code == 200

    orig_run_ids = {r["id"] for r in orig_runs.json()["runs"]}
    clone_run_ids = {r["id"] for r in clone_runs.json()["runs"]}
    assert orig_run_ids.isdisjoint(clone_run_ids)


def test_full_crud_lifecycle(client):
    create_resp = client.post("/api/function", json={
        "name": "crud_fn",
        "code": "print('created')",
        "description": "initial",
    })
    assert create_resp.status_code == 200
    fn = create_resp.json()["function"]
    fn_id = fn["id"]

    list_resp = client.get("/api/function")
    assert list_resp.status_code == 200
    listed_ids = [f["id"] for f in list_resp.json()["functions"]]
    assert fn_id in listed_ids

    get_resp = client.get(f"/api/function/{fn_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["function"]["name"] == "crud_fn"
    assert get_resp.json()["function"]["code"] == "print('created')"

    update_resp = client.put(f"/api/function/{fn_id}", json={
        "code": "print('updated')",
        "description": "modified",
    })
    assert update_resp.status_code == 200
    updated = update_resp.json()["function"]
    assert updated["code"] == "print('updated')"
    assert updated["description"] == "modified"
    assert updated["name"] == "crud_fn"

    get_resp2 = client.get(f"/api/function/{fn_id}")
    assert get_resp2.status_code == 200
    assert get_resp2.json()["function"]["code"] == "print('updated')"
    assert get_resp2.json()["function"]["description"] == "modified"

    del_resp = client.delete(f"/api/function/{fn_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["function"]["id"] == fn_id

    gone_resp = client.get(f"/api/function/{fn_id}")
    assert gone_resp.status_code == 404

    list_resp2 = client.get("/api/function")
    assert list_resp2.status_code == 200
    listed_ids2 = [f["id"] for f in list_resp2.json()["functions"]]
    assert fn_id not in listed_ids2


def test_monitor_returns_valid_snapshot(client):
    resp = client.get("/api/monitor")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"

    snap = data["snapshot"]
    assert snap["cpu_percent"] >= 0
    assert snap["memory_percent"] >= 0
    assert snap["memory_used_mb"] >= 0
    assert snap["memory_total_mb"] > 0
    assert snap["timestamp"] != ""

    assert isinstance(data["history"], list)
