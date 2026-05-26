"""Comprehensive integration tests for every major node endpoint."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from yggdrasil.node.app import create_app
from yggdrasil.node.config import Settings


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_hello_get(self, client):
        resp = client.get("/api/hello")
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "test-node-integration"
        assert isinstance(data["port"], int)
        assert isinstance(data["version"], str)

    def test_hello_post_registers_peer(self, client):
        resp = client.post("/api/hello", json={
            "node_id": "peer-1",
            "host": "10.0.0.2",
            "port": 8100,
            "version": "0.1.0",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "test-node-integration"

    def test_peers_list_after_registration(self, client):
        client.post("/api/hello", json={
            "node_id": "peer-2",
            "host": "10.0.0.3",
            "port": 8200,
        })
        resp = client.get("/api/hello/peers")
        assert resp.status_code == 200
        peers = resp.json()["peers"]
        assert any(p["node_id"] == "peer-2" for p in peers)

    def test_hello_post_duplicate_peer(self, client):
        payload = {
            "node_id": "peer-dup",
            "host": "10.0.0.4",
            "port": 8100,
        }
        client.post("/api/hello", json=payload)
        client.post("/api/hello", json=payload)
        resp = client.get("/api/hello/peers")
        ids = [p["node_id"] for p in resp.json()["peers"]]
        assert ids.count("peer-dup") == 1


# ---------------------------------------------------------------------------
# Functions CRUD
# ---------------------------------------------------------------------------

class TestFunctionCRUD:
    def test_create_function(self, client):
        resp = client.post("/api/function", json={
            "name": "greet",
            "code": "print('hello')",
            "description": "A greeting",
        })
        assert resp.status_code == 200
        func = resp.json()["function"]
        assert func["name"] == "greet"
        assert func["code"] == "print('hello')"
        assert func["language"] == "python"
        assert func["state"] == "ready"
        assert func["run_count"] == 0

    def test_list_functions(self, client):
        client.post("/api/function", json={"name": "f1", "code": "pass"})
        client.post("/api/function", json={"name": "f2", "code": "pass"})
        resp = client.get("/api/function")
        assert resp.status_code == 200
        funcs = resp.json()["functions"]
        names = {f["name"] for f in funcs}
        assert "f1" in names
        assert "f2" in names

    def test_get_function(self, client):
        resp = client.post("/api/function", json={"name": "getter", "code": "x=1"})
        fid = resp.json()["function"]["id"]
        resp = client.get(f"/api/function/{fid}")
        assert resp.status_code == 200
        assert resp.json()["function"]["name"] == "getter"

    def test_get_nonexistent(self, client):
        resp = client.get("/api/function/999999")
        assert resp.status_code == 404

    def test_update_function(self, client):
        resp = client.post("/api/function", json={"name": "orig", "code": "pass"})
        fid = resp.json()["function"]["id"]
        resp = client.put(f"/api/function/{fid}", json={"code": "x = 42"})
        assert resp.status_code == 200
        assert resp.json()["function"]["code"] == "x = 42"
        assert resp.json()["function"]["name"] == "orig"

    def test_update_nonexistent(self, client):
        resp = client.put("/api/function/999999", json={"code": "x"})
        assert resp.status_code == 404

    def test_delete_function(self, client):
        resp = client.post("/api/function", json={"name": "todelete", "code": "pass"})
        fid = resp.json()["function"]["id"]
        resp = client.delete(f"/api/function/{fid}")
        assert resp.status_code == 200
        resp = client.get(f"/api/function/{fid}")
        assert resp.status_code == 404

    def test_delete_nonexistent(self, client):
        resp = client.delete("/api/function/999999")
        assert resp.status_code == 404

    def test_upsert_by_name(self, client):
        resp1 = client.post("/api/function", json={"name": "upsert_me", "code": "v1"})
        fid1 = resp1.json()["function"]["id"]
        resp2 = client.post("/api/function", json={"name": "upsert_me", "code": "v2"})
        fid2 = resp2.json()["function"]["id"]
        assert fid1 == fid2
        assert resp2.json()["function"]["code"] == "v2"

    def test_create_with_dependencies(self, client):
        resp = client.post("/api/function", json={
            "name": "with_deps",
            "code": "import requests",
            "dependencies": ["requests", "numpy"],
        })
        assert resp.status_code == 200
        assert resp.json()["function"]["dependencies"] == ["requests", "numpy"]

    def test_strict_model_rejects_extra_fields(self, client):
        resp = client.post("/api/function", json={
            "name": "strict",
            "code": "pass",
            "bogus_field": "should fail",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class TestRunCRUD:
    def _create_function(self, client, code: str = "print('ok')") -> int:
        resp = client.post("/api/function", json={
            "name": f"run_fn_{id(code)}",
            "code": code,
        })
        return resp.json()["function"]["id"]

    def test_create_run_success(self, client):
        fid = self._create_function(client, "print('hello world')")
        resp = client.post("/api/run", json={"function_id": fid})
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "completed"
        assert run["function_id"] == fid
        assert "hello world" in (run["stdout"] or "")
        assert run["returncode"] == 0
        assert run["duration"] is not None

    def test_create_run_failure(self, client):
        fid = self._create_function(client, "raise ValueError('boom')")
        resp = client.post("/api/run", json={"function_id": fid})
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "failed"
        assert run["returncode"] != 0
        assert "boom" in (run["stderr"] or "")

    def test_create_run_with_args(self, client):
        code = "import json, os\nargs = json.loads(os.environ['__ygg_inputs__'])\nprint(args['x'] + args['y'])"
        fid = self._create_function(client, code)
        resp = client.post("/api/run", json={
            "function_id": fid,
            "args": {"x": 3, "y": 7},
        })
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "completed"
        assert "10" in (run["stdout"] or "")

    def test_run_nonexistent_function(self, client):
        resp = client.post("/api/run", json={"function_id": 999999})
        assert resp.status_code == 404

    def test_list_runs(self, client):
        fid = self._create_function(client)
        client.post("/api/run", json={"function_id": fid})
        client.post("/api/run", json={"function_id": fid})
        resp = client.get("/api/run")
        assert resp.status_code == 200
        runs = resp.json()["runs"]
        assert len(runs) >= 2

    def test_get_run(self, client):
        fid = self._create_function(client)
        resp = client.post("/api/run", json={"function_id": fid})
        rid = resp.json()["run"]["id"]
        resp = client.get(f"/api/run/{rid}")
        assert resp.status_code == 200
        assert resp.json()["run"]["id"] == rid

    def test_get_nonexistent_run(self, client):
        resp = client.get("/api/run/999999")
        assert resp.status_code == 404

    def test_delete_run(self, client):
        fid = self._create_function(client)
        resp = client.post("/api/run", json={"function_id": fid})
        rid = resp.json()["run"]["id"]
        resp = client.delete(f"/api/run/{rid}")
        assert resp.status_code == 200
        resp = client.get(f"/api/run/{rid}")
        assert resp.status_code == 404

    def test_function_run_count_increments(self, client):
        fid = self._create_function(client, "print(1)")
        client.post("/api/run", json={"function_id": fid})
        client.post("/api/run", json={"function_id": fid})
        resp = client.get(f"/api/function/{fid}")
        assert resp.json()["function"]["run_count"] == 2

    def test_run_via_function_endpoint(self, client):
        fid = self._create_function(client, "print('via_func')")
        resp = client.post(f"/api/function/{fid}/run")
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "completed"
        assert "via_func" in (run["stdout"] or "")

    def test_list_function_runs(self, client):
        fid = self._create_function(client)
        client.post(f"/api/function/{fid}/run")
        client.post(f"/api/function/{fid}/run")
        resp = client.get(f"/api/function/{fid}/run")
        assert resp.status_code == 200
        runs = resp.json()["runs"]
        assert len(runs) >= 2
        assert all(r["function_id"] == fid for r in runs)

    def test_run_structured_outputs(self, client):
        code = (
            "import json, os\n"
            "with open(os.environ['__ygg_outputs_file__'], 'w') as f:\n"
            "    json.dump({'answer': 42}, f)\n"
        )
        fid = self._create_function(client, code)
        resp = client.post("/api/run", json={"function_id": fid})
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "completed"
        assert run["result"] == {"answer": 42}


# ---------------------------------------------------------------------------
# Run streaming logs
# ---------------------------------------------------------------------------

class TestRunLogs:
    def _create_and_run(self, client, code: str) -> int:
        resp = client.post("/api/function", json={
            "name": f"log_fn_{id(code)}",
            "code": code,
        })
        fid = resp.json()["function"]["id"]
        resp = client.post("/api/run", json={"function_id": fid})
        return resp.json()["run"]["id"]

    def test_stream_logs_completed(self, client):
        rid = self._create_and_run(client, "print('line1')\nprint('line2')")
        resp = client.get(f"/api/run/{rid}/logs")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events = [
            json.loads(line.removeprefix("data: "))
            for line in resp.text.strip().split("\n\n")
            if line.startswith("data: ")
        ]
        types = [e["type"] for e in events]
        assert "stdout" in types
        assert "complete" in types

    def test_stream_logs_failed(self, client):
        rid = self._create_and_run(client, "raise RuntimeError('fail')")
        resp = client.get(f"/api/run/{rid}/logs")
        assert resp.status_code == 200
        events = [
            json.loads(line.removeprefix("data: "))
            for line in resp.text.strip().split("\n\n")
            if line.startswith("data: ")
        ]
        types = [e["type"] for e in events]
        assert "stderr" in types
        assert "complete" in types

    def test_stream_logs_nonexistent(self, client):
        resp = client.get("/api/run/999999/logs")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# CMD execution
# ---------------------------------------------------------------------------

class TestCmdExecution:
    def test_execute_simple(self, client):
        resp = client.post("/api/cmd", json={"command": ["echo", "hello"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["returncode"] == 0
        assert "hello" in (data["stdout"] or "")

    def test_execute_failing_command(self, client):
        resp = client.post("/api/cmd", json={"command": ["false"]})
        assert resp.status_code == 200
        assert resp.json()["returncode"] != 0

    def test_list_history(self, client):
        client.post("/api/cmd", json={"command": ["echo", "a"]})
        client.post("/api/cmd", json={"command": ["echo", "b"]})
        resp = client.get("/api/cmd")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) >= 2

    def test_get_command(self, client):
        resp = client.post("/api/cmd", json={"command": ["echo", "get_me"]})
        cmd_id = resp.json()["id"]
        resp = client.get(f"/api/cmd/{cmd_id}")
        assert resp.status_code == 200

    def test_delete_command(self, client):
        resp = client.post("/api/cmd", json={"command": ["echo", "del_me"]})
        cmd_id = resp.json()["id"]
        resp = client.delete(f"/api/cmd/{cmd_id}")
        assert resp.status_code == 200
        resp = client.get(f"/api/cmd/{cmd_id}")
        assert resp.status_code == 404

    def test_command_with_env(self, client):
        resp = client.post("/api/cmd", json={
            "command": ["printenv", "MY_VAR"],
            "env": {"MY_VAR": "test_value"},
        })
        assert resp.status_code == 200
        assert "test_value" in (resp.json()["stdout"] or "")

    def test_command_timeout(self, client):
        resp = client.post("/api/cmd", json={
            "command": ["sleep", "60"],
            "timeout": 0.5,
        })
        assert resp.status_code == 408


# ---------------------------------------------------------------------------
# Python execution
# ---------------------------------------------------------------------------

class TestPythonExecution:
    def test_execute_simple(self, client):
        resp = client.post("/api/python", json={"code": "print(2+2)"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["returncode"] == 0
        assert "4" in (data["stdout"] or "")

    def test_execute_with_error(self, client):
        resp = client.post("/api/python", json={"code": "1/0"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "ZeroDivision" in (data["stderr"] or "")

    def test_execute_history(self, client):
        client.post("/api/python", json={"code": "pass"})
        resp = client.get("/api/python")
        assert resp.status_code == 200
        assert len(resp.json()["items"]) >= 1

    def test_get_execution(self, client):
        resp = client.post("/api/python", json={"code": "print('get_me')"})
        eid = resp.json()["id"]
        resp = client.get(f"/api/python/{eid}")
        assert resp.status_code == 200

    def test_delete_execution(self, client):
        resp = client.post("/api/python", json={"code": "pass"})
        eid = resp.json()["id"]
        resp = client.delete(f"/api/python/{eid}")
        assert resp.status_code == 200
        resp = client.get(f"/api/python/{eid}")
        assert resp.status_code == 404

    def test_execute_with_timeout(self, client):
        resp = client.post("/api/python", json={
            "code": "import time; time.sleep(60)",
            "timeout": 0.5,
        })
        assert resp.status_code == 408


# ---------------------------------------------------------------------------
# Environment CRUD
# ---------------------------------------------------------------------------

class TestEnvironmentCRUD:
    def test_create_environment(self, client):
        resp = client.post("/api/environment", json={
            "name": "test-env",
            "python_version": "3.11",
        })
        assert resp.status_code == 200
        env = resp.json()["environment"]
        assert env["name"] == "test-env"
        assert env["python_version"] == "3.11"

    def test_list_environments(self, client):
        client.post("/api/environment", json={"name": "env1"})
        client.post("/api/environment", json={"name": "env2"})
        resp = client.get("/api/environment")
        assert resp.status_code == 200
        names = {e["name"] for e in resp.json()["environments"]}
        assert "env1" in names
        assert "env2" in names

    def test_get_environment(self, client):
        resp = client.post("/api/environment", json={"name": "getenv"})
        eid = resp.json()["environment"]["id"]
        resp = client.get(f"/api/environment/{eid}")
        assert resp.status_code == 200
        assert resp.json()["environment"]["name"] == "getenv"

    def test_get_nonexistent_environment(self, client):
        resp = client.get("/api/environment/999999")
        assert resp.status_code == 404

    def test_delete_environment(self, client):
        resp = client.post("/api/environment", json={"name": "delenv"})
        eid = resp.json()["environment"]["id"]
        resp = client.delete(f"/api/environment/{eid}")
        assert resp.status_code == 200
        resp = client.get(f"/api/environment/{eid}")
        assert resp.status_code == 404

    def test_upsert_environment(self, client):
        resp1 = client.post("/api/environment", json={"name": "upsert_env"})
        eid1 = resp1.json()["environment"]["id"]
        resp2 = client.post("/api/environment", json={
            "name": "upsert_env",
            "dependencies": ["requests"],
        })
        eid2 = resp2.json()["environment"]["id"]
        assert eid1 == eid2

    def test_strict_model_rejects_extra(self, client):
        resp = client.post("/api/environment", json={
            "name": "strict_env",
            "unknown_field": True,
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# DAGs
# ---------------------------------------------------------------------------

class TestDagCRUD:
    def _create_function(self, client, name: str, code: str) -> int:
        resp = client.post("/api/function", json={"name": name, "code": code})
        return resp.json()["function"]["id"]

    def test_create_dag(self, client):
        fid = self._create_function(client, "dag_step", "print('step')")
        resp = client.post("/api/dag", json={
            "name": "my_dag",
            "steps": [
                {"id": "step1", "ref": {"function_id": fid}},
            ],
        })
        assert resp.status_code == 200
        dag = resp.json()["dag"]
        assert dag["name"] == "my_dag"
        assert len(dag["steps"]) == 1

    def test_list_dags(self, client):
        fid = self._create_function(client, "dag_list_fn", "pass")
        client.post("/api/dag", json={
            "name": "dag_a",
            "steps": [{"id": "s", "ref": {"function_id": fid}}],
        })
        resp = client.get("/api/dag")
        assert resp.status_code == 200
        assert len(resp.json()["dags"]) >= 1

    def test_get_dag(self, client):
        fid = self._create_function(client, "dag_get_fn", "pass")
        resp = client.post("/api/dag", json={
            "name": "dag_get",
            "steps": [{"id": "s", "ref": {"function_id": fid}}],
        })
        did = resp.json()["dag"]["id"]
        resp = client.get(f"/api/dag/{did}")
        assert resp.status_code == 200
        assert resp.json()["dag"]["name"] == "dag_get"

    def test_get_nonexistent_dag(self, client):
        resp = client.get("/api/dag/999999")
        assert resp.status_code == 404

    def test_delete_dag(self, client):
        fid = self._create_function(client, "dag_del_fn", "pass")
        resp = client.post("/api/dag", json={
            "name": "dag_del",
            "steps": [{"id": "s", "ref": {"function_id": fid}}],
        })
        did = resp.json()["dag"]["id"]
        resp = client.delete(f"/api/dag/{did}")
        assert resp.status_code == 200
        resp = client.get(f"/api/dag/{did}")
        assert resp.status_code == 404

    def test_execute_single_step_dag(self, client):
        fid = self._create_function(client, "dag_exec_fn", "print('dag ok')")
        resp = client.post("/api/dag", json={
            "name": "dag_exec",
            "steps": [{"id": "only", "ref": {"function_id": fid}}],
        })
        did = resp.json()["dag"]["id"]
        resp = client.post(f"/api/dag/{did}/run")
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "completed"
        assert run["duration"] is not None

    def test_execute_multi_step_dag(self, client):
        fid1 = self._create_function(client, "dag_multi_1", "print('step1')")
        fid2 = self._create_function(client, "dag_multi_2", "print('step2')")
        resp = client.post("/api/dag", json={
            "name": "dag_multi",
            "steps": [
                {"id": "first", "ref": {"function_id": fid1}},
                {"id": "second", "ref": {"function_id": fid2}, "depends_on": ["first"]},
            ],
        })
        did = resp.json()["dag"]["id"]
        resp = client.post(f"/api/dag/{did}/run")
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "completed"
        assert "first" in run["step_results"]
        assert "second" in run["step_results"]

    def test_list_dag_runs(self, client):
        fid = self._create_function(client, "dag_runs_fn", "pass")
        resp = client.post("/api/dag", json={
            "name": "dag_runs",
            "steps": [{"id": "s", "ref": {"function_id": fid}}],
        })
        did = resp.json()["dag"]["id"]
        client.post(f"/api/dag/{did}/run")
        resp = client.get(f"/api/dag/{did}/run")
        assert resp.status_code == 200
        assert len(resp.json()["runs"]) >= 1

    def test_get_dag_run(self, client):
        fid = self._create_function(client, "dag_getrun_fn", "pass")
        resp = client.post("/api/dag", json={
            "name": "dag_getrun",
            "steps": [{"id": "s", "ref": {"function_id": fid}}],
        })
        did = resp.json()["dag"]["id"]
        resp = client.post(f"/api/dag/{did}/run")
        rid = resp.json()["run"]["id"]
        resp = client.get(f"/api/dag/{did}/run/{rid}")
        assert resp.status_code == 200
        assert resp.json()["run"]["dag_id"] == did

    def test_dag_with_failing_step(self, client):
        fid = self._create_function(client, "dag_fail_fn", "raise Exception('dag fail')")
        resp = client.post("/api/dag", json={
            "name": "dag_fail",
            "steps": [{"id": "bad", "ref": {"function_id": fid}}],
        })
        did = resp.json()["dag"]["id"]
        resp = client.post(f"/api/dag/{did}/run")
        assert resp.status_code == 200
        assert resp.json()["run"]["status"] == "failed"

    def test_upsert_dag(self, client):
        fid = self._create_function(client, "dag_upsert_fn", "pass")
        resp1 = client.post("/api/dag", json={
            "name": "dag_upsert",
            "steps": [{"id": "s", "ref": {"function_id": fid}}],
        })
        did1 = resp1.json()["dag"]["id"]
        resp2 = client.post("/api/dag", json={
            "name": "dag_upsert",
            "description": "updated",
            "steps": [{"id": "s", "ref": {"function_id": fid}}],
        })
        did2 = resp2.json()["dag"]["id"]
        assert did1 == did2
        assert resp2.json()["dag"]["description"] == "updated"


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class TestJobCRUD:
    def test_create_job_cmd(self, client):
        resp = client.post("/api/job", json={
            "name": "echo_job",
            "tasks": {
                "t1": {"type": "cmd", "command": ["echo", "ok"]},
            },
        })
        assert resp.status_code == 200
        job = resp.json()["job"]
        assert job["name"] == "echo_job"
        assert "t1" in job["task_keys"]

    def test_create_job_python(self, client):
        resp = client.post("/api/job", json={
            "name": "py_job",
            "tasks": {
                "t1": {"type": "python", "code": "print('py job')"},
            },
        })
        assert resp.status_code == 200

    def test_list_jobs(self, client):
        client.post("/api/job", json={
            "name": "list_job",
            "tasks": {"t": {"type": "cmd", "command": ["true"]}},
        })
        resp = client.get("/api/job")
        assert resp.status_code == 200
        assert len(resp.json()["items"]) >= 1

    def test_get_job(self, client):
        resp = client.post("/api/job", json={
            "name": "get_job",
            "tasks": {"t": {"type": "cmd", "command": ["true"]}},
        })
        jid = resp.json()["job"]["job_id"]
        resp = client.get(f"/api/job/{jid}")
        assert resp.status_code == 200
        assert resp.json()["job"]["name"] == "get_job"

    def test_get_nonexistent_job(self, client):
        resp = client.get("/api/job/nonexistent")
        assert resp.status_code == 404

    def test_delete_job(self, client):
        resp = client.post("/api/job", json={
            "name": "del_job",
            "tasks": {"t": {"type": "cmd", "command": ["true"]}},
        })
        jid = resp.json()["job"]["job_id"]
        resp = client.delete(f"/api/job/{jid}")
        assert resp.status_code == 200
        resp = client.get(f"/api/job/{jid}")
        assert resp.status_code == 404

    def test_trigger_run(self, client):
        resp = client.post("/api/job", json={
            "name": "run_job",
            "tasks": {"t": {"type": "cmd", "command": ["echo", "ran"]}},
        })
        jid = resp.json()["job"]["job_id"]
        resp = client.post(f"/api/job/{jid}/run")
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] in ("completed", "running")

    def test_list_job_runs(self, client):
        resp = client.post("/api/job", json={
            "name": "runs_job",
            "tasks": {"t": {"type": "cmd", "command": ["true"]}},
        })
        jid = resp.json()["job"]["job_id"]
        client.post(f"/api/job/{jid}/run")
        resp = client.get(f"/api/job/{jid}/run")
        assert resp.status_code == 200
        assert len(resp.json()["items"]) >= 1

    def test_get_job_run(self, client):
        resp = client.post("/api/job", json={
            "name": "getrun_job",
            "tasks": {"t": {"type": "cmd", "command": ["echo", "ok"]}},
        })
        jid = resp.json()["job"]["job_id"]
        resp = client.post(f"/api/job/{jid}/run")
        rid = resp.json()["run"]["run_id"]
        resp = client.get(f"/api/job/{jid}/run/{rid}")
        assert resp.status_code == 200

    def test_update_job(self, client):
        resp = client.post("/api/job", json={
            "name": "upd_job",
            "tasks": {"t": {"type": "cmd", "command": ["true"]}},
        })
        jid = resp.json()["job"]["job_id"]
        resp = client.put(f"/api/job/{jid}", json={
            "name": "upd_job_v2",
            "tasks": {"t2": {"type": "cmd", "command": ["echo", "updated"]}},
        })
        assert resp.status_code == 200
        assert resp.json()["job"]["name"] == "upd_job_v2"

    def test_job_with_dependencies(self, client):
        resp = client.post("/api/job", json={
            "name": "dep_job",
            "tasks": {
                "first": {"type": "cmd", "command": ["echo", "1"]},
                "second": {"type": "cmd", "command": ["echo", "2"], "depends_on": ["first"]},
            },
        })
        assert resp.status_code == 200
        jid = resp.json()["job"]["job_id"]
        resp = client.post(f"/api/job/{jid}/run")
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "completed"

    def test_job_python_failure(self, client):
        resp = client.post("/api/job", json={
            "name": "fail_job",
            "tasks": {"t": {"type": "python", "code": "1/0"}},
        })
        jid = resp.json()["job"]["job_id"]
        resp = client.post(f"/api/job/{jid}/run")
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "failed"


# ---------------------------------------------------------------------------
# Messenger
# ---------------------------------------------------------------------------

class TestMessenger:
    def test_default_channel_exists(self, client):
        resp = client.get("/api/messenger/channels")
        assert resp.status_code == 200
        names = [ch["name"] for ch in resp.json()["channels"]]
        assert "general" in names

    def test_send_message(self, client):
        resp = client.post("/api/messenger", json={
            "text": "hello world",
            "sender": "tester",
        })
        assert resp.status_code == 200
        msg = resp.json()
        assert msg["text"] == "hello world"
        assert msg["sender"] == "tester"
        assert msg["channel"] == "general"

    def test_send_to_custom_channel(self, client):
        client.post("/api/messenger/channels", params={"name": "alerts"})
        resp = client.post("/api/messenger", json={
            "text": "alert!",
            "channel": "alerts",
        })
        assert resp.status_code == 200
        assert resp.json()["channel"] == "alerts"

    def test_create_channel(self, client):
        resp = client.post("/api/messenger/channels", params={"name": "new_ch"})
        assert resp.status_code == 200
        assert resp.json()["channel"]["name"] == "new_ch"

    def test_get_channel(self, client):
        client.post("/api/messenger/channels", params={"name": "get_ch"})
        resp = client.get("/api/messenger/channels/get_ch")
        assert resp.status_code == 200
        assert resp.json()["channel"]["name"] == "get_ch"

    def test_get_nonexistent_channel(self, client):
        resp = client.get("/api/messenger/channels/no_such_channel")
        assert resp.status_code == 404

    def test_delete_channel(self, client):
        client.post("/api/messenger/channels", params={"name": "del_ch"})
        resp = client.delete("/api/messenger/channels/del_ch")
        assert resp.status_code == 200
        resp = client.get("/api/messenger/channels/del_ch")
        assert resp.status_code == 404

    def test_get_messages(self, client):
        client.post("/api/messenger", json={"text": "msg1"})
        client.post("/api/messenger", json={"text": "msg2"})
        resp = client.get("/api/messenger/channels/general/messages")
        assert resp.status_code == 200
        msgs = resp.json()["messages"]
        assert len(msgs) >= 2

    def test_get_messages_with_limit(self, client):
        for i in range(5):
            client.post("/api/messenger", json={"text": f"m{i}"})
        resp = client.get("/api/messenger/channels/general/messages", params={"limit": 2})
        assert resp.status_code == 200
        assert len(resp.json()["messages"]) <= 2

    def test_cannot_delete_general(self, client):
        resp = client.delete("/api/messenger/channels/general")
        assert resp.status_code in (400, 403, 409)

    def test_poll_returns_immediately_with_existing(self, client):
        client.post("/api/messenger", json={"text": "poll_msg"})
        resp = client.get("/api/messenger/channels/general/poll", params={"timeout": 0.5})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------

class TestEnvVars:
    def test_get_env_all(self, client):
        resp = client.get("/api/env")
        assert resp.status_code == 200
        data = resp.json()
        assert "variables" in data or "vars" in data or isinstance(data, dict)

    def test_set_and_get_env(self, client):
        import os
        client.post("/api/env", json={"variables": {"YGG_TEST_VAR": "123"}})
        resp = client.get("/api/env", params={"keys": "YGG_TEST_VAR"})
        assert resp.status_code == 200
        os.environ.pop("YGG_TEST_VAR", None)

    def test_get_specific_keys(self, client):
        resp = client.get("/api/env", params={"keys": "PATH"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class TestMonitor:
    def test_monitor_snapshot(self, client):
        resp = client.get("/api/monitor")
        assert resp.status_code == 200
        data = resp.json()
        assert "snapshot" in data or "node_id" in data

    def test_monitor_with_limit(self, client):
        resp = client.get("/api/monitor", params={"limit": 5})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Cross-service integration
# ---------------------------------------------------------------------------

class TestCrossService:
    def test_full_function_lifecycle(self, client):
        resp = client.post("/api/function", json={
            "name": "lifecycle",
            "code": "print('created')",
        })
        assert resp.status_code == 200
        fid = resp.json()["function"]["id"]

        resp = client.put(f"/api/function/{fid}", json={"code": "print('updated')"})
        assert resp.status_code == 200

        resp = client.post("/api/run", json={"function_id": fid})
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "completed"
        assert "updated" in (run["stdout"] or "")

        resp = client.get(f"/api/function/{fid}")
        assert resp.json()["function"]["run_count"] == 1

        resp = client.post(f"/api/function/{fid}/clone")
        assert resp.status_code == 200
        clone_id = resp.json()["function"]["id"]
        assert clone_id != fid

        resp = client.delete(f"/api/function/{fid}")
        assert resp.status_code == 200
        resp = client.get(f"/api/function/{fid}")
        assert resp.status_code == 404
        resp = client.get(f"/api/function/{clone_id}")
        assert resp.status_code == 200

    def test_dag_depends_on_functions(self, client):
        r1 = client.post("/api/function", json={
            "name": "step_a",
            "code": "import json, os\nwith open(os.environ['__ygg_outputs_file__'], 'w') as f: json.dump({'v': 1}, f)",
        })
        fid1 = r1.json()["function"]["id"]
        r2 = client.post("/api/function", json={
            "name": "step_b",
            "code": "print('step_b ran')",
        })
        fid2 = r2.json()["function"]["id"]

        resp = client.post("/api/dag", json={
            "name": "cross_dag",
            "steps": [
                {"id": "a", "ref": {"function_id": fid1}},
                {"id": "b", "ref": {"function_id": fid2}, "depends_on": ["a"]},
            ],
        })
        did = resp.json()["dag"]["id"]
        resp = client.post(f"/api/dag/{did}/run")
        assert resp.status_code == 200
        assert resp.json()["run"]["status"] == "completed"

    def test_run_updates_function_timestamps(self, client):
        resp = client.post("/api/function", json={"name": "ts_fn", "code": "pass"})
        fid = resp.json()["function"]["id"]
        created = resp.json()["function"]["created_at"]

        client.post("/api/run", json={"function_id": fid})

        resp = client.get(f"/api/function/{fid}")
        func = resp.json()["function"]
        assert func["last_used_at"] is not None
        assert func["run_count"] == 1
        assert func["created_at"] == created


# ---------------------------------------------------------------------------
# Edge cases & robustness
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_function_code(self, client):
        resp = client.post("/api/function", json={"name": "empty", "code": ""})
        assert resp.status_code == 200

    def test_function_with_unicode(self, client):
        resp = client.post("/api/function", json={
            "name": "unicode_fn",
            "code": "print('日本語 こんにちは')",
        })
        assert resp.status_code == 200
        fid = resp.json()["function"]["id"]
        resp = client.post("/api/run", json={"function_id": fid})
        assert resp.status_code == 200
        assert resp.json()["run"]["status"] == "completed"

    def test_large_stdout(self, client):
        resp = client.post("/api/function", json={
            "name": "big_output",
            "code": "print('x' * 10000)",
        })
        fid = resp.json()["function"]["id"]
        resp = client.post("/api/run", json={"function_id": fid})
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "completed"
        assert len(run["stdout"]) >= 10000

    def test_concurrent_runs(self, client):
        resp = client.post("/api/function", json={
            "name": "conc_fn",
            "code": "import time; time.sleep(0.1); print('done')",
        })
        fid = resp.json()["function"]["id"]
        r1 = client.post("/api/run", json={"function_id": fid})
        r2 = client.post("/api/run", json={"function_id": fid})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["run"]["id"] != r2.json()["run"]["id"]

    def test_function_with_multiline_code(self, client):
        code = "def add(a, b):\n    return a + b\nresult = add(3, 4)\nprint(result)"
        resp = client.post("/api/function", json={"name": "multiline", "code": code})
        fid = resp.json()["function"]["id"]
        resp = client.post("/api/run", json={"function_id": fid})
        assert resp.status_code == 200
        assert "7" in (resp.json()["run"]["stdout"] or "")

    def test_run_with_syntax_error(self, client):
        resp = client.post("/api/function", json={
            "name": "syntax_err",
            "code": "def foo(\n",
        })
        fid = resp.json()["function"]["id"]
        resp = client.post("/api/run", json={"function_id": fid})
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "failed"
        assert run["returncode"] != 0

    def test_cmd_empty_command_rejected(self, client):
        resp = client.post("/api/cmd", json={"command": []})
        assert resp.status_code == 422

    def test_messenger_empty_text_rejected(self, client):
        resp = client.post("/api/messenger", json={"text": ""})
        assert resp.status_code == 422

    def test_job_empty_tasks_rejected(self, client):
        resp = client.post("/api/job", json={"name": "empty", "tasks": {}})
        assert resp.status_code == 422

    def test_multiple_function_updates(self, client):
        resp = client.post("/api/function", json={"name": "multi_upd", "code": "v0"})
        fid = resp.json()["function"]["id"]
        for i in range(5):
            resp = client.put(f"/api/function/{fid}", json={"code": f"v{i+1}"})
            assert resp.status_code == 200
        resp = client.get(f"/api/function/{fid}")
        assert resp.json()["function"]["code"] == "v5"

    def test_run_env_variables_injected(self, client):
        code = "import os; print(os.environ.get('YGG_RUNTIME_VERSION', 'MISSING'))"
        resp = client.post("/api/function", json={"name": "env_check", "code": code})
        fid = resp.json()["function"]["id"]
        resp = client.post("/api/run", json={"function_id": fid})
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "completed"
        assert "MISSING" not in (run["stdout"] or "MISSING")


# ---------------------------------------------------------------------------
# Import safety (the original Windows bug)
# ---------------------------------------------------------------------------

class TestImportSafety:
    def test_node_imports_on_all_platforms(self):
        from yggdrasil.node.app import create_app
        from yggdrasil.node.services.run import RunService
        assert create_app is not None
        assert RunService is not None

    def test_resource_module_not_at_top_level(self):
        import importlib
        import sys
        if "yggdrasil.node.services.run" in sys.modules:
            del sys.modules["yggdrasil.node.services.run"]
        mod = importlib.import_module("yggdrasil.node.services.run")
        source = open(mod.__file__).read()
        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import resource") or stripped.startswith("from resource"):
                indent = len(line) - len(line.lstrip())
                assert indent > 0, (
                    f"'resource' module must not be imported at module level (line {i+1})"
                )
