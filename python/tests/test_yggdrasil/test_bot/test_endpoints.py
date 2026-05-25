from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from yggdrasil.node.app import create_app
from yggdrasil.node.config import Settings


class TestBotEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = Settings(allow_remote=True)
        cls.app = create_app(settings)
        cls.client = TestClient(cls.app)

    # -- env ---------------------------------------------------------------

    def test_env_get(self):
        resp = self.client.get("/api/env?keys=PATH,HOME")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("variables", data)
        self.assertIn("node_id", data)

    def test_env_set_and_get(self):
        resp = self.client.post(
            "/api/env",
            json={"variables": {"YGG_BOT_TEST_VAR": "hello"}},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["applied"]["YGG_BOT_TEST_VAR"], "hello")

        resp = self.client.get("/api/env?keys=YGG_BOT_TEST_VAR")
        self.assertEqual(resp.json()["variables"]["YGG_BOT_TEST_VAR"], "hello")

        self.client.post(
            "/api/env",
            json={"variables": {"YGG_BOT_TEST_VAR": None}},
        )

    # -- cmd ---------------------------------------------------------------

    def test_cmd_execute(self):
        resp = self.client.post(
            "/api/cmd",
            json={"command": ["echo", "hello bot"]},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["returncode"], 0)
        self.assertIn("hello bot", data["stdout"])
        self.assertEqual(data["status"], "completed")

    def test_cmd_get_and_list(self):
        resp = self.client.post(
            "/api/cmd",
            json={"command": ["echo", "for-list"]},
        )
        cmd_id = resp.json()["id"]

        resp = self.client.get(f"/api/cmd/{cmd_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["id"], cmd_id)

        resp = self.client.get("/api/cmd")
        self.assertEqual(resp.status_code, 200)
        ids = [e["id"] for e in resp.json()["items"]]
        self.assertIn(cmd_id, ids)

    def test_cmd_delete(self):
        resp = self.client.post(
            "/api/cmd",
            json={"command": ["echo", "to-delete"]},
        )
        cmd_id = resp.json()["id"]

        resp = self.client.delete(f"/api/cmd/{cmd_id}")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get(f"/api/cmd/{cmd_id}")
        self.assertEqual(resp.status_code, 404)

    def test_cmd_not_found(self):
        resp = self.client.get("/api/cmd/nonexistent")
        self.assertEqual(resp.status_code, 404)

    # -- python ------------------------------------------------------------

    def test_python_execute(self):
        resp = self.client.post(
            "/api/python",
            json={"code": "print('from python')"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "completed")
        self.assertIn("from python", data["stdout"])

    def test_python_result(self):
        resp = self.client.post(
            "/api/python",
            json={"code": "__result__ = [1, 2, 3]"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["result"], [1, 2, 3])

    def test_python_arrow_ipc_response(self):
        resp = self.client.post(
            "/api/python",
            json={
                "code": "__result__ = {'col': [1, 2, 3]}",
                "result_format": "arrow_ipc",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.headers["content-type"],
            "application/vnd.apache.arrow.file",
        )
        self.assertTrue(resp.content.startswith(b"ARROW1"))

    def test_python_list_and_delete(self):
        resp = self.client.post(
            "/api/python",
            json={"code": "x = 1"},
        )
        exec_id = resp.json()["id"]

        resp = self.client.get("/api/python")
        ids = [e["id"] for e in resp.json()["items"]]
        self.assertIn(exec_id, ids)

        resp = self.client.delete(f"/api/python/{exec_id}")
        self.assertEqual(resp.status_code, 200)

    # -- job ---------------------------------------------------------------

    def test_job_lifecycle(self):
        resp = self.client.post(
            "/api/job",
            json={
                "name": "test-job",
                "tasks": {
                    "step1": {"type": "cmd", "command": ["echo", "step1"]},
                    "step2": {
                        "type": "python",
                        "code": "print('step2')",
                        "depends_on": ["step1"],
                    },
                },
            },
        )
        self.assertEqual(resp.status_code, 200)
        job_id = resp.json()["job"]["job_id"]

        resp = self.client.get(f"/api/job/{job_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["job"]["name"], "test-job")
        self.assertEqual(resp.json()["job"]["task_keys"], ["step1", "step2"])

        resp = self.client.get("/api/job")
        job_ids = [j["job_id"] for j in resp.json()["items"]]
        self.assertIn(job_id, job_ids)

    def test_job_run(self):
        resp = self.client.post(
            "/api/job",
            json={
                "name": "run-test",
                "tasks": {
                    "echo": {"type": "cmd", "command": ["echo", "hi"]},
                },
            },
        )
        job_id = resp.json()["job"]["job_id"]

        resp = self.client.post(f"/api/job/{job_id}/run")
        self.assertEqual(resp.status_code, 200)
        run = resp.json()["run"]
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["task_results"]["echo"]["status"], "completed")

        run_id = run["run_id"]
        resp = self.client.get(f"/api/job/{job_id}/run/{run_id}")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get(f"/api/job/{job_id}/run")
        self.assertEqual(len(resp.json()["items"]), 1)

    def test_job_run_with_dependencies(self):
        resp = self.client.post(
            "/api/job",
            json={
                "name": "dep-test",
                "tasks": {
                    "a": {"type": "cmd", "command": ["echo", "a"]},
                    "b": {"type": "cmd", "command": ["echo", "b"], "depends_on": ["a"]},
                    "c": {"type": "python", "code": "print('c')", "depends_on": ["a", "b"]},
                },
            },
        )
        job_id = resp.json()["job"]["job_id"]

        resp = self.client.post(f"/api/job/{job_id}/run")
        run = resp.json()["run"]
        self.assertEqual(run["status"], "completed")
        for key in ("a", "b", "c"):
            self.assertEqual(run["task_results"][key]["status"], "completed")

    def test_job_update_and_delete(self):
        resp = self.client.post(
            "/api/job",
            json={
                "name": "to-update",
                "tasks": {"t": {"type": "cmd", "command": ["true"]}},
            },
        )
        job_id = resp.json()["job"]["job_id"]

        resp = self.client.put(
            f"/api/job/{job_id}",
            json={
                "name": "updated-name",
                "tasks": {"t": {"type": "cmd", "command": ["echo", "updated"]}},
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["job"]["name"], "updated-name")

        resp = self.client.delete(f"/api/job/{job_id}")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get(f"/api/job/{job_id}")
        self.assertEqual(resp.status_code, 404)

    def test_run_delete(self):
        resp = self.client.post(
            "/api/job",
            json={
                "name": "run-del",
                "tasks": {"t": {"type": "cmd", "command": ["true"]}},
            },
        )
        job_id = resp.json()["job"]["job_id"]

        resp = self.client.post(f"/api/job/{job_id}/run")
        run_id = resp.json()["run"]["run_id"]

        resp = self.client.delete(f"/api/job/{job_id}/run/{run_id}")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get(f"/api/job/{job_id}/run/{run_id}")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
