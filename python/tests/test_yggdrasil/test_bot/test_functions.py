"""Tests for FunctionService, EnvironmentService, RunService and their endpoints."""
from __future__ import annotations

import pytest


class TestFunctionEndpoints:
    @pytest.fixture(autouse=True)
    def setup(self):
        from httpx import ASGITransport, AsyncClient
        from yggdrasil.node.app import create_app
        self.app = create_app()
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://test",
        )

    @pytest.mark.asyncio
    async def test_list_functions_empty(self):
        resp = await self.client.get("/api/function")
        assert resp.status_code == 200
        data = resp.json()
        assert "functions" in data
        assert len(data["functions"]) == 0

    @pytest.mark.asyncio
    async def test_create_function(self):
        resp = await self.client.post("/api/function", json={
            "name": "hello",
            "code": "print('hello world')",
            "language": "python",
            "description": "A test function",
        })
        assert resp.status_code == 200
        data = resp.json()
        func = data["function"]
        assert func["name"] == "hello"
        assert func["code"] == "print('hello world')"
        assert func["language"] == "python"
        assert isinstance(func["id"], int)
        assert func["created_at"]

    @pytest.mark.asyncio
    async def test_upsert_function(self):
        resp1 = await self.client.post("/api/function", json={
            "name": "upsert_test",
            "code": "print(1)",
        })
        func_id = resp1.json()["function"]["id"]
        assert isinstance(func_id, int)

        resp2 = await self.client.post("/api/function", json={
            "name": "upsert_test",
            "code": "print(2)",
            "description": "updated via upsert",
        })
        assert resp2.json()["function"]["id"] == func_id
        assert resp2.json()["function"]["code"] == "print(2)"
        assert resp2.json()["function"]["description"] == "updated via upsert"

    @pytest.mark.asyncio
    async def test_create_and_get_function(self):
        create_resp = await self.client.post("/api/function", json={
            "name": "greet",
            "code": "print('hi')",
        })
        func_id = create_resp.json()["function"]["id"]

        get_resp = await self.client.get(f"/api/function/{func_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["function"]["name"] == "greet"

    @pytest.mark.asyncio
    async def test_update_function(self):
        create_resp = await self.client.post("/api/function", json={
            "name": "updatable_fn",
            "code": "print(1)",
        })
        func_id = create_resp.json()["function"]["id"]

        update_resp = await self.client.put(f"/api/function/{func_id}", json={
            "code": "print(2)",
            "description": "updated",
        })
        assert update_resp.status_code == 200
        func = update_resp.json()["function"]
        assert func["code"] == "print(2)"
        assert func["description"] == "updated"

    @pytest.mark.asyncio
    async def test_delete_function(self):
        create_resp = await self.client.post("/api/function", json={
            "name": "deleteme_fn",
            "code": "pass",
        })
        func_id = create_resp.json()["function"]["id"]

        del_resp = await self.client.delete(f"/api/function/{func_id}")
        assert del_resp.status_code == 200

        get_resp = await self.client.get(f"/api/function/{func_id}")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_run_function(self):
        create_resp = await self.client.post("/api/function", json={
            "name": "runnable_fn",
            "code": "print('executed')",
        })
        func_id = create_resp.json()["function"]["id"]

        run_resp = await self.client.post(f"/api/function/{func_id}/run", json={})
        assert run_resp.status_code == 200
        run = run_resp.json()["run"]
        assert run["function_id"] == func_id
        assert isinstance(run["id"], int)
        assert run["status"] in ("pending", "running", "completed", "failed")


class TestEnvironmentEndpoints:
    @pytest.fixture(autouse=True)
    def setup(self):
        from httpx import ASGITransport, AsyncClient
        from yggdrasil.node.app import create_app
        self.app = create_app()
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://test",
        )

    @pytest.mark.asyncio
    async def test_list_environments_empty(self):
        resp = await self.client.get("/api/environment")
        assert resp.status_code == 200
        assert "environments" in resp.json()

    @pytest.mark.asyncio
    async def test_create_environment(self):
        resp = await self.client.post("/api/environment", json={
            "name": "test-env",
            "python_version": "3.11",
        })
        assert resp.status_code == 200
        env = resp.json()["environment"]
        assert env["name"] == "test-env"
        assert isinstance(env["id"], int)

    @pytest.mark.asyncio
    async def test_upsert_environment(self):
        resp1 = await self.client.post("/api/environment", json={
            "name": "upsert-env",
            "python_version": "3.11",
        })
        env_id = resp1.json()["environment"]["id"]

        resp2 = await self.client.post("/api/environment", json={
            "name": "upsert-env",
            "python_version": "3.11",
            "dependencies": ["requests"],
        })
        assert resp2.json()["environment"]["id"] == env_id


class TestRunEndpoints:
    @pytest.fixture(autouse=True)
    def setup(self):
        from httpx import ASGITransport, AsyncClient
        from yggdrasil.node.app import create_app
        self.app = create_app()
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://test",
        )

    @pytest.mark.asyncio
    async def test_list_runs_empty(self):
        resp = await self.client.get("/api/run")
        assert resp.status_code == 200
        assert "runs" in resp.json()


class TestDagEndpoints:
    @pytest.fixture(autouse=True)
    def setup(self):
        from httpx import ASGITransport, AsyncClient
        from yggdrasil.node.app import create_app
        self.app = create_app()
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://test",
        )

    @pytest.mark.asyncio
    async def test_list_dags_empty(self):
        resp = await self.client.get("/api/dag")
        assert resp.status_code == 200
        assert "dags" in resp.json()

    @pytest.mark.asyncio
    async def test_create_dag(self):
        func_resp = await self.client.post("/api/function", json={
            "name": "dag_step_fn",
            "code": "print('step')",
        })
        func_id = func_resp.json()["function"]["id"]

        resp = await self.client.post("/api/dag", json={
            "name": "test-dag",
            "description": "A test DAG",
            "steps": [
                {"id": "step1", "ref": {"function_id": func_id, "args": {}}}
            ],
        })
        assert resp.status_code == 200
        dag = resp.json()["dag"]
        assert dag["name"] == "test-dag"
        assert isinstance(dag["id"], int)
        assert len(dag["steps"]) == 1
