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
        assert func["id"]
        assert func["created_at"]
        self._func_id = func["id"]

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
            "name": "updatable",
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
            "name": "deleteme",
            "code": "pass",
        })
        func_id = create_resp.json()["function"]["id"]

        del_resp = await self.client.delete(f"/api/function/{func_id}")
        assert del_resp.status_code == 200

        get_resp = await self.client.get(f"/api/function/{func_id}")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_nonexistent_function(self):
        resp = await self.client.get("/api/function/doesnotexist")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_run_function(self):
        create_resp = await self.client.post("/api/function", json={
            "name": "runnable",
            "code": "print('executed')",
        })
        func_id = create_resp.json()["function"]["id"]

        run_resp = await self.client.post(f"/api/function/{func_id}/run", json={})
        assert run_resp.status_code == 200
        run = run_resp.json()["run"]
        assert run["function_id"] == func_id
        assert run["status"] in ("pending", "running", "completed", "failed")

    @pytest.mark.asyncio
    async def test_list_function_runs(self):
        create_resp = await self.client.post("/api/function", json={
            "name": "multi_run",
            "code": "print('run')",
        })
        func_id = create_resp.json()["function"]["id"]
        await self.client.post(f"/api/function/{func_id}/run", json={})

        runs_resp = await self.client.get(f"/api/function/{func_id}/run")
        assert runs_resp.status_code == 200
        assert len(runs_resp.json()["runs"]) >= 1


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
        data = resp.json()
        assert "environments" in data

    @pytest.mark.asyncio
    async def test_create_environment(self):
        resp = await self.client.post("/api/environment", json={
            "name": "test-env",
            "python_version": "3.11",
        })
        assert resp.status_code == 200
        env = resp.json()["environment"]
        assert env["name"] == "test-env"
        assert env["python_version"] == "3.11"
        assert env["status"] in ("pending", "creating", "ready", "failed")


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
        data = resp.json()
        assert "runs" in data
