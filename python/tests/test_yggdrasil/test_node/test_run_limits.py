"""Tests for run resource limits (timeout, memory, cpu)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from yggdrasil.node.app import create_app
from yggdrasil.node.config import Settings


@pytest.fixture
def tmp_home(tmp_path):
    return tmp_path / "ygg_home"


@pytest.fixture
def settings(tmp_home):
    return Settings(
        node_home=tmp_home,
        node_id="test-node-limits",
        max_python_timeout=30.0,
    )


@pytest.fixture
def client(settings):
    app = create_app(settings)
    return TestClient(app)


class TestRunResourceLimits:
    def _create_function(self, client, code: str) -> int:
        resp = client.post("/api/function", json={
            "name": f"test_func_{id(code)}",
            "code": code,
        })
        assert resp.status_code == 200
        return resp.json()["function"]["id"]

    def test_run_with_timeout_success(self, client):
        """A fast function should complete within the timeout."""
        func_id = self._create_function(client, "print('fast')")

        resp = client.post("/api/run", json={
            "function_id": func_id,
            "timeout": 10.0,
        })
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "completed"
        assert run["timeout"] == 10.0
        assert "fast" in (run["stdout"] or "")

    def test_run_with_timeout_expiry(self, client):
        """A slow function should fail when timeout is exceeded."""
        func_id = self._create_function(client, "import time; time.sleep(10)")

        resp = client.post("/api/run", json={
            "function_id": func_id,
            "timeout": 0.5,
        })
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "failed"
        assert "Timed out" in (run["stderr"] or "")

    def test_run_limits_stored_in_entry(self, client):
        """Resource limit fields should be persisted in the run entry."""
        func_id = self._create_function(client, "print('ok')")

        resp = client.post("/api/run", json={
            "function_id": func_id,
            "max_memory_mb": 512,
            "max_cpu_percent": 50.0,
            "timeout": 30.0,
        })
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["max_memory_mb"] == 512
        assert run["max_cpu_percent"] == 50.0
        assert run["timeout"] == 30.0

    def test_run_without_limits_uses_defaults(self, client):
        """When no limits are specified, fields should be None."""
        func_id = self._create_function(client, "print('default')")

        resp = client.post("/api/run", json={
            "function_id": func_id,
        })
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["max_memory_mb"] is None
        assert run["max_cpu_percent"] is None
        assert run["timeout"] is None
        assert run["status"] == "completed"


class TestRunMetadata:
    def test_function_last_used_at_updated(self, client):
        """Running a function should update its last_used_at."""
        resp = client.post("/api/function", json={
            "name": "track_usage",
            "code": "print('used')",
        })
        assert resp.status_code == 200
        func = resp.json()["function"]
        func_id = func["id"]
        assert func["last_used_at"] is None

        # Run it
        resp = client.post("/api/run", json={"function_id": func_id})
        assert resp.status_code == 200

        # Check last_used_at is now set
        resp = client.get(f"/api/function/{func_id}")
        assert resp.status_code == 200
        func = resp.json()["function"]
        assert func["last_used_at"] is not None
        assert func["run_count"] == 1

    def test_function_state_field(self, client):
        """Functions should have a state field defaulting to 'ready'."""
        resp = client.post("/api/function", json={
            "name": "stateful",
            "code": "pass",
        })
        assert resp.status_code == 200
        func = resp.json()["function"]
        assert func["state"] == "ready"
