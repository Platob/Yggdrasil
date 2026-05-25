"""Tests for function and environment clone endpoints."""
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
        node_id="test-node-clone",
    )


@pytest.fixture
def client(settings):
    app = create_app(settings)
    return TestClient(app)


class TestFunctionClone:
    def test_clone_function_default_name(self, client):
        # Create a function
        resp = client.post("/api/function", json={
            "name": "hello",
            "code": "print('hello')",
            "description": "A greeting function",
            "dependencies": ["requests"],
        })
        assert resp.status_code == 200
        func = resp.json()["function"]
        func_id = func["id"]

        # Clone it
        resp = client.post(f"/api/function/{func_id}/clone")
        assert resp.status_code == 200
        cloned = resp.json()["function"]

        assert cloned["id"] != func_id
        assert cloned["name"] == "hello_clone"
        assert cloned["code"] == "print('hello')"
        assert cloned["description"] == "A greeting function"
        assert cloned["dependencies"] == ["requests"]
        assert cloned["state"] == "ready"

    def test_clone_function_custom_name(self, client):
        resp = client.post("/api/function", json={
            "name": "original",
            "code": "x = 1",
        })
        assert resp.status_code == 200
        func_id = resp.json()["function"]["id"]

        resp = client.post(f"/api/function/{func_id}/clone", json={
            "name": "my_copy",
        })
        assert resp.status_code == 200
        cloned = resp.json()["function"]
        assert cloned["name"] == "my_copy"

    def test_clone_nonexistent_function(self, client):
        resp = client.post("/api/function/999999/clone")
        assert resp.status_code == 404


class TestEnvironmentClone:
    def test_clone_environment_default_name(self, client):
        # Create an environment (will fail to actually build venv in test, but entry is created)
        resp = client.post("/api/environment", json={
            "name": "myenv",
            "python_version": "3.11",
            "dependencies": ["numpy"],
        })
        assert resp.status_code == 200
        env = resp.json()["environment"]
        env_id = env["id"]

        # Clone it
        resp = client.post(f"/api/environment/{env_id}/clone")
        assert resp.status_code == 200
        cloned = resp.json()["environment"]

        assert cloned["id"] != env_id
        assert cloned["name"] == "myenv_clone"
        assert cloned["python_version"] == "3.11"
        assert cloned["dependencies"] == ["numpy"]

    def test_clone_environment_custom_name(self, client):
        resp = client.post("/api/environment", json={
            "name": "base_env",
            "python_version": "3.11",
        })
        assert resp.status_code == 200
        env_id = resp.json()["environment"]["id"]

        resp = client.post(f"/api/environment/{env_id}/clone", json={
            "name": "derived_env",
        })
        assert resp.status_code == 200
        cloned = resp.json()["environment"]
        assert cloned["name"] == "derived_env"

    def test_clone_nonexistent_environment(self, client):
        resp = client.post("/api/environment/999999/clone")
        assert resp.status_code == 404
