from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_get_all_env_vars(client: TestClient):
    resp = client.get("/api/env")
    assert resp.status_code == 200
    data = resp.json()
    assert "variables" in data
    assert isinstance(data["variables"], dict)
    assert "PATH" in data["variables"]


def test_get_specific_keys(client: TestClient):
    resp = client.get("/api/env", params={"keys": "PATH,HOME"})
    assert resp.status_code == 200
    variables = resp.json()["variables"]
    assert set(variables.keys()) == {"PATH", "HOME"}
    assert variables["PATH"] is not None
    assert variables["HOME"] is not None


def test_get_nonexistent_key(client: TestClient):
    resp = client.get("/api/env", params={"keys": "YGG_SURELY_DOES_NOT_EXIST"})
    assert resp.status_code == 200
    variables = resp.json()["variables"]
    assert "YGG_SURELY_DOES_NOT_EXIST" in variables
    assert variables["YGG_SURELY_DOES_NOT_EXIST"] is None


def test_set_and_get(client: TestClient):
    set_resp = client.post("/api/env", json={"variables": {"YGG_TEST_VAR": "hello"}})
    assert set_resp.status_code == 200
    assert set_resp.json()["applied"] == {"YGG_TEST_VAR": "hello"}

    get_resp = client.get("/api/env", params={"keys": "YGG_TEST_VAR"})
    assert get_resp.status_code == 200
    assert get_resp.json()["variables"]["YGG_TEST_VAR"] == "hello"


def test_unset_variable(client: TestClient):
    client.post("/api/env", json={"variables": {"YGG_TO_REMOVE": "temporary"}})

    resp = client.post("/api/env", json={"variables": {"YGG_TO_REMOVE": None}})
    assert resp.status_code == 200
    assert resp.json()["applied"] == {"YGG_TO_REMOVE": None}

    get_resp = client.get("/api/env", params={"keys": "YGG_TO_REMOVE"})
    assert get_resp.json()["variables"]["YGG_TO_REMOVE"] is None


def test_set_multiple_variables(client: TestClient):
    payload = {"variables": {"YGG_MULTI_A": "alpha", "YGG_MULTI_B": "beta"}}
    resp = client.post("/api/env", json=payload)
    assert resp.status_code == 200
    applied = resp.json()["applied"]
    assert applied == {"YGG_MULTI_A": "alpha", "YGG_MULTI_B": "beta"}

    get_resp = client.get("/api/env", params={"keys": "YGG_MULTI_A,YGG_MULTI_B"})
    variables = get_resp.json()["variables"]
    assert variables["YGG_MULTI_A"] == "alpha"
    assert variables["YGG_MULTI_B"] == "beta"


def test_response_includes_node_id(client: TestClient):
    get_resp = client.get("/api/env")
    assert get_resp.json()["node_id"] == "test-node"

    post_resp = client.post("/api/env", json={"variables": {"YGG_ID_CHECK": "x"}})
    assert post_resp.json()["node_id"] == "test-node"
