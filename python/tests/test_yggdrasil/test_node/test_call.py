from __future__ import annotations

import pytest
import pyarrow as pa
from fastapi.testclient import TestClient

from yggdrasil.node.app import create_app
from yggdrasil.node.config import Settings
from yggdrasil.node.remote import _REGISTRY, _RemoteSpec, _infer_modules, ensure_modules, remote
from yggdrasil.node.transport import (
    CONTENT_TYPE_ARROW_STREAM,
    CONTENT_TYPE_PICKLE,
    deserialize_pickle,
    deserialize_result,
    is_tabular,
    read_arrow_stream,
    serialize_pickle,
    serialize_result,
    to_arrow_table,
)

try:
    serialize_pickle({})
except Exception:
    pass


@remote
def add(x: int, y: int) -> int:
    return x + y


@remote
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"


@remote
def make_table(n: int) -> pa.Table:
    return pa.table({"id": list(range(n)), "value": [float(i * 10) for i in range(n)]})


@remote(name="test_call:multiply")
def multiply(a: float, b: float) -> float:
    return a * b


@remote
def nested_result() -> dict:
    return {"numbers": [1, 2, 3], "nested": {"key": "value"}}


@remote
def failing_func() -> None:
    raise ValueError("intentional")


def _func_key_for(fn) -> str:
    return fn._remote_key


@pytest.fixture(scope="module")
def call_client():
    settings = Settings(allow_remote=True)
    app = create_app(settings)
    return TestClient(app)


def _call(client, func_key, args=(), kwargs=None, stream=False):
    payload = serialize_pickle({"func": func_key, "args": args, "kwargs": kwargs or {}})
    headers = {"Content-Type": CONTENT_TYPE_PICKLE}
    if stream:
        headers["Accept"] = CONTENT_TYPE_ARROW_STREAM
    endpoint = "/api/call/stream" if stream else "/api/call"
    return client.post(endpoint, content=payload, headers=headers)


# -- Remote decorator tests --------------------------------------------------


def test_remote_local_call():
    assert add(10, 20) == 30
    assert greet("World") == "Hello, World!"


def test_remote_registry():
    assert _func_key_for(add) in _REGISTRY
    assert _func_key_for(greet) in _REGISTRY
    assert _func_key_for(nested_result) in _REGISTRY


def test_remote_custom_name():
    assert "test_call:multiply" in _REGISTRY
    spec = _REGISTRY["test_call:multiply"]
    assert spec.func is multiply._remote_func


def test_remote_preserves_metadata():
    assert add.__name__ == "add"
    assert greet.__name__ == "greet"
    assert multiply.__name__ == "multiply"


# -- Transport tests ----------------------------------------------------------


def test_pickle_roundtrip():
    original = {"func": "test", "args": (1, 2), "kwargs": {}}
    data = serialize_pickle(original)
    assert isinstance(data, bytes)
    restored = deserialize_pickle(data)
    assert restored == original


def test_is_tabular():
    table = pa.table({"x": [1, 2, 3]})
    batch = pa.record_batch({"x": [1, 2, 3]})
    assert is_tabular(table) is True
    assert is_tabular(batch) is True
    assert is_tabular([1, 2, 3]) is False
    assert is_tabular("hello") is False


def test_to_arrow_table():
    batch = pa.record_batch({"col": [10, 20, 30]})
    table = to_arrow_table(batch)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 3
    assert table.column("col").to_pylist() == [10, 20, 30]


def test_serialize_tabular_result():
    table = pa.table({"a": [1, 2], "b": [3.0, 4.0]})
    data, content_type = serialize_result(table)
    assert content_type == CONTENT_TYPE_ARROW_STREAM
    assert isinstance(data, bytes)
    restored = read_arrow_stream(data)
    assert restored.num_rows == 2


def test_serialize_complex_result():
    obj = {"status": "ok", "values": [1, 2, 3]}
    data, content_type = serialize_result(obj)
    assert content_type == CONTENT_TYPE_PICKLE
    assert isinstance(data, bytes)
    restored = deserialize_pickle(data)
    assert restored == obj


# -- Module inference tests ---------------------------------------------------


def test_infer_stdlib_excluded():
    def _stdlib_only():
        import os
        import json
        return os.getpid(), json.dumps({})

    modules = _infer_modules(_stdlib_only)
    assert modules == []


def test_infer_third_party():
    def _uses_numpy():
        import numpy as np
        return np.array([1, 2, 3])

    modules = _infer_modules(_uses_numpy)
    assert "numpy" in modules


def test_infer_from_import():
    def _uses_scipy():
        from scipy.stats import norm
        return norm.pdf(0)

    modules = _infer_modules(_uses_scipy)
    assert "scipy" in modules


def test_infer_yggdrasil_excluded():
    def _uses_yggdrasil():
        from yggdrasil.node.config import Settings
        return Settings()

    modules = _infer_modules(_uses_yggdrasil)
    assert "yggdrasil" not in modules


def test_auto_infer_on_remote():
    @remote
    def _inferred():
        import requests
        return requests.get("http://example.com")

    spec = _REGISTRY[_func_key_for(_inferred)]
    assert "requests" in spec.modules


def test_explicit_modules_override():
    @remote(modules=["custom"])
    def _explicit():
        import requests
        return requests.get("http://example.com")

    spec = _REGISTRY[_func_key_for(_explicit)]
    assert spec.modules == ["custom"]


# -- Call endpoint tests ------------------------------------------------------


def test_call_simple(call_client):
    resp = _call(call_client, _func_key_for(add), args=(10, 20))
    assert resp.status_code == 200
    result = deserialize_result(resp.content, resp.headers["content-type"])
    assert result == 30


def test_call_with_kwargs(call_client):
    resp = _call(call_client, _func_key_for(greet), args=("World",), kwargs={"greeting": "Hi"})
    assert resp.status_code == 200
    result = deserialize_result(resp.content, resp.headers["content-type"])
    assert result == "Hi, World!"


def test_call_custom_name(call_client):
    resp = _call(call_client, "test_call:multiply", args=(3.0, 7.0))
    assert resp.status_code == 200
    result = deserialize_result(resp.content, resp.headers["content-type"])
    assert result == 21.0


def test_call_returns_arrow_for_tabular(call_client):
    resp = _call(call_client, _func_key_for(make_table), args=(100,))
    assert resp.status_code == 200
    assert CONTENT_TYPE_ARROW_STREAM in resp.headers["content-type"]
    table = read_arrow_stream(resp.content)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 100
    assert table.column("id").to_pylist() == list(range(100))


def test_call_nested_result(call_client):
    resp = _call(call_client, _func_key_for(nested_result))
    assert resp.status_code == 200
    result = deserialize_result(resp.content, resp.headers["content-type"])
    assert result == {"numbers": [1, 2, 3], "nested": {"key": "value"}}


def test_call_not_found_404(call_client):
    resp = _call(call_client, "nonexistent:function", args=())
    assert resp.status_code == 404


def test_call_response_headers(call_client):
    resp = _call(call_client, _func_key_for(add), args=(1, 2))
    assert resp.status_code == 200
    assert "x-bot-call-id" in resp.headers
    assert "x-bot-call-func" in resp.headers
    assert "x-bot-call-duration" in resp.headers
    assert "x-bot-node-id" in resp.headers


def test_call_stream_tabular(call_client):
    resp = _call(call_client, _func_key_for(make_table), args=(50,), stream=True)
    assert resp.status_code == 200
    assert CONTENT_TYPE_ARROW_STREAM in resp.headers["content-type"]
    table = read_arrow_stream(resp.content)
    assert table.num_rows == 50


def test_registry_endpoint(call_client):
    resp = call_client.get("/api/call/registry")
    assert resp.status_code == 200
    data = resp.json()
    assert _func_key_for(add) in data
    assert _func_key_for(greet) in data
    assert "test_call:multiply" in data


# -- ensure_modules tests ----------------------------------------------------


def test_ensure_modules_already_installed():
    spec = _RemoteSpec(func=lambda: None, key="test:noop", timeout=None, modules=["sys", "os"])
    ensure_modules(spec)


def test_ensure_modules_empty():
    spec_none = _RemoteSpec(func=lambda: None, key="test:empty1", timeout=None, modules=None)
    ensure_modules(spec_none)
    spec_list = _RemoteSpec(func=lambda: None, key="test:empty2", timeout=None, modules=[])
    ensure_modules(spec_list)
