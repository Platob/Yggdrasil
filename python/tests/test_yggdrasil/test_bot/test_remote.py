from __future__ import annotations

import unittest

import pyarrow as pa
from fastapi.testclient import TestClient

from yggdrasil.bot.app import create_app
from yggdrasil.bot.config import Settings
from yggdrasil.bot.remote import _REGISTRY, _RemoteSpec, ensure_modules, remote
from yggdrasil.bot.transport import (
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


# -- Register test functions -----------------------------------------------

@remote
def add(x: int, y: int) -> int:
    return x + y


@remote
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"


@remote
def make_table(n: int) -> pa.Table:
    return pa.table({"id": list(range(n)), "value": [float(i * 10) for i in range(n)]})


@remote(name="custom:multiply")
def multiply(a: float, b: float) -> float:
    return a * b


@remote
def nested_result() -> dict:
    return {
        "numbers": [1, 2, 3],
        "nested": {"key": "value"},
        "flag": True,
    }


@remote
def failing_func() -> None:
    raise ValueError("intentional error")


@remote
def return_list() -> list:
    return [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]


# -- Tests -----------------------------------------------------------------

class TestRemoteDecorator(unittest.TestCase):
    def test_local_call_works(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(greet("world"), "Hello, world!")
        self.assertEqual(multiply(3, 4), 12.0)

    def test_registry(self):
        key = add._remote_key
        self.assertIn(key, _REGISTRY)
        spec = _REGISTRY[key]
        self.assertIsInstance(spec, _RemoteSpec)
        self.assertIs(spec.func, add._remote_func)

    def test_custom_name(self):
        self.assertEqual(multiply._remote_key, "custom:multiply")
        self.assertIn("custom:multiply", _REGISTRY)

    def test_preserves_metadata(self):
        self.assertEqual(add.__name__, "add")
        self.assertEqual(greet.__name__, "greet")


class TestTransport(unittest.TestCase):
    def test_pickle_roundtrip(self):
        obj = {"key": [1, 2, 3], "nested": {"a": True}}
        data = serialize_pickle(obj)
        self.assertIsInstance(data, bytes)
        result = deserialize_pickle(data)
        self.assertEqual(result, obj)

    def test_is_tabular(self):
        self.assertTrue(is_tabular(pa.table({"a": [1]})))
        self.assertTrue(is_tabular(pa.record_batch({"a": [1]})))
        self.assertFalse(is_tabular([1, 2, 3]))
        self.assertFalse(is_tabular("hello"))

    def test_to_arrow_table(self):
        batch = pa.record_batch({"x": [1, 2]})
        table = to_arrow_table(batch)
        self.assertIsInstance(table, pa.Table)
        self.assertEqual(table.num_rows, 2)

    def test_serialize_result_tabular(self):
        table = pa.table({"a": [1, 2, 3]})
        data, ct = serialize_result(table)
        self.assertEqual(ct, CONTENT_TYPE_ARROW_STREAM)
        recovered = read_arrow_stream(data)
        self.assertEqual(recovered.num_rows, 3)

    def test_serialize_result_complex(self):
        obj = {"key": "value", "list": [1, 2]}
        data, ct = serialize_result(obj)
        self.assertEqual(ct, CONTENT_TYPE_PICKLE)
        recovered = deserialize_result(data, ct)
        self.assertEqual(recovered, obj)


class TestCallEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = Settings(allow_remote=True)
        cls.app = create_app(settings)
        cls.client = TestClient(cls.app)

    def _call(self, func_key: str, args=(), kwargs=None, stream=False):
        payload = serialize_pickle({
            "func": func_key,
            "args": args,
            "kwargs": kwargs or {},
            "stream": stream,
        })
        headers = {"Content-Type": CONTENT_TYPE_PICKLE}
        if stream:
            headers["Accept"] = CONTENT_TYPE_ARROW_STREAM
        return self.client.post("/api/call", content=payload, headers=headers)

    def test_call_simple_function(self):
        resp = self._call(add._remote_key, args=(10, 20))
        self.assertEqual(resp.status_code, 200)
        result = deserialize_result(resp.content, resp.headers["content-type"])
        self.assertEqual(result, 30)

    def test_call_with_kwargs(self):
        resp = self._call(greet._remote_key, args=("Alice",), kwargs={"greeting": "Hi"})
        self.assertEqual(resp.status_code, 200)
        result = deserialize_result(resp.content, resp.headers["content-type"])
        self.assertEqual(result, "Hi, Alice!")

    def test_call_custom_name(self):
        resp = self._call("custom:multiply", args=(5.0, 3.0))
        self.assertEqual(resp.status_code, 200)
        result = deserialize_result(resp.content, resp.headers["content-type"])
        self.assertEqual(result, 15.0)

    def test_call_returns_arrow_for_tabular(self):
        resp = self._call(make_table._remote_key, args=(100,))
        self.assertEqual(resp.status_code, 200)
        self.assertIn(CONTENT_TYPE_ARROW_STREAM, resp.headers["content-type"])
        table = read_arrow_stream(resp.content)
        self.assertEqual(table.num_rows, 100)
        self.assertEqual(table.column_names, ["id", "value"])
        self.assertEqual(resp.headers["x-arrow-num-rows"], "100")
        self.assertEqual(resp.headers["x-arrow-num-columns"], "2")

    def test_call_nested_result(self):
        resp = self._call(nested_result._remote_key)
        self.assertEqual(resp.status_code, 200)
        result = deserialize_pickle(resp.content)
        self.assertEqual(result["numbers"], [1, 2, 3])
        self.assertEqual(result["nested"]["key"], "value")

    def test_call_not_found(self):
        resp = self._call("nonexistent:func")
        self.assertEqual(resp.status_code, 404)

    def test_call_headers(self):
        resp = self._call(add._remote_key, args=(1, 2))
        self.assertIn("x-bot-call-id", resp.headers)
        self.assertIn("x-bot-call-func", resp.headers)
        self.assertIn("x-bot-call-duration", resp.headers)
        self.assertIn("x-bot-node-id", resp.headers)

    def test_call_stream_tabular(self):
        resp = self._call(make_table._remote_key, args=(50,), stream=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(CONTENT_TYPE_ARROW_STREAM, resp.headers["content-type"])
        table = read_arrow_stream(resp.content)
        self.assertEqual(table.num_rows, 50)

    def test_registry_endpoint(self):
        resp = self.client.get("/api/call/registry")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(add._remote_key, data)
        self.assertIn("custom:multiply", data)
        sig = data[add._remote_key]
        self.assertIn("x", sig)
        self.assertIn("y", sig)
        self.assertIn("int", sig)

    def test_call_list_result(self):
        resp = self._call(return_list._remote_key)
        self.assertEqual(resp.status_code, 200)
        result = deserialize_pickle(resp.content)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["a"], 1)


class TestModuleAutoInstall(unittest.TestCase):
    def test_ensure_modules_already_installed(self):
        spec = _RemoteSpec(func=lambda: None, key="test:noop", timeout=None, modules=["sys", "os"])
        ensure_modules(spec)

    def test_ensure_modules_empty(self):
        spec = _RemoteSpec(func=lambda: None, key="test:noop", timeout=None, modules=None)
        ensure_modules(spec)

    def test_ensure_modules_no_list(self):
        spec = _RemoteSpec(func=lambda: None, key="test:noop", timeout=None, modules=[])
        ensure_modules(spec)

    def test_remote_with_modules_decorator(self):
        @remote(modules=["json", "os"])
        def needs_modules() -> str:
            return "ok"

        self.assertEqual(needs_modules(), "ok")
        self.assertEqual(needs_modules._remote_modules, ["json", "os"])

    def test_call_with_modules_metadata(self):
        @remote(name="test:with_mods", modules=["json"])
        def with_mods() -> str:
            import json
            return json.dumps({"ok": True})

        settings = Settings(allow_remote=True)
        app = create_app(settings)
        client = TestClient(app)

        payload = serialize_pickle({
            "func": "test:with_mods",
            "args": (),
            "kwargs": {},
        })
        resp = client.post("/api/call", content=payload, headers={"Content-Type": CONTENT_TYPE_PICKLE})
        self.assertEqual(resp.status_code, 200)
        result = deserialize_pickle(resp.content)
        self.assertEqual(result, '{"ok": true}')


class TestPolarsTransport(unittest.TestCase):
    def test_polars_dataframe(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")

        df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        self.assertTrue(is_tabular(df))

        data, ct = serialize_result(df)
        self.assertEqual(ct, CONTENT_TYPE_ARROW_STREAM)
        table = read_arrow_stream(data)
        self.assertEqual(table.num_rows, 3)
        self.assertEqual(table.column_names, ["x", "y"])

    def test_polars_series(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")

        s = pl.Series("vals", [10, 20, 30])
        self.assertTrue(is_tabular(s))
        table = to_arrow_table(s)
        self.assertEqual(table.num_rows, 3)


if __name__ == "__main__":
    unittest.main()
