from __future__ import annotations

import subprocess
import sys
import unittest

import pyarrow as pa
from fastapi.testclient import TestClient

from yggdrasil.bot.app import create_app
from yggdrasil.bot.config import Settings
from yggdrasil.bot.remote import _REGISTRY, _infer_modules, remote
from yggdrasil.bot.transport import (
    CONTENT_TYPE_ARROW_STREAM,
    CONTENT_TYPE_PICKLE,
    deserialize_pickle,
    read_arrow_stream,
    serialize_pickle,
)


# -- module inference tests ------------------------------------------------

class TestModuleInference(unittest.TestCase):
    def test_infer_stdlib_excluded(self):
        def fn():
            import os
            import sys
            import json
            return os.getcwd()

        modules = _infer_modules(fn)
        self.assertEqual(modules, [])

    def test_infer_third_party(self):
        def fn():
            import numpy as np
            import pandas as pd
            return np.array([1, 2, 3])

        modules = _infer_modules(fn)
        self.assertIn("numpy", modules)
        self.assertIn("pandas", modules)

    def test_infer_from_import(self):
        def fn():
            from scipy.stats import norm
            return norm.cdf(0)

        modules = _infer_modules(fn)
        self.assertIn("scipy", modules)

    def test_infer_mixed(self):
        def fn():
            import os
            import requests
            from bs4 import BeautifulSoup
            return requests.get("http://example.com")

        modules = _infer_modules(fn)
        self.assertIn("requests", modules)
        self.assertIn("bs4", modules)
        self.assertNotIn("os", modules)

    def test_infer_yggdrasil_excluded(self):
        def fn():
            from yggdrasil.data import Schema
            return Schema

        modules = _infer_modules(fn)
        self.assertEqual(modules, [])

    def test_auto_infer_on_remote(self):
        @remote
        def uses_polars():
            import polars as pl
            return pl.DataFrame({"a": [1]})

        self.assertIn("polars", uses_polars._remote_modules)

    def test_explicit_modules_override(self):
        @remote(modules=["custom_pkg"])
        def explicit():
            import polars
            return 1

        self.assertEqual(explicit._remote_modules, ["custom_pkg"])

    def test_empty_modules_disables_inference(self):
        @remote(modules=[])
        def no_infer():
            import polars
            return 1

        self.assertEqual(no_infer._remote_modules, [])


# -- arg coercion tests ---------------------------------------------------

class TestArgCoercion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        @remote(name="coerce:typed_add")
        def typed_add(x: int, y: int) -> int:
            return x + y

        @remote(name="coerce:greet")
        def greet(name: str, count: int = 1) -> str:
            return f"Hello {name}! " * count

        settings = Settings(allow_remote=True)
        cls.app = create_app(settings)
        cls.client = TestClient(cls.app)

    def _call(self, func_key, args=(), kwargs=None):
        payload = serialize_pickle({
            "func": func_key,
            "args": args,
            "kwargs": kwargs or {},
        })
        return self.client.post(
            "/api/call",
            content=payload,
            headers={"Content-Type": CONTENT_TYPE_PICKLE},
        )

    def test_string_args_coerced_to_int(self):
        resp = self._call("coerce:typed_add", args=("3", "7"))
        self.assertEqual(resp.status_code, 200)
        result = deserialize_pickle(resp.content)
        self.assertEqual(result, 10)

    def test_mixed_types_coerced(self):
        resp = self._call("coerce:greet", args=("world",), kwargs={"count": "3"})
        self.assertEqual(resp.status_code, 200)
        result = deserialize_pickle(resp.content)
        self.assertEqual(result, "Hello world! Hello world! Hello world! ")


# -- /api/call/stream endpoint tests --------------------------------------

class TestStreamEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        @remote(name="stream:table")
        def stream_table(n: int) -> pa.Table:
            return pa.table({"x": list(range(n))})

        @remote(name="stream:scalar")
        def stream_scalar() -> int:
            return 42

        settings = Settings(allow_remote=True)
        cls.app = create_app(settings)
        cls.client = TestClient(cls.app)

    def _call_stream(self, func_key, args=(), kwargs=None):
        payload = serialize_pickle({
            "func": func_key,
            "args": args,
            "kwargs": kwargs or {},
        })
        return self.client.post(
            "/api/call/stream",
            content=payload,
            headers={"Content-Type": CONTENT_TYPE_PICKLE},
        )

    def test_stream_tabular(self):
        resp = self._call_stream("stream:table", args=(100,))
        self.assertEqual(resp.status_code, 200)
        self.assertIn(CONTENT_TYPE_ARROW_STREAM, resp.headers["content-type"])
        table = read_arrow_stream(resp.content)
        self.assertEqual(table.num_rows, 100)

    def test_stream_scalar(self):
        resp = self._call_stream("stream:scalar")
        self.assertEqual(resp.status_code, 200)
        result = deserialize_pickle(resp.content)
        self.assertEqual(result, 42)


# -- CLI tests (ygg entry point) ------------------------------------------

class TestCLI(unittest.TestCase):
    def _run_ygg(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "yggdrasil.cli.main", *args],
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_help(self):
        result = self._run_ygg("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("bot", result.stdout)
        self.assertIn("genie", result.stdout)

    def test_bot_help(self):
        result = self._run_ygg("bot", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("serve", result.stdout)
        self.assertIn("run", result.stdout)

    def test_bot_serve_help(self):
        result = self._run_ygg("bot", "serve", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--host", result.stdout)
        self.assertIn("--port", result.stdout)

    def test_bot_run_help(self):
        result = self._run_ygg("bot", "run", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("func", result.stdout)
        self.assertIn("--url", result.stdout)
        self.assertIn("--kwarg", result.stdout)

    def test_no_args_shows_help(self):
        result = self._run_ygg()
        self.assertEqual(result.returncode, 0)
        self.assertIn("bot", result.stdout)


if __name__ == "__main__":
    unittest.main()
