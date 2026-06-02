"""Live call-efficiency + open/non-open coverage for :class:`VolumePath`.

Runs against a real Unity Catalog volume (no mocks — the "prefer real calls"
contract), exercising both **non-opened** path operations (mkdir, exists, size /
stat, is_dir / is_file, ls) and **opened** IO (``open`` rb / wb / ab, read,
write, round-trip), and asserting the **number of backend calls** each makes so
regressions in chattiness are caught. The call counter wraps the shared
:class:`HTTPSession.fetch` — the single transport every Files-API op funnels
through.

Skipped wholesale unless ``DATABRICKS_HOST`` is set. The volume is read from
``DATABRICKS_INTEGRATION_CATALOG`` / ``_SCHEMA`` / ``_VOLUME`` (default
``trading`` / ``unittest`` / ``tmp``); a permission error degrades to a skip.
"""
from __future__ import annotations

import os
import secrets
import unittest
from contextlib import contextmanager

from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import PermissionDenied

from .. import DatabricksIntegrationCase


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class TestVolumePathCallsIntegration(DatabricksIntegrationCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog = _env("DATABRICKS_INTEGRATION_CATALOG", "trading")
        cls.schema = _env("DATABRICKS_INTEGRATION_SCHEMA", "unittest")
        cls.volume = _env("DATABRICKS_INTEGRATION_VOLUME", "tmp")
        base = (
            f"/Volumes/{cls.catalog}/{cls.schema}/{cls.volume}"
            f"/_ygg_calls_{secrets.token_hex(4)}"
        )
        cls.base = cls.client.path(base)
        try:
            cls.base.mkdir(parents=True, exist_ok=True)
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(f"cannot write to {base}: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.base.remove(recursive=True)
        except Exception:
            pass
        super().tearDownClass()

    @contextmanager
    def _count(self):
        """Count HTTPSession.fetch calls by method for the wrapped block."""
        import collections
        from yggdrasil.http_.session import HTTPSession

        calls: collections.Counter = collections.Counter()
        orig = HTTPSession.fetch

        def wrapper(s, method, url, **k):
            calls[method] += 1
            return orig(s, method, url, **k)

        HTTPSession.fetch = wrapper
        try:
            yield calls
        finally:
            HTTPSession.fetch = orig

    def _fresh(self, name: str):
        # A path handle with no warmed caches, so call counts reflect the op.
        p = self.base / name
        p.invalidate_singleton()
        return p

    # ---- non-opened path operations ----------------------------------
    def test_write_bytes_is_single_overwrite_put(self):
        p = self._fresh("a.bin")
        with self._count() as calls:
            p.write_bytes(b"hello world")
        # whole-file write → exactly one PUT, no read-modify-write
        self.assertEqual(calls.get("PUT"), 1, dict(calls))
        self.assertEqual(calls.get("GET", 0), 0, dict(calls))
        # round-trip correctness from a cold handle
        self.assertEqual(bytes(self._fresh("a.bin").read_bytes()), b"hello world")

    def test_metadata_after_write_is_cached(self):
        p = self._fresh("b.bin")
        p.write_bytes(b"x" * 32)
        # exists / size / is_dir read the stat the write seeded — no new calls
        with self._count() as calls:
            self.assertTrue(p.exists())
            self.assertEqual(p.size, 32)
            self.assertFalse(p.is_dir())
            self.assertTrue(p.is_file())
        self.assertEqual(sum(calls.values()), 0, dict(calls))

    def test_stat_on_cold_handle_is_one_call(self):
        self._fresh("c.bin").write_bytes(b"data!")
        p = self._fresh("c.bin")
        with self._count() as calls:
            self.assertEqual(p.size, 5)
        self.assertLessEqual(sum(calls.values()), 1, dict(calls))

    def test_ls_is_single_listing(self):
        self._fresh("d1.bin").write_bytes(b"1")
        self._fresh("d2.bin").write_bytes(b"2")
        with self._count() as calls:
            names = sorted(c.name for c in self.base.ls())
        self.assertIn("d1.bin", names)
        self.assertEqual(calls.get("GET"), 1, dict(calls))

    # ---- opened IO ---------------------------------------------------
    def test_open_write_then_read_roundtrip(self):
        p = self._fresh("e.bin")
        with p.open("wb") as io:
            io.write(b"opened-write-payload")
        cold = self._fresh("e.bin")
        with cold.open("rb") as io:
            self.assertEqual(io.read(), b"opened-write-payload")

    def test_open_append(self):
        p = self._fresh("f.bin")
        p.write_bytes(b"head-")
        with p.open("ab") as io:
            io.write(b"tail")
        self.assertEqual(bytes(self._fresh("f.bin").read_bytes()), b"head-tail")
