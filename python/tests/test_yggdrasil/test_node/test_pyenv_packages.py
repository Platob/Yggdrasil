"""Tests for PyEnv installed-library listing + its TTL cache.

The node UI shows each PyEnv's resolved interpreter version and the
libraries actually installed in its venv. Collecting that list shells
out to ``uv pip list`` / ``pip list``, so the service serves it from a
TTL cache — repeated polls must not re-spawn the subprocess. These tests
pin that contract (cache hit, ``refresh``, invalidation-on-install) plus
a real listing against a live interpreter, with no pytest-asyncio
dependency (each coroutine runs under ``asyncio.run``).
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

from yggdrasil.node.api.schemas.pyenv import PyEnvEntry, PyEnvPackagesResponse
from yggdrasil.node.api.services.pyenv import PyEnvService
from yggdrasil.node.config import Settings
from yggdrasil.node.exceptions import NotFoundError


def _make_settings(tmp_home: Path, **overrides) -> Settings:
    base = dict(
        node_id="test-node",
        node_home=tmp_home,
        front_home=tmp_home,
        pyenv_packages_cache_ttl=60.0,
    )
    base.update(overrides)
    return Settings(**base)


def _register_entry(svc: PyEnvService, env_id: int, path: Path) -> PyEnvEntry:
    """Drop a ready PyEnvEntry into the service without building a venv."""
    entry = PyEnvEntry(
        id=env_id, name=f"env{env_id}", python_version="3.11",
        dependencies=[], path=str(path), status="ready",
        created_at="t0", updated_at="t0",
    )
    svc._envs.set(env_id, entry)
    svc._name_to_id[entry.name] = env_id
    return entry


def _fake_interpreter_dir(tmp: Path) -> Path:
    """A dir whose ``bin/python`` points at the live interpreter, so
    ``_collect_packages`` resolves a real version + a real package list
    without the cost of building an isolated venv."""
    bindir = tmp / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    link = bindir / "python"
    try:
        os.symlink(sys.executable, link)
    except (OSError, NotImplementedError):
        # Platforms without symlink: a tiny exec shim.
        link.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        link.chmod(0o755)
    return tmp


class TestPyEnvPackagesCache(unittest.TestCase):
    def test_listing_resolves_version_and_packages(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as envd:
            svc = PyEnvService(_make_settings(Path(home)))
            _register_entry(svc, 1, _fake_interpreter_dir(Path(envd)))

            res = asyncio.run(svc.packages(1))
            self.assertIsInstance(res, PyEnvPackagesResponse)
            # Resolved to the live interpreter's full version, not "3.11".
            expected = ".".join(str(p) for p in sys.version_info[:3])
            self.assertEqual(res.python_version, expected)
            self.assertIsNone(res.error)
            self.assertGreater(res.package_count, 0)
            self.assertEqual(res.package_count, len(res.packages))
            # Sorted, case-insensitive by name.
            names = [p.name.lower() for p in res.packages]
            self.assertEqual(names, sorted(names))

    def test_second_call_is_cache_hit(self):
        with tempfile.TemporaryDirectory() as home:
            svc = PyEnvService(_make_settings(Path(home)))
            _register_entry(svc, 1, Path(home))

            calls = {"n": 0}
            stub = PyEnvPackagesResponse(
                env_id=1, name="env1", python_version="3.11.9",
                package_count=0, packages=[], cached_at="t",
            )

            def _collect(env_id, entry):
                calls["n"] += 1
                return stub

            svc._collect_packages = _collect

            asyncio.run(svc.packages(1))
            asyncio.run(svc.packages(1))
            self.assertEqual(calls["n"], 1, "second call must hit the cache")

            # refresh=True bypasses the cache.
            asyncio.run(svc.packages(1, refresh=True))
            self.assertEqual(calls["n"], 2)

    def test_install_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as home:
            svc = PyEnvService(_make_settings(Path(home)))
            entry = _register_entry(svc, 7, Path(home))

            calls = {"n": 0}

            def _collect(env_id, e):
                calls["n"] += 1
                return PyEnvPackagesResponse(
                    env_id=env_id, name=e.name, python_version="3.11.9",
                    package_count=calls["n"], packages=[], cached_at="t",
                )

            svc._collect_packages = _collect

            first = asyncio.run(svc.packages(7))
            self.assertEqual(first.package_count, 1)

            # Simulate an install: it pops the cache, so the next read
            # re-collects (no real pip needed — pip_install no-ops on the
            # empty package list path is avoided by stubbing the runner).
            svc._pip_install = lambda *a, **k: None
            svc._install_packages(7, ["nothing"])

            second = asyncio.run(svc.packages(7))
            self.assertEqual(second.package_count, 2, "cache must be invalidated")

    def test_packages_on_missing_env_raises(self):
        with tempfile.TemporaryDirectory() as home:
            svc = PyEnvService(_make_settings(Path(home)))
            with self.assertRaises(NotFoundError):
                asyncio.run(svc.packages(999))

    def test_missing_interpreter_reports_error_not_crash(self):
        with tempfile.TemporaryDirectory() as home:
            svc = PyEnvService(_make_settings(Path(home)))
            _register_entry(svc, 2, Path(home) / "does-not-exist")
            res = asyncio.run(svc.packages(2))
            self.assertEqual(res.package_count, 0)
            self.assertIsNotNone(res.error)


if __name__ == "__main__":
    unittest.main()
