"""Live PyEnv install path integration test.

Runs only when ``YGGDRASIL_PYENV_INSTALL_TEST=1`` is set so default CI
doesn't pay the round-trip cost or risk mutating the host venv. The
test exercises the full install → import → uninstall cycle through
:meth:`PyEnv.runtime_import_module` and asserts that:

* The package lands in the **currently running** interpreter's
  site-packages (``sys.executable`` — the active venv), not in a
  sibling Python like ``/usr/bin/python``.
* The lazy-imports ``install=True`` opt-in actually triggers the
  PyEnv install path (and ``install=False`` raises cleanly when the
  package is missing).

Uses ``cowsay`` as the throwaway target — pure-Python, no compiled
ext modules, ~5 KB wheel — so the round-trip cost stays well under
the 30 s default pytest timeout even on a cold network.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import sysconfig
import unittest

import pytest

from yggdrasil.environ import PyEnv
from yggdrasil.lazy_imports import _LAZY_CACHE, _lazy_import


_GATE_ENV = "YGGDRASIL_PYENV_INSTALL_TEST"
_TARGET_PIP = "cowsay"
_TARGET_MODULE = "cowsay"


def _gated() -> bool:
    """Skip unless the env gate is explicitly set."""
    return os.environ.get(_GATE_ENV, "").strip() not in ("", "0", "false", "False")


def _is_installed(module_name: str) -> bool:
    """True iff *module_name* is importable in the active interpreter."""
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return False
    return spec is not None


def _site_packages_dir() -> str:
    """Path to the active interpreter's site-packages."""
    return sysconfig.get_paths()["purelib"]


def _force_uninstall(module_name: str, pip_name: str) -> None:
    """Best-effort uninstall via the active interpreter's pip.

    Routed through ``sys.executable -m pip`` directly (not PyEnv) so
    the cleanup is independent of the code under test — a regression
    in PyEnv mustn't leave the host venv polluted with ``cowsay``.
    Drops the cached module (and any submodules) from ``sys.modules``
    so the next ``importlib.import_module`` actually re-resolves the
    spec against the now-empty site-packages.
    """
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", pip_name],
        check=False,
        capture_output=True,
    )
    _LAZY_CACHE.pop(module_name, None)
    for cached in list(sys.modules):
        if cached == module_name or cached.startswith(module_name + "."):
            sys.modules.pop(cached, None)
    importlib.invalidate_caches()


@pytest.mark.pyenv_install_integration
@unittest.skipUnless(_gated(), f"set {_GATE_ENV}=1 to run PyEnv install tests")
class TestPyEnvInstallIntegration(unittest.TestCase):
    """End-to-end exercises of the PyEnv runtime install path."""

    def setUp(self) -> None:
        _force_uninstall(_TARGET_MODULE, _TARGET_PIP)
        self.assertFalse(
            _is_installed(_TARGET_MODULE),
            f"setUp invariant: {_TARGET_MODULE!r} must be absent before each test "
            "(uninstall failed?)",
        )

    def tearDown(self) -> None:
        _force_uninstall(_TARGET_MODULE, _TARGET_PIP)

    def test_runtime_import_module_installs_into_active_venv(self) -> None:
        """``PyEnv.runtime_import_module`` targets ``sys.executable``."""
        mod = PyEnv.runtime_import_module(
            module_name=_TARGET_MODULE,
            pip_name=_TARGET_PIP,
            install=True,
            warn=False,
            use_cache=False,
        )
        self.assertIsNotNone(mod)
        # Package file lives under the active interpreter's site-packages.
        mod_file = getattr(mod, "__file__", "") or ""
        self.assertTrue(
            mod_file.startswith(_site_packages_dir()),
            f"{_TARGET_MODULE} installed outside the active venv: "
            f"{mod_file!r} not under {_site_packages_dir()!r}",
        )

    def test_lazy_import_install_false_raises_when_missing(self) -> None:
        """``install=False`` must not trigger PyEnv install."""
        with self.assertRaises(ImportError):
            _lazy_import(_TARGET_MODULE, _TARGET_PIP, install=False)
        # Still missing — install=False didn't accidentally pull it in.
        self.assertFalse(_is_installed(_TARGET_MODULE))

    def test_lazy_import_install_true_installs_and_caches(self) -> None:
        """``install=True`` triggers PyEnv install and seeds the module cache."""
        mod = _lazy_import(_TARGET_MODULE, _TARGET_PIP, install=True)
        self.assertIsNotNone(mod)
        # Subsequent no-arg call must hit the cache (no second install attempt).
        self.assertIs(mod, _lazy_import(_TARGET_MODULE, _TARGET_PIP, install=False))

    def test_current_pyenv_points_at_sys_executable(self) -> None:
        """``PyEnv.current()`` anchors on the running interpreter.

        Sanity check — every install routes through ``python_path``,
        which must match ``sys.executable`` so the install lands in
        the same venv the test is reading from.
        """
        env = PyEnv.current()
        from pathlib import Path
        self.assertEqual(env.python_path.resolve(), Path(sys.executable).resolve())


if __name__ == "__main__":
    unittest.main()
