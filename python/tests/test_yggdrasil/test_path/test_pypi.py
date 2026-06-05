"""Unit tests for the generic :class:`PyPIPath` artefact index."""
from __future__ import annotations

import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path as _LocalPath

from yggdrasil.path.pypi import (
    PyPIPath,
    normalize_pep503_name,
    parse_wheel_filename,
)


class TestNormalizeAndParse(unittest.TestCase):
    def test_pep503_lowercases_and_collapses_separators(self):
        self.assertEqual(normalize_pep503_name("My_Package.Name"), "my-package-name")
        self.assertEqual(normalize_pep503_name("foo--bar"), "foo-bar")

    def test_parse_wheel_filename(self):
        dist, version = parse_wheel_filename(
            "my_pkg-1.2.3-py3-none-any.whl"
        )
        self.assertEqual(dist, "my_pkg")
        self.assertEqual(version, "1.2.3")

    def test_parse_non_wheel(self):
        self.assertEqual(parse_wheel_filename("not-a-wheel.zip"), (None, None))


class TestPyPIPathLocal(unittest.TestCase):
    """Exercise PyPIPath end-to-end against a LocalPath root."""

    def setUp(self) -> None:
        self.tmp = _LocalPath(tempfile.mkdtemp(prefix="ygg-pypi-test-"))
        self.root = self.tmp / "simple"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_pkg(self) -> _LocalPath:
        """Create a tiny installable Python package on disk."""
        pkg_root = self.tmp / "src" / "tiny_pkg"
        pkg_root.mkdir(parents=True)
        (pkg_root.parent / "pyproject.toml").write_text(textwrap.dedent("""
            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"

            [project]
            name = "tiny_pkg"
            version = "0.4.2"

            [tool.setuptools.packages.find]
            where = ["."]
        """))
        (pkg_root / "__init__.py").write_text("VALUE = 1\n")
        return pkg_root.parent

    def test_publish_uploads_wheel_and_index(self):
        pkg_root = self._make_pkg()
        pypi = PyPIPath(str(self.root))

        published = pypi.publish("tiny_pkg", source_path=str(pkg_root))

        # Wheel landed under <root>/<normalized>/<wheel>.whl
        full = published.full_path()
        self.assertIn("simple/tiny-pkg/", full)
        self.assertTrue(full.endswith(".whl"))

        # index.html exists and lists the wheel
        index = (self.root / "tiny-pkg" / "index.html")
        body = index.read_bytes().decode()
        self.assertIn("Links for tiny_pkg", body)
        self.assertIn(_LocalPath(full).name, body)

    def test_publish_is_idempotent_on_wheel_upload(self):
        """Re-publishing the same source skips re-uploading the wheel."""
        pkg_root = self._make_pkg()
        pypi = PyPIPath(str(self.root))

        first = pypi.publish("tiny_pkg", source_path=str(pkg_root))
        wheel_target = self.root / "tiny-pkg" / _LocalPath(first.full_path()).name

        # Mark the wheel write time and re-publish — the wheel byte
        # stamp must stay constant; only the index page may be
        # re-rendered (it depends on the directory listing).
        first_wheel_bytes = wheel_target.read_bytes()
        original_size = len(first_wheel_bytes)

        pypi.publish("tiny_pkg", source_path=str(pkg_root))

        # Wheel preserved verbatim (no re-upload of fresh bytes).
        self.assertEqual(wheel_target.read_bytes(), first_wheel_bytes)
        self.assertEqual(len(wheel_target.read_bytes()), original_size)

    def test_import_module_installs_and_returns_package(self):
        import importlib
        import sys

        pkg_root = self._make_pkg()
        pypi = PyPIPath(str(self.root))
        pypi.publish("tiny_pkg", source_path=str(pkg_root))

        # Ensure the package isn't in sys.modules / sys.path already
        sys.modules.pop("tiny_pkg", None)

        # install=False is the new default; pass True to exercise the
        # full pip-install + import roundtrip on the local wheel.
        mod = pypi.import_module("tiny_pkg", install=True)
        try:
            self.assertEqual(getattr(mod, "VALUE", None), 1)
        finally:
            sys.modules.pop("tiny_pkg", None)
            importlib.invalidate_caches()

    def test_import_module_unknown_raises(self):
        pypi = PyPIPath(str(self.root))
        with self.assertRaises(ModuleNotFoundError):
            pypi.import_module("nonexistent_pkg")

    def test_publish_archive_uses_upload_module(self):
        pkg_root = self._make_pkg() / "tiny_pkg"
        pypi = PyPIPath(str(self.root))

        target = pypi.publish_archive(
            str(pkg_root), name="tiny_pkg", version="0.4.2",
        )
        full = target.full_path()
        self.assertTrue(full.endswith("tiny-pkg-0.4.2.zip"))
        # Index also updated.
        index = (self.root / "tiny-pkg" / "index.html")
        self.assertIn("tiny-pkg-0.4.2.zip", index.read_bytes().decode())


if __name__ == "__main__":
    unittest.main()
