"""Tests for the abstract :class:`Path` ``upload_module`` /
``import_module`` surface and the shared :mod:`yggdrasil.io.path._module_pack`
helper.

These exercise the generic flow on :class:`LocalPath` so backend
mocks aren't needed; the WorkspacePath specialization picks up
its own dedicated tests in ``test_databricks``.
"""
from __future__ import annotations

import os
import sys
import textwrap
import zipfile

import pytest

from yggdrasil.io.path import LocalPath
from yggdrasil.io.path._module_pack import (
    build_module_archive,
    resolve_module_root,
)


@pytest.fixture
def demo_package(tmp_path):
    """Materialize a tiny package on disk and return its directory."""
    pkg = tmp_path / "src" / "demo_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("VALUE = 7\n")
    (pkg / "math.py").write_text("def add(a, b): return a + b\n")
    # The junk we expect to skip on archive build.
    cache = pkg / "__pycache__"
    cache.mkdir()
    (cache / "junk.pyc").write_bytes(b"\x00")
    dist = pkg / "demo_pkg.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: demo_pkg\n")
    return pkg


class TestResolveModuleRoot:

    def test_resolve_pathlike(self, demo_package) -> None:
        assert resolve_module_root(demo_package).resolve() == demo_package.resolve()

    def test_resolve_path_string(self, demo_package) -> None:
        assert resolve_module_root(str(demo_package)).resolve() == demo_package.resolve()

    def test_resolve_known_module(self) -> None:
        root = resolve_module_root("yggdrasil")
        assert root.exists()
        assert root.name == "yggdrasil"

    def test_resolve_missing_path(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_module_root(tmp_path / "does-not-exist")


class TestBuildModuleArchive:

    def test_emits_deflated_zip(self, demo_package, tmp_path) -> None:
        out = build_module_archive(demo_package, dest=tmp_path)
        assert out.suffix == ".zip"
        assert out.name == "demo_pkg.zip"
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        assert "demo_pkg/__init__.py" in names
        assert "demo_pkg/math.py" in names

    def test_skips_pycache_and_distinfo(self, demo_package, tmp_path) -> None:
        out = build_module_archive(demo_package, dest=tmp_path)
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert not any("__pycache__" in n for n in names)
        assert not any(".dist-info" in n for n in names)

    def test_dest_directory_keeps_default_name(self, demo_package, tmp_path) -> None:
        out_dir = tmp_path / "stage"
        out_dir.mkdir()
        out = build_module_archive(demo_package, dest=out_dir)
        assert out == out_dir / "demo_pkg.zip"

    def test_existing_archive_passthrough(self, demo_package, tmp_path) -> None:
        first = build_module_archive(demo_package, dest=tmp_path)
        dest = tmp_path / "copy"
        dest.mkdir()
        second = build_module_archive(first, dest=dest)
        assert second == dest / first.name
        assert second.read_bytes() == first.read_bytes()


class TestPathUploadModule:

    def test_upload_to_directory(self, demo_package, tmp_path) -> None:
        target_dir = LocalPath(tmp_path / "remote")
        target_dir.mkdir()

        archive = target_dir.upload_module(demo_package)
        assert isinstance(archive, LocalPath)
        assert archive.full_path() == str(tmp_path / "remote" / "demo_pkg.zip")
        assert archive.exists()
        assert archive.size > 0
        with zipfile.ZipFile(archive.full_path()) as zf:
            assert "demo_pkg/__init__.py" in zf.namelist()

    def test_upload_to_zip_path(self, demo_package, tmp_path) -> None:
        target = LocalPath(tmp_path / "out" / "custom.zip")
        archive = target.upload_module(demo_package)
        assert archive.full_path() == str(tmp_path / "out" / "custom.zip")
        with zipfile.ZipFile(archive.full_path()) as zf:
            assert "demo_pkg/__init__.py" in zf.namelist()

    def test_upload_no_overwrite_raises(self, demo_package, tmp_path) -> None:
        target_dir = LocalPath(tmp_path / "remote")
        target_dir.mkdir()
        target_dir.upload_module(demo_package)
        with pytest.raises(FileExistsError):
            target_dir.upload_module(demo_package, overwrite=False)

    def test_upload_custom_name(self, demo_package, tmp_path) -> None:
        target_dir = LocalPath(tmp_path / "remote")
        target_dir.mkdir()
        archive = target_dir.upload_module(demo_package, name="renamed.zip")
        assert archive.name == "renamed.zip"


class TestPathImportModule:

    def test_round_trip(self, demo_package, tmp_path) -> None:
        target_dir = LocalPath(tmp_path / "remote")
        target_dir.mkdir()
        archive = target_dir.upload_module(demo_package)

        cache = tmp_path / "cache"
        # Make sure stale imports don't leak between tests.
        for mod in list(sys.modules):
            if mod == "demo_pkg" or mod.startswith("demo_pkg."):
                del sys.modules[mod]

        mod = archive.import_module("demo_pkg", cache_dir=cache)
        try:
            assert mod.VALUE == 7
            from importlib import import_module
            math_mod = import_module("demo_pkg.math")
            assert math_mod.add(2, 3) == 5
        finally:
            for name in list(sys.modules):
                if name == "demo_pkg" or name.startswith("demo_pkg."):
                    del sys.modules[name]
            # Drop the cache entry from sys.path so it doesn't leak.
            archive_local = str(cache / archive.name)
            if archive_local in sys.path:
                sys.path.remove(archive_local)

    def test_import_rejects_non_archive(self, tmp_path) -> None:
        plain = LocalPath(tmp_path / "data.txt")
        plain.write_bytes(b"not a zip")
        with pytest.raises(ValueError, match="does not look like a Python archive"):
            plain.import_module("anything")
