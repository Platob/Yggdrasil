# tests/test_modules_realworld.py
#
# Real-world tests: no monkeypatching/mocking of module internals.
# We only modify environment variables, and we do it via a tiny context manager.
#
# Run:
#   pytest -q

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Iterator, Optional

import yggdrasil.pyutils.modules as m
import pytest

from yggdrasil.pyutils.modules import PipIndexSettings


@contextlib.contextmanager
def temp_environ(**updates: Optional[str]) -> Iterator[None]:
    """
    Temporarily set/unset environment variables.
    Pass value=None to unset.
    """
    old = {}
    for k in updates:
        old[k] = os.environ.get(k)

    try:
        for k, v in updates.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, prev in old.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


@contextlib.contextmanager
def temp_sys_path(p: Path) -> Iterator[None]:
    s = str(p)
    had = s in sys.path
    if not had:
        sys.path.insert(0, s)
    try:
        yield
    finally:
        if not had:
            # remove first occurrence
            try:
                sys.path.remove(s)
            except ValueError:
                pass


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# test\n", encoding="utf-8")
    return p


# -------------------------
# module_name_to_project_name
# -------------------------

def test_module_name_to_project_name_alias_and_default():
    assert m.module_name_to_project_name("yggdrasil") == "ygg"
    assert m.module_name_to_project_name("jwt") == "PyJWT"
    assert m.module_name_to_project_name("numpy") == "numpy"


# -------------------------
# resolve_local_lib_path
# -------------------------

def test_resolve_local_lib_path_topmost_package(tmp_path: Path):
    pkgA = tmp_path / "pkgA"
    pkgB = pkgA / "pkgB"
    _touch(pkgA / "__init__.py")
    _touch(pkgB / "__init__.py")
    mod = _touch(pkgB / "mod.py")

    assert m.resolve_local_lib_path(pkgB / "__init__.py") == pkgA.resolve()
    assert m.resolve_local_lib_path(mod) == pkgA.resolve()
    assert m.resolve_local_lib_path(pkgB) == pkgA.resolve()


def test_resolve_local_lib_path_non_package_file(tmp_path: Path):
    f = _touch(tmp_path / "plain.py")
    assert m.resolve_local_lib_path(f) == f.resolve()


def test_resolve_local_lib_path_non_package_dir(tmp_path: Path):
    d = tmp_path / "data"
    d.mkdir()
    assert m.resolve_local_lib_path(d) == d.resolve()


def test_resolve_local_lib_path_module_import(tmp_path: Path):
    # Create a temp importable package and ensure module-name path resolution works IRL.
    pkg = tmp_path / "mypkg"
    sub = pkg / "sub"
    _touch(pkg / "__init__.py")
    _touch(sub / "__init__.py")
    _touch(sub / "m.py")

    with temp_sys_path(tmp_path):
        # import should work now
        p = m.resolve_local_lib_path("mypkg.sub.m")
        assert p == pkg.resolve()


def test_resolve_local_lib_path_module_object_without___file__raises():
    import types

    mod = types.ModuleType("no_file")
    with pytest.raises(ValueError):
        m.resolve_local_lib_path(mod)


# -------------------------
# _req_project_name (behavior depends on packaging availability)
# -------------------------

def _packaging_available() -> bool:
    try:
        from packaging.requirements import Requirement  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.parametrize(
    "line,expected_when_packaging,expected_when_no_packaging",
    [
        ("requests (>=2.0)", "requests", "requests"),
        ("PyJWT[crypto] (>=2.0)", "PyJWT", "PyJWT"),
        ("cryptography>=41 ; python_version>='3.10'", "cryptography", "cryptography"),
        ("  numpy==1.26.0  ", "numpy", "numpy"),
        # With packaging: invalid => None. Without packaging: regex grabs first token.
        ("invalid req line!!!", None, "invalid"),
        ("bad%%%line", None, "bad"),
        ("", None, None),
    ],
)
def test_req_project_name(line: str, expected_when_packaging, expected_when_no_packaging):
    expected = expected_when_packaging if _packaging_available() else expected_when_no_packaging
    assert m._req_project_name(line) == expected


# -------------------------
# _run_pip + get_pip_index_settings
# -------------------------

def test_run_pip_version_smoke():
    rc, out, err = m._run_pip("--version")
    assert rc == 0
    assert "pip" in out.lower()
    assert isinstance(err, str)


def test_get_pip_index_settings_returns_dataclass():
    # No env forcing; just check shape/types, since config varies by machine/CI image.
    with temp_environ(PIP_INDEX_URL=None, PIP_EXTRA_INDEX_URL=None):
        out = m.get_pip_index_settings()

    assert isinstance(out, m.PipIndexSettings)
    assert (out.index_url is None) or isinstance(out.index_url, str)
    assert isinstance(out.extra_index_urls, list)
    assert all(isinstance(x, str) for x in out.extra_index_urls)
    assert isinstance(out.sources, dict)
    assert "env" in out.sources and "config" in out.sources


def test_get_pip_index_settings_default():
    # No env forcing; just check shape/types, since config varies by machine/CI image.
    with temp_environ(PIP_INDEX_URL=None, PIP_EXTRA_INDEX_URL=None):
        out = PipIndexSettings.default_settings()

    assert isinstance(out, m.PipIndexSettings)
    assert (out.index_url is None) or isinstance(out.index_url, str)
    assert isinstance(out.extra_index_urls, list)
    assert all(isinstance(x, str) for x in out.extra_index_urls)
    assert isinstance(out.sources, dict)
    assert "env" in out.sources and "config" in out.sources


def test_get_pip_index_settings_env_override_is_deterministic():
    # Deterministic even if pip.conf exists somewhere.
    env_index = "https://example.invalid/simple"
    env_extra = "https://a.invalid/simple https://a.invalid/simple https://b.invalid/simple"

    with temp_environ(PIP_INDEX_URL=env_index, PIP_EXTRA_INDEX_URL=env_extra):
        out = m.get_pip_index_settings()

    assert out.index_url == env_index
    assert out.extra_index_urls == ["https://a.invalid/simple", "https://b.invalid/simple"]
    assert out.sources["env"]["PIP_INDEX_URL"] == env_index
    assert out.sources["env"]["PIP_EXTRA_INDEX_URL"] == env_extra


# -------------------------
# module_dependencies (real installed dists; assert invariants, not exact deps)
# -------------------------

@pytest.mark.skipif(m.ilm is None, reason="importlib.metadata not available")
def test_module_dependencies_smoke_for_pip_or_skip():
    # In some weird python builds, pip may be absent. If so, skip cleanly.
    try:
        deps = m.module_dependencies("pip")
    except Exception as e:
        pytest.skip(f"module_dependencies('pip') not supported here: {e}")

    assert isinstance(deps, list)
    for d in deps:
        assert isinstance(d.project, str) and d.project
        assert isinstance(d.requirement, str) and d.requirement
        assert isinstance(d.installed, bool)
        assert (d.version is None) or isinstance(d.version, str)
        assert (d.dist_root is None) or isinstance(d.dist_root, Path)
        assert (d.metadata_path is None) or isinstance(d.metadata_path, Path)


def test_module_dependencies_nonexistent_module_raises():
    with pytest.raises(ModuleNotFoundError):
        m.module_dependencies("definitely_not_a_real_module_name____zzz")
