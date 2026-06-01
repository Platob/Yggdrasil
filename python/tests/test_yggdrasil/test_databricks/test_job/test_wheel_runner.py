"""Unit tests for the wheel build/upload (no live cluster)."""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.databricks.job import wheel


# --------------------------------------------------------------------------- #
# shared: synthesize + build a dep-free wheel whose console script prints a marker
# --------------------------------------------------------------------------- #
_MARKER = "HELLO_FROM_WHEEL"


def _whl_child(name: str, full_path: str) -> MagicMock:
    """A listing child standing in for a workspace wheel path."""
    child = MagicMock()
    child.name = name
    child.full_path.return_value = full_path
    return child


def _build_fake_wheel(tmp_path, monkeypatch) -> Path:
    """Really ``pip wheel`` a tiny dep-free project carrying a ``fake-job``
    console script that prints :data:`_MARKER`. Skips if no build backend."""
    monkeypatch.setenv("PIP_NO_BUILD_ISOLATION", "1")   # use installed setuptools/wheel
    proj = tmp_path / "proj"
    pkg = proj / "fakepkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        f"def hi():\n    print({_MARKER!r})\n    return 0\n"
    )
    (proj / "pyproject.toml").write_text(
        wheel._render_pyproject(
            "fakepkg", "0.0.1", "fakepkg",
            deps=[],                                   # dep-free → offline build
            scripts={"fake-job": "fakepkg:hi"},
        )
    )
    try:
        with patch("yggdrasil.databricks.job.wheel.synthesize_project", return_value=proj):
            wheels = wheel.build_wheel("fakepkg")
    except subprocess.CalledProcessError as exc:        # no build backend here
        pytest.skip(f"pip wheel unavailable in this env: {exc}")
    return next(w for w in wheels if w.name.startswith("fakepkg-0.0.1-"))


# --------------------------------------------------------------------------- #
# wheel — build + upload + ensure
# --------------------------------------------------------------------------- #
class TestWheel:
    def test_project_dependencies_flattens_requested_extras(self):
        reqs = [
            "pyarrow>=10",                              # base
            'pandas>=2; extra == "data"',              # requested extra → keep
            'pyspark>=3; extra == "bigdata"',          # other extra → drop
            'tomli; python_version < "3.11"',          # non-extra marker → keep
        ]
        with patch("yggdrasil.databricks.job.wheel.ilmd.requires", return_value=reqs):
            deps = wheel._project_dependencies("ygg", {"data"})
        assert "pyarrow>=10" in deps
        assert "pandas>=2" in deps
        assert all("pyspark" not in d for d in deps)
        assert any("tomli" in d for d in deps)

    def test_render_pyproject_has_scripts_and_deps(self):
        text = wheel._render_pyproject(
            "ygg", "1.2.3", "yggdrasil",
            ["pyarrow>=10"],
            {"ygg": "yggdrasil.cli.main:main"},
        )
        assert 'name = "ygg"' in text and 'version = "1.2.3"' in text
        assert '"pyarrow>=10",' in text
        assert 'ygg = "yggdrasil.cli.main:main"' in text
        assert 'include = ["yggdrasil*"]' in text

    def test_synthesize_project_copies_package_and_writes_pyproject(self, tmp_path):
        # point at a fake on-disk package; real ygg metadata fills the pyproject
        module = MagicMock()
        module.__file__ = str(tmp_path / "yggdrasil" / "__init__.py")
        (tmp_path / "yggdrasil").mkdir()
        (tmp_path / "yggdrasil" / "__init__.py").write_text("# pkg\n")
        out = tmp_path / "synth"
        with patch("yggdrasil.databricks.job.wheel.importlib.import_module", return_value=module), \
             patch("yggdrasil.databricks.job.wheel.distribution_for", return_value="ygg"):
            project = wheel.synthesize_project("yggdrasil", dest_dir=out)
        assert (project / "yggdrasil" / "__init__.py").exists()    # live files copied
        py = (project / "pyproject.toml").read_text()
        assert 'name = "ygg"' in py and "[project.scripts]" in py
        assert 'include = ["yggdrasil*"]' in py

    def test_upload_writes_to_workspace_dir(self, tmp_path):
        client = MagicMock()
        wf = tmp_path / "ygg-1.2.3-py3-none-any.whl"
        wf.write_bytes(b"WHEELBYTES")
        path = MagicMock()
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP:
            DP.from_.return_value = path
            dest = wheel.upload_wheel(client, wf)
        assert dest == "/Workspace/Shared/.ygg/whl/ygg-1.2.3-py3-none-any.whl"
        DP.from_.assert_called_once_with(dest, client=client)
        path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        path.write_bytes.assert_called_once_with(b"WHEELBYTES")

    def test_build_wheel_synthesizes_then_pip_wheels(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "ygg-1.0-py3-none-any.whl").write_bytes(b"")
        (out / "pyarrow-1-py3-none-any.whl").write_bytes(b"")
        with patch("yggdrasil.databricks.job.wheel.synthesize_project", return_value=Path("/synth")) as sp, \
             patch("yggdrasil.databricks.job.wheel.subprocess.run") as run, \
             patch("yggdrasil.databricks.job.wheel.tempfile.mkdtemp", return_value=str(out)):
            wheels = wheel.build_wheel("yggdrasil", extras=["databricks"], requirements=["databricks-sdk"])
        sp.assert_called_once_with("yggdrasil", extras=["databricks"])
        cmd = run.call_args.args[0]
        assert "wheel" in cmd and "/synth" in cmd and "databricks-sdk" in cmd
        assert sorted(w.name for w in wheels) == ["pyarrow-1-py3-none-any.whl", "ygg-1.0-py3-none-any.whl"]

    def test_ensure_builds_with_deps_and_uploads_all(self):
        client = MagicMock()
        built = [Path("/tmp/ygg-1.0-py3-none-any.whl"), Path("/tmp/pyarrow-1-py3-none-any.whl")]
        with patch("yggdrasil.databricks.job.wheel.build_wheel", return_value=built) as bw, \
             patch("yggdrasil.databricks.job.wheel.upload_wheel", side_effect=lambda c, w, *, workspace_dir: f"{workspace_dir}/{w.name}"):
            dests = wheel.ensure_wheel(client, "yggdrasil", workspace_dir="/ws/job", extras=["databricks"])
        bw.assert_called_once_with("yggdrasil", extras=["databricks"], requirements=())
        assert dests == ["/ws/job/ygg-1.0-py3-none-any.whl", "/ws/job/pyarrow-1-py3-none-any.whl"]

    def test_import_packages_for_inverts_distribution_for(self):
        # ``ygg`` (pip/dist name) → its top-level import package ``yggdrasil``.
        assert wheel.import_packages_for("ygg") == ["yggdrasil"]
        assert wheel.distribution_for("yggdrasil") == "ygg"
        assert wheel.import_packages_for("no-such-dist-xyz") == []

    def test_synthesize_accepts_distribution_name(self, tmp_path):
        # ``"ygg"`` isn't importable (the import package is ``yggdrasil``);
        # synthesize_project must resolve it to the import package and build.
        module = MagicMock()
        module.__file__ = str(tmp_path / "yggdrasil" / "__init__.py")
        (tmp_path / "yggdrasil").mkdir()
        (tmp_path / "yggdrasil" / "__init__.py").write_text("# pkg\n")
        out = tmp_path / "synth"

        def _import(name):
            if name == "ygg":
                raise ModuleNotFoundError("No module named 'ygg'")
            return module

        with patch("yggdrasil.databricks.job.wheel.importlib.import_module", side_effect=_import), \
             patch("yggdrasil.databricks.job.wheel.import_packages_for", return_value=["yggdrasil"]):
            project = wheel.synthesize_project("ygg", dest_dir=out)
        assert (project / "yggdrasil" / "__init__.py").exists()
        py = (project / "pyproject.toml").read_text()
        assert 'name = "ygg"' in py and 'include = ["yggdrasil*"]' in py

    def test_ensure_ygg_wheel_builds_into_version_subdir_when_absent(self):
        # No deployed bundle for the current version → build into the
        # version-scoped subfolder of the shared wheel path.
        client = MagicMock()
        with patch("yggdrasil.databricks.job.wheel.ilmd.version", return_value="9.9"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels", return_value=[]) as dw, \
             patch("yggdrasil.databricks.job.wheel.ensure_wheel", return_value=["/ws/x.whl"]) as ew:
            out = wheel.ensure_ygg_wheel(client, workspace_dir="/ws/job")
        dw.assert_called_once_with(client, "ygg", "9.9", workspace_dir="/ws/job/9.9")
        ew.assert_called_once_with(
            client, "ygg",
            workspace_dir="/ws/job/9.9",
            extras=("databricks",),
            requirements=("databricks-sdk",),
        )
        assert out == ["/ws/x.whl"]

    def test_ensure_ygg_wheel_reuses_deployed_bundle(self):
        # A bundle already deployed for the current version is reused — no build.
        client = MagicMock()
        deployed = ["/ws/job/9.9/ygg-9.9-py3-none-any.whl",
                    "/ws/job/9.9/pyarrow-1-py3-none-any.whl"]
        with patch("yggdrasil.databricks.job.wheel.ilmd.version", return_value="9.9"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels", return_value=deployed), \
             patch("yggdrasil.databricks.job.wheel.ensure_wheel") as ew:
            out = wheel.ensure_ygg_wheel(client, workspace_dir="/ws/job")
        ew.assert_not_called()
        assert out == deployed

    def test_ensure_ygg_wheel_rebuild_skips_reuse(self):
        client = MagicMock()
        with patch("yggdrasil.databricks.job.wheel.ilmd.version", return_value="9.9"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels") as dw, \
             patch("yggdrasil.databricks.job.wheel.ensure_wheel", return_value=["/ws/x.whl"]) as ew:
            out = wheel.ensure_ygg_wheel(client, workspace_dir="/ws/job", rebuild=True)
        dw.assert_not_called()                 # rebuild bypasses the reuse probe
        ew.assert_called_once_with(
            client, "ygg",
            workspace_dir="/ws/job/9.9",
            extras=("databricks",),
            requirements=("databricks-sdk",),
        )
        assert out == ["/ws/x.whl"]

    def test_deployed_wheels_present_and_partial(self):
        # Present: the dist wheel for the version is there → all wheels returned.
        client = MagicMock()
        folder = MagicMock()
        folder.exists.return_value = True
        folder.iterdir.return_value = [
            _whl_child("ygg-9.9-py3-none-any.whl", "/ws/9.9/ygg-9.9-py3-none-any.whl"),
            _whl_child("pyarrow-1-py3-none-any.whl", "/ws/9.9/pyarrow-1-py3-none-any.whl"),
            _whl_child("notes.txt", "/ws/9.9/notes.txt"),
        ]
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", return_value=folder):
            out = wheel.deployed_wheels(client, "ygg", "9.9", workspace_dir="/ws/9.9")
        assert out == ["/ws/9.9/ygg-9.9-py3-none-any.whl", "/ws/9.9/pyarrow-1-py3-none-any.whl"]

        # Partial: dist wheel for the version missing (only deps) → treated as absent.
        folder.iterdir.return_value = [
            _whl_child("pyarrow-1-py3-none-any.whl", "/ws/9.9/pyarrow-1-py3-none-any.whl"),
            _whl_child("ygg-9.8-py3-none-any.whl", "/ws/9.9/ygg-9.8-py3-none-any.whl"),
        ]
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", return_value=folder):
            assert wheel.deployed_wheels(client, "ygg", "9.9", workspace_dir="/ws/9.9") == []

    def test_deployed_wheels_missing_dir(self):
        client = MagicMock()
        folder = MagicMock()
        folder.exists.return_value = False
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", return_value=folder):
            assert wheel.deployed_wheels(client, "ygg", "9.9", workspace_dir="/ws/9.9") == []

    def test_build_wheel_produces_a_real_wheel(self, tmp_path, monkeypatch):
        """Isolated, offline end-to-end: a synthesized dep-free project is really
        ``pip wheel``-ed into a wheel carrying the entry point."""
        built = _build_fake_wheel(tmp_path, monkeypatch)
        assert built.name.startswith("fakepkg-0.0.1-") and built.name.endswith(".whl")
        # the wheel actually declares the entry point we synthesized
        with zipfile.ZipFile(built) as zf:
            eps = next(n for n in zf.namelist() if n.endswith("entry_points.txt"))
            assert "fake-job = fakepkg:hi" in zf.read(eps).decode()

    @pytest.mark.skipif(os.name != "posix", reason="venv script layout is posix-specific")
    def test_built_wheel_installs_and_runs_its_entry_point(self, tmp_path, monkeypatch):
        """Full pipeline: synthesize → build → install the wheel into a fresh
        offline venv → run its console script → it prints the marker."""
        built = _build_fake_wheel(tmp_path, monkeypatch)

        venv = tmp_path / "venv"
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
            pip = venv / "bin" / "pip"
            subprocess.run([str(pip), "install", "--no-index", str(built)], check=True)
        except subprocess.CalledProcessError as exc:        # offline / no venv support
            pytest.skip(f"cannot create/install into venv here: {exc}")

        fake_job = venv / "bin" / "fake-job"
        assert fake_job.exists(), "console script not installed by the wheel"
        out = subprocess.run([str(fake_job)], check=True, capture_output=True, text=True)
        assert _MARKER in out.stdout
