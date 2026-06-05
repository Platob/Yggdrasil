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
        assert dest == "/Workspace/Shared/pypi/ygg-1.2.3-py3-none-any.whl"
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
        assert "--no-deps" not in cmd                 # full build by default
        assert sorted(w.name for w in wheels) == ["pyarrow-1-py3-none-any.whl", "ygg-1.0-py3-none-any.whl"]

    def test_build_wheel_no_deps_uses_uv(self):
        out = Path("/does-not-matter")
        with patch("yggdrasil.databricks.job.wheel.synthesize_project", return_value=Path("/synth")), \
             patch("yggdrasil.databricks.job.wheel.subprocess.run") as run, \
             patch("yggdrasil.databricks.job.wheel.tempfile.mkdtemp", return_value=str(out)), \
             patch("pathlib.Path.glob", return_value=[Path("/synth/ygg-1.0-py3-none-any.whl")]):
            wheel.build_wheel("yggdrasil", no_deps=True)
        cmd = run.call_args.args[0]
        assert cmd[:3] == ["uv", "build", "--wheel"]    # uv, not pip
        assert "/synth" in cmd

    def test_build_wheel_no_deps_falls_back_to_pip_without_uv(self):
        out = Path("/does-not-matter")
        calls = []

        def _run(cmd, **kw):
            calls.append(cmd)
            if cmd[:2] == ["uv", "build"]:
                raise FileNotFoundError("uv")           # uv not on PATH
            return None

        with patch("yggdrasil.databricks.job.wheel.synthesize_project", return_value=Path("/synth")), \
             patch("yggdrasil.databricks.job.wheel.subprocess.run", side_effect=_run), \
             patch("yggdrasil.databricks.job.wheel.tempfile.mkdtemp", return_value=str(out)), \
             patch("pathlib.Path.glob", return_value=[Path("/synth/ygg-1.0-py3-none-any.whl")]):
            wheel.build_wheel("yggdrasil", no_deps=True)
        assert calls[0][:3] == ["uv", "build", "--wheel"]
        assert "wheel" in calls[1] and "--no-deps" in calls[1]   # pip fallback

    def test_ensure_builds_with_deps_and_uploads_all(self):
        client = MagicMock()
        built = [Path("/tmp/ygg-1.0-py3-none-any.whl"), Path("/tmp/pyarrow-1-py3-none-any.whl")]
        with patch("yggdrasil.databricks.job.wheel.build_wheel", return_value=built) as bw, \
             patch("yggdrasil.databricks.job.wheel.upload_wheel", side_effect=lambda c, w, *, workspace_dir: f"{workspace_dir}/{w.name}"):
            dests = wheel.ensure_wheel(client, "yggdrasil", workspace_dir="/ws/job", extras=["databricks"])
        bw.assert_called_once_with("yggdrasil", extras=["databricks"], requirements=(), no_deps=False)
        assert dests == ["/ws/job/ygg-1.0-py3-none-any.whl", "/ws/job/pyarrow-1-py3-none-any.whl"]

    def test_ensure_named_environment_writes_env_yaml_and_returns_path(self):
        client = MagicMock()
        path = MagicMock()
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP, \
             patch("yggdrasil.databricks.job.wheel.serverless_environment_version", return_value="5"):
            DP.from_.return_value = path
            dest = wheel.ensure_named_environment(
                client, "yellow",
                dependencies=["/ws/pypi/ygg-1.0-py3-none-any.whl", "pyarrow==1"],
            )
        assert dest == "/Workspace/Shared/environments/yellow.env.yaml"
        DP.from_.assert_called_once_with(dest, client=client)
        path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        body = path.write_text.call_args.args[0]
        assert body == (
            "environment_version: '5'\n"
            "dependencies:\n"
            "  - /ws/pypi/ygg-1.0-py3-none-any.whl\n"
            "  - pyarrow==1\n"
        )

    def test_ensure_cluster_requirements_writes_flat_requirements_txt(self):
        client = MagicMock()
        path = MagicMock()
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP:
            DP.from_.return_value = path
            dest = wheel.ensure_cluster_requirements(
                client, "yellow",
                dependencies=["/ws/pypi/ygg-1.0-py3-none-any.whl", "pyarrow==1"],
            )
        assert dest == "/Workspace/Shared/environments/yellow.requirements.txt"
        DP.from_.assert_called_once_with(dest, client=client)
        path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        body = path.write_text.call_args.args[0]
        # Flat pip requirements — no environment_version, no list indentation.
        assert body == "/ws/pypi/ygg-1.0-py3-none-any.whl\npyarrow==1\n"

    def test_deployed_environments_filters_env_and_requirements_files(self):
        client = MagicMock()
        folder = MagicMock()
        folder.exists.return_value = True

        def _child(name):
            c = MagicMock()
            c.name = name
            c.full_path.return_value = f"/ws/env/{name}"
            return c

        folder.iterdir.return_value = [
            _child("yellow.env.yaml"),
            _child("yellow.requirements.txt"),
            _child("README.md"),          # ignored
        ]
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP:
            DP.from_.return_value = folder
            paths = wheel.deployed_environments(client)
        assert paths == ["/ws/env/yellow.env.yaml", "/ws/env/yellow.requirements.txt"]

    def test_deployed_environments_empty_when_dir_absent(self):
        client = MagicMock()
        folder = MagicMock()
        folder.exists.return_value = False
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP:
            DP.from_.return_value = folder
            assert wheel.deployed_environments(client) == []

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
        # No deployed wheel for the current version → build per-Python wheels and
        # upload them into the version-scoped subfolder; ensure_ygg_wheel returns
        # the single matched wheel.
        client = MagicMock()
        with patch("yggdrasil.databricks.job.wheel.ilmd.version", return_value="9.9"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels", return_value=[]) as dw, \
             patch("yggdrasil.databricks.job.wheel.build_wheels_for_versions",
                   return_value=[Path("/tmp/ygg-9.9-py3-none-any.whl")]) as bw, \
             patch("yggdrasil.databricks.job.wheel.upload_wheel",
                   side_effect=lambda c, w, *, workspace_dir: f"{workspace_dir}/{w.name}") as up:
            out = wheel.ensure_ygg_wheel(client, workspace_dir="/ws/job")
        dw.assert_called_once_with(
            client, "ygg", "9.9", workspace_dir="/ws/job/ygg", dist_only=True,
        )
        bw.assert_called_once_with("ygg", versions=wheel.SUPPORTED_PYTHONS, extras=("databricks",))
        up.assert_called_once()
        assert out == ["/ws/job/ygg/ygg-9.9-py3-none-any.whl"]

    def test_ensure_ygg_wheel_reuses_deployed_wheel(self):
        # A wheel already deployed for the current version is reused — no build.
        client = MagicMock()
        deployed = ["/ws/job/9.9/ygg-9.9-py3-none-any.whl"]
        with patch("yggdrasil.databricks.job.wheel.ilmd.version", return_value="9.9"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels", return_value=deployed) as dw, \
             patch("yggdrasil.databricks.job.wheel.ensure_wheel") as ew:
            out = wheel.ensure_ygg_wheel(client, workspace_dir="/ws/job")
        dw.assert_called_once_with(
            client, "ygg", "9.9", workspace_dir="/ws/job/ygg", dist_only=True,
        )
        ew.assert_not_called()
        assert out == deployed

    def test_ensure_ygg_wheel_rebuild_skips_reuse(self):
        client = MagicMock()
        with patch("yggdrasil.databricks.job.wheel.ilmd.version", return_value="9.9"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels") as dw, \
             patch("yggdrasil.databricks.job.wheel.build_wheels_for_versions",
                   return_value=[Path("/tmp/ygg-9.9-py3-none-any.whl")]) as bw, \
             patch("yggdrasil.databricks.job.wheel.upload_wheel",
                   side_effect=lambda c, w, *, workspace_dir: f"{workspace_dir}/{w.name}"):
            out = wheel.ensure_ygg_wheel(client, workspace_dir="/ws/job", rebuild=True)
        dw.assert_not_called()                 # rebuild bypasses the reuse probe
        bw.assert_called_once_with("ygg", versions=wheel.SUPPORTED_PYTHONS, extras=("databricks",))
        assert out == ["/ws/job/ygg/ygg-9.9-py3-none-any.whl"]

    def test_ensure_bundle_reuses_full_when_present_and_not_rebuild(self):
        client = MagicMock()
        deployed = ["/ws/ygg-bundle/ygg-9.9-py3-none-any.whl",
                    "/ws/ygg-bundle/pyarrow-1-cp311.whl"]
        with patch("yggdrasil.databricks.job.wheel.ilmd.version", return_value="9.9"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels", return_value=deployed), \
             patch("yggdrasil.databricks.job.wheel.build_wheel") as bw:
            out = wheel.ensure_bundle(client, "ygg", workspace_dir="/ws")
        bw.assert_not_called()                 # full cache hit → no build/upload
        assert out == deployed

    def test_ensure_bundle_rebuild_uploads_only_project_reuses_deps(self):
        # Warm rebuild (e.g. editable ygg): deps already deployed are reused and
        # only the project wheel is rebuilt (no_deps) + re-uploaded → fast.
        client = MagicMock()
        deployed = ["/ws/ygg-bundle/ygg-9.9-py3-none-any.whl",
                    "/ws/ygg-bundle/pyarrow-1-cp311.whl",
                    "/ws/ygg-bundle/polars-2-cp311.whl"]
        uploaded: list[str] = []

        def _upload(c, w, *, workspace_dir):
            uploaded.append(w.name)
            return f"{workspace_dir}/{w.name}"

        def _build(pkg, *, extras=(), requirements=(), no_deps=False):
            assert no_deps is True   # warm rebuild builds only the project wheel
            return [Path("/tmp/ygg-9.9-py3-none-any.whl")]

        with patch("yggdrasil.databricks.job.wheel.ilmd.version", return_value="9.9"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels", return_value=deployed), \
             patch("yggdrasil.databricks.job.wheel.build_wheel", side_effect=_build), \
             patch("yggdrasil.databricks.job.wheel.upload_wheel", side_effect=_upload):
            out = wheel.ensure_bundle(client, "ygg", workspace_dir="/ws", rebuild=True)
        # Only the project wheel was built + uploaded; the two deps reused by path.
        assert uploaded == ["ygg-9.9-py3-none-any.whl"]
        assert out == [
            "/ws/ygg-bundle/ygg-9.9-py3-none-any.whl",   # freshly built + uploaded
            "/ws/ygg-bundle/pyarrow-1-cp311.whl",         # reused
            "/ws/ygg-bundle/polars-2-cp311.whl",          # reused
        ]

    def test_ensure_bundle_cold_uploads_everything(self):
        client = MagicMock()
        built = [Path("/tmp/ygg-9.9-py3-none-any.whl"), Path("/tmp/pyarrow-1-cp311.whl")]
        with patch("yggdrasil.databricks.job.wheel.ilmd.version", return_value="9.9"), \
             patch("yggdrasil.databricks.job.wheel.deployed_wheels", return_value=[]), \
             patch("yggdrasil.databricks.job.wheel.build_wheel", return_value=built), \
             patch("yggdrasil.databricks.job.wheel.upload_wheel",
                   side_effect=lambda c, w, *, workspace_dir: f"{workspace_dir}/{w.name}"):
            out = wheel.ensure_bundle(client, "ygg", workspace_dir="/ws")
        assert out == ["/ws/ygg-bundle/ygg-9.9-py3-none-any.whl",
                       "/ws/ygg-bundle/pyarrow-1-cp311.whl"]

    def test_deployed_wheels_dist_only_and_full(self):
        client = MagicMock()
        folder = MagicMock()
        folder.exists.return_value = True
        folder.iterdir.return_value = [
            _whl_child("ygg-9.9-py3-none-any.whl", "/ws/9.9/ygg-9.9-py3-none-any.whl"),
            _whl_child("pyarrow-1-py3-none-any.whl", "/ws/9.9/pyarrow-1-py3-none-any.whl"),
            _whl_child("notes.txt", "/ws/9.9/notes.txt"),
        ]
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", return_value=folder):
            # full bundle (legacy) — every wheel
            assert wheel.deployed_wheels(client, "ygg", "9.9", workspace_dir="/ws/9.9") == [
                "/ws/9.9/ygg-9.9-py3-none-any.whl", "/ws/9.9/pyarrow-1-py3-none-any.whl",
            ]
            # dist_only — just the ygg wheel, even if stale dep wheels linger
            assert wheel.deployed_wheels(
                client, "ygg", "9.9", workspace_dir="/ws/9.9", dist_only=True,
            ) == ["/ws/9.9/ygg-9.9-py3-none-any.whl"]

        # Partial: dist wheel for the version missing → treated as absent.
        folder.iterdir.return_value = [
            _whl_child("pyarrow-1-py3-none-any.whl", "/ws/9.9/pyarrow-1-py3-none-any.whl"),
            _whl_child("ygg-9.8-py3-none-any.whl", "/ws/9.9/ygg-9.8-py3-none-any.whl"),
        ]
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", return_value=folder):
            assert wheel.deployed_wheels(client, "ygg", "9.9", workspace_dir="/ws/9.9") == []
            assert wheel.deployed_wheels(
                client, "ygg", "9.9", workspace_dir="/ws/9.9", dist_only=True,
            ) == []

    def test_deployed_wheels_missing_dir(self):
        client = MagicMock()
        folder = MagicMock()
        folder.exists.return_value = False
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", return_value=folder):
            assert wheel.deployed_wheels(client, "ygg", "9.9", workspace_dir="/ws/9.9") == []

    def test_ygg_environment_ships_wheel_by_path_plus_index_deps(self):
        # The versioned ygg image: latest serverless v5, the ygg wheel by path,
        # and its runtime deps as index requirements (so they resolve as
        # platform-correct builds on the cluster).
        client = MagicMock()
        wheels = ["/ws/9.9/ygg-9.9-py3-none-any.whl"]
        deps = ["pyarrow>=20", "polars>=1.3", "databricks-sdk>=0.107"]
        with patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel", return_value=wheels) as ew, \
             patch("yggdrasil.databricks.job.wheel.ygg_runtime_dependencies", return_value=list(deps)):
            env = wheel.ygg_environment(client)
        ew.assert_called_once_with(
            client, workspace_dir=wheel.WORKSPACE_WHL_DIR, rebuild=False,
        )
        assert env.environment_key == "default"
        # defaults to the Python-matched serverless env version
        assert env.spec.environment_version == wheel.serverless_environment_version()
        # wheel path first (installed by path), then the index-resolved deps
        assert env.spec.dependencies == wheels + deps

    def test_serverless_environment_version_maps_python(self):
        import sys
        from unittest.mock import patch as _patch
        cases = {(3, 10): "1", (3, 11): "2", (3, 12): "5", (3, 13): "5"}
        for (maj, minr), expected in cases.items():
            with _patch.object(sys, "version_info", (maj, minr, 0)):
                assert wheel.serverless_environment_version() == expected

    def test_ygg_runtime_dependencies_are_index_names(self):
        deps = wheel.ygg_runtime_dependencies()
        # names/pins, never workspace wheel paths
        assert deps and all("/" not in d and d.endswith(".whl") is False for d in deps)
        assert any(d.startswith("databricks-sdk") for d in deps)

    def test_ygg_environment_forwards_key_and_rebuild(self):
        client = MagicMock()
        with patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel", return_value=["w.whl"]) as ew, \
             patch("yggdrasil.databricks.job.wheel.ygg_runtime_dependencies", return_value=[]):
            env = wheel.ygg_environment(
                client, environment_key="etl", environment_version="6", rebuild=True,
            )
        ew.assert_called_once_with(
            client, workspace_dir=wheel.WORKSPACE_WHL_DIR, rebuild=True,
        )
        assert env.environment_key == "etl"
        assert env.spec.environment_version == "6"

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


# --------------------------------------------------------------------------- #
# editable detection + per-user pypi folder
# --------------------------------------------------------------------------- #
class TestEditableAndUserPypi:
    def test_is_editable_install_reads_direct_url(self):
        dist = MagicMock()
        dist.read_text.return_value = '{"url": "file:///x", "dir_info": {"editable": true}}'
        with patch("yggdrasil.databricks.job.wheel.ilmd.distribution", return_value=dist):
            assert wheel.is_editable_install("anything") is True

    def test_is_editable_install_false_for_regular(self):
        dist = MagicMock()
        dist.read_text.return_value = None
        dist.files = []
        with patch("yggdrasil.databricks.job.wheel.ilmd.distribution", return_value=dist):
            assert wheel.is_editable_install("anything") is False

    def test_is_editable_install_missing_dist(self):
        import importlib.metadata as ilmd
        with patch(
            "yggdrasil.databricks.job.wheel.ilmd.distribution",
            side_effect=ilmd.PackageNotFoundError,
        ):
            assert wheel.is_editable_install("nope") is False

    def test_user_pypi_dir_uses_current_user(self):
        client = MagicMock()
        client.workspace_client.return_value.current_user.me.return_value.user_name = "me@x.io"
        assert wheel.user_pypi_dir(client) == "/Workspace/Users/me@x.io/pypi"


def test_synthesize_descends_when_finder_gives_project_root(tmp_path):
    # Editable finders can surface a package with __file__=None and __path__ set
    # to the *project root* (no __init__ there) — synthesize must descend into the
    # package dir so the build doesn't double-nest (pkg/pkg/__init__.py).
    import sys
    import types

    root = tmp_path / "proj"
    (root / "demo").mkdir(parents=True)
    (root / "demo" / "__init__.py").write_text("# pkg\n")
    (root / "demo" / "mod.py").write_text("x = 1\n")
    fake = types.ModuleType("demo")
    fake.__file__ = None
    fake.__path__ = [str(root)]                       # project ROOT, not the pkg dir
    sys.modules["demo"] = fake                         # so import_module('demo') finds it
    meta = {"Name": "demo", "Version": "0.0.1"}
    try:
        with patch("yggdrasil.databricks.job.wheel.ilmd.metadata", return_value=meta), \
             patch("yggdrasil.databricks.job.wheel.ilmd.requires", return_value=[]), \
             patch("yggdrasil.databricks.job.wheel.ilmd.entry_points", return_value=[]):
            proj = wheel.synthesize_project("demo", dest_dir=str(tmp_path / "out"))
    finally:
        sys.modules.pop("demo", None)
    assert (proj / "demo" / "__init__.py").exists()    # single-nested
    assert not (proj / "demo" / "demo").exists()       # not double-nested


# --------------------------------------------------------------------------- #
# per-Python wheel matrix + environments
# --------------------------------------------------------------------------- #
class TestVersionMatrix:
    def test_serverless_environment_version_table_and_fallback(self):
        assert wheel.serverless_environment_version("3.10") == "1"
        assert wheel.serverless_environment_version("3.11") == "2"
        assert wheel.serverless_environment_version("3.12") == wheel.SERVERLESS_ENVIRONMENT_VERSION
        assert wheel.serverless_environment_version("py313") == wheel.SERVERLESS_ENVIRONMENT_VERSION

    def test_environment_key_for(self):
        assert wheel.environment_key_for("3.11") == "py311"
        assert wheel.environment_key_for("3.12.4") == "py312"

    def test_wheel_for_python_prefers_tag_else_universal(self):
        native = ["/x/p-1-cp310-cp310-linux.whl", "/x/p-1-cp311-cp311-linux.whl"]
        assert wheel.wheel_for_python(native, "3.11").endswith("cp311-cp311-linux.whl")
        universal = ["/x/p-1-py3-none-any.whl"]
        assert wheel.wheel_for_python(universal, "3.13") == "/x/p-1-py3-none-any.whl"

    def test_build_wheels_for_versions_stops_after_universal(self):
        # A universal py3-none-any wheel is built once and reused for every Python.
        def _build(*a, **k):
            out = Path(k.get("out-dir") if False else "/")  # not used
        with patch("yggdrasil.databricks.job.wheel.synthesize_project", return_value=Path("/synth")), \
             patch("yggdrasil.databricks.job.wheel.subprocess.run") as run, \
             patch("yggdrasil.databricks.job.wheel.tempfile.mkdtemp", return_value="/out"), \
             patch("pathlib.Path.glob", return_value=[Path("/out/ygg-1.0-py3-none-any.whl")]):
            wheels = wheel.build_wheels_for_versions("yggdrasil", versions=("3.10", "3.11", "3.12"))
        assert run.call_count == 1                       # stopped after universal wheel
        assert [w.name for w in wheels] == ["ygg-1.0-py3-none-any.whl"]
        assert run.call_args.args[0][:3] == ["uv", "build", "--wheel"]
        assert "3.10" in run.call_args.args[0]           # built --python 3.10 first

    def test_ygg_environments_one_per_python_plus_default(self):
        ws = ["/ws/ygg-1.0-py3-none-any.whl"]
        with patch.object(wheel, "ensure_ygg_wheels", return_value=ws), \
             patch.object(wheel, "ygg_runtime_dependencies", return_value=["pyarrow>=20"]):
            envs = wheel.ygg_environments("CLIENT", default_python="3.11")
        assert [e.environment_key for e in envs] == ["default", "py310", "py311", "py312", "py313"]
        by_key = {e.environment_key: e for e in envs}
        assert by_key["default"].spec.environment_version == "2"     # 3.11
        assert by_key["py310"].spec.environment_version == "1"
        assert by_key["py312"].spec.environment_version == wheel.SERVERLESS_ENVIRONMENT_VERSION
        assert by_key["py311"].spec.dependencies == ws + ["pyarrow>=20"]
