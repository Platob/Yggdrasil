"""Tests for the serverless environment spec generator
(``python/scripts/build_serverless_env.py``).

Two things matter:

* the generated YAML is a valid Databricks serverless ``Environment`` spec
  (``environment_version`` + ``dependencies``, ygg wheel first); and
* its ``python → environment_version`` mapping does not drift from the
  package's :func:`yggdrasil.databricks.job.wheel.serverless_environment_version`,
  the runtime source of truth the deploy path uses.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

# Load the standalone build script by path — ``python/scripts`` is not a package.
_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "build_serverless_env.py"
_spec = importlib.util.spec_from_file_location("build_serverless_env", _SCRIPT)
build_serverless_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_serverless_env)


class TestMappingParity:
    """The script mirrors the package's serverless-version mapping exactly."""

    def test_latest_constant_matches_package(self) -> None:
        from yggdrasil.databricks.job import wheel

        assert (
            build_serverless_env.SERVERLESS_ENVIRONMENT_VERSION
            == wheel.SERVERLESS_ENVIRONMENT_VERSION
        )

    @pytest.mark.parametrize("minor", [10, 11, 12, 13])
    def test_version_for_minor_matches_package(self, minor, monkeypatch) -> None:
        from yggdrasil.databricks.job import wheel

        # The package function keys off the live interpreter; pin it per minor.
        monkeypatch.setattr(sys, "version_info", (3, minor, 0, "final", 0))
        assert (
            build_serverless_env.serverless_environment_version(minor)
            == wheel.serverless_environment_version()
        )

    def test_py312_resolves_to_latest(self) -> None:
        assert build_serverless_env.serverless_environment_version(12) == "5"


class TestBuildYaml:
    DEPS = ["pyarrow>=20", "polars>=1.3", "databricks-sdk>=0.107"]

    def test_yaml_is_valid_environment_spec(self) -> None:
        text = build_serverless_env.build_yaml(
            "0.8.49", "3.12", self.DEPS, "ygg-0.8.49-py3-none-any.whl",
        )
        spec = yaml.safe_load(text)
        # environment_version is a *string* (the API contract), not an int.
        assert spec["environment_version"] == "5"
        assert isinstance(spec["environment_version"], str)
        # The ygg wheel installs first, then the runtime deps as index reqs.
        assert spec["dependencies"][0] == "ygg-0.8.49-py3-none-any.whl"
        assert spec["dependencies"][1:] == self.DEPS

    def test_python311_picks_environment_version_two(self) -> None:
        spec = yaml.safe_load(
            build_serverless_env.build_yaml(
                "0.8.49", "3.11", self.DEPS, "ygg-0.8.49-py3-none-any.whl",
            )
        )
        assert spec["environment_version"] == "2"


class TestMain:
    def test_main_writes_versioned_spec(self, tmp_path, capsys) -> None:
        rc = build_serverless_env.main(
            ["--version", "9.9.9", "--python", "3.12", "--out-dir", str(tmp_path)]
        )
        assert rc == 0
        out = tmp_path / "ygg-9.9.9-py3.12-serverless.yml"
        assert out.exists()
        spec = yaml.safe_load(out.read_text())
        assert spec["environment_version"] == "5"
        assert spec["dependencies"][0] == "ygg-9.9.9-py3-none-any.whl"
        # The declared runtime deps (pyarrow / polars / …) plus the databricks
        # extra (databricks-sdk) all land as index requirements.
        joined = " ".join(spec["dependencies"])
        assert "pyarrow" in joined and "databricks-sdk" in joined

    def test_main_reads_version_from_pyproject(self, tmp_path) -> None:
        import tomllib

        pyproject = Path(_SCRIPT).resolve().parents[1] / "pyproject.toml"
        version = tomllib.loads(pyproject.read_text())["project"]["version"]

        build_serverless_env.main(["--out-dir", str(tmp_path)])
        produced = list(tmp_path.glob("ygg-*-py3.12-serverless.yml"))
        assert len(produced) == 1
        # No --version → the filename version is read from python/pyproject.toml.
        assert produced[0].name == f"ygg-{version}-py3.12-serverless.yml"
