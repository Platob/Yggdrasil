"""Tests for the serverless environment spec generator
(``python/scripts/build_serverless_env.py``).

Two things matter:

* the generated YAML is a valid Databricks serverless ``Environment`` spec
  (``environment_version`` + ``dependencies``, ygg wheel first); and
* its ``python → environment_version`` mapping does not drift from the
  package's :func:`yggdrasil.databricks.wheels.service.serverless_environment_version`,
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
        from yggdrasil.databricks.wheels import service as wheel

        assert (
            build_serverless_env.SERVERLESS_ENVIRONMENT_VERSION
            == wheel.SERVERLESS_ENVIRONMENT_VERSION
        )

    @pytest.mark.parametrize("minor", [10, 11, 12, 13])
    def test_version_for_minor_matches_package(self, minor, monkeypatch) -> None:
        from yggdrasil.databricks.wheels import service as wheel

        # The package function keys off the live interpreter; pin it per minor.
        monkeypatch.setattr(sys, "version_info", (3, minor, 0, "final", 0))
        assert (
            build_serverless_env.serverless_environment_version(minor)
            == wheel.serverless_environment_version()
        )

    def test_py312_resolves_to_latest(self) -> None:
        assert build_serverless_env.serverless_environment_version(12) == "5"


class TestCollectWheels:
    BUNDLE = [
        "databricks_sdk-0.114.0-py3-none-any.whl",
        "orjson-3.11.9-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "polars-1.41.2-py3-none-any.whl",
        "polars_runtime_32-1.41.2-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "xxhash-3.7.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
    ]

    def _stage(self, tmp_path):
        for name in self.BUNDLE:
            (tmp_path / name).write_bytes(b"")
        return tmp_path

    def test_wheel_dir_globs_sorted_after_ygg(self, tmp_path) -> None:
        wheels = build_serverless_env.collect_wheels(
            "0.8.49", wheel_dir=self._stage(tmp_path),
        )
        assert wheels[0] == "ygg-0.8.49-py3-none-any.whl"
        assert wheels[1:] == sorted(self.BUNDLE)

    def test_explicit_wheels_come_before_dir(self, tmp_path) -> None:
        wheels = build_serverless_env.collect_wheels(
            "0.8.49", wheel_dir=self._stage(tmp_path), extra_wheels=["extra-1.0.whl"],
        )
        assert wheels[0] == "ygg-0.8.49-py3-none-any.whl"
        assert wheels[1] == "extra-1.0.whl"

    def test_order_preserving_dedupe(self, tmp_path) -> None:
        # The ygg wheel sitting in the dir (and a repeated explicit) appears once.
        (tmp_path / "ygg-0.8.49-py3-none-any.whl").write_bytes(b"")
        (tmp_path / "databricks_sdk-0.114.0-py3-none-any.whl").write_bytes(b"")
        wheels = build_serverless_env.collect_wheels(
            "0.8.49",
            wheel_dir=tmp_path,
            extra_wheels=["databricks_sdk-0.114.0-py3-none-any.whl"],
        )
        assert wheels == [
            "ygg-0.8.49-py3-none-any.whl",
            "databricks_sdk-0.114.0-py3-none-any.whl",
        ]


class TestBuildYaml:
    WHEELS = [
        "ygg-0.8.49-py3-none-any.whl",
        "databricks_sdk-0.114.0-py3-none-any.whl",
        "polars-1.41.2-py3-none-any.whl",
        "polars_runtime_32-1.41.2-cp310-abi3-manylinux_2_17_x86_64.whl",
        "orjson-3.11.9-cp312-cp312-manylinux_2_17_x86_64.whl",
        "xxhash-3.7.0-cp312-cp312-manylinux_2_17_x86_64.whl",
    ]

    def test_yaml_is_valid_environment_spec(self) -> None:
        spec = yaml.safe_load(build_serverless_env.build_yaml("0.8.49", "3.12", self.WHEELS))
        # environment_version is a *string* (the API contract), not an int.
        assert spec["environment_version"] == "5"
        assert isinstance(spec["environment_version"], str)
        # Wheels only, by path, ygg first — no index requirements at all.
        assert spec["dependencies"] == self.WHEELS

    def test_dependencies_are_wheels_only(self) -> None:
        spec = yaml.safe_load(build_serverless_env.build_yaml("0.8.49", "3.12", self.WHEELS))
        assert all(dep.endswith(".whl") for dep in spec["dependencies"])
        # The compiled polars runtime is bundled, not just the polars shim.
        assert any(d.startswith("polars_runtime_") for d in spec["dependencies"])

    def test_python311_picks_environment_version_two(self) -> None:
        spec = yaml.safe_load(build_serverless_env.build_yaml("0.8.49", "3.11", self.WHEELS))
        assert spec["environment_version"] == "2"


class TestBuildRequirements:
    WHEELS = [
        "ygg-0.8.49-py3-none-any.whl",
        "databricks_sdk-0.114.0-py3-none-any.whl",
        "polars-1.41.2-py3-none-any.whl",
    ]

    def test_flat_wheel_list_no_environment_version(self) -> None:
        body = build_serverless_env.build_requirements("0.8.49", "3.12", self.WHEELS)
        lines = [ln for ln in body.splitlines() if ln and not ln.startswith("#")]
        # The cluster requirements are the same wheels, by path, ygg first — and
        # carry no ``environment_version`` line (a classic cluster has no such key).
        assert lines == self.WHEELS
        assert "environment_version" not in body


class TestMain:
    def test_main_writes_serverless_and_cluster(self, tmp_path) -> None:
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()
        (wheel_dir / "databricks_sdk-0.114.0-py3-none-any.whl").write_bytes(b"")
        out_dir = tmp_path / "out"
        rc = build_serverless_env.main(
            ["--version", "9.9.9", "--python", "3.12",
             "--wheel-dir", str(wheel_dir), "--out-dir", str(out_dir)]
        )
        assert rc == 0
        # Both a serverless spec and a classic-cluster requirements file are written.
        serverless = yaml.safe_load((out_dir / "ygg-9.9.9-py3.12-serverless.yml").read_text())
        assert serverless["environment_version"] == "5"
        cluster = (out_dir / "ygg-9.9.9-py3.12-cluster.requirements.txt").read_text()
        wheels = [ln for ln in cluster.splitlines() if ln and not ln.startswith("#")]
        assert wheels == ["ygg-9.9.9-py3-none-any.whl", "databricks_sdk-0.114.0-py3-none-any.whl"]

    def test_all_versions_writes_a_pair_per_python(self, tmp_path) -> None:
        rc = build_serverless_env.main(
            ["--version", "9.9.9", "--all-versions", "--out-dir", str(tmp_path)]
        )
        assert rc == 0
        # Capped at MAX_PYTHON (3.12) — Databricks doesn't run 3.13+ yet.
        for py in ("3.10", "3.11", "3.12"):
            assert (tmp_path / f"ygg-9.9.9-py{py}-serverless.yml").exists()
            assert (tmp_path / f"ygg-9.9.9-py{py}-cluster.requirements.txt").exists()
        assert not (tmp_path / "ygg-9.9.9-py3.13-serverless.yml").exists()
        # py3.10 → environment_version "1"; py3.12 → latest "5".
        assert yaml.safe_load(
            (tmp_path / "ygg-9.9.9-py3.10-serverless.yml").read_text()
        )["environment_version"] == "1"

    def test_python_target_is_capped_at_max(self) -> None:
        # Databricks doesn't run 3.13+ yet — every build target clamps to 3.12.
        from yggdrasil.databricks.wheels.service import (
            MAX_PYTHON, SUPPORTED_PYTHONS, _py_minor, environment_key_for,
        )
        assert MAX_PYTHON == "3.12"
        assert "3.13" not in SUPPORTED_PYTHONS and SUPPORTED_PYTHONS[-1] == "3.12"
        assert _py_minor("3.13") == "3.12"        # clamped
        assert _py_minor("py314") == "3.12"       # clamped
        assert _py_minor("3.11") == "3.11"        # under the cap, untouched
        assert environment_key_for("3.13") == "py312"   # env key follows the clamp

    def test_main_writes_versioned_spec(self, tmp_path) -> None:
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()
        for name in ["databricks_sdk-0.114.0-py3-none-any.whl",
                     "polars-1.41.2-py3-none-any.whl"]:
            (wheel_dir / name).write_bytes(b"")
        out_dir = tmp_path / "out"
        rc = build_serverless_env.main(
            ["--version", "9.9.9", "--python", "3.12",
             "--wheel-dir", str(wheel_dir), "--out-dir", str(out_dir)]
        )
        assert rc == 0
        spec = yaml.safe_load((out_dir / "ygg-9.9.9-py3.12-serverless.yml").read_text())
        assert spec["environment_version"] == "5"
        assert spec["dependencies"] == [
            "ygg-9.9.9-py3-none-any.whl",
            "databricks_sdk-0.114.0-py3-none-any.whl",
            "polars-1.41.2-py3-none-any.whl",
        ]

    def test_without_bundle_only_ygg(self, tmp_path) -> None:
        build_serverless_env.main(
            ["--version", "9.9.9", "--out-dir", str(tmp_path)]
        )
        spec = yaml.safe_load(
            (tmp_path / "ygg-9.9.9-py3.12-serverless.yml").read_text()
        )
        assert spec["dependencies"] == ["ygg-9.9.9-py3-none-any.whl"]

    def test_main_reads_version_from_pyproject(self, tmp_path) -> None:
        import tomllib

        pyproject = Path(_SCRIPT).resolve().parents[1] / "pyproject.toml"
        version = tomllib.loads(pyproject.read_text())["project"]["version"]

        build_serverless_env.main(["--out-dir", str(tmp_path)])
        produced = list(tmp_path.glob("ygg-*-py3.12-serverless.yml"))
        assert len(produced) == 1
        # No --version → the filename version is read from python/pyproject.toml.
        assert produced[0].name == f"ygg-{version}-py3.12-serverless.yml"
