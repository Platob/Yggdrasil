"""Tests for :meth:`DatabricksClient.spark` dependency wiring.

The Spark Connect session itself is mocked out; the focus is the
two compute-aware paths:

* **Serverless** (default, no ``cluster_id``) — PyPI specs and
  local archives ride through ``DatabricksEnv.withDependencies``
  and the builder applies it via
  ``DatabricksSession.builder.withEnvironment``.
* **Classic** (``cluster_id`` set) — same inputs, but the cluster
  installs nothing for us, so dependencies are resolved into
  local archives + wheels and attached with
  ``session.addArtifacts(..., pyfile=True)``.
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks import DatabricksClient


@pytest.fixture
def demo_pkg(tmp_path):
    pkg = tmp_path / "src" / "demo_sc_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("VALUE = 99\n")
    return pkg


@pytest.fixture
def serverless_client():
    """Default shape — no ``cluster_id`` → serverless compute."""
    return DatabricksClient(host="https://ws.example.com", token="dapi-x")


@pytest.fixture
def classic_client():
    return DatabricksClient(
        host="https://ws.example.com",
        token="dapi-x",
        cluster_id="0123-456789-abcdef",
    )


class TestServerlessDetection:

    def test_no_cluster_id_is_serverless(self, serverless_client) -> None:
        assert serverless_client.is_serverless_compute is True

    def test_explicit_cluster_id_is_classic(self, classic_client) -> None:
        assert classic_client.is_serverless_compute is False

    def test_explicit_serverless_compute_id_is_serverless(self) -> None:
        client = DatabricksClient(
            host="https://ws.example.com",
            token="dapi-x",
            cluster_id="abc",
            serverless_compute_id="auto",
        )
        # serverless_compute_id wins — explicit opt-in.
        assert client.is_serverless_compute is True


class TestCollectLocalModuleArchives:

    def test_zips_each_dep(self, serverless_client, demo_pkg, tmp_path) -> None:
        archives = serverless_client._collect_local_module_archives(
            [demo_pkg], cache_dir=tmp_path / "cache",
        )
        assert len(archives) == 1
        out = archives[0]
        assert out.suffix == ".zip"
        with zipfile.ZipFile(out) as zf:
            assert "demo_sc_pkg/__init__.py" in zf.namelist()

    def test_empty_input_returns_empty(self, serverless_client, tmp_path) -> None:
        assert serverless_client._collect_local_module_archives(
            [], cache_dir=tmp_path / "cache",
        ) == []


class TestCollectArtifactsClassic:
    """Pure resolution path — no SparkSession needed."""

    def test_local_module_only(self, classic_client, demo_pkg, tmp_path) -> None:
        artifacts = classic_client._collect_spark_artifacts(
            dependencies=[demo_pkg],
            pip_dependencies=(),
            cache_dir=tmp_path / "cache",
        )
        assert len(artifacts) == 1
        out = artifacts[0]
        assert out.suffix == ".zip"
        with zipfile.ZipFile(out) as zf:
            assert "demo_sc_pkg/__init__.py" in zf.namelist()

    def test_pip_dependency_invokes_pip_download(
        self, classic_client, tmp_path, monkeypatch,
    ) -> None:
        cache = tmp_path / "cache"
        observed: list[tuple] = []
        wheel = cache / "ygg-0.0.0-py3-none-any.whl"

        def fake_pip(self, *args, **kwargs):
            observed.append(args)
            cache.mkdir(parents=True, exist_ok=True)
            wheel.write_bytes(b"PK\x03\x04fake")
            return None

        from yggdrasil.environ import PyEnv
        monkeypatch.setattr(PyEnv, "pip", fake_pip)
        monkeypatch.setattr(
            PyEnv, "current",
            classmethod(lambda cls: PyEnv(python_path=Path("/usr/bin/python"))),
        )

        artifacts = classic_client._collect_spark_artifacts(
            dependencies=(),
            pip_dependencies=("ygg",),
            cache_dir=cache,
        )

        assert observed, "PyEnv.pip should have fired for the PyPI spec"
        args = observed[0]
        assert args[0] == "download"
        assert "--no-deps" in args
        assert "ygg" in args
        assert artifacts == [wheel]


@pytest.fixture
def stubbed_workspace_config(monkeypatch):
    """Skip the live SDK ``Config(...)`` call — it tries to resolve
    credentials and stalls in offline tests."""
    monkeypatch.setattr(
        DatabricksClient,
        "workspace_config",
        property(lambda self: MagicMock(name="WorkspaceConfig")),
    )


@pytest.fixture
def mocked_builder_factory(monkeypatch):
    """Patch ``databricks.connect`` with a builder we can assert on.

    Returns a callable so tests can grab the ``(builder, session,
    env_class)`` triple inside the test body.
    """
    def install():
        session = MagicMock(name="SparkSession")
        builder = MagicMock(name="Builder")
        builder.sdkConfig.return_value = builder
        builder.withEnvironment.return_value = builder
        builder.getOrCreate.return_value = session

        env_instances: list = []

        class FakeEnv:
            def __init__(self):
                self._deps: list = []
                env_instances.append(self)

            def withDependencies(self, specs):
                self._deps.append(list(specs))
                return self

        databricks_session = MagicMock(name="DatabricksSession", builder=builder)
        fake_module = type(
            "FakeConnect", (), {
                "DatabricksSession": databricks_session,
                "DatabricksEnv": FakeEnv,
            },
        )
        monkeypatch.setitem(sys.modules, "databricks.connect", fake_module)
        return builder, session, FakeEnv, env_instances

    return install


class TestServerlessBranch:
    """Serverless: declarative DatabricksEnv path."""

    def test_pip_dep_routes_through_with_dependencies(
        self, serverless_client, mocked_builder_factory,
        stubbed_workspace_config,
    ) -> None:
        builder, session, _env_cls, env_instances = mocked_builder_factory()
        result = serverless_client.spark(pip_dependencies=("ygg",))

        assert result is session
        assert len(env_instances) == 1
        env = env_instances[0]
        # One ``withDependencies`` call for the pip specs.
        assert env._deps == [["ygg"]]
        builder.withEnvironment.assert_called_once_with(env)
        # Serverless never falls back to addArtifacts.
        session.addArtifacts.assert_not_called()

    def test_local_module_routes_as_local_prefix(
        self, serverless_client, demo_pkg, mocked_builder_factory,
        stubbed_workspace_config, tmp_path,
    ) -> None:
        builder, session, _env_cls, env_instances = mocked_builder_factory()
        result = serverless_client.spark(
            dependencies=[demo_pkg],
            pip_dependencies=(),
            cache_dir=tmp_path / "cache",
        )

        assert result is session
        env = env_instances[0]
        # The single ``withDependencies`` call carries the local-prefixed
        # archive path.
        assert len(env._deps) == 1
        specs = env._deps[0]
        assert len(specs) == 1
        spec = specs[0]
        assert spec.startswith("local:")
        assert spec.endswith("demo_sc_pkg.zip")
        # The archive really exists on disk.
        assert os.path.exists(spec[len("local:"):])
        builder.withEnvironment.assert_called_once_with(env)
        session.addArtifacts.assert_not_called()

    def test_no_dependencies_skips_with_environment(
        self, serverless_client, mocked_builder_factory,
        stubbed_workspace_config,
    ) -> None:
        builder, session, _env_cls, env_instances = mocked_builder_factory()
        result = serverless_client.spark(pip_dependencies=())
        assert result is session
        # Nothing to wire — neither withEnvironment nor addArtifacts.
        builder.withEnvironment.assert_not_called()
        session.addArtifacts.assert_not_called()
        assert env_instances == []


class TestClassicBranch:
    """Classic compute: eager ``addArtifacts`` path."""

    def test_session_receives_artifacts(
        self, classic_client, demo_pkg, tmp_path, mocked_builder_factory,
        stubbed_workspace_config,
    ) -> None:
        builder, session, _env_cls, env_instances = mocked_builder_factory()
        result = classic_client.spark(
            dependencies=[demo_pkg],
            pip_dependencies=(),
            cache_dir=tmp_path / "cache",
        )

        assert result is session
        session.addArtifacts.assert_called_once()
        args, kwargs = session.addArtifacts.call_args
        assert kwargs == {"pyfile": True}
        assert len(args) == 1
        assert args[0].endswith("demo_sc_pkg.zip")
        assert os.path.exists(args[0])
        # Classic never goes through the env path.
        builder.withEnvironment.assert_not_called()
        assert env_instances == []

    def test_no_dependencies_skips_add_artifacts(
        self, classic_client, mocked_builder_factory,
        stubbed_workspace_config,
    ) -> None:
        builder, session, _env_cls, _envs = mocked_builder_factory()
        result = classic_client.spark(pip_dependencies=())
        assert result is session
        session.addArtifacts.assert_not_called()
        builder.withEnvironment.assert_not_called()
