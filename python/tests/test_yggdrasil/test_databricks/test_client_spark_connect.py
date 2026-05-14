"""Tests for :meth:`DatabricksClient.spark_connect` artifact bundling.

The Spark Connect session itself is mocked out; the focus here is
the deterministic pieces:

* local modules resolve to deflated ``.zip`` archives on disk;
* PyPI specs route through ``pip download --no-deps`` and the
  resulting wheels are picked up as extra artifacts;
* ``session.addArtifacts`` is called once with every resolved
  artifact and ``pyfile=True``;
* the optional ``workspace_cache`` mirror persists each artifact
  to the bound :class:`WorkspacePath`.
"""
from __future__ import annotations

import os
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
def client():
    return DatabricksClient(host="https://ws.example.com", token="dapi-x")


class TestCollectArtifacts:
    """Pure resolution path — no SparkSession needed."""

    def test_local_module_only(self, client, demo_pkg, tmp_path) -> None:
        artifacts = client._collect_spark_connect_artifacts(
            dependencies=[demo_pkg],
            pip_dependencies=(),
            cache_dir=tmp_path / "cache",
        )
        assert len(artifacts) == 1
        out = artifacts[0]
        assert out.suffix == ".zip"
        with zipfile.ZipFile(out) as zf:
            assert "demo_sc_pkg/__init__.py" in zf.namelist()

    def test_empty_inputs_skip_pip(self, client, tmp_path, monkeypatch) -> None:
        called = []

        def fake_pip(*args, **kwargs):  # pragma: no cover - should never fire
            called.append((args, kwargs))

        from yggdrasil.environ import PyEnv
        monkeypatch.setattr(PyEnv, "pip", lambda self, *a, **kw: fake_pip(*a, **kw))

        artifacts = client._collect_spark_connect_artifacts(
            dependencies=(),
            pip_dependencies=(),
            cache_dir=tmp_path / "cache",
        )
        assert artifacts == []
        assert called == []

    def test_pip_dependency_invokes_pip_download(
        self, client, tmp_path, monkeypatch,
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
            PyEnv, "current", classmethod(lambda cls: PyEnv(python_path=Path("/usr/bin/python"))),
        )

        artifacts = client._collect_spark_connect_artifacts(
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

    def test_local_and_pip_combined(
        self, client, demo_pkg, tmp_path, monkeypatch,
    ) -> None:
        cache = tmp_path / "cache"
        wheel = cache / "ygg-0.0.0-py3-none-any.whl"

        def fake_pip(self, *args, **kwargs):
            cache.mkdir(parents=True, exist_ok=True)
            wheel.write_bytes(b"PK\x03\x04fake")

        from yggdrasil.environ import PyEnv
        monkeypatch.setattr(PyEnv, "pip", fake_pip)
        monkeypatch.setattr(
            PyEnv, "current",
            classmethod(lambda cls: PyEnv(python_path=Path("/usr/bin/python"))),
        )

        artifacts = client._collect_spark_connect_artifacts(
            dependencies=[demo_pkg],
            pip_dependencies=("ygg",),
            cache_dir=cache,
        )
        names = sorted(p.name for p in artifacts)
        assert names == ["demo_sc_pkg.zip", "ygg-0.0.0-py3-none-any.whl"]


class TestSparkConnectBuilder:
    """Full :meth:`spark_connect` flow with the Spark builder mocked."""

    @pytest.fixture
    def mocked_builder(self, monkeypatch):
        session = MagicMock()
        builder = MagicMock()
        builder.sdkConfig.return_value = builder
        builder.getOrCreate.return_value = session

        fake_module = type(
            "FakeConnect", (), {"DatabricksSession": MagicMock(builder=builder)},
        )
        monkeypatch.setitem(
            __import__("sys").modules, "databricks.connect", fake_module,
        )
        return builder, session

    @pytest.fixture
    def stubbed_workspace_config(self, monkeypatch):
        """Skip the live SDK ``Config(...)`` call — it tries to
        resolve credentials and stalls in offline tests."""
        monkeypatch.setattr(
            DatabricksClient,
            "workspace_config",
            property(lambda self: MagicMock(name="WorkspaceConfig")),
        )

    def test_session_receives_artifacts(
        self, client, demo_pkg, tmp_path, mocked_builder,
        stubbed_workspace_config,
    ) -> None:
        _builder, session = mocked_builder
        result = client.spark_connect(
            dependencies=[demo_pkg],
            pip_dependencies=(),
            cache_dir=tmp_path / "cache",
        )

        assert result is session
        session.addArtifacts.assert_called_once()
        args, kwargs = session.addArtifacts.call_args
        assert kwargs == {"pyfile": True}
        # One artifact, the .zip we built.
        assert len(args) == 1
        assert args[0].endswith("demo_sc_pkg.zip")
        assert os.path.exists(args[0])

    def test_no_dependencies_skips_add_artifacts(
        self, client, mocked_builder, stubbed_workspace_config,
    ) -> None:
        _builder, session = mocked_builder
        result = client.spark_connect(pip_dependencies=())
        assert result is session
        session.addArtifacts.assert_not_called()
