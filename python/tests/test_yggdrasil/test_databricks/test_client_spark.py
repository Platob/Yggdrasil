"""Tests for :meth:`DatabricksClient.spark` — dependency resolution
through the :class:`WorkspacePyPIRegistry`.

Each test mocks just enough of ``databricks.connect`` (the
``DatabricksSession.builder`` chain + a stand-in ``DatabricksEnv``)
and the registry's wheel-build helper so the suite stays offline.
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks import DatabricksClient
from yggdrasil.version import __version__ as _ygg_version
from yggdrasil.databricks.registry import (
    DependencyInfo,
    DependencyKind,
    WorkspacePyPIRegistry,
    classify_dependency,
)


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def serverless_client():
    """Explicit serverless compute."""
    return DatabricksClient(
        host="https://ws.example.com",
        token="dapi-x",
        serverless_compute_id="auto",
    )


@pytest.fixture
def classic_client():
    return DatabricksClient(
        host="https://ws.example.com",
        token="dapi-x",
        cluster_id="0123-456789-abcdef",
    )


@pytest.fixture
def stubbed_workspace_config(monkeypatch):
    """Skip the live SDK ``Config(...)`` call and cluster resolution."""
    monkeypatch.setattr(
        DatabricksClient,
        "workspace_config",
        property(lambda self: MagicMock(name="WorkspaceConfig")),
    )
    monkeypatch.setattr(
        DatabricksClient,
        "workspace_client",
        lambda self: MagicMock(name="WorkspaceClient"),
    )

    from yggdrasil.databricks.cluster.service import Clusters

    def _stub_get_or_create(self_clusters, obj=None, **kwargs):
        stub = MagicMock(name="Cluster")
        cluster_id = obj if isinstance(obj, str) else kwargs.get("cluster_id", "stub-id")
        stub.cluster_id = cluster_id
        return stub

    monkeypatch.setattr(Clusters, "get_or_create", _stub_get_or_create)


@pytest.fixture
def mocked_builder(monkeypatch):
    """Patch ``databricks.connect`` with controllable
    ``DatabricksSession.builder`` and ``DatabricksEnv`` shapes.

    Also force ``SparkSession.getActiveSession()`` → ``None`` so the
    early-return path in :meth:`DatabricksClient.spark` (which
    bypasses the builder when a live session is already pinned on
    the process) doesn't shortcut the mock. Real-world test
    environments — notebook drivers, PyCharm's Spark Connect plugin,
    leaked state from an earlier suite — routinely have an active
    session, which would otherwise make ``serverless_client.spark()``
    return the real session instead of the mocked one.
    """
    session = MagicMock(name="SparkSession")
    builder = MagicMock(name="Builder")
    # Keep the fluent chain on the same builder mock regardless of which
    # connection setters the code calls (sdkConfig for classic; serverless +
    # host + token for serverless).
    builder.sdkConfig.return_value = builder
    builder.withEnvironment.return_value = builder
    builder.serverless.return_value = builder
    builder.host.return_value = builder
    builder.token.return_value = builder
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

    # Neutralise any pre-existing live SparkSession in the process — both
    # the classic and the Spark Connect classes expose
    # ``getActiveSession()`` and ``DatabricksClient.spark`` probes the
    # classic one before reaching the builder.
    try:
        from pyspark.sql import SparkSession as _SparkSession
        monkeypatch.setattr(_SparkSession, "getActiveSession", staticmethod(lambda: None))
    except Exception:
        pass
    try:
        from pyspark.sql.connect.session import SparkSession as _ConnectSparkSession
        monkeypatch.setattr(
            _ConnectSparkSession, "getActiveSession", staticmethod(lambda: None),
        )
    except Exception:
        pass
    # Also clear yggdrasil's own cached session so the create-once helpers
    # don't hand back a leaked Connect session from a previous test.
    try:
        from yggdrasil.environ import PyEnv
        from yggdrasil.environ.environment import MISSING
        monkeypatch.setattr(PyEnv, "_SPARK_SESSION", MISSING, raising=False)
    except Exception:
        pass

    return builder, session, FakeEnv, env_instances


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassifyDependency:

    def test_pip_spec_string(self) -> None:
        info = classify_dependency("numpy==1.0")
        assert info.kind == DependencyKind.PUBLIC
        assert info.spec == "numpy==1.0"
        assert info.name == "numpy"

    def test_installed_dist_default_is_local(self, monkeypatch) -> None:
        # ``ygg`` is editable in this repo; force the editable
        # detector off so we exercise the LOCAL branch alone.
        from importlib.metadata import PackageNotFoundError, distribution
        try:
            distribution("ygg")
        except PackageNotFoundError:
            pytest.skip(
                "ygg is not installed as a distribution in this environment; "
                "the LOCAL-classification path requires `pip install -e .`."
            )
        import yggdrasil.databricks.registry as reg
        monkeypatch.setattr(reg, "_detect_editable", lambda dist: False)
        info = classify_dependency("ygg")
        assert info.kind == DependencyKind.LOCAL
        assert info.name == "ygg"
        assert info.version is not None

    def test_editable_install_is_editable(self) -> None:
        # Skip when the dev environment installed ``ygg`` non-editably
        # (e.g. ``pip install .`` instead of ``pip install -e .``), or
        # not at all. The editable bit comes from PEP 610
        # ``direct_url.json`` — probe the same source the production
        # classifier uses so the test only runs in environments that
        # can actually satisfy its premise.
        from importlib.metadata import PackageNotFoundError, distribution
        from yggdrasil.databricks.registry import _detect_editable

        try:
            dist = distribution("ygg")
        except PackageNotFoundError:
            pytest.skip(
                "ygg is not installed as a distribution in this environment; "
                "the EDITABLE-classification path requires `pip install -e .`."
            )
        if not _detect_editable(dist):
            pytest.skip(
                "ygg is installed non-editably in this environment; the "
                "EDITABLE-classification path can only be exercised after "
                "`pip install -e .`."
            )

        info = classify_dependency("ygg")
        # The repo install IS editable; classification should
        # honour direct_url.json and stamp the hostname.
        assert info.kind == DependencyKind.EDITABLE
        host = socket.gethostname() or "unknown"
        host = "".join(c if c.isalnum() else "-" for c in host).strip("-")
        assert info.version is not None
        assert info.version.endswith(f"+host.{host}")

    def test_unknown_falls_back_to_public(self) -> None:
        info = classify_dependency("zzz-nonexistent-xyz")
        assert info.kind == DependencyKind.PUBLIC

    def test_pathlike_is_local(self, tmp_path) -> None:
        info = classify_dependency(tmp_path)
        assert info.kind == DependencyKind.LOCAL
        assert info.source == tmp_path.resolve()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestWorkspacePyPIRegistry:

    def test_default_base_is_shared_registry(self, serverless_client, monkeypatch):
        # No base_path → the shared workspace registry (same root the job image
        # uses), so Spark Connect and the job image share one index.
        from yggdrasil.databricks.wheels.service import WORKSPACE_PYPI_DIR
        monkeypatch.setattr(
            type(serverless_client), "workspaces",
            property(lambda self: MagicMock(name="Workspaces")),
        )
        assert WorkspacePyPIRegistry.DEFAULT_BASE == WORKSPACE_PYPI_DIR == "/Workspace/Shared/pypi"
        reg = WorkspacePyPIRegistry(client=serverless_client)
        assert reg.base_path.full_path() == "/Workspace/Shared/pypi"

    @pytest.fixture
    def registry(self, serverless_client, tmp_path, monkeypatch):
        """Registry with a mocked workspace upload path.

        Workspace I/O is stubbed via direct ``WorkspacePath``
        monkeypatching so the registry layout assertions don't
        need a live workspace client.

        Returns ``(registry, store, write_log)`` — ``store``
        models the persisted workspace state; ``write_log`` is a
        list of every write so tests can distinguish "exists ==
        True" from "wrote again this call".
        """
        from yggdrasil.databricks.fs.workspace_path import WorkspacePath

        store: dict[str, bytes] = {}
        writes: list[str] = []

        def fake_write_bytes(self, data, offset=0):
            store[self.full_path()] = bytes(data)
            writes.append(self.full_path())
            return len(data)

        def fake_mkdir(self, parents=True, exist_ok=True):
            return self

        def fake_exists(self):
            return self.full_path() in store

        def fake_read_bytes(self, size=-1, offset=0):
            data = store[self.full_path()]
            return data if size < 0 else data[offset:offset + size]

        monkeypatch.setattr(WorkspacePath, "write_bytes", fake_write_bytes)
        monkeypatch.setattr(WorkspacePath, "mkdir", fake_mkdir)
        monkeypatch.setattr(WorkspacePath, "exists", fake_exists)
        monkeypatch.setattr(WorkspacePath, "read_bytes", fake_read_bytes)

        reg = WorkspacePyPIRegistry(
            client=serverless_client,
            base_path="/Workspace/Shared/.ygg/pypi/simple",
            local_cache=tmp_path / "cache",
        )
        return reg, store, writes

    def test_public_spec_short_circuits(self, registry) -> None:
        reg, _store, writes = registry
        spec, remote = reg.publish("numpy==1.0")
        assert spec == "numpy==1.0"
        assert remote is None
        assert writes == []

    def test_local_publishes_lazily(self, registry, monkeypatch, tmp_path) -> None:
        reg, _store, writes = registry

        wheel_bytes = b"PK\x03\x04fake-wheel"
        wheel_name = "demo-1.0.0-py3-none-any.whl"

        def fake_pip_wheel(self_reg, info):
            target = self_reg.local_cache / wheel_name
            target.write_bytes(wheel_bytes)
            return target

        monkeypatch.setattr(WorkspacePyPIRegistry, "_pip_wheel", fake_pip_wheel)

        info = DependencyInfo(
            name="demo", version="1.0.0", kind=DependencyKind.LOCAL,
        )
        monkeypatch.setattr(
            "yggdrasil.databricks.registry.classify_dependency",
            lambda obj, *, check_public=False: info,
        )

        # First call uploads once.
        spec, remote = reg.publish("demo")
        assert spec.startswith("local:")
        assert remote is not None
        expected_remote = (
            "/Workspace/Shared/.ygg/pypi/simple/demo/demo-1.0.0-py3-none-any.whl"
        )
        assert remote.full_path() == expected_remote
        assert writes == [expected_remote]

        # Second call is lazy — the registry sees the existing
        # entry and skips the workspace write.
        spec2, remote2 = reg.publish("demo")
        assert spec2 == spec
        assert remote2.full_path() == remote.full_path()
        assert writes == [expected_remote]  # no second write

    def test_editable_always_overwrites(self, registry, monkeypatch, tmp_path) -> None:
        reg, _store, writes = registry

        wheel_bytes = b"PK\x03\x04editable-wheel"
        wheel_name = "demo-1.0.0+host.h-py3-none-any.whl"

        def fake_pip_wheel(self_reg, info):
            target = self_reg.local_cache / wheel_name
            target.write_bytes(wheel_bytes)
            return target

        monkeypatch.setattr(WorkspacePyPIRegistry, "_pip_wheel", fake_pip_wheel)

        info = DependencyInfo(
            name="demo",
            version="1.0.0+host.h",
            kind=DependencyKind.EDITABLE,
            source=tmp_path,
        )
        monkeypatch.setattr(
            "yggdrasil.databricks.registry.classify_dependency",
            lambda obj, *, check_public=False: info,
        )

        reg.publish("demo")
        reg.publish("demo")
        # Editable installs refresh the workspace slot on every load.
        assert len(writes) == 2
        assert writes[0] == writes[1]


# ---------------------------------------------------------------------------
# Client.spark — serverless + classic branches
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_registry(monkeypatch):
    """Patch :class:`WorkspacePyPIRegistry` with an in-process stub
    so ``client.spark`` doesn't touch the workspace at all."""
    calls: list[tuple] = []

    class FakeRegistry:
        def __init__(self, client=None, base_path=None, local_cache=None):
            self.client = client
            self.base_path = base_path
            self.local_cache = local_cache

        def publish_many(self, deps, *, check_public=False):
            specs: list[str] = []
            remotes: list = []
            for d in deps:
                calls.append((d, check_public))
                if isinstance(d, str):
                    specs.append(d)
            return specs, remotes

    monkeypatch.setattr(
        "yggdrasil.databricks.registry.WorkspacePyPIRegistry", FakeRegistry,
    )
    return FakeRegistry, calls


class TestServerlessBranch:

    def test_default_ships_ygg_wheel_plus_index_deps(
        self, serverless_client, mocked_builder, stubbed_workspace_config,
        fake_registry,
    ) -> None:
        from yggdrasil.databricks.wheels.service import runtime_dependencies

        builder, session, _env_cls, env_instances = mocked_builder
        result = serverless_client.spark()
        assert result is session

        env = env_instances[0]
        # ygg ships as a wheel (bare ``ygg`` → the registry builds + uploads it
        # and attaches it via ``local:``), NOT a pip ``ygg==version`` index spec
        # — so a private ygg installs on the cluster. Its runtime deps ride
        # along as index requirement names.
        assert env._deps == [["ygg", *runtime_dependencies("ygg")]]
        assert not any("==" in d and d.startswith("ygg") for d in env._deps[0])
        builder.withEnvironment.assert_called_once_with(env)
        # The client is stashed on the session for downstream use.
        assert session.ygg_client is serverless_client

    def test_user_specs_merge_with_default_ygg(
        self, serverless_client, mocked_builder, stubbed_workspace_config,
        fake_registry,
    ) -> None:
        from yggdrasil.databricks.wheels.service import runtime_dependencies

        _builder, _session, _env_cls, env_instances = mocked_builder
        serverless_client.spark("numpy==1.0")
        env = env_instances[0]
        assert env._deps == [["ygg", *runtime_dependencies("ygg"), "numpy==1.0"]]

    def test_ygg_not_doubled_when_caller_passes_it(
        self, serverless_client, mocked_builder, stubbed_workspace_config,
        fake_registry,
    ) -> None:
        # Explicit ``"ygg"`` (any shape — version-pinned, with
        # extras, bare name) from the caller already covers the
        # default; we shouldn't see ``ygg[data,databricks]`` added
        # on top.
        _builder, _session, _env_cls, env_instances = mocked_builder
        _registry_cls, calls = fake_registry
        serverless_client.spark("ygg==0.7.85")
        names = [d for d, _ in calls]
        assert names == ["ygg==0.7.85"]


class TestSparkConnectBuilder:
    """``_spark_connect_builder`` — serverless uses ``.serverless(True)`` + auth;
    classic uses sdkConfig with a fallback for newer databricks-connect that
    rejects it alongside env/profile connection."""

    def test_serverless_uses_serverless_and_explicit_auth(
        self, serverless_client, mocked_builder, stubbed_workspace_config,
    ) -> None:
        builder, _session, _env, _envs = mocked_builder
        out = serverless_client._spark_connect_builder(serverless=True)
        builder.serverless.assert_called_once_with(True)
        # serverless can't take sdkConfig — auth is set explicitly from the PAT
        builder.host.assert_called_once_with(serverless_client.host)
        builder.token.assert_called_once_with(serverless_client.token)
        builder.sdkConfig.assert_not_called()
        assert out is builder

    def test_classic_uses_sdkconfig_when_accepted(
        self, classic_client, mocked_builder, stubbed_workspace_config,
    ) -> None:
        builder, _session, _env, _envs = mocked_builder
        out = classic_client._spark_connect_builder(serverless=False)
        builder.sdkConfig.assert_called_once()
        builder.serverless.assert_not_called()
        assert out is builder.sdkConfig.return_value

    def test_classic_falls_back_to_plain_builder_on_sdkconfig_conflict(
        self, classic_client, mocked_builder, stubbed_workspace_config,
    ) -> None:
        builder, _session, _env, _envs = mocked_builder
        builder.sdkConfig.side_effect = Exception(
            "sdkConfig must not be set when connection parameters are "
            "explicitly configured."
        )
        out = classic_client._spark_connect_builder(serverless=False)
        assert out is builder           # parameter-free builder (env/profile auth)

    def test_classic_other_builder_errors_propagate(
        self, classic_client, mocked_builder, stubbed_workspace_config,
    ) -> None:
        builder, _session, _env, _envs = mocked_builder
        builder.sdkConfig.side_effect = RuntimeError("network down")
        with pytest.raises(RuntimeError, match="network down"):
            classic_client._spark_connect_builder(serverless=False)


class TestGrpcCaBundle:
    def test_sets_grpc_roots_from_ssl_cert_file(self, monkeypatch, tmp_path):
        ca = tmp_path / "ca.crt"
        ca.write_text("-----BEGIN CERTIFICATE-----\n")
        monkeypatch.delenv("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", raising=False)
        monkeypatch.setenv("SSL_CERT_FILE", str(ca))
        from yggdrasil.databricks.client import DatabricksClient
        DatabricksClient._ensure_grpc_ca_bundle()
        import os
        assert os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] == str(ca)

    def test_does_not_override_existing(self, monkeypatch):
        monkeypatch.setenv("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", "/already/set.pem")
        from yggdrasil.databricks.client import DatabricksClient
        DatabricksClient._ensure_grpc_ca_bundle()
        import os
        assert os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] == "/already/set.pem"


class TestClassicBranch:

    def test_public_specs_dont_call_add_artifacts(
        self, classic_client, mocked_builder, stubbed_workspace_config,
        fake_registry,
    ) -> None:
        # The fake registry echoes the specs back without a ``local:``
        # prefix, so the classic-compute branch has nothing to upload.
        _builder, session, _env_cls, env_instances = mocked_builder
        classic_client.spark()
        assert env_instances == []
        session.addArtifacts.assert_not_called()

    def test_local_specs_route_to_add_artifacts(
        self, classic_client, mocked_builder, stubbed_workspace_config,
        fake_registry, tmp_path, monkeypatch,
    ) -> None:
        # When the registry returns a ``local:`` spec, the classic
        # branch ships the local wheel through addArtifacts with
        # ``pyfile=True``. We rebind the fake registry to emit one
        # such spec on top of the default.
        _builder, session, _env_cls, env_instances = mocked_builder
        from yggdrasil.databricks import registry as reg_mod

        wheel = tmp_path / "demo-1.0-py3-none-any.whl"
        wheel.write_bytes(b"PK\x03\x04")

        class LocalRegistry:
            def __init__(self, **_kwargs):
                pass

            def publish_many(self, deps, *, check_public=False):
                specs = [f"ygg[data,databricks]=={_ygg_version}", f"local:{wheel}"]
                return specs, []

        monkeypatch.setattr(reg_mod, "WorkspacePyPIRegistry", LocalRegistry)
        classic_client.spark("demo")
        session.addArtifacts.assert_called_once_with(
            str(wheel), pyfile=True,
        )
