"""Tests for the :class:`DynamicFrame` function-dependency scan.

The scan helper :func:`function_top_modules` lives in
:mod:`yggdrasil.spark.dependencies` — a pyspark-free module — so
the bulk of the test surface runs even when pyspark isn't
installed. The integration tests that exercise the full
``DynamicFrame._ensure_installed`` flow skip when pyspark is
unavailable.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasil.spark.dependencies import function_top_modules


def _make_fn_with_globals(code: str, **globs):
    """Build a function with a hand-built ``__globals__`` so the
    scan tests see exactly the bindings under inspection."""
    namespace: dict = dict(globs)
    exec(compile(code, "<test>", "exec"), namespace)
    return namespace["fn"]


def _has_pyspark() -> bool:
    try:
        import pyspark  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Scan helper — pyspark-free
# ---------------------------------------------------------------------------


class TestFunctionTopModules:

    def test_skips_stdlib_and_builtins(self) -> None:
        import json
        import os

        fn = _make_fn_with_globals(
            "def fn(x):\n    return json.dumps({'x': x, 'cwd': os.getcwd()})\n",
            json=json, os=os,
        )
        assert function_top_modules(fn) == set()

    def test_picks_up_third_party_top_package(self) -> None:
        import yggdrasil

        fn = _make_fn_with_globals(
            "def fn(x):\n    return yggdrasil.__version__\n",
            yggdrasil=yggdrasil,
        )
        assert function_top_modules(fn) == {"yggdrasil"}

    def test_picks_up_closure_modules(self) -> None:
        import yggdrasil

        # Closure-based capture — uses real ``def`` so __closure__
        # is wired up by Python.
        def outer():
            captured = yggdrasil  # noqa: F841

            def inner(x):
                return captured.__name__

            return inner

        fn = outer()
        # ``inner`` carries this test file's __globals__ (includes
        # pytest, yggdrasil, tests …); the assertion just checks
        # that closure cells contributed yggdrasil too.
        assert "yggdrasil" in function_top_modules(fn)

    def test_picks_up_callable_globals(self) -> None:
        from yggdrasil.environ import PyEnv

        fn = _make_fn_with_globals(
            "def fn(x):\n    return PyEnv.current()\n",
            PyEnv=PyEnv,
        )
        assert "yggdrasil" in function_top_modules(fn)


# ---------------------------------------------------------------------------
# DynamicFrame integration — only runs when pyspark is available
# ---------------------------------------------------------------------------


pyspark_required = pytest.mark.skipif(
    not _has_pyspark(), reason="pyspark not installed",
)


@pyspark_required
class TestInstalledModulesSeed:

    def test_constructor_accepts_seed(self) -> None:
        from yggdrasil.spark.frame import DynamicFrame

        df = MagicMock()
        frame = DynamicFrame(df, installed_modules={"ygg", "polars"})
        assert frame.installed_modules == {"ygg", "polars"}

    def test_default_is_empty_set(self) -> None:
        from yggdrasil.spark.frame import DynamicFrame

        df = MagicMock()
        frame = DynamicFrame(df)
        assert frame.installed_modules == set()
        assert isinstance(frame.installed_modules, set)


@pyspark_required
class TestEnsureInstalled:

    @pytest.fixture
    def fake_frame(self, monkeypatch):
        """``DynamicFrame`` bound to a mocked Spark session."""
        from yggdrasil.spark.frame import DynamicFrame

        session = MagicMock(name="SparkSession")
        # ``MagicMock`` auto-creates non-None attributes for every probed
        # name, including ``ygg_client``. That trips the production
        # ``_install_modules_on_executors`` branch that wraps a real
        # ``WorkspacePyPIRegistry`` around the (mock) client, which then
        # publishes through the actual filesystem instead of the
        # ``fake_build_archive`` monkeypatch this fixture sets up. Pin
        # ``ygg_client = None`` so the registry path stays inert and the
        # archive shim drives the call.
        session.ygg_client = None
        df = MagicMock()
        df.sparkSession = session

        from yggdrasil.io.path import _module_pack
        import pathlib

        monkeypatch.setattr(
            _module_pack, "resolve_module_root",
            lambda name: pathlib.Path(f"/fake/{name}"),
        )

        def fake_build_archive(root, dest=None):
            return pathlib.Path(f"/fake/{root.name}.zip")

        monkeypatch.setattr(_module_pack, "build_module_archive", fake_build_archive)
        return DynamicFrame(df), session

    def test_no_modules_in_function_still_ships_yggdrasil(self, fake_frame) -> None:
        """Even a closure with no third-party deps must ship ygg to executors.

        UDFs always go through ``mapInArrow`` which unpickles
        ``yggdrasil.pickle`` bytes on the executor — so ``yggdrasil`` is a
        load-time dependency of every transform regardless of what the user
        function does. Without auto-shipping it, a cluster running an older
        ygg surfaces as ``UnpicklingError: invalid load key, 'Y'``.
        """
        frame, session = fake_frame

        def fn(x):
            return x + 1

        new = frame._ensure_installed(fn)
        assert "yggdrasil" in new
        session.addArtifacts.assert_called_with(
            "/fake/yggdrasil.zip", pyfile=True,
        )

    def test_third_party_modules_get_shipped(self, fake_frame) -> None:
        frame, session = fake_frame
        import yggdrasil

        def fn(x):
            return yggdrasil.__version__ + str(x)

        new = frame._ensure_installed(fn)
        assert "yggdrasil" in new
        session.addArtifacts.assert_called_with(
            "/fake/yggdrasil.zip", pyfile=True,
        )
        assert "yggdrasil" in frame.installed_modules

    def test_repeat_call_is_noop(self, fake_frame) -> None:
        frame, session = fake_frame
        import yggdrasil

        def fn(x):
            return yggdrasil.__version__

        frame._ensure_installed(fn)
        first = session.addArtifacts.call_count
        frame._ensure_installed(fn)
        assert session.addArtifacts.call_count == first

    def test_registry_path_when_ygg_client_present(
        self, fake_frame, monkeypatch,
    ) -> None:
        frame, session = fake_frame
        import yggdrasil

        client = MagicMock(name="DatabricksClient")
        session.ygg_client = client

        publish_calls: list[str] = []

        class FakeRegistry:
            def __init__(self, client=None, base_path=None, local_cache=None):
                self.client = client

            def publish(self, obj, *, check_public=False):
                publish_calls.append(obj)
                return f"local:/fake/wheels/{obj}-1.0.whl", MagicMock(name="remote")

        from yggdrasil.databricks import registry as reg_mod
        monkeypatch.setattr(reg_mod, "WorkspacePyPIRegistry", FakeRegistry)

        def fn(x):
            return yggdrasil.__version__

        new = frame._ensure_installed(fn)
        assert "yggdrasil" in new
        assert "yggdrasil" in publish_calls
        # The wheel returned by the registry is shipped through
        # ``addArtifacts`` — not the fake fallback zip.
        session.addArtifacts.assert_called_with(
            "/fake/wheels/yggdrasil-1.0.whl", pyfile=True,
        )
