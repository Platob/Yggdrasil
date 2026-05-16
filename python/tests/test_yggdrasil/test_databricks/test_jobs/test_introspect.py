"""Unit tests for static introspection helpers used by JobTask.from_callable."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from yggdrasil.databricks.jobs.introspect import (
    ModuleDependency,
    dependencies_to_pip_specs,
    resolve_module_dependency,
    sniff_env_vars,
    sniff_imports,
)


class TestSniffImports:
    def test_finds_top_level_imports_from_both_shapes(self):
        src = (
            "import polars as pl\n"
            "import pandas\n"
            "from pyarrow.compute import cast\n"
            "from os import path\n"
        )
        assert sniff_imports(src) == {"polars", "pandas", "pyarrow", "os"}

    def test_skips_relative_imports(self):
        src = "from . import helpers\nfrom ..util import x\n"
        assert sniff_imports(src) == set()

    def test_invalid_source_returns_empty(self):
        assert sniff_imports("import :::\n!!!") == set()


class TestSniffEnvVars:
    def test_finds_three_idiomatic_shapes(self):
        src = (
            "import os\n"
            "a = os.getenv('FOO')\n"
            "b = os.getenv('BAR', 'default')\n"
            "c = os.environ['BAZ']\n"
            "d = os.environ.get('QUX')\n"
        )
        assert sniff_env_vars(src) == {"FOO", "BAR", "BAZ", "QUX"}

    def test_skips_dynamic_calls(self):
        src = (
            "import os\n"
            "k = 'FOO'\n"
            "a = os.getenv(k)\n"
            "b = os.environ[k]\n"
        )
        assert sniff_env_vars(src) == set()


class TestResolveModuleDependency:
    def test_stdlib_short_circuits(self):
        dep = resolve_module_dependency("os")
        assert dep.kind == "stdlib"
        assert dep.project is None

    def test_unknown_module_returns_unknown(self):
        dep = resolve_module_dependency("definitely_not_a_real_module_xyz")
        assert dep.kind == "unknown"

    def test_real_pypi_install_marked_pypi(self):
        # pyarrow is a hard dependency of the project; reliable to test.
        dep = resolve_module_dependency("pyarrow")
        assert dep.kind == "pypi"
        assert dep.project == "pyarrow"
        assert dep.version is not None


class TestDependenciesToPipSpecs:
    def test_pins_pypi_versions_by_default(self):
        with patch(
            "yggdrasil.databricks.jobs.introspect.resolve_module_dependency"
        ) as mock_resolve:
            mock_resolve.side_effect = lambda m: ModuleDependency(
                module=m, project=m, version="1.2.3", kind="pypi",
            )
            specs = dependencies_to_pip_specs(["polars", "pandas"])
        assert specs == ["pandas==1.2.3", "polars==1.2.3"]

    def test_drops_stdlib_and_excluded(self):
        with patch(
            "yggdrasil.databricks.jobs.introspect.resolve_module_dependency"
        ) as mock_resolve:
            mock_resolve.side_effect = lambda m: ModuleDependency(
                module=m,
                project=m,
                version="1.0",
                kind="stdlib" if m == "os" else "pypi",
            )
            specs = dependencies_to_pip_specs(
                ["os", "polars", "mlflow"],  # mlflow is in DEFAULT_EXCLUDED_MODULES
            )
        assert specs == ["polars==1.0"]

    def test_editable_falls_back_to_bare_without_workspace_pypi(self):
        with patch(
            "yggdrasil.databricks.jobs.introspect.resolve_module_dependency"
        ) as mock_resolve:
            mock_resolve.return_value = ModuleDependency(
                module="my_pkg", project="my_pkg", version="0.0.1",
                kind="editable", source_path="/tmp/my_pkg",
            )
            specs = dependencies_to_pip_specs(["my_pkg"])
        assert specs == ["my_pkg==0.0.1"]

    def test_editable_uploads_via_workspace_pypi(self):
        publisher = MagicMock()
        # publish() returns a Path-like object with .full_path()
        published = MagicMock()
        published.full_path.return_value = (
            "/Workspace/Shared/.ygg/pypi/simple/my-pkg/my_pkg-0.0.1-py3-none-any.whl"
        )
        publisher.publish.return_value = published
        with patch(
            "yggdrasil.databricks.jobs.introspect.resolve_module_dependency"
        ) as mock_resolve:
            mock_resolve.return_value = ModuleDependency(
                module="my_pkg", project="my_pkg", version="0.0.1",
                kind="editable", source_path="/tmp/my_pkg",
            )
            specs = dependencies_to_pip_specs(
                ["my_pkg"], workspace_pypi=publisher,
            )
        publisher.publish.assert_called_once_with(
            "my_pkg", source_path="/tmp/my_pkg", version="0.0.1",
        )
        assert specs == [
            "my_pkg @ /Workspace/Shared/.ygg/pypi/simple/my-pkg/"
            "my_pkg-0.0.1-py3-none-any.whl",
        ]

    def test_publisher_failure_falls_back_to_bare(self):
        publisher = MagicMock()
        publisher.publish.side_effect = RuntimeError("upload failed")
        with patch(
            "yggdrasil.databricks.jobs.introspect.resolve_module_dependency"
        ) as mock_resolve:
            mock_resolve.return_value = ModuleDependency(
                module="my_pkg", project="my_pkg", version="0.0.1",
                kind="editable",
            )
            specs = dependencies_to_pip_specs(
                ["my_pkg"], workspace_pypi=publisher,
            )
        assert specs == ["my_pkg==0.0.1"]
