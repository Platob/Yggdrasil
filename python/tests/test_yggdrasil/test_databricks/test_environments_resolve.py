"""Unit tests for the centralized base-environment discovery on
:class:`~yggdrasil.databricks.environments.service.Environments` —
``resolve`` (direct path / named stem / auto) and ``client_project``.

Discovery used to live in ``job/service.py``; it now belongs to the dedicated
``dbc.environments`` service. These exercise the resolution branches directly
against mocked ``list`` / ``get`` / workspace lookups.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from yggdrasil.databricks.environments.environment import Environment
from yggdrasil.databricks.environments.service import Environments


def _service() -> Environments:
    return Environments(client=MagicMock())


class TestResolveDirectPath:
    def test_existing_yml_path_wraps_as_environment(self) -> None:
        svc = _service()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_") as from_:
            path = from_.return_value
            path.exists.return_value = True
            path.full_path.return_value = (
                "/Workspace/Shared/environment/ygg/0.8.58/ygg-0.8.58-py311.yml"
            )
            env = svc.resolve("/Workspace/Shared/environment/ygg/0.8.58/ygg-0.8.58-py311.yml")
        assert isinstance(env, Environment)
        assert env.serverless.endswith("ygg-0.8.58-py311.yml")
        assert env.name == "ygg-0.8.58-py311"

    def test_missing_path_returns_none(self) -> None:
        svc = _service()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_") as from_:
            from_.return_value.exists.return_value = False
            assert svc.resolve("/Workspace/Shared/environment/nope.yml") is None


class TestResolveNamedStem:
    def test_matches_deployed_stem(self) -> None:
        svc = _service()
        target = Environment(name="meteologica-1.0.0-py311",
                             serverless="/ws/meteologica.yml")
        with patch.object(svc, "list", return_value=[
            Environment(name="ygg-0.8.58-py311", serverless="/ws/ygg.yml"),
            target,
        ]):
            assert svc.resolve("meteologica-1.0.0-py311") is target

    def test_unknown_stem_returns_none(self) -> None:
        svc = _service()
        with patch.object(svc, "list", return_value=[]):
            assert svc.resolve("ghost-9.9.9-py311") is None


class TestResolveAuto:
    def test_prefers_client_project(self) -> None:
        svc = _service()
        proj = Environment(name="myproj-2.0.0-py311", serverless="/ws/myproj.yml")
        with patch.object(svc, "client_project", return_value=proj) as cp, \
                patch.object(svc, "get") as get:
            assert svc.resolve() is proj
            cp.assert_called_once()
            get.assert_not_called()

    def test_falls_back_to_ygg(self) -> None:
        svc = _service()
        ygg = Environment(name="ygg-0.8.58-py311", serverless="/ws/ygg.yml")
        with patch.object(svc, "client_project", return_value=None), \
                patch.object(svc, "get", return_value=ygg) as get:
            assert svc.resolve() is ygg
            get.assert_called_once_with("ygg", workspace_dir=None)

    def test_returns_none_when_nothing_deployed(self) -> None:
        svc = _service()
        with patch.object(svc, "client_project", return_value=None), \
                patch.object(svc, "get", return_value=None):
            assert svc.resolve() is None


class TestClientProject:
    def test_reads_local_pyproject_and_looks_up(self) -> None:
        svc = _service()
        proj = Environment(name="ygg-0.8.58-py311", serverless="/ws/ygg.yml")
        with patch("yggdrasil.databricks.environments.service.find_pyproject",
                   return_value="/repo/pyproject.toml"), \
                patch("yggdrasil.databricks.environments.service.read_pyproject",
                      return_value={"name": "ygg", "version": "0.8.58"}), \
                patch.object(svc, "get", return_value=proj) as get:
            assert svc.client_project() is proj
            get.assert_called_once_with("ygg", "0.8.58", workspace_dir=None)

    def test_no_pyproject_returns_none(self) -> None:
        svc = _service()
        with patch("yggdrasil.databricks.environments.service.find_pyproject",
                   side_effect=OSError):
            assert svc.client_project() is None
