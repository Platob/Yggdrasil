"""The uniform ``default()`` accessor across the project's compute resources —
``dbc.environments.default()`` / ``warehouses.default()`` /
``compute.clusters.default()`` all resolve the resource named for the running
client project (its canonical distribution name / display name)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cluster.service import Clusters
from yggdrasil.databricks.environments.environment import Environment
from yggdrasil.databricks.environments.service import Environments
from yggdrasil.databricks.warehouse.service import Warehouses


class TestEnvironmentsDefault:
    def _service(self, project: str = "ygg") -> Environments:
        svc = Environments(client=MagicMock())
        svc.client.project = project
        return svc

    def test_prefers_local_pyproject(self) -> None:
        svc = self._service()
        proj = Environment(name="myproj-2.0.0-py311", serverless="/ws/myproj.yml")
        with patch.object(svc, "client_project", return_value=proj), \
                patch.object(svc, "get") as get:
            assert svc.default() is proj
            get.assert_not_called()

    def test_falls_back_to_client_project_env(self) -> None:
        svc = self._service("meteologica")
        env = Environment(name="meteologica-1.0.0-py311", serverless="/ws/m.yml")
        with patch.object(svc, "client_project", return_value=None), \
                patch.object(svc, "get", side_effect=[env]) as get:
            assert svc.default() is env
            get.assert_called_once_with("meteologica", None, python=None,
                                        workspace_dir=None, refresh=False)

    def test_falls_back_to_seeded_ygg(self) -> None:
        svc = self._service("meteologica")
        ygg = Environment(name="ygg-0.8.58-py311", serverless="/ws/ygg.yml")
        # client project env missing → seeded ygg.
        with patch.object(svc, "client_project", return_value=None), \
                patch.object(svc, "get", side_effect=[None, ygg]) as get:
            assert svc.default() is ygg
            assert get.call_count == 2
            assert get.call_args_list[-1].args[0] == "ygg"

    def test_none_when_nothing_deployed(self) -> None:
        svc = self._service("ygg")
        with patch.object(svc, "client_project", return_value=None), \
                patch.object(svc, "get", return_value=None):
            assert svc.default() is None


class TestWarehousesDefault:
    def test_delegates_to_find_default(self) -> None:
        svc = Warehouses(client=MagicMock())
        sentinel = object()
        with patch.object(svc, "find_default", return_value=sentinel) as fd:
            assert svc.default() is sentinel
            fd.assert_called_once_with(raise_error=True)

    def test_passes_raise_error_through(self) -> None:
        svc = Warehouses(client=MagicMock())
        with patch.object(svc, "find_default", return_value=None) as fd:
            assert svc.default(raise_error=False) is None
            fd.assert_called_once_with(raise_error=False)


class TestClustersDefault:
    def test_resolves_cluster_named_for_project(self) -> None:
        svc = Clusters(client=MagicMock())
        svc.client.product_name = "Meteologica"
        sentinel = object()
        with patch.object(svc, "find_cluster", return_value=sentinel) as fc:
            assert svc.default() is sentinel
            fc.assert_called_once_with(cluster_name="Meteologica", raise_error=False)

    def test_defaults_to_ygg_display_when_no_project(self) -> None:
        svc = Clusters(client=MagicMock())
        svc.client.product_name = None
        with patch.object(svc, "find_cluster", return_value=None) as fc:
            assert svc.default() is None
            fc.assert_called_once_with(cluster_name="Ygg", raise_error=False)
