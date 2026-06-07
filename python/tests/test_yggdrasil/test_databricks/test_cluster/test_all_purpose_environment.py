"""``Clusters.all_purpose_cluster`` library composition — by default a cluster
installs the seeded **generic environment** (zero-PyPI wheels) rather than
``pip install``-ing ygg from PyPI; ``environment`` overrides the layer."""
from __future__ import annotations

from unittest.mock import MagicMock

from yggdrasil.databricks.environments.service import (
    WORKSPACE_ENV_DIR,
    environment_folder,
    ygg_base_environment_name,
)
from yggdrasil.databricks.tests import DatabricksTestCase

_REQS = "/Workspace/Shared/environment/ygg/ygg-1.0-py312.requirements.txt"


def _seeded_env_requirements() -> str:
    # The mock client can't resolve a cluster runtime → _default_ygg_layer falls
    # back to the local interpreter's Python, the new project-folder layout.
    name = ygg_base_environment_name()
    return f"{WORKSPACE_ENV_DIR}/{environment_folder('ygg')}/{name}.requirements.txt"


class TestAllPurposeClusterEnvironment(DatabricksTestCase):
    def _clusters(self):
        # all_purpose_cluster lists existing (none) then calls create — stub both
        # so we can capture the libraries it composed.
        clusters = self.clusters
        clusters.list = MagicMock(return_value=iter([]))
        created = MagicMock()
        created.cluster_name = "test-cluster"
        clusters.create = MagicMock(return_value=created)
        return clusters

    def test_environment_replaces_pypi_ygg_with_generic_env(self):
        clusters = self._clusters()
        clusters.all_purpose_cluster(
            name="test-cluster", environment=_REQS,
            single_user_name="me@co.com", wait=False,
        )
        libs = clusters.create.call_args.kwargs["libraries"]
        # The generic env requirements file is installed (Library(requirements=…)),
        # uv/dill ride along, and no PyPI ``ygg[…]`` resolve is added.
        self.assertIn(_REQS, libs)
        self.assertIn("uv", libs)
        self.assertIn("dill", libs)
        self.assertFalse(any(str(lib).startswith("ygg[") for lib in libs))

    def test_default_uses_seeded_environment_not_pypi(self):
        clusters = self._clusters()
        clusters.all_purpose_cluster(
            name="test-cluster", single_user_name="me@co.com", wait=False,
        )
        libs = clusters.create.call_args.kwargs["libraries"]
        # Default installs the seeded generic-env requirements (zero-PyPI), not a
        # PyPI ``ygg[…]`` resolve. uv/dill still ride along.
        self.assertIn(_seeded_env_requirements(), libs)
        self.assertIn("uv", libs)
        self.assertIn("dill", libs)
        self.assertFalse(any(str(lib).startswith("ygg[") for lib in libs))

    def test_get_or_create_defaults_to_seeded_environment(self):
        clusters = self._clusters()
        clusters.find_cluster = MagicMock(return_value=None)
        clusters.get_or_create(cluster_name="test-cluster", wait=False)
        libs = clusters.create.call_args.kwargs["libraries"]
        self.assertIn(_seeded_env_requirements(), libs)
        self.assertFalse(any(str(lib).startswith("ygg[") for lib in libs))
