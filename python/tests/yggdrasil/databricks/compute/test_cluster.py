import unittest

from databricks.sdk.service.compute import RuntimeEngine

from yggdrasil.databricks import Workspace
from yggdrasil.databricks.compute import Cluster


class TestCluster(unittest.TestCase):

    def setUp(self):
        self.workspace = Workspace(host="xxx.cloud.databricks.com")
        self.cluster = Cluster(workspace=self.workspace).create_or_update(
            cluster_name=self.workspace.current_user.user_name,
            runtime_engine=RuntimeEngine.PHOTON
        )

    def test_list_spark_versions(self):
        result = self.cluster.spark_versions()
        latest = self.cluster.latest_spark_version(python_version="current")

        assert result
        assert latest

    def test_get_or_create(self):
        cluster = self.cluster.create_or_update(
            cluster_name=self.cluster.workspace.current_user.user_name,
            single_user_name=self.cluster.workspace.current_user.user_name,
            runtime_engine=RuntimeEngine.PHOTON,
            autotermination_minutes=30,
            libraries=[
                "ygg",
                "datamanagement"
            ],
        )

        assert cluster is not None
