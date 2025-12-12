import os
import unittest

from yggdrasil.databricks import Workspace
from yggdrasil.databricks.compute.cluster import Cluster
from yggdrasil.databricks.compute.remote import databricks_remote_compute

class TestCluster(unittest.TestCase):

    def setUp(self):
        self.workspace = Workspace(host="xxx.cloud.databricks.com")
        self.cluster = Cluster.replicated_current_environment(workspace=self.workspace)

    def test_cluster_dyn_properties(self):
        assert self.cluster.details
        assert self.cluster.python_version

    def test_list_spark_versions(self):
        result = self.cluster.spark_versions()
        latest = self.cluster.latest_spark_version(python_version="current")

        assert result
        assert latest

    def test_execute(self):
        def test():
            return "ok"

        with self.cluster.execution_context() as context:
            result = context.execute(test)

        assert result is not None

    def test_execute_error(self):
        def test():
            raise ValueError("error")

        with self.cluster.execution_context() as context:
            result = context.execute(test)

        assert result is not None

    def test_decorator(self):
        @self.cluster.remote_execute
        def decorated(a: int):
            return {
                "os": os.getenv("DATABRICKS_RUNTIME_VERSION"),
                "value": a
            }

        result = decorated(1)

        assert result["os"]
        assert result["value"] == 1

    def test_repeated_decorator(self):
        @self.cluster.remote_execute
        def decorated(a: int):
            return {
                "os": os.getenv("DATABRICKS_RUNTIME_VERSION"),
                "value": a
            }

        for i in range(2):
            result = decorated(i)

        assert result["os"]
        assert result["value"] == 1

    def test_databricks_remote_compute_decorator(self):
        @databricks_remote_compute(workspace=Workspace(host="xxx.cloud.databricks.com"))
        def decorated(a: int):
            return {
                "os": os.getenv("DATABRICKS_RUNTIME_VERSION"),
                "value": a
            }

        result = decorated(1)

        assert result["os"]
        assert result["value"] == 1
