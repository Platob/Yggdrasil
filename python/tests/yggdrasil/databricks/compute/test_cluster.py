import os
import unittest

from databricks.sdk.service.compute import Language
from mongoengine import DynamicDocument

from yggdrasil.databricks import Workspace
from yggdrasil.databricks.compute.cluster import Cluster
from yggdrasil.databricks.compute.remote import databricks_remote_compute


class Cities(DynamicDocument):
    meta = {'collection': 'cities'}

class TestCluster(unittest.TestCase):

    def setUp(self):
        self.workspace = Workspace(host="xxx.cloud.databricks.com").connect()
        self.cluster = Cluster.replicated_current_environment(workspace=self.workspace)
        # self.cluster.restart()

    def test_get_current_token(self):
        assert Workspace(
            host=self.workspace.host,
            token=self.workspace.current_token()
        ).current_user.user_name == self.workspace.current_user.user_name

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
        @self.cluster.execution_decorator
        def decorated(a: int):
            return {
                "os": os.environ,
                "value": a
            }

        result = decorated(1)

        assert result["os"]
        assert result["value"] == 1

    def test_install_temporary_lib(self):
        Cluster(
            workspace=self.workspace,
            cluster_id="xxx"
        ).install_temporary_libraries(["path/to/folder", "pandas"])

    def test_repeated_decorator(self):
        @self.cluster.execution_decorator
        def decorated(a: int):
            return {
                "os": os.environ,
                "value": a
            }

        for i in range(2):
            result = decorated(i)

            assert result["os"]
            assert result["value"] == i

    def test_databricks_remote_compute_decorator(self):
        @databricks_remote_compute(workspace=self.workspace)
        def decorated(a: int):
            return os.environ

        for i in range(2):
            result = decorated(i)

            assert result is not None

    def test_decorator_broadcast_credentials(self):
        wk = self.workspace

        @databricks_remote_compute(workspace=self.workspace)
        def decorated(a: int):
            return wk.connect().current_user

        for i in range(4):
            result = decorated(i)
            print(result)
            assert result is not None

    def test_execute_sql(self):
        result = self.cluster.execution_context(language=Language.SQL).execute("SELECT 1")

        print(result)