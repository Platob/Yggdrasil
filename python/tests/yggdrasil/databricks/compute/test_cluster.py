import logging
import os
import sys
import unittest

import pytest
from databricks.sdk.service.compute import Language

from yggdrasil.databricks import Workspace

# ---- logging to stdout ----
logger = logging.getLogger("test")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class TestCluster(unittest.TestCase):

    def setUp(self):
        self.workspace = Workspace().connect()
        self.cluster = self.workspace.clusters().push_python_environment()
        # self.cluster.restart()
        self.venv = self.cluster.pull_python_environment()

    def test_cluster_dyn_properties(self):
        assert self.cluster.details
        assert self.cluster.python_version

    def test_list_spark_versions(self):
        result = self.cluster.spark_versions()
        latest = self.cluster.latest_spark_version(python_version=sys.version_info)

        assert result
        assert latest

    def test_execute(self):
        def test():
            return "ok"

        with self.cluster.context() as context:
            result = context.execute(test)

        self.assertEqual("ok", result)

    def test_execute_error(self):
        def f():
            raise ValueError("error")

        with pytest.raises(ValueError):
            with self.cluster.context() as context:
                _ = context.execute(f)

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
        self.cluster.install_temporary_libraries(["path/to/folder", "pandas"])

    def test_execute_sql(self):
        result = self.cluster.context(language=Language.SQL).execute("SELECT 1")

        self.assertTrue(result)
