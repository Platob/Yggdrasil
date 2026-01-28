import logging
import os
import sys
import unittest
import datetime as dt

import pytest
from databricks.sdk.service.compute import Language

from yggdrasil.databricks import Workspace
from yggdrasil.databricks.compute import databricks_remote_compute


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

    def test_cluster_dyn_properties(self):
        assert self.cluster.details
        assert self.cluster.python_version

    def test_list_spark_versions(self):
        result = self.cluster.spark_versions()
        latest = self.cluster.latest_spark_version(python_version=sys.version_info)

        assert result
        assert latest

    def test_command(self):
        def test(a: str, date: dt.date):
            return {
                "value": a,
                "date": date
            }

        cmd = self.cluster.context().command(func=test)
        today = dt.date.today()

        result = cmd(a="test", date=today)

        assert result
        assert result["value"] == "test"
        assert result["date"] == today

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
            with self.cluster.system_context as context:
                _ = context.execute(f)

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

    def test_static_decorator(self):
        @databricks_remote_compute(workspace=self.workspace)
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
