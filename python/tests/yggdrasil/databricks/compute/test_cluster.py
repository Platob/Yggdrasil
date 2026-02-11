import logging
import os
import sys
import unittest
import datetime as dt

import numpy as np
import pandas
import pandas as pd
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


def rand_hex(rng: np.random.Generator, rows: int, nbytes: int = 8) -> pd.Series:
    # generate random bytes -> hex strings
    b = rng.integers(0, 256, size=(rows, nbytes), dtype=np.uint8)
    return pd.Series([row.tobytes().hex() for row in b], dtype="string")


class TestCluster(unittest.TestCase):

    @staticmethod
    def make_big_df(
        rows=2_000_000,
        float_cols=20,
        int_cols=10,
        str_cols=5,
        seed=42,
        string_mode="string",  # "string" or "object"
    ):
        rng = np.random.default_rng(seed)
        data = {}

        # floats
        for i in range(float_cols):
            data[f"f{i}"] = rng.standard_normal(rows)

        # ints
        for i in range(int_cols):
            data[f"i{i}"] = rng.integers(0, 1_000_000, size=rows, dtype=np.int64)

        # timestamps
        start = np.datetime64("2020-01-01")
        data["ts"] = start + rng.integers(0, 1_000_000, size=rows).astype("timedelta64[s]")

        # categorical
        data["venue"] = pd.Categorical(rng.choice(["ICE", "CME", "EEX", "NASDAQ"], size=rows))

        # strings
        for i in range(str_cols):
            s = rand_hex(rng, rows, nbytes=8)  # swap to rand_ascii / rand_object_strings if you want
            if string_mode == "object":
                s = s.astype(object)
            data[f"s{i}"] = s

        return pd.DataFrame(data)

    @classmethod
    def setUpClass(cls):
        cls.workspace = Workspace().connect()
        cls.cluster = cls.workspace.clusters().push_python_environment()
        cls.cluster.install_temporary_libraries(libraries=["yggdrasil"])

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
        def f():
            return "ok"

        with self.cluster.context() as context:
            result = context.command(func=f).start().wait()

        self.assertEqual("ok", result)

    def test_execute_error(self):
        def f():
            raise ValueError("error")

        with pytest.raises(ValueError):
            with self.cluster.system_context as context:
                _ = context.command(func=f).start().wait()

        with pytest.raises(ValueError):
            with self.cluster.context() as context:
                _ = context.command(func=f).start().wait()

    def test_decorator(self):
        @self.cluster.system_context.decorate(
            environ={
                "TEST_ENV": "testenv"
            }
        )
        def decorated(a: int):
            env = os.environ["TEST_ENV"]

            return {
                "os": os.environ,
                "value": a,
                "env": env
            }

        result = decorated(1)

        assert result["os"]
        assert "DATABRICKS_RUNTIME_VERSION" in result["os"].keys()
        assert result["value"] == 1
        assert result["env"] == "testenv"

    def test_data_decorator(self):
        @self.cluster.system_context.decorate
        def decorated(df: pd.DataFrame):
            assert os.getenv("DATABRICKS_RUNTIME_VERSION") is not None
            return df

        pdf = self.make_big_df(1024)
        result = decorated(df=pdf)

        self.assertTrue(isinstance(result, pandas.DataFrame))

    def test_static_decorator(self):
        os.environ["TEST_ENV"] = "testenv"

        @databricks_remote_compute(
            env_keys=["TEST_ENV"]
        )
        def decorated(a: int):
            env = os.environ["TEST_ENV"]

            return {
                "os": os.environ,
                "value": a,
                "env": env
            }

        result = decorated(1)

        assert result["os"]
        assert "DATABRICKS_RUNTIME_VERSION" in result["os"].keys()
        assert result["value"] == 1
        assert result["env"] == "testenv"

    def test_install_temporary_lib(self):
        self.cluster.install_temporary_libraries(["path/to/folder", "pandas"])

    def test_execute_sql(self):
        result = self.cluster.context(language=Language.SQL).command("SELECT 1", language=Language.SQL).start()

        self.assertTrue(result)
