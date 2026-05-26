"""PySpark read/write integration tests on LocalPath."""
from __future__ import annotations

import pathlib

import pyarrow as pa
import pytest

pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession

from yggdrasil.enums import Mode
from yggdrasil.path.local_path import LocalPath


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("test")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    # Register the session so read_spark_frame can resolve it via PyEnv.
    from yggdrasil.environ import PyEnv
    PyEnv.set_spark_session(session)
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# TestSparkWrite
# ---------------------------------------------------------------------------


class TestSparkWrite:

    def test_write_spark_dataframe_to_parquet(self, spark, tmp_path: pathlib.Path):
        sdf = spark.createDataFrame(
            [(1, "a"), (2, "b"), (3, "c")],
            schema=["id", "letter"],
        )
        p = LocalPath(str(tmp_path / "spark_out.parquet"), singleton_ttl=False)
        p.write_table(sdf)

        result = p.read_arrow_table()
        assert result.num_rows == 3
        assert result.column("id").to_pylist() == [1, 2, 3]
        assert result.column("letter").to_pylist() == ["a", "b", "c"]

    def test_write_spark_dataframe_to_ipc(self, spark, tmp_path: pathlib.Path):
        sdf = spark.createDataFrame(
            [(10, 1.5), (20, 2.5)],
            schema=["x", "y"],
        )
        p = LocalPath(str(tmp_path / "spark_out.ipc"), singleton_ttl=False)
        p.write_table(sdf)

        result = p.read_arrow_table()
        assert result.num_rows == 2
        assert result.column("x").to_pylist() == [10, 20]

    def test_write_spark_dataframe_overwrite(self, spark, tmp_path: pathlib.Path):
        sdf_first = spark.createDataFrame([(1,), (2,), (3,)], schema=["v"])
        sdf_second = spark.createDataFrame([(10,), (20,)], schema=["v"])

        p = LocalPath(str(tmp_path / "overwrite.parquet"), singleton_ttl=False)
        p.write_table(sdf_first)
        assert p.read_arrow_table().num_rows == 3

        p.write_table(sdf_second, mode=Mode.OVERWRITE)
        result = p.read_arrow_table()
        assert result.num_rows == 2
        assert result.column("v").to_pylist() == [10, 20]


# ---------------------------------------------------------------------------
# TestSparkRead
# ---------------------------------------------------------------------------


class TestSparkRead:

    def test_read_spark_frame_returns_spark_dataframe(self, spark, tmp_path: pathlib.Path):
        table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        p = LocalPath(str(tmp_path / "read_spark.parquet"), singleton_ttl=False)
        p.write_table(table)

        sdf = p.read_spark_frame()
        assert type(sdf).__module__.startswith("pyspark")
        assert type(sdf).__name__ == "DataFrame"

    def test_read_spark_frame_row_count(self, spark, tmp_path: pathlib.Path):
        table = pa.table({"val": list(range(50))})
        p = LocalPath(str(tmp_path / "count.parquet"), singleton_ttl=False)
        p.write_table(table)

        sdf = p.read_spark_frame()
        assert sdf.count() == 50

    def test_read_spark_frame_column_values(self, spark, tmp_path: pathlib.Path):
        table = pa.table({"name": ["alice", "bob"], "score": [95, 82]})
        p = LocalPath(str(tmp_path / "vals.parquet"), singleton_ttl=False)
        p.write_table(table)

        sdf = p.read_spark_frame()
        rows = sorted(sdf.collect(), key=lambda r: r["name"])
        assert rows[0]["name"] == "alice"
        assert rows[0]["score"] == 95
        assert rows[1]["name"] == "bob"
        assert rows[1]["score"] == 82


# ---------------------------------------------------------------------------
# TestSparkRoundTrip
# ---------------------------------------------------------------------------


class TestSparkRoundTrip:

    def test_spark_parquet_spark_values_match(self, spark, tmp_path: pathlib.Path):
        sdf_in = spark.createDataFrame(
            [(1, "one"), (2, "two"), (3, "three")],
            schema=["num", "word"],
        )
        p = LocalPath(str(tmp_path / "round.parquet"), singleton_ttl=False)
        p.write_table(sdf_in)

        sdf_out = p.read_spark_frame()
        rows_in = sorted(sdf_in.collect(), key=lambda r: r["num"])
        rows_out = sorted(sdf_out.collect(), key=lambda r: r["num"])
        assert len(rows_in) == len(rows_out)
        for r_in, r_out in zip(rows_in, rows_out):
            assert r_in["num"] == r_out["num"]
            assert r_in["word"] == r_out["word"]

    def test_spark_ipc_arrow_spark(self, spark, tmp_path: pathlib.Path):
        sdf_in = spark.createDataFrame(
            [(100, 3.14), (200, 2.71)],
            schema=["k", "v"],
        )
        p = LocalPath(str(tmp_path / "round.ipc"), singleton_ttl=False)
        p.write_table(sdf_in)

        arrow_table = p.read_arrow_table()
        assert arrow_table.num_rows == 2

        sdf_out = spark.createDataFrame(arrow_table.to_pandas())
        rows_out = sorted(sdf_out.collect(), key=lambda r: r["k"])
        assert rows_out[0]["k"] == 100
        assert abs(rows_out[0]["v"] - 3.14) < 1e-6
        assert rows_out[1]["k"] == 200
        assert abs(rows_out[1]["v"] - 2.71) < 1e-6

    def test_large_spark_dataframe(self, spark, tmp_path: pathlib.Path):
        rows = [(i, float(i * 2)) for i in range(1000)]
        sdf_in = spark.createDataFrame(rows, schema=["idx", "doubled"])

        p = LocalPath(str(tmp_path / "large.parquet"), singleton_ttl=False)
        p.write_table(sdf_in)

        sdf_out = p.read_spark_frame()
        assert sdf_out.count() == 1000

        # Spot-check a few values via Arrow for speed.
        arrow_out = p.read_arrow_table()
        idx_col = arrow_out.column("idx").to_pylist()
        doubled_col = arrow_out.column("doubled").to_pylist()
        assert 0 in idx_col
        assert 999 in idx_col
        pos = idx_col.index(500)
        assert doubled_col[pos] == 1000.0
