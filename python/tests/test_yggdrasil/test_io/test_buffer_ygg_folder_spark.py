"""Tests for the Spark connector on :class:`YGGFolderIO`.

PySpark is an optional extra; the entire module is skipped when it
isn't importable. The connector covers two surfaces:

- **Batch** — ``YGGFolderSparkConnector.read_batch`` pumps Arrow
  batches into Spark via ``mapInArrow``.
- **Stream** — ``read_stream`` returns a streaming DataFrame
  rooted on Spark's parquet streaming source.
- **Optional registration** — ``register_datasource`` plugs a
  ``"yggfolder"`` data source into the active session on PySpark
  4.0+; older versions return ``False`` and are skipped.
"""

from __future__ import annotations

import pathlib
import tempfile

import pyarrow as pa
import pytest

pytest.importorskip("pyspark")

from yggdrasil.io.buffer.nested import (  # noqa: E402
    YGGFolderIO,
    YGGFolderSparkConnector,
    register_datasource,
)
from yggdrasil.spark.tests import SparkTestCase  # noqa: E402


def _make_table(start: int, n: int = 4) -> pa.Table:
    return pa.table({
        "id": pa.array(list(range(start, start + n)), type=pa.int64()),
        "tag": pa.array([f"r-{i}" for i in range(start, start + n)]),
    })


class TestYGGFolderSparkBatch(SparkTestCase):
    def _populated_folder(self) -> pathlib.Path:
        folder = self.tmp_path / "data"
        folder.mkdir()
        with YGGFolderIO(path=str(folder)) as io:
            io.write_arrow_table(_make_table(0, 4))
            io.write_arrow_table(_make_table(4, 4), mode="APPEND")
        return folder

    def test_connector_requires_yggfolderio(self):
        with pytest.raises(TypeError):
            YGGFolderSparkConnector(object())  # type: ignore[arg-type]

    def test_read_batch_returns_dataframe(self):
        folder = self._populated_folder()
        with YGGFolderIO(path=str(folder)) as io:
            df = YGGFolderSparkConnector(io).read_batch(self.spark)
            assert df is not None
            rows = df.orderBy("id").collect()
        ids = [r["id"] for r in rows]
        assert ids == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_read_batch_via_io_spark_connector(self):
        folder = self._populated_folder()
        with YGGFolderIO(path=str(folder)) as io:
            df = io.spark_connector().read_batch(self.spark)
            count = df.count()
        assert count == 8

    def test_read_spark_frame_routes_through_connector(self):
        folder = self._populated_folder()
        with YGGFolderIO(path=str(folder)) as io:
            df = io.read_spark_frame()
            count = df.count()
        assert count == 8

    def test_predicate_pushed_through_arrow(self):
        # The predicate is applied at the Arrow layer (per batch
        # inside the mapInArrow function) — Spark sees only the
        # surviving rows, not the unfiltered stream.
        folder = self._populated_folder()
        with YGGFolderIO(path=str(folder)) as io:
            from yggdrasil.data.expr import col
            df = io.spark_connector().read_batch(
                self.spark, predicate=col("id") >= 4,
            )
            ids = sorted(r["id"] for r in df.collect())
        assert ids == [4, 5, 6, 7]


class TestYGGFolderSparkStream(SparkTestCase):
    def test_stream_dataframe_is_streaming(self):
        folder = self.tmp_path / "stream"
        folder.mkdir()
        # Pre-populate so the streaming source has a committed schema.
        with YGGFolderIO(path=str(folder)) as io:
            io.write_arrow_table(_make_table(0, 4))

        with YGGFolderIO(path=str(folder)) as io:
            df = io.spark_connector().read_stream(self.spark)
        assert df.isStreaming

    def test_stream_without_committed_schema_raises(self):
        folder = self.tmp_path / "empty"
        folder.mkdir()
        with YGGFolderIO(path=str(folder)) as io:
            with pytest.raises(RuntimeError):
                io.spark_connector().read_stream(self.spark)


class TestRegisterDatasource(SparkTestCase):
    def test_register_returns_bool(self):
        # Registration succeeds on PySpark 4+ (where
        # pyspark.sql.datasource exists) and gracefully returns
        # False on older versions. Either way, no exception.
        result = register_datasource(self.spark)
        assert isinstance(result, bool)

    def test_format_yggfolder_loads_when_registered(self):
        registered = register_datasource(self.spark)
        if not registered:
            pytest.skip(
                "PySpark version doesn't expose pyspark.sql.datasource"
            )

        folder = self.tmp_path / "ds"
        folder.mkdir()
        with YGGFolderIO(path=str(folder)) as io:
            io.write_arrow_table(_make_table(0, 5))

        df = (
            self.spark.read.format("yggfolder")
            .option("path", str(folder))
            .load()
        )
        assert sorted(r["id"] for r in df.collect()) == [0, 1, 2, 3, 4]
