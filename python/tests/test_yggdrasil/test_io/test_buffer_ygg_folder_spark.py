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
import time

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


# ---------------------------------------------------------------------------
# End-to-end streaming — worker writes Arrow batches; Spark consumes via
# structured streaming
# ---------------------------------------------------------------------------


class TestYGGFolderSparkStreamingIntegration(SparkTestCase):
    """End-to-end producer → folder → Spark structured streaming.

    A daemon producer thread writes successive Arrow batches into
    a :class:`YGGFolderIO` (one parquet child per write). The Spark
    side runs a structured-streaming query with a ``processingTime``
    trigger so the source materialises files incrementally as
    micro-batches. The test asserts:

    - every produced row is eventually consumed by the streaming
      query (poll until catch-up rather than racing on a fixed
      timeout),
    - the producer thread's checkpoint records carry the *driver's*
      compute URL — propagated through the ``YGG_OWNER_URL`` env var
      so worker-side calls to :func:`compute_identifier_url` match
      the driver's identity.
    """

    @staticmethod
    def _wait_for(predicate, timeout: float, interval: float = 0.5) -> bool:
        """Poll ``predicate()`` until it returns truthy or *timeout* elapses."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if predicate():
                    return True
            except Exception:
                pass
            time.sleep(interval)
        return bool(predicate())

    def test_streaming_consumes_concurrent_worker_writes(self):
        import threading

        folder = self.tmp_path / "stream_e2e"
        folder.mkdir()

        connector = YGGFolderSparkConnector(YGGFolderIO(path=str(folder)))
        driver_url = connector.driver_owner_url()
        # Propagate the driver's id onto the Spark conf and into the
        # in-process env so the producer thread (and any executor that
        # later starts) returns the same value from compute_identifier_url.
        connector.propagate_owner_url(self.spark, driver_url)
        assert self.spark.conf.get("spark.yggdrasil.owner_url") == driver_url

        # Sync seed write: a single row so the streaming source has
        # a committed schema before we hand it the streaming reader.
        with YGGFolderIO(path=str(folder)) as io:
            io.write_arrow_table(_make_table(0, 1))
        produced: list[int] = [0]

        # Streaming reader must be set up after the seed lands so
        # ``collect_schema`` succeeds — but BEFORE the producer
        # writes its remaining batches, so they trigger genuine
        # micro-batches.
        with YGGFolderIO(path=str(folder)) as io:
            stream_df = io.spark_connector().read_stream(self.spark)
        assert stream_df.isStreaming

        out_dir = self.tmp_path / "stream_out"
        chk_dir = self.tmp_path / "stream_chkpt"

        query = (
            stream_df.writeStream
            .format("parquet")
            .option("path", str(out_dir))
            .option("checkpointLocation", str(chk_dir))
            .trigger(processingTime="500 milliseconds")
            .start()
        )

        # Producer thread — writes 4 more batches of 3 rows each
        # while the streaming query is live, with a small sleep so
        # multiple micro-batches actually fire.
        producer_error: list[BaseException] = []

        def producer() -> None:
            import os as _os
            prev = _os.environ.get("YGG_OWNER_URL")
            _os.environ["YGG_OWNER_URL"] = driver_url
            try:
                with YGGFolderIO(path=str(folder)) as io:
                    for batch_idx in range(1, 5):
                        start = batch_idx * 4
                        table = _make_table(start, 3)
                        io.write_arrow_table(table, mode="APPEND")
                        produced.extend(range(start, start + 3))
                        time.sleep(0.2)
                    # Final commit — checkpoint records the
                    # driver's URL because YGG_OWNER_URL is set.
                    io.checkpoint("producer-drain")
            except BaseException as exc:  # noqa: BLE001
                producer_error.append(exc)
            finally:
                if prev is None:
                    _os.environ.pop("YGG_OWNER_URL", None)
                else:
                    _os.environ["YGG_OWNER_URL"] = prev

        worker = threading.Thread(target=producer, daemon=True)
        worker.start()

        try:
            worker.join(timeout=30.0)
            assert not worker.is_alive(), "producer thread did not finish"
            assert not producer_error, f"producer raised: {producer_error[0]!r}"

            # Poll until the streaming query has consumed every row.
            def caught_up() -> bool:
                try:
                    return (
                        self.spark.read.parquet(str(out_dir)).count()
                        >= len(produced)
                    )
                except Exception:
                    return False

            assert self._wait_for(caught_up, timeout=60.0), (
                "Spark streaming query never caught up with "
                f"producer (expected {len(produced)} rows)."
            )
        finally:
            if query.isActive:
                query.stop()

        # Final assertions — compare the consumed row set to what
        # the producer actually wrote.
        consumed_rows = (
            self.spark.read.parquet(str(out_dir)).orderBy("id").collect()
        )
        consumed_ids = sorted(r["id"] for r in consumed_rows)
        assert consumed_ids == sorted(produced), (
            f"missing rows: {sorted(set(produced) - set(consumed_ids))}"
        )

        # Owner attribution: the producer's checkpoint records
        # the driver's URL because YGG_OWNER_URL was set to that
        # value while the thread ran.
        with YGGFolderIO(path=str(folder)) as io:
            records = io.list_checkpoints()
        owners = {r.get("owner") for r in records}
        assert driver_url in owners, (
            f"driver URL {driver_url!r} missing from {owners!r}"
        )

    def test_worker_owner_url_env_matches_driver(self):
        """When ``YGG_OWNER_URL`` is set in a worker env, the worker's
        :func:`compute_identifier_url` returns the propagated value
        verbatim — the override beats Databricks-env detection so a
        coordinator can pin all workers to the driver's identity."""
        import os as _os
        from yggdrasil.io.buffer._concurrency import compute_identifier_url

        folder = self.tmp_path / "owner_env"
        folder.mkdir()
        connector = YGGFolderSparkConnector(YGGFolderIO(path=str(folder)))
        driver_url = connector.driver_owner_url()

        prev = _os.environ.get("YGG_OWNER_URL")
        try:
            _os.environ["YGG_OWNER_URL"] = driver_url
            assert compute_identifier_url() == driver_url
        finally:
            if prev is None:
                _os.environ.pop("YGG_OWNER_URL", None)
            else:
                _os.environ["YGG_OWNER_URL"] = prev


# ---------------------------------------------------------------------------
# Owner-URL propagation — driver-only behaviour, no real Spark needed
# ---------------------------------------------------------------------------


class _StubConf:
    """In-memory stand-in for ``SparkSession.conf``.

    Just enough surface for :meth:`YGGFolderSparkConnector.propagate_owner_url`
    and :meth:`commit_checkpoint` — they only call ``conf.set(key, value)``.
    """

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self.values[key] = value


class _StubSpark:
    """In-memory stand-in for a SparkSession (driver-only surface)."""

    def __init__(self) -> None:
        self.conf = _StubConf()


class TestOwnerURLPropagation:
    """Driver-only paths exercised against a stub SparkSession.

    Real Spark is the integration target (covered by the
    ``SparkTestCase`` suites above); these tests focus on what the
    connector itself does — capture the driver URL, push it to the
    Spark conf under both keys, fall back gracefully when the
    session is absent.
    """

    def _connector(self, tmp_path):
        folder = tmp_path / "data"
        folder.mkdir()
        io = YGGFolderIO(path=str(folder))
        return YGGFolderSparkConnector(io), io, folder

    def test_driver_owner_url_returns_compute_url(self, tmp_path):
        connector, _io, _folder = self._connector(tmp_path)
        url = connector.driver_owner_url()
        assert isinstance(url, str)
        assert url.startswith("host://") or url.startswith("databricks://")

    def test_propagate_owner_url_writes_both_conf_keys(self, tmp_path):
        connector, _io, _folder = self._connector(tmp_path)
        spark = _StubSpark()
        url = connector.propagate_owner_url(
            spark, owner_url="host://driver/123",
        )
        assert url == "host://driver/123"
        # Driver-readable conf key.
        assert spark.conf.values["spark.yggdrasil.owner_url"] == url
        # Executor-side env propagation.
        assert (
            spark.conf.values["spark.executorEnv.YGG_OWNER_URL"] == url
        )

    def test_propagate_owner_url_defaults_to_driver_url(self, tmp_path):
        connector, _io, _folder = self._connector(tmp_path)
        spark = _StubSpark()
        captured = connector.propagate_owner_url(spark)
        assert captured.startswith("host://") or captured.startswith("databricks://")
        assert spark.conf.values["spark.yggdrasil.owner_url"] == captured

    def test_commit_checkpoint_records_driver_owner(self, tmp_path):
        connector, io, _folder = self._connector(tmp_path)
        spark = _StubSpark()
        record = connector.commit_checkpoint(
            spark,
            message="batch-1",
            owner="databricks://driver-cluster/42?host=driver&job=99",
        )
        assert (
            record["owner"]
            == "databricks://driver-cluster/42?host=driver&job=99"
        )
        # Propagation happened too.
        assert (
            spark.conf.values["spark.yggdrasil.owner_url"]
            == record["owner"]
        )
        # And the log file got the same content.
        records = io.list_checkpoints()
        assert records[-1]["owner"] == record["owner"]

    def test_commit_checkpoint_propagate_false_skips_conf(self, tmp_path):
        connector, _io, _folder = self._connector(tmp_path)
        spark = _StubSpark()
        connector.commit_checkpoint(
            spark,
            message="quiet",
            owner="host://nopropagate/1",
            propagate=False,
        )
        assert spark.conf.values == {}

    def test_commit_checkpoint_without_spark_session_still_records(self, tmp_path):
        """No active Spark session and ``propagate=False`` → fall
        back to a driver-local checkpoint with the local URL."""
        connector, io, _folder = self._connector(tmp_path)
        record = connector.commit_checkpoint(
            spark=None, message="local", propagate=False,
        )
        assert "owner" in record
        assert record["message"] == "local"
        assert io.list_checkpoints()[-1]["owner"] == record["owner"]
