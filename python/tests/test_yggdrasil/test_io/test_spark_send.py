"""Spark-mode ``send_many`` tabular tests.

Drive :meth:`Session.send_many` with ``as_tabular=True`` +
``spark_session=...`` end-to-end through a shared :class:`SparkTestCase`
session. Requests cross the wire as a :data:`REQUEST_SCHEMA`-shaped
Arrow table — no driver-side :class:`Session` subclass is pickled to
the workers, so the path works over Spark Connect / Databricks Connect
where the cluster has no idea what ``tests.test_yggdrasil.…StubSession``
is. The worker side rebuilds requests from the Arrow rows and dispatches
them through a vanilla :class:`HTTPSession`, so the assertions stick to
the contract we *can* observe without a stub-on-worker:

* the returned :class:`Tabular` wraps a Spark frame matching
  :data:`RESPONSE_SCHEMA`,
* one input request produces one output row,
* multi-chunk batches union into one frame without losing rows,
* the returned frame is lazy (no executor work until an action), and
* an empty input iterator returns a schema-bearing empty frame.

Tests assert on the row-shape (URL-hash preservation, method
propagation) rather than specific status codes — those depend on
whatever the real endpoint replies with.
"""

from __future__ import annotations

from pyspark.sql import DataFrame as SparkDataFrame

from yggdrasil.io.response import RESPONSE_SCHEMA
from yggdrasil.io.tabular import Dataset
from yggdrasil.spark.tests import SparkTestCase

from ._helpers import StubSession, make_request


class TestSparkSend(SparkTestCase):
    """Driver-side smoke tests for ``send_many(as_tabular=True)`` on Spark."""

    def test_single_request_returns_lazy_one_row_tabular(self) -> None:
        s = StubSession()
        req = make_request("https://example.com/spark-send")

        tabular = s.send_many(
            iter([req]),
            as_tabular=True,
            spark_session=self.spark,
        )

        assert isinstance(tabular, Dataset)
        df = tabular.frame
        assert isinstance(df, SparkDataFrame)
        assert df.schema == RESPONSE_SCHEMA.to_spark_schema()

        rows = df.collect()
        assert len(rows) == 1
        row = rows[0]
        assert row["status_code"] is not None
        assert row["request_method"] == "GET"
        assert row["request_public_url_hash"] == req.public_url_hash

    def test_many_requests_union_into_one_frame(self) -> None:
        s = StubSession()
        reqs = [
            make_request(f"https://example.com/many/{i}")
            for i in range(5)
        ]

        tabular = s.send_many(
            iter(reqs),
            as_tabular=True,
            spark_session=self.spark,
        )

        df = tabular.frame
        assert df.schema == RESPONSE_SCHEMA.to_spark_schema()
        assert df.count() == len(reqs)

        out_hashes = {
            row["request_public_url_hash"]
            for row in df.select("request_public_url_hash").collect()
        }
        for req in reqs:
            assert req.public_url_hash in out_hashes

    def test_multi_chunk_union_preserves_rows(self) -> None:
        # Force ``_send_many_batches`` to emit multiple chunks by
        # setting ``batch_size`` well below the request count. The
        # per-chunk Spark frames must concat without losing rows.
        s = StubSession()
        reqs = [
            make_request(f"https://example.com/chunked/{i}")
            for i in range(12)
        ]

        tabular = s.send_many(
            iter(reqs),
            as_tabular=True,
            batch_size=4,
            spark_session=self.spark,
        )

        df = tabular.frame
        assert df.schema == RESPONSE_SCHEMA.to_spark_schema()
        assert df.count() == len(reqs)

        out_hashes = {
            row["request_public_url_hash"]
            for row in df.select("request_public_url_hash").collect()
        }
        assert out_hashes == {r.public_url_hash for r in reqs}

    def test_empty_iter_yields_empty_frame(self) -> None:
        s = StubSession()
        tabular = s.send_many(
            iter([]),
            as_tabular=True,
            spark_session=self.spark,
        )
        df = tabular.frame
        assert df.schema == RESPONSE_SCHEMA.to_spark_schema()
        assert df.count() == 0

    def test_returns_lazy_dataframe(self) -> None:
        # The contract: ``send_many(as_tabular=True)`` builds a Spark
        # plan and returns it without firing any executor work. Workers
        # run a vanilla ``HTTPSession`` — never the driver-side stub —
        # so the stub's ``calls`` list stays empty before *and* after
        # the action.
        s = StubSession()
        reqs = [
            make_request(f"https://example.com/lazy/{i}")
            for i in range(4)
        ]

        tabular = s.send_many(
            iter(reqs),
            as_tabular=True,
            spark_session=self.spark,
        )

        df = tabular.frame
        assert isinstance(df, SparkDataFrame)
        assert s.calls == []

        assert df.count() == len(reqs)
        assert s.calls == []

    def test_status_code_propagates(self) -> None:
        s = StubSession()
        tabular = s.send_many(
            iter(make_request(f"https://example.com/status/{i}") for i in range(3)),
            as_tabular=True,
            spark_session=self.spark,
        )
        df = tabular.frame
        codes = [row["status_code"] for row in df.select("status_code").collect()]
        assert len(codes) == 3
        assert all(isinstance(c, int) for c in codes)

    def test_spark_session_true_auto_discovers_active(self) -> None:
        # ``spark_session=True`` (or ``...``) delegates to
        # :meth:`PyEnv.spark_session`, which finds the active session
        # ``SparkTestCase`` has wired up. The returned tabular wraps
        # the same Spark frame as an explicit ``spark_session=`` arg.
        s = StubSession()
        req = make_request("https://example.com/auto-spark")

        tabular = s.send_many(
            iter([req]),
            as_tabular=True,
            spark_session=True,
        )

        assert isinstance(tabular, Dataset)
        assert tabular.frame.schema == RESPONSE_SCHEMA.to_spark_schema()
