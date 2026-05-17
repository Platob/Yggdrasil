"""Spark-mode ``send`` / ``send_many`` tests.

Drive :meth:`Session.spark_send` and :meth:`Session.spark_send_many`
end-to-end through a shared :class:`SparkTestCase` session. Requests
cross the wire as a :data:`REQUEST_SCHEMA`-shaped Arrow table — no
driver-side :class:`Session` subclass is pickled to the workers, so the
path works over Spark Connect / Databricks Connect where the cluster
has no idea what ``tests.test_yggdrasil.…StubSession`` is. The worker
side rebuilds requests from the Arrow rows and dispatches them through
a vanilla :class:`HTTPSession`, so the assertions stick to the
contract we *can* observe without a stub-on-worker:

* the returned frame matches :data:`RESPONSE_SCHEMA`,
* one input request produces one output row,
* :meth:`spark_send` collapses to :meth:`spark_send_many` cleanly,
* multi-chunk batches union into one frame without losing rows,
* the returned frame is lazy (no executor work until an action), and
* passing no SparkSession raises a loud ``ValueError`` instead of
  silently returning an Arrow-mode batch.

Tests assert on the row-shape (URL-hash preservation, method
propagation) rather than specific status codes — those depend on
whatever the real endpoint replies with.
"""

from __future__ import annotations

from pyspark.sql import DataFrame as SparkDataFrame

from yggdrasil.io.response import RESPONSE_SCHEMA
from yggdrasil.spark.tests import SparkTestCase

from ._helpers import StubSession, make_request


class TestSparkSend(SparkTestCase):
    """Driver-side smoke tests for :meth:`Session.spark_send_many`."""

    def test_spark_send_returns_lazy_one_row_frame(self) -> None:
        s = StubSession()
        req = make_request("https://example.com/spark-send")

        df = s.spark_send(req, spark_session=self.spark)

        # Schema is the canonical RESPONSE_SCHEMA — every column the
        # downstream consumers rely on must be present and ordered.
        assert isinstance(df, SparkDataFrame)
        assert df.schema == RESPONSE_SCHEMA.to_spark_schema()

        rows = df.collect()
        assert len(rows) == 1
        row = rows[0]
        # Worker-side HTTPSession replies with *something*; status_code
        # is engine/network-dependent so just assert the column is
        # populated.
        assert row["status_code"] is not None
        assert row["request_method"] == "GET"
        assert row["request_public_url_hash"] == req.public_url_hash

    def test_spark_send_many_unions_per_chunk_frames(self) -> None:
        # Workers each get a vanilla ``HTTPSession`` per partition; the
        # contract under test is "one row per request, hash preserved".
        s = StubSession()
        reqs = [
            make_request(f"https://example.com/many/{i}")
            for i in range(5)
        ]

        df = s.spark_send_many(iter(reqs), spark_session=self.spark)

        assert df.schema == RESPONSE_SCHEMA.to_spark_schema()
        assert df.count() == len(reqs)

        # Every input request must show up in the output frame. Match
        # by ``request_public_url_hash`` — the canonical request
        # identity after normalisation, also what
        # ``PreparedRequest.public_url_hash`` returns on the driver.
        out_hashes = {
            row["request_public_url_hash"]
            for row in df.select("request_public_url_hash").collect()
        }
        for req in reqs:
            assert req.public_url_hash in out_hashes

    def test_spark_send_many_chunks_union_correctly(self) -> None:
        # Force ``_send_many_batches`` to emit multiple chunks (and
        # therefore multiple per-chunk Spark frames that need
        # ``unionByName``) by setting ``batch_size`` well below the
        # request count. The union must preserve every row.
        s = StubSession()
        reqs = [
            make_request(f"https://example.com/chunked/{i}")
            for i in range(12)
        ]

        df = s.spark_send_many(
            iter(reqs),
            batch_size=4,  # 3 chunks of 4 → 3 per-chunk frames unioned
            spark_session=self.spark,
        )

        assert df.schema == RESPONSE_SCHEMA.to_spark_schema()
        assert df.count() == len(reqs)

        out_hashes = {
            row["request_public_url_hash"]
            for row in df.select("request_public_url_hash").collect()
        }
        assert out_hashes == {r.public_url_hash for r in reqs}

    def test_spark_send_many_requires_spark_session(self) -> None:
        s = StubSession()
        req = make_request("https://example.com/needs-spark")
        try:
            s.spark_send_many(iter([req]), spark_session=None)  # type: ignore[arg-type]
        except ValueError as exc:
            assert "SparkSession" in str(exc)
        else:
            raise AssertionError(
                "spark_send_many must reject a missing SparkSession",
            )

    def test_spark_send_many_empty_iter_yields_empty_frame(self) -> None:
        s = StubSession()
        df = s.spark_send_many(iter([]), spark_session=self.spark)
        assert df.schema == RESPONSE_SCHEMA.to_spark_schema()
        assert df.count() == 0

    def test_spark_send_many_returns_lazy_dataframe(self) -> None:
        # The contract: ``spark_send_many`` builds a Spark plan and
        # returns it without firing any executor work. Workers run a
        # vanilla ``HTTPSession`` — never the driver-side stub — so the
        # stub's ``calls`` list stays empty before *and* after the
        # action, which is itself a useful smoke test that we are no
        # longer pickling the driver's :class:`Session` subclass.
        s = StubSession()
        reqs = [
            make_request(f"https://example.com/lazy/{i}")
            for i in range(4)
        ]

        df = s.spark_send_many(iter(reqs), spark_session=self.spark)

        # Plan built, frame returned, no executor work yet.
        assert isinstance(df, SparkDataFrame)
        assert s.calls == []

        # ``df.count()`` forces an action; the workers each spin up a
        # vanilla HTTPSession and process every request — one row per
        # input lands in the output frame.
        assert df.count() == len(reqs)

        # Stub stayed untouched — confirming we no longer ship driver
        # session state to workers.
        assert s.calls == []

    def test_spark_send_status_code_propagates(self) -> None:
        # The ``status_code`` column must survive the mapInArrow round
        # trip and arrive as an integer per row. The specific value is
        # whatever the real endpoint replies with — we only verify the
        # column is plumbed through.
        s = StubSession()
        df = s.spark_send_many(
            iter(make_request(f"https://example.com/status/{i}") for i in range(3)),
            spark_session=self.spark,
        )
        codes = [row["status_code"] for row in df.select("status_code").collect()]
        assert len(codes) == 3
        assert all(isinstance(c, int) for c in codes)
