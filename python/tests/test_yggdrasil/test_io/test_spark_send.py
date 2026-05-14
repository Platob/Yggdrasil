"""Spark-mode ``send`` / ``send_many`` tests.

Drive :meth:`Session.spark_send` and :meth:`Session.spark_send_many`
end-to-end through a shared :class:`SparkTestCase` session, with a
:class:`StubSession` standing in for the network. The stub's
``_local_send`` synthesises a default :class:`Response` for any request
without a queued reply, which is enough to verify the lazy
``DataFrame[Response]`` contract:

* the returned frame matches :data:`RESPONSE_SCHEMA`,
* one input request produces one output row,
* :meth:`spark_send` collapses to :meth:`spark_send_many` cleanly, and
* passing no SparkSession raises a loud ``ValueError`` instead of
  silently returning an Arrow-mode batch.
"""

from __future__ import annotations

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
        assert df.schema == RESPONSE_SCHEMA.to_spark_schema()

        rows = df.collect()
        assert len(rows) == 1
        row = rows[0]
        assert row["status_code"] == 200
        assert row["request_method"] == "GET"

    def test_spark_send_many_unions_per_chunk_frames(self) -> None:
        # No ``queue`` — let :class:`StubSession._local_send` fall through
        # to ``make_response(request=request)`` so each worker's response
        # mirrors the request it actually received. Pre-queuing would
        # bind responses to a driver-side FIFO that the broadcast-shared
        # queue can't honour across partitions.
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
