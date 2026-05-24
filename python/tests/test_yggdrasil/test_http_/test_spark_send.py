"""Spark integration tests for :meth:`Session.send_many` with Spark fan-out.

Validates that ``send_many(spark_session=spark)`` scatters requests
via ``mapInArrow`` and produces the expected Spark DataFrames.
"""
from __future__ import annotations

import pytest

from yggdrasil.io.response import RESPONSE_SCHEMA
from yggdrasil.io.session import Session

from ._helpers import StubSession, make_request

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import DataFrame as SparkDataFrame  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    from yggdrasil.spark.tests import _get_test_spark, SparkTestCase

    try:
        return _get_test_spark(app_name="ygg-spark-send-test")
    except Exception as exc:
        pytest.skip(f"local SparkSession unavailable: {exc!r}")


@pytest.fixture(autouse=True)
def _clear_singletons():
    Session._INSTANCES.clear()
    yield
    Session._INSTANCES.clear()


class TestSparkSend:

    def test_single_request_produces_one_row(self, spark) -> None:
        s = StubSession()
        req = make_request("https://example.com/spark-send")

        batches = list(s.send_many_batches(
            iter([req]), spark_session=spark,
        ))
        assert len(batches) >= 1
        total = sum(b.counts.get("new", 0) for b in batches)
        assert total == 1

    def test_many_requests_all_fetched(self, spark) -> None:
        s = StubSession()
        reqs = [
            make_request(f"https://example.com/many/{i}")
            for i in range(5)
        ]

        responses = list(s.send_many(iter(reqs), spark_session=spark))
        assert len(responses) == len(reqs)

    def test_multi_chunk_preserves_rows(self, spark) -> None:
        s = StubSession()
        reqs = [
            make_request(f"https://example.com/chunked/{i}")
            for i in range(12)
        ]

        responses = list(s.send_many(
            iter(reqs), spark_session=spark, batch_size=4,
        ))
        assert len(responses) == 12
