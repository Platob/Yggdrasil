"""Verify Table._write_spark_frame forwards spark_session to spark_insert.

When ``_insert_cache`` (the cache writeback helper) passes a
``CastOptions`` carrying a ``spark_session``, the Table's
``_write_spark_frame`` must forward that session to ``spark_insert``
so it doesn't fall back to ``resolve_session(create=True)`` and
potentially pick a stale / wrong session or fail entirely.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestWriteSparkFrameForwardsSession:

    def test_spark_session_forwarded(self):
        """_write_spark_frame must pass options.spark_session to spark_insert."""
        from yggdrasil.data.options import CastOptions
        from yggdrasil.enums import Mode

        fake_spark = MagicMock(name="FakeSparkSession")
        fake_frame = MagicMock(name="FakeSparkDataFrame")

        opts = CastOptions(
            mode=Mode.APPEND,
            spark_session=fake_spark,
        )

        table = MagicMock()
        table.spark_insert = MagicMock()

        from yggdrasil.databricks.table.table import Table
        Table._write_spark_frame(table, fake_frame, opts)

        table.spark_insert.assert_called_once()
        call_kwargs = table.spark_insert.call_args
        assert call_kwargs.kwargs.get("spark_session") is fake_spark, (
            "spark_session must be forwarded from options to spark_insert"
        )

    def test_spark_session_none_when_absent(self):
        """_write_spark_frame passes None when options has no spark_session."""
        from yggdrasil.data.options import CastOptions
        from yggdrasil.enums import Mode

        opts = CastOptions(mode=Mode.APPEND)

        table = MagicMock()
        table.spark_insert = MagicMock()

        from yggdrasil.databricks.table.table import Table
        Table._write_spark_frame(table, MagicMock(), opts)

        call_kwargs = table.spark_insert.call_args
        assert call_kwargs.kwargs.get("spark_session") is None
