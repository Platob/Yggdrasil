"""Unit tests for :class:`MemorySparkIO`.

The class wraps a Spark DataFrame, so the tests gate on pyspark
availability via ``importorskip``. We only exercise the non-Spark
surface (registration, empty state) without the optional dep.
"""

from __future__ import annotations

from yggdrasil.io.buffer.base import TabularIO
from yggdrasil.io.buffer.memory import MemorySparkIO


class TestMemorySparkIOBasics:
    def test_default_media_type_is_none_so_not_registered(self):
        # In-memory holders never want to win factory dispatch.
        assert MemorySparkIO.default_media_type() is None

    def test_is_a_tabular_io(self):
        assert isinstance(MemorySparkIO(), TabularIO)

    def test_empty_construction(self):
        io = MemorySparkIO()
        assert io.is_empty()
        assert io.frame is None
        assert io.spark is None
        assert not io.cached
        assert not io
