"""ArrowIPCIO: Arrow Table / RecordBatch round-trip."""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.primitive import ArrowIPCIO
from .._helpers import sample_batches, sample_table


class TestArrowIPCRoundTrip:
    def test_table_round_trip(self, tmp_path):
        p = tmp_path / "a.arrow"
        ArrowIPCIO(path=str(p)).write_arrow_table(sample_table())
        out = ArrowIPCIO(path=str(p)).read_arrow_table()
        assert out.num_rows == 3
        assert out.column_names == ["a", "b"]

    def test_batches_round_trip(self, tmp_path):
        p = tmp_path / "b.arrow"
        ArrowIPCIO(path=str(p)).write_arrow_batches(iter(sample_batches()))
        out = list(ArrowIPCIO(path=str(p)).read_arrow_batches())
        assert sum(b.num_rows for b in out) == 3

    def test_collect_schema(self, tmp_path):
        p = tmp_path / "c.arrow"
        ArrowIPCIO(path=str(p)).write_arrow_table(sample_table())
        schema = ArrowIPCIO(path=str(p)).collect_schema()
        assert list(schema.field_names()) == ["a", "b"]

    def test_in_memory_buffer_round_trip(self):
        # ArrowIPCIO's writer needs a seekable/path-bound target;
        # in-memory write-then-read works through its bytes buffer.
        io = ArrowIPCIO()
        io.write_arrow_table(sample_table())
        io.seek(0)
        out = io.read_arrow_table()
        io.close()
        assert out.num_rows == 3
