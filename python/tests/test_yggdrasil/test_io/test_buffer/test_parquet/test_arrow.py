"""ParquetIO: Arrow Table / RecordBatch round-trip."""

from __future__ import annotations

from yggdrasil.io.primitive import ParquetIO
from .._helpers import sample_batches, sample_table


class TestParquetArrow:
    def test_table_round_trip(self, tmp_path):
        p = tmp_path / "a.parquet"
        ParquetIO(path=str(p)).write_arrow_table(sample_table())
        out = ParquetIO(path=str(p)).read_arrow_table()
        assert out.num_rows == 3
        assert out.column_names == ["a", "b"]

    def test_batches_round_trip(self, tmp_path):
        p = tmp_path / "b.parquet"
        ParquetIO(path=str(p)).write_arrow_batches(iter(sample_batches()))
        out = list(ParquetIO(path=str(p)).read_arrow_batches())
        assert sum(b.num_rows for b in out) == 3

    def test_collect_schema(self, tmp_path):
        p = tmp_path / "c.parquet"
        ParquetIO(path=str(p)).write_arrow_table(sample_table())
        schema = ParquetIO(path=str(p)).collect_schema()
        assert list(schema.field_names()) == ["a", "b"]

    def test_in_memory_round_trip(self):
        io = ParquetIO()
        io.write_arrow_table(sample_table())
        io.seek(0)
        out = io.read_arrow_table()
        io.close()
        assert out.num_rows == 3
