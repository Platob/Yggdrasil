"""CsvIO: Arrow round-trip."""

from __future__ import annotations

from yggdrasil.io.primitive import CsvIO
from .._helpers import sample_table


class TestCsvArrow:
    def test_table_round_trip(self, tmp_path):
        p = tmp_path / "a.csv"
        CsvIO(path=str(p)).write_arrow_table(sample_table())
        out = CsvIO(path=str(p)).read_arrow_table()
        assert out.num_rows == 3
        assert out.column_names == ["a", "b"]

    def test_collect_schema(self, tmp_path):
        p = tmp_path / "c.csv"
        CsvIO(path=str(p)).write_arrow_table(sample_table())
        schema = CsvIO(path=str(p)).collect_schema()
        assert list(schema.field_names()) == ["a", "b"]
