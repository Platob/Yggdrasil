"""CsvIO: Polars round-trip."""

from __future__ import annotations

from yggdrasil.io.primitive import CsvIO
from .._helpers import require_polars, sample_polars_frame


class TestCsvPolars:
    def test_polars_round_trip(self, tmp_path):
        require_polars()
        p = tmp_path / "a.csv"
        CsvIO(path=str(p)).write_polars_frame(sample_polars_frame())
        out = CsvIO(path=str(p)).read_polars_frame()
        assert out.shape == (3, 2)
        assert out.columns == ["a", "b"]
