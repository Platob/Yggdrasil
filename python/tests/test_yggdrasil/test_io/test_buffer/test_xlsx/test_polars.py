"""XlsxIO: Polars round-trip (requires openpyxl)."""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.primitive import XlsxIO
from .._helpers import require_polars, sample_polars_frame


class TestXlsxPolars:
    def test_polars_round_trip(self, tmp_path):
        pytest.importorskip("openpyxl")
        require_polars()
        p = tmp_path / "a.xlsx"
        XlsxIO(path=str(p)).write_polars_frame(sample_polars_frame())
        out = XlsxIO(path=str(p)).read_polars_frame()
        assert out.shape == (3, 2)
