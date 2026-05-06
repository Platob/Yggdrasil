"""XlsxIO: Arrow round-trip (requires openpyxl)."""

from __future__ import annotations

import pytest

from yggdrasil.io.primitive import XlsxIO
from .._helpers import sample_table


class TestXlsxArrow:
    def test_table_round_trip(self, tmp_path):
        pytest.importorskip("openpyxl")
        p = tmp_path / "a.xlsx"
        XlsxIO(path=str(p)).write_arrow_table(sample_table())
        out = XlsxIO(path=str(p)).read_arrow_table()
        assert out.num_rows == 3
        assert out.column_names == ["a", "b"]
