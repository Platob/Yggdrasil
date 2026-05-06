"""NDJsonIO: Arrow round-trip."""

from __future__ import annotations

from yggdrasil.io.primitive import NDJsonIO
from .._helpers import sample_table


class TestNDJsonArrow:
    def test_table_round_trip(self, tmp_path):
        p = tmp_path / "a.ndjson"
        NDJsonIO(path=str(p)).write_arrow_table(sample_table())
        out = NDJsonIO(path=str(p)).read_arrow_table()
        assert out.num_rows == 3
        assert set(out.column_names) == {"a", "b"}
