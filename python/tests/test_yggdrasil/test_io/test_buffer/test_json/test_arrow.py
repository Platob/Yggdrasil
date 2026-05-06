"""JsonIO: Arrow round-trip."""

from __future__ import annotations

from yggdrasil.io.primitive import JsonIO
from .._helpers import sample_table


class TestJsonArrow:
    def test_table_round_trip(self, tmp_path):
        p = tmp_path / "a.json"
        JsonIO(path=str(p)).write_arrow_table(sample_table())
        out = JsonIO(path=str(p)).read_arrow_table()
        assert out.num_rows == 3
        assert set(out.column_names) == {"a", "b"}
