"""NDJsonIO: Polars round-trip."""

from __future__ import annotations

from yggdrasil.io.primitive import NDJsonIO
from .._helpers import require_polars, sample_polars_frame


class TestNDJsonPolars:
    def test_polars_round_trip(self, tmp_path):
        require_polars()
        p = tmp_path / "a.ndjson"
        NDJsonIO(path=str(p)).write_polars_frame(sample_polars_frame())
        out = NDJsonIO(path=str(p)).read_polars_frame()
        assert out.shape == (3, 2)
        assert set(out.columns) == {"a", "b"}
