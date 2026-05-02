"""JsonIO: Polars round-trip."""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.primitive import JsonIO
from .._helpers import require_polars, sample_polars_frame


class TestJsonPolars:
    def test_polars_round_trip(self, tmp_path):
        require_polars()
        p = tmp_path / "a.json"
        JsonIO(path=str(p)).write_polars_frame(sample_polars_frame())
        out = JsonIO(path=str(p)).read_polars_frame()
        assert out.shape == (3, 2)
        assert set(out.columns) == {"a", "b"}
