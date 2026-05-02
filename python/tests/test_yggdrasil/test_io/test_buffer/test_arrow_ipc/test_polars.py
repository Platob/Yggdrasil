"""ArrowIPCIO: Polars round-trip."""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.primitive import ArrowIPCIO
from .._helpers import require_polars, sample_polars_frame


class TestArrowIPCPolars:
    def test_polars_round_trip(self, tmp_path):
        require_polars()
        p = tmp_path / "a.arrow"
        ArrowIPCIO(path=str(p)).write_polars_frame(sample_polars_frame())
        out = ArrowIPCIO(path=str(p)).read_polars_frame()
        assert out.shape == (3, 2)
        assert out.columns == ["a", "b"]
