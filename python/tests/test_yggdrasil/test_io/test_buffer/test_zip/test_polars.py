"""ZipIO: Polars round-trip."""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.nested import ZipIO
from .._helpers import require_polars, sample_polars_frame


class TestZipPolars:
    def test_polars_round_trip(self, tmp_path):
        require_polars()
        path = tmp_path / "a.zip"
        ZipIO(path=str(path)).write_polars_frame(sample_polars_frame())
        out = ZipIO(path=str(path)).read_polars_frame()
        assert out.shape == (3, 2)
        assert set(out.columns) == {"a", "b"}
