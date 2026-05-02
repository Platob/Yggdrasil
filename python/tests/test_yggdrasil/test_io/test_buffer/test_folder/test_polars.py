"""FolderIO: Polars round-trip across multiple child files."""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.nested import FolderIO
from .._helpers import require_polars, sample_polars_frame


class TestFolderPolars:
    def test_polars_round_trip(self, tmp_path):
        require_polars()
        FolderIO(path=str(tmp_path)).write_polars_frame(sample_polars_frame())
        out = FolderIO(path=str(tmp_path)).read_polars_frame()
        assert out.shape == (3, 2)
        assert set(out.columns) == {"a", "b"}
