"""Polars round-trip via :class:`BytesIO` + media-typed view.

Same redirect rule as ``test_arrow.py``: a raw buffer can't speak
Polars without a format. With a tabular media set, the buffer
defers to the format leaf which knows how to translate Polars
frames to Arrow batches and back.
"""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer import BytesIO
from yggdrasil.data.enums import MimeTypes
from .._helpers import require_polars, sample_polars_frame


class TestBytesIOPolarsOpaqueRaises:
    def test_read_polars_raises_without_media(self):
        pytest.importorskip("polars")
        bio = BytesIO(b"opaque")
        with pytest.raises(NotImplementedError):
            bio.read_polars_frame()


class TestBytesIOPolarsViaMedia:
    def test_polars_round_trip_arrow_ipc(self, tmp_path):
        require_polars()
        path = tmp_path / "x.arrow"
        bio = BytesIO(path=str(path), media_type=MimeTypes.ARROW_IPC)
        bio.write_polars_frame(sample_polars_frame())
        bio.close()

        reader = BytesIO(path=str(path), media_type=MimeTypes.ARROW_IPC)
        out = reader.read_polars_frame()
        reader.close()
        assert out.shape == (3, 2)
        assert out.columns == ["a", "b"]
