"""XlsxIO core (requires openpyxl)."""

from __future__ import annotations

import pytest

from yggdrasil.io.primitive import XlsxIO
from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.primitive import XlsxOptions
from yggdrasil.data.enums import MimeTypes


class TestXlsxBase:
    def test_default_media_type(self):
        assert XlsxIO.default_media_type() == MimeTypes.XLSX

    def test_options_class(self):
        assert XlsxIO.options_class() is XlsxOptions

    def test_dispatch_via_path(self, tmp_path):
        pytest.importorskip("openpyxl")
        p = tmp_path / "x.xlsx"
        p.touch()
        io = BytesIO(path=str(p))
        assert isinstance(io, XlsxIO)
