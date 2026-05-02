"""XlsxIO core (requires openpyxl)."""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.primitive import PrimitiveIO, XlsxIO
from yggdrasil.io.buffer.primitive.xlsx_io import XlsxOptions
from yggdrasil.io.enums import MimeTypes


class TestXlsxBase:
    def test_default_mime_type(self):
        assert XlsxIO.default_mime_type() == MimeTypes.XLSX

    def test_options_class(self):
        assert XlsxIO.options_class() is XlsxOptions

    def test_dispatch_via_path(self, tmp_path):
        pytest.importorskip("openpyxl")
        p = tmp_path / "x.xlsx"
        p.touch()
        io = PrimitiveIO(path=str(p))
        assert isinstance(io, XlsxIO)
