"""CsvIO core: registration, mime, append/upsert support flags."""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.primitive import CsvIO
from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.buffer.primitive.csv_io import CsvOptions
from yggdrasil.io.enums import Mode, MimeTypes
from .._helpers import sample_table


class TestCsvBase:
    def test_default_mime_type(self):
        assert CsvIO.default_mime_type() == MimeTypes.CSV

    def test_options_class(self):
        assert CsvIO.options_class() is CsvOptions

    def test_dispatch_via_path(self, tmp_path):
        p = tmp_path / "x.csv"
        p.touch()
        io = BytesIO(path=str(p))
        assert isinstance(io, CsvIO)

    def test_supports_native_append(self):
        # CSV is the only leaf with honest APPEND/UPSERT support
        # at the buffer level — Parquet/IPC fall back to rewrite.
        assert CsvIO._SUPPORTED_APPEND is True
        assert CsvIO._SUPPORTED_UPSERT is True


class TestCsvAppend:
    def test_append_concatenates(self, tmp_path):
        p = tmp_path / "x.csv"
        CsvIO(path=str(p)).write_arrow_table(sample_table())
        CsvIO(path=str(p)).write_arrow_table(
            sample_table().slice(0, 1), mode=Mode.APPEND,
        )
        out = CsvIO(path=str(p)).read_arrow_table()
        assert out.num_rows == 4

    def test_append_on_empty_includes_header(self, tmp_path):
        p = tmp_path / "x.csv"
        CsvIO(path=str(p)).write_arrow_table(sample_table(), mode=Mode.APPEND)
        out = CsvIO(path=str(p)).read_arrow_table()
        assert out.num_rows == 3
