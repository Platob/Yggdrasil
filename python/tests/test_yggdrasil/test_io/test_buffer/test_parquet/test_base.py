"""ParquetIO core: registration, mime, save modes, in-memory round-trip."""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.primitive import ParquetIO, PrimitiveIO
from yggdrasil.io.buffer.primitive.parquet_io import ParquetOptions
from yggdrasil.io.enums import Mode, MimeTypes
from .._helpers import sample_table


class TestParquetIOBase:
    def test_default_mime_type(self):
        assert ParquetIO.default_mime_type() == MimeTypes.PARQUET

    def test_options_class(self):
        assert ParquetIO.options_class() is ParquetOptions

    def test_dispatch_via_path(self, tmp_path):
        p = tmp_path / "x.parquet"
        p.touch()
        io = PrimitiveIO(path=str(p))
        assert isinstance(io, ParquetIO)

    def test_concrete_class_skips_dispatch(self, tmp_path):
        p = tmp_path / "x.parquet"
        p.touch()
        io = ParquetIO(path=str(p))
        assert type(io) is ParquetIO


class TestParquetSaveModes:
    def test_overwrite_replaces(self, tmp_path):
        p = tmp_path / "x.parquet"
        ParquetIO(path=str(p)).write_arrow_table(sample_table())
        ParquetIO(path=str(p)).write_arrow_table(sample_table().slice(0, 1))
        assert ParquetIO(path=str(p)).read_arrow_table().num_rows == 1

    def test_ignore_skips_when_non_empty(self, tmp_path):
        p = tmp_path / "x.parquet"
        ParquetIO(path=str(p)).write_arrow_table(sample_table())
        ParquetIO(path=str(p)).write_arrow_table(
            sample_table().slice(0, 1), mode=Mode.IGNORE,
        )
        assert ParquetIO(path=str(p)).read_arrow_table().num_rows == 3
