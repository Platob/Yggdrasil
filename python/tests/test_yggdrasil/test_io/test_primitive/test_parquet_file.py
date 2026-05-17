"""Tests for :class:`yggdrasil.io.primitive.parquet_file.ParquetFile`."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.io.memory import Memory
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.parquet_file import ParquetFile


class TestRegistration:

    def test_class_for_media_type_parquet(self) -> None:
        assert Holder.class_for_media_type("parquet") is ParquetFile

    def test_class_for_media_type_mime(self) -> None:
        assert (
            Holder.class_for_media_type("application/vnd.apache.parquet")
            is ParquetFile
        )

    def test_path_dispatches_via_extension(self, tmp_path) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        b = BytesIO(path=str(tmp_path / "x.parquet"))
        assert isinstance(b, ParquetFile)

    def test_open_local_path_dispatches(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "x.parquet"))
        cursor = lp.open("rb", auto_open=False)
        assert isinstance(cursor, ParquetFile)


class TestMemoryRoundTrip:

    @pytest.fixture
    def table(self) -> pa.Table:
        return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})

    def test_write_then_read_arrow_table(self, table) -> None:
        mem = Memory()
        leaf = ParquetFile(holder=mem, owns_holder=False)
        leaf.write_arrow_table(table)
        assert mem.size > 0

        leaf2 = ParquetFile(holder=mem, owns_holder=False)
        assert leaf2.read_arrow_table().equals(table)

    def test_dispatch_via_stamped_media(self, table) -> None:
        from yggdrasil.data.enums import MediaType, MimeTypes
        from yggdrasil.io.bytes_io import BytesIO

        mem = Memory()
        mem.media_type = MediaType(MimeTypes.PARQUET)
        writer = BytesIO(holder=mem, owns_holder=False)
        assert isinstance(writer, ParquetFile)
        writer.write_arrow_table(table)

        reader = BytesIO(holder=mem, owns_holder=False)
        assert reader.read_arrow_table().equals(table)


class TestLocalPathRoundTrip:

    def test_write_and_read_back(self, tmp_path) -> None:
        table = pa.table({"x": [1, 2, 3]})
        path = LocalPath(str(tmp_path / "out.parquet"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(table)
        with path.open("rb") as cursor:
            assert cursor.read_arrow_table().equals(table)

    def test_open_returns_parquet_file_cursor(self, tmp_path) -> None:
        path = LocalPath(str(tmp_path / "x.parquet"))
        cursor = path.open("rb", auto_open=False)
        assert isinstance(cursor, ParquetFile)
        assert cursor.parent is path


class TestOptions:

    def test_options_class(self) -> None:
        from yggdrasil.io.primitive.parquet_file import ParquetOptions

        assert ParquetFile.options_class() is ParquetOptions
