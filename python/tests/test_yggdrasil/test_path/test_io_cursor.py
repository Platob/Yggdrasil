"""IO cursor and Tabular integration tests with real file I/O."""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.path.local_path import LocalPath
from yggdrasil.enums import Mode


class TestCursorLifecycle:

    def test_open_returns_cursor_with_parent(self, tmp_path):
        p = LocalPath(str(tmp_path / "f.bin"))
        p.write_bytes(b"hello")
        c = p.open()
        assert c._parent is p
        c.close()

    def test_cursor_read_sequential(self, tmp_path):
        p = LocalPath(str(tmp_path / "f.bin"))
        p.write_bytes(b"abcdefgh")
        c = p.open()
        assert c.read(3) == b"abc"
        assert c.read(3) == b"def"
        assert c.read(3) == b"gh"
        c.close()

    def test_cursor_seek_and_tell(self, tmp_path):
        p = LocalPath(str(tmp_path / "f.bin"))
        p.write_bytes(b"0123456789")
        c = p.open()
        assert c.tell() == 0
        c.seek(5)
        assert c.tell() == 5
        assert c.read(3) == b"567"
        c.close()

    def test_cursor_seek_from_end(self, tmp_path):
        p = LocalPath(str(tmp_path / "f.bin"))
        p.write_bytes(b"abcdef")
        c = p.open()
        c.seek(-3, 2)
        assert c.read(3) == b"def"
        c.close()

    def test_cursor_seek_relative(self, tmp_path):
        p = LocalPath(str(tmp_path / "f.bin"))
        p.write_bytes(b"abcdef")
        c = p.open()
        c.read(2)
        c.seek(1, 1)
        assert c.read(1) == b"d"
        c.close()

    def test_multiple_cursors_independent(self, tmp_path):
        p = LocalPath(str(tmp_path / "f.bin"))
        p.write_bytes(b"abcdefghij")
        c1 = p.open()
        c2 = p.open()
        assert c1.read(3) == b"abc"
        assert c2.read(5) == b"abcde"
        assert c1.read(2) == b"de"
        assert c2.read(2) == b"fg"
        c1.close()
        c2.close()

    def test_cursor_close_does_not_close_path(self, tmp_path):
        p = LocalPath(str(tmp_path / "f.bin"))
        p.write_bytes(b"data")
        c = p.open()
        c.read(4)
        c.close()
        assert p.size == 4

    def test_cursor_with_statement(self, tmp_path):
        p = LocalPath(str(tmp_path / "f.bin"))
        p.write_bytes(b"hello")
        with p.open() as c:
            assert c.read(5) == b"hello"

    def test_cursor_read_empty_file(self, tmp_path):
        p = LocalPath(str(tmp_path / "empty.bin"))
        p.write_bytes(b"")
        c = p.open()
        assert c.read(10) == b""
        c.close()

    def test_cursor_write_then_read(self, tmp_path):
        p = LocalPath(str(tmp_path / "rw.bin"))
        p.write_bytes(b"initial")
        c = p.open("rb+")
        c.seek(0, 2)
        c.write(b" appended")
        c.seek(0)
        assert c.read(100) == b"initial appended"
        c.close()


class TestTabularArrowIntegration:

    def test_write_read_arrow_table(self, tmp_path):
        p = LocalPath(str(tmp_path / "data.ipc"))
        table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        leaf = p.as_media("arrow")
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 3
        assert result.column("x").to_pylist() == [1, 2, 3]

    def test_write_read_arrow_batches(self, tmp_path):
        p = LocalPath(str(tmp_path / "data.ipc"))
        table = pa.table({"a": [10, 20]})
        leaf = p.as_media("arrow")
        leaf.write_arrow_batches(table.to_batches(), mode=Mode.OVERWRITE)
        batches = list(leaf.read_arrow_batches())
        total = sum(b.num_rows for b in batches)
        assert total == 2

    def test_overwrite_replaces_data(self, tmp_path):
        p = LocalPath(str(tmp_path / "ow.ipc"))
        leaf = p.as_media("arrow")
        leaf.write_arrow_table(pa.table({"x": [1, 2, 3]}), mode=Mode.OVERWRITE)
        leaf.write_arrow_table(pa.table({"x": [99]}), mode=Mode.OVERWRITE)
        assert leaf.read_arrow_table().num_rows == 1
        assert leaf.read_arrow_table().column("x").to_pylist() == [99]

    def test_collect_schema(self, tmp_path):
        p = LocalPath(str(tmp_path / "schema.ipc"))
        leaf = p.as_media("arrow")
        table = pa.table({"id": pa.array([1, 2], type=pa.int64()), "name": ["a", "b"]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        schema = leaf.collect_schema()
        assert "id" in schema

    def test_parquet_write_read(self, tmp_path):
        p = LocalPath(str(tmp_path / "data.parquet"))
        leaf = p.as_media("parquet")
        table = pa.table({"v": [1.0, 2.0, 3.0]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 3

    def test_csv_write_read(self, tmp_path):
        p = LocalPath(str(tmp_path / "data.csv"))
        leaf = p.as_media("csv")
        table = pa.table({"col": ["hello", "world"]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 2

    def test_ndjson_write_read(self, tmp_path):
        p = LocalPath(str(tmp_path / "data.jsonl"))
        leaf = p.as_media("ndjson")
        table = pa.table({"k": [1, 2], "v": ["a", "b"]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 2


class TestTabularPolarsIntegration:

    def test_write_polars_read_arrow(self, tmp_path):
        import polars as pl
        p = LocalPath(str(tmp_path / "pl.ipc"))
        leaf = p.as_media("arrow")
        df = pl.DataFrame({"x": [1, 2, 3]})
        leaf.write_table(df, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 3

    def test_read_polars_dataframe(self, tmp_path):
        import polars as pl
        p = LocalPath(str(tmp_path / "pl2.ipc"))
        leaf = p.as_media("arrow")
        table = pa.table({"a": [10, 20, 30]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        table = leaf.read_arrow_table()
        df = pl.from_arrow(table)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3

    def test_write_arrow_read_polars_roundtrip(self, tmp_path):
        import polars as pl
        p = LocalPath(str(tmp_path / "rt.parquet"))
        leaf = p.as_media("parquet")
        table = pa.table({"id": [1, 2], "val": [3.14, 2.72]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        table = leaf.read_arrow_table()
        df = pl.from_arrow(table)
        assert df["id"].to_list() == [1, 2]


class TestTabularPandasIntegration:

    def test_write_pandas_read_arrow(self, tmp_path):
        import pandas as pd
        p = LocalPath(str(tmp_path / "pd.ipc"))
        leaf = p.as_media("arrow")
        df = pd.DataFrame({"x": [1, 2, 3]})
        leaf.write_table(df, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 3

    def test_read_pandas_dataframe(self, tmp_path):
        import pandas as pd
        p = LocalPath(str(tmp_path / "pd2.ipc"))
        leaf = p.as_media("arrow")
        table = pa.table({"a": [10, 20]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        table = leaf.read_arrow_table()
        df = table.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2


class TestMultiFormatRoundTrip:

    @pytest.mark.parametrize("ext,media", [
        (".ipc", "arrow"),
        (".parquet", "parquet"),
        (".csv", "csv"),
    ])
    def test_format_roundtrip(self, tmp_path, ext, media):
        p = LocalPath(str(tmp_path / f"data{ext}"))
        leaf = p.as_media(media)
        table = pa.table({"x": [1, 2, 3]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 3

    def test_large_table_roundtrip(self, tmp_path):
        p = LocalPath(str(tmp_path / "big.ipc"))
        leaf = p.as_media("arrow")
        table = pa.table({"id": list(range(10000)), "val": [f"row_{i}" for i in range(10000)]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 10000

    def test_empty_table_roundtrip(self, tmp_path):
        p = LocalPath(str(tmp_path / "empty.ipc"))
        leaf = p.as_media("arrow")
        table = pa.table({"x": pa.array([], type=pa.int64())})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 0


class TestCursorOnTabular:

    def test_ipc_file_as_cursor(self, tmp_path):
        p = LocalPath(str(tmp_path / "cursor.ipc"))
        leaf = p.as_media("arrow")
        table = pa.table({"x": [1, 2, 3]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        raw = p.read_bytes()
        assert len(raw) > 0
        assert raw[:4] == b"ARRO" or raw[:6] == b"ARROW1"

    def test_parquet_file_raw_bytes(self, tmp_path):
        p = LocalPath(str(tmp_path / "raw.parquet"))
        leaf = p.as_media("parquet")
        table = pa.table({"v": [42]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        raw = p.read_bytes()
        assert raw[:4] == b"PAR1"
