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


class TestPandasIndexRoundTrip(__import__(
    "yggdrasil.pandas.tests", fromlist=["PandasTestCase"],
).PandasTestCase):
    """ParquetFile round-trips every non-default pandas index shape.

    The base ``_write_pandas_frame`` preserves the index only when at
    least one level is named, which silently dropped non-default
    ``RangeIndex`` and unnamed non-range ``Index`` values. The Parquet
    override uses ``preserve_index=None`` so pyarrow's auto mode
    stamps the ``b"pandas"`` schema-metadata channel — ``to_pandas()``
    rebuilds the index from it on the read side.
    """

    def _roundtrip(self, df):
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(df)
        return ParquetFile(holder=mem, owns_holder=False).read_pandas_frame()

    def test_default_range_index(self) -> None:
        df = self.df({"a": [1, 2, 3]})
        result = self._roundtrip(df)
        self.assertFrameEqual(result, df, check_index=True)
        assert isinstance(result.index, self.pd.RangeIndex)
        assert result.index.start == 0

    def test_non_default_range_index(self) -> None:
        df = self.df({"a": [1, 2, 3]}, index=self.pd.RangeIndex(5, 8))
        result = self._roundtrip(df)
        self.assertFrameEqual(result, df, check_index=True)
        assert list(result.index) == [5, 6, 7]

    def test_named_range_index(self) -> None:
        df = self.df({"a": [10, 20, 30]})
        df.index.name = "i"
        result = self._roundtrip(df)
        self.assertFrameEqual(result, df, check_index=True)
        assert result.index.name == "i"

    def test_datetime_index(self) -> None:
        df = self.df(
            {"v": [1, 2, 3]},
            index=self.pd.date_range("2024-01-01", periods=3, name="ts"),
        )
        result = self._roundtrip(df)
        # ``freq`` round-trips as None — parquet's pandas metadata
        # carries the values, not the index's freq descriptor.
        self.assertFrameEqual(result, df, check_index=True, check_freq=False)
        assert isinstance(result.index, self.pd.DatetimeIndex)
        assert result.index.name == "ts"

    def test_multi_index(self) -> None:
        idx = self.pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=["k1", "k2"],
        )
        df = self.df({"v": [10, 20, 30, 40]}, index=idx)
        result = self._roundtrip(df)
        self.assertFrameEqual(result, df, check_index=True)
        assert isinstance(result.index, self.pd.MultiIndex)
        assert result.index.names == ["k1", "k2"]

    def test_unnamed_non_range_index(self) -> None:
        df = self.df({"a": [1, 2, 3]}, index=[10, 20, 30])
        result = self._roundtrip(df)
        self.assertFrameEqual(result, df, check_index=True)
        assert list(result.index) == [10, 20, 30]

    def test_local_path_roundtrip(self) -> None:
        df = self.df({"a": [1, 2, 3]}, index=self.pd.RangeIndex(100, 103))
        path = LocalPath(str(self.tmp_path / "indexed.parquet"))
        with path.open("wb") as cursor:
            cursor.write_pandas_frame(df)
        with path.open("rb") as cursor:
            result = cursor.read_pandas_frame()
        self.assertFrameEqual(result, df, check_index=True)
        assert list(result.index) == [100, 101, 102]

    def test_target_bound_write_drops_index(self) -> None:
        """A bound target schema strictly defines columns — index drops out."""
        from yggdrasil.data.schema import Schema
        from yggdrasil.io.primitive.parquet_file import ParquetOptions

        df = self.df({"a": [1, 2, 3]}, index=self.pd.RangeIndex(5, 8))
        target = Schema.from_arrow(pa.schema([("a", pa.int64())]))
        opts = ParquetOptions(target=target)
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(df, options=opts)
        result = ParquetFile(holder=mem, owns_holder=False).read_pandas_frame()
        assert isinstance(result.index, self.pd.RangeIndex)
        assert result.index.start == 0

    def test_reads_parquet_written_by_pandas(self) -> None:
        """Cross-tool: pandas.to_parquet → ParquetFile.read_pandas_frame restores index."""
        import io

        df = self.df(
            {"a": [1, 2, 3]},
            index=self.pd.Index([100, 200, 300], name="custom_idx"),
        )
        buf = io.BytesIO()
        df.to_parquet(buf)
        mem = Memory()
        mem.write(buf.getvalue())
        result = ParquetFile(holder=mem, owns_holder=False).read_pandas_frame()
        self.assertFrameEqual(result, df, check_index=True)
        assert result.index.name == "custom_idx"
