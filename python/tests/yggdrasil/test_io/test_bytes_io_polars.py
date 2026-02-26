"""Unit tests for BytesIO Polars integration.

Tests cover:
- read_polars / write_polars round-trips for every supported MediaType
- lazy=True path (LazyFrame return, scan_parquet/scan_ipc for file-backed buffers)
- ZIP multi-entry concat
- ZIP empty archive error
- content_type auto-detection (no explicit override)
- codec decompression path (outer Codec wrapper)
- unsupported MediaType raises ValueError
- LazyFrame input to write_polars is collected transparently
- cursor position is preserved after read_polars
"""
from __future__ import annotations

import io
import zipfile
from typing import Any

import polars as pl
import pytest

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.enums.codec import Codec
from yggdrasil.io.enums.media_type import MediaType, MediaTypes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df() -> pl.DataFrame:
    return pl.DataFrame({
        "sym":   ["TTF", "JKM", "NBP", "HH"],
        "price": [42.5,  18.3,  29.7,  3.1],
        "vol":   [100,   200,   150,   300],
        "flag":  [True,  False, True,  False],
    })


def _round_trip(
    df: pl.DataFrame,
    media_type: MediaType,
    *,
    lazy: bool = False,
    codec: Codec | None = None,
    **write_kwargs: Any,
) -> pl.DataFrame | pl.LazyFrame:
    buf = BytesIO()
    buf.write_polars(df, media_type, codec=codec, **write_kwargs)
    buf.seek(0)
    return buf.read_polars(media_type, lazy=lazy)


def _collect(result: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(result, pl.LazyFrame):
        return result.collect()
    return result


def _assert_equal(result: pl.DataFrame | pl.LazyFrame, expected: pl.DataFrame) -> None:
    got = _collect(result)
    assert got.shape == expected.shape, f"shape mismatch: {got.shape} != {expected.shape}"
    assert got.columns == expected.columns, f"column mismatch: {got.columns}"
    for col in expected.columns:
        assert got[col].to_list() == expected[col].to_list(), f"column {col!r} differs"


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

class TestParquet:
    def test_round_trip_eager(self):
        df = _make_df()
        _assert_equal(_round_trip(df, MediaTypes.PARQUET), df)

    def test_round_trip_lazy(self):
        df = _make_df()
        result = _round_trip(df, MediaTypes.PARQUET, lazy=True)
        assert isinstance(result, pl.LazyFrame)
        _assert_equal(result, df)

    def test_scan_parquet_uses_path_when_available(self, tmp_path):
        """When buffer is file-backed and no codec, lazy should use scan_parquet."""
        df = _make_df()
        p = tmp_path / "data.parquet"
        df.write_parquet(p)
        buf = BytesIO(p)
        result = buf.read_polars(MediaTypes.PARQUET, lazy=True)
        assert isinstance(result, pl.LazyFrame)
        _assert_equal(result, df)

    def test_compression_zstd(self):
        df = _make_df()
        _assert_equal(_round_trip(df, MediaTypes.PARQUET, compression="zstd"), df)

    def test_compression_snappy(self):
        df = _make_df()
        _assert_equal(_round_trip(df, MediaTypes.PARQUET, compression="snappy"), df)

    def test_row_group_size(self):
        df = _make_df()
        _assert_equal(_round_trip(df, MediaTypes.PARQUET, row_group_size=2), df)

    def test_auto_detect_format(self):
        """content_type auto-detection should identify Parquet magic bytes."""
        df = _make_df()
        buf = BytesIO()
        buf.write_polars(df, MediaTypes.PARQUET)
        buf.seek(0)
        result = buf.read_polars()  # no explicit content_type
        _assert_equal(result, df)

    def test_lazyframe_input_collected(self):
        df = _make_df()
        buf = BytesIO()
        buf.write_polars(df.lazy(), MediaTypes.PARQUET)  # LazyFrame input
        buf.seek(0)
        _assert_equal(buf.read_polars(MediaTypes.PARQUET), df)

    def test_cursor_preserved_after_read(self):
        df = _make_df()
        buf = BytesIO()
        buf.write_polars(df, MediaTypes.PARQUET)
        end_pos = buf.tell()
        buf.seek(0)
        buf.read_polars(MediaTypes.PARQUET)
        assert buf.tell() == 0


# ---------------------------------------------------------------------------
# IPC / Feather
# ---------------------------------------------------------------------------

class TestIPC:
    def test_round_trip_ipc_eager(self):
        df = _make_df()
        _assert_equal(_round_trip(df, MediaTypes.IPC), df)

    def test_round_trip_feather_eager(self):
        df = _make_df()
        _assert_equal(_round_trip(df, MediaTypes.FEATHER), df)

    def test_round_trip_ipc_lazy(self):
        df = _make_df()
        result = _round_trip(df, MediaTypes.IPC, lazy=True)
        assert isinstance(result, pl.LazyFrame)
        _assert_equal(result, df)

    def test_scan_ipc_uses_path_when_available(self, tmp_path):
        df = _make_df()
        p = tmp_path / "data.ipc"
        df.write_ipc(p)
        buf = BytesIO(p)
        result = buf.read_polars(MediaTypes.IPC, lazy=True)
        assert isinstance(result, pl.LazyFrame)
        _assert_equal(result, df)

    def test_auto_detect_ipc(self):
        df = _make_df()
        buf = BytesIO()
        buf.write_polars(df, MediaTypes.IPC)
        buf.seek(0)
        _assert_equal(buf.read_polars(), df)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

class TestCSV:
    def test_round_trip_eager(self):
        # CSV loses type information for booleans/ints — compare string-safe subset
        df = pl.DataFrame({"sym": ["TTF", "JKM"], "price": [42.5, 18.3]})
        _assert_equal(_round_trip(df, MediaTypes.CSV), df)

    def test_round_trip_lazy_wraps_lazy(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = _round_trip(df, MediaTypes.CSV, lazy=True)
        assert isinstance(result, pl.LazyFrame)

    def test_auto_detect_csv_explicit(self):
        """CSV has no magic bytes — explicit content_type required."""
        df = pl.DataFrame({"x": [1, 2]})
        buf = BytesIO()
        buf.write_polars(df, MediaTypes.CSV)
        buf.seek(0)
        _assert_equal(buf.read_polars(MediaTypes.CSV), df)


# ---------------------------------------------------------------------------
# JSON / NDJSON
# ---------------------------------------------------------------------------

class TestJSON:
    def test_round_trip_json_eager(self):
        df = pl.DataFrame({"sym": ["TTF"], "price": [42.5]})
        _assert_equal(_round_trip(df, MediaTypes.JSON), df)

    def test_round_trip_ndjson_eager(self):
        df = pl.DataFrame({"sym": ["TTF", "JKM"], "price": [42.5, 18.3]})
        _assert_equal(_round_trip(df, MediaTypes.NDJSON), df)

    def test_round_trip_json_lazy(self):
        df = pl.DataFrame({"a": [1, 2]})
        result = _round_trip(df, MediaTypes.JSON, lazy=True)
        assert isinstance(result, pl.LazyFrame)
        _assert_equal(result, df)

    def test_round_trip_ndjson_lazy(self):
        df = pl.DataFrame({"a": [1, 2]})
        result = _round_trip(df, MediaTypes.NDJSON, lazy=True)
        assert isinstance(result, pl.LazyFrame)
        _assert_equal(result, df)


# ---------------------------------------------------------------------------
# Avro
# ---------------------------------------------------------------------------

class TestAvro:
    def test_round_trip_eager(self):
        df = pl.DataFrame({"sym": ["TTF", "JKM"], "price": [42.5, 18.3]})
        _assert_equal(_round_trip(df, MediaTypes.AVRO), df)

    def test_round_trip_lazy(self):
        df = pl.DataFrame({"a": [1, 2]})
        result = _round_trip(df, MediaTypes.AVRO, lazy=True)
        assert isinstance(result, pl.LazyFrame)
        _assert_equal(result, df)


# ---------------------------------------------------------------------------
# ZIP
# ---------------------------------------------------------------------------

def _make_zip_bytes(*dfs: pl.DataFrame, fmt: MediaType = MediaTypes.PARQUET) -> bytes:
    """Pack one DataFrame per entry into a ZIP archive."""
    sink = io.BytesIO()
    with zipfile.ZipFile(sink, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, df in enumerate(dfs):
            entry_buf = io.BytesIO()
            if fmt == MediaTypes.PARQUET:
                df.write_parquet(entry_buf)
            elif fmt == MediaTypes.CSV:
                df.write_csv(entry_buf)
            entry_buf.seek(0)
            zf.writestr(f"data_{i}.{fmt.full_extension}", entry_buf.read())
    return sink.getvalue()


class TestZIP:
    def test_single_entry_parquet(self):
        df = _make_df()
        raw = _make_zip_bytes(df, fmt=MediaTypes.PARQUET)
        buf = BytesIO(raw)
        result = buf.read_polars(MediaTypes.ZIP)
        _assert_equal(result, df)

    def test_multi_entry_concat(self):
        df1 = pl.DataFrame({"a": [1, 2], "b": [10, 20]})
        df2 = pl.DataFrame({"a": [3, 4], "b": [30, 40]})
        expected = pl.concat([df1, df2], rechunk=False)
        raw = _make_zip_bytes(df1, df2, fmt=MediaTypes.PARQUET)
        buf = BytesIO(raw)
        result = buf.read_polars()
        _assert_equal(result, expected)

    def test_empty_zip_raises(self):
        sink = io.BytesIO()
        with zipfile.ZipFile(sink, mode="w") as _:
            pass
        buf = BytesIO(sink.getvalue())
        with pytest.raises(ValueError, match="ZIP archive is empty"):
            buf.read_polars(MediaTypes.ZIP)

    def test_lazy_zip_returns_lazyframe(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        raw = _make_zip_bytes(df, fmt=MediaTypes.PARQUET)
        buf = BytesIO(raw)
        result = buf.read_polars(MediaTypes.ZIP, lazy=True)
        assert isinstance(result, pl.LazyFrame)
        _assert_equal(result, df)


# ---------------------------------------------------------------------------
# Outer codec decompression
# ---------------------------------------------------------------------------

class TestCodec:
    @pytest.mark.parametrize("codec", [Codec.ZSTD, Codec.GZIP, Codec.LZ4])
    def test_codec_round_trip_parquet(self, codec: Codec):
        df = _make_df()
        buf = BytesIO()
        buf.write_polars(df, MediaTypes.PARQUET, codec=codec)
        buf.seek(0)
        # content_type should carry the codec; read_polars decompresses transparently
        result = buf.read_polars(content_type=".parquet." + codec.name)
        _assert_equal(result, df)

    @pytest.mark.parametrize("codec", [Codec.ZSTD, Codec.GZIP])
    def test_codec_round_trip_csv(self, codec: Codec):
        df = pl.DataFrame({"sym": ["TTF"], "price": [42.5]})
        buf = BytesIO()
        buf.write_polars(df, MediaTypes.CSV, codec=codec)
        buf.seek(0)
        result = buf.read_polars(MediaType.of(MediaTypes.CSV, codec=codec))
        _assert_equal(result, df)


# ---------------------------------------------------------------------------
# Unsupported MediaType
# ---------------------------------------------------------------------------

class TestUnsupportedMediaType:
    def test_write_unsupported_raises(self):
        df = _make_df()
        buf = BytesIO()
        with pytest.raises(ValueError, match="unsupported MediaType"):
            buf.write_polars(df, MediaTypes.ZIP)

    def test_read_unsupported_raises(self):
        buf = BytesIO(b"\x00\x01\x02\x03")
        with pytest.raises(ValueError, match="unsupported MediaType"):
            buf.read_polars(None)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_dataframe_round_trip(self):
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.Utf8)})
        buf = BytesIO()
        buf.write_polars(df, MediaTypes.PARQUET)
        buf.seek(0)
        result = buf.read_polars(MediaTypes.PARQUET)
        assert result.shape == (0, 2)
        assert result.columns == ["a", "b"]

    def test_single_row(self):
        df = pl.DataFrame({"sym": ["TTF"], "price": [42.5]})
        _assert_equal(_round_trip(df, MediaTypes.PARQUET), df)

    def test_large_dataframe_spill(self):
        """DataFrame exceeding spill threshold should still round-trip correctly."""
        from yggdrasil.io.config import BufferConfig
        df = pl.DataFrame({"val": list(range(100_000))})
        buf = BytesIO(config=BufferConfig(spill_bytes=1024))  # 1 KB threshold — forces spill
        buf.write_polars(df, MediaTypes.PARQUET)
        buf.seek(0)
        _assert_equal(buf.read_polars(MediaTypes.PARQUET), df)

    def test_multiple_writes_concat_zip(self):
        """Verify write_polars can be called multiple times (cursor management)."""
        df1 = pl.DataFrame({"x": [1]})
        df2 = pl.DataFrame({"x": [2]})
        for df in (df1, df2):
            buf = BytesIO()
            buf.write_polars(df, MediaTypes.PARQUET)
            buf.seek(0)
            _assert_equal(buf.read_polars(MediaTypes.PARQUET), df)

    def test_nested_struct_column(self):
        """Struct columns should survive a Parquet round-trip."""
        df = pl.DataFrame({"meta": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]})
        _assert_equal(_round_trip(df, MediaTypes.PARQUET), df)

    def test_list_column(self):
        df = pl.DataFrame({"prices": [[1.0, 2.0], [3.0]]})
        _assert_equal(_round_trip(df, MediaTypes.PARQUET), df)

    def test_null_column(self):
        df = pl.DataFrame({"a": [1, None, 3], "b": [None, None, None]})
        _assert_equal(_round_trip(df, MediaTypes.PARQUET), df)

    def test_datetime_column(self):
        import datetime
        df = pl.DataFrame({"ts": [
            datetime.datetime(2024, 1, 1, 12, 0, 0),
            datetime.datetime(2024, 6, 15, 8, 30, 0),
        ]})
        _assert_equal(_round_trip(df, MediaTypes.PARQUET), df)