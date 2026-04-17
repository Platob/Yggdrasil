"""Robust unit tests for the yggdrasil.io.buffer media_io module and subclasses.

Covers:
  - MediaIO.make() factory dispatch (Parquet, JSON, IPC, ZIP, unknown)
  - MediaIO.codec property
  - MediaIO._decompressed_buffer / _compress_into_buffer helpers
  - MediaIO.skip_write guard (IGNORE, ERROR_IF_EXISTS, normal)
  - ParquetIO: read/write roundtrip, column projection, empty buffer, compressed roundtrip
  - JsonIO: read/write roundtrip, empty buffer, single-object fallback, compressed roundtrip
  - IPCIO: read/write roundtrip, column projection, empty buffer, compressed roundtrip
  - ZipIO: read/write roundtrip, member selection, glob, KeyError, inner media
  - MediaIO convenience wrappers: read/write pylist, pydict, polars, pandas
  - Transparent codec (gzip) roundtrip for every format
"""
from __future__ import annotations

import gzip
import io
import json
import zipfile

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest
from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.buffer.arrow_ipc_io import IPCIO, IPCOptions
from yggdrasil.io.buffer.csv_io import CsvIO, CsvOptions
from yggdrasil.io.buffer.json_io import JsonIO, JsonOptions
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.buffer.parquet_io import ParquetIO, ParquetOptions
from yggdrasil.io.buffer.xml_io import XmlIO
from yggdrasil.io.buffer.zip_io import ZipIO, ZipOptions
from yggdrasil.io.enums import MediaType, MimeType, SaveMode, GZIP
from yggdrasil.io.enums.mime_type import MimeTypes

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TABLE = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
SAMPLE_DICTS = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]


def _parquet_bytes(table: pa.Table = SAMPLE_TABLE) -> bytes:
    """Serialise *table* to Parquet bytes."""
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _ipc_bytes(table: pa.Table = SAMPLE_TABLE) -> bytes:
    """Serialise *table* to Arrow IPC file bytes."""
    buf = io.BytesIO()
    with ipc.new_file(buf, table.schema) as w:
        w.write_table(table)
    return buf.getvalue()


def _json_bytes(records: list[dict] = SAMPLE_DICTS) -> bytes:
    return json.dumps(records).encode("utf-8")


def _gzip_bytes(raw: bytes) -> bytes:
    return gzip.compress(raw)


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    """Create a ZIP archive with the given {name: payload} members."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return buf.getvalue()


# ===================================================================
# MediaIO.make() factory
# ===================================================================

class TestMediaIOMake:
    def test_make_parquet(self):
        buf = BytesIO(_parquet_bytes())
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        assert isinstance(mio, ParquetIO)
        assert mio.media_type.mime_type is MimeTypes.PARQUET
        assert mio.codec is None

    def test_make_json(self):
        buf = BytesIO(_json_bytes())
        mio = MediaIO.make(buf, MimeTypes.JSON)
        assert isinstance(mio, JsonIO)

    def test_make_csv(self):
        buf = BytesIO(b"a,b\n1,2\n")
        mio = MediaIO.make(buf, MimeTypes.CSV)
        assert isinstance(mio, CsvIO)

    def test_make_ndjson_dispatches_to_json_io(self):
        buf = BytesIO(b'{"a":1}\n{"a":2}\n')
        mio = MediaIO.make(buf, MimeTypes.NDJSON)
        assert isinstance(mio, JsonIO)

    def test_make_arrow_ipc(self):
        buf = BytesIO(_ipc_bytes())
        mio = MediaIO.make(buf, MimeTypes.ARROW_IPC)
        assert isinstance(mio, IPCIO)

    def test_make_xml(self):
        buf = BytesIO(b"<rows><row><a>1</a></row></rows>")
        mio = MediaIO.make(buf, MimeTypes.XML)
        assert isinstance(mio, XmlIO)

    def test_make_zip(self):
        buf = BytesIO(_zip_bytes({"data.parquet": _parquet_bytes()}))
        mio = MediaIO.make(buf, MimeTypes.ZIP)
        assert isinstance(mio, ZipIO)

    def test_make_with_codec_preserves_it(self):
        mt = MediaType(MimeTypes.PARQUET, codec=GZIP)
        buf = BytesIO(_gzip_bytes(_parquet_bytes()))
        mio = MediaIO.make(buf, mt)
        assert isinstance(mio, ParquetIO)
        assert mio.codec is GZIP

    def test_make_unsupported_raises(self):
        buf = BytesIO(b"hello")
        with pytest.raises(NotImplementedError):
            MediaIO.make(buf, "text/plain")

    def test_make_sets_buffer_media_type(self):
        buf = BytesIO(_parquet_bytes())
        MediaIO.make(buf, MimeTypes.PARQUET)
        assert buf.media_type.mime_type is MimeTypes.PARQUET

    def test_make_from_string_path_infers_media(self, tmp_path):
        path = tmp_path / "sample.parquet"
        path.write_bytes(_parquet_bytes())

        mio = MediaIO.make(str(path))

        assert isinstance(mio, ParquetIO)
        assert mio.media_type.mime_type is MimeTypes.PARQUET
        assert mio.read_arrow_table().equals(SAMPLE_TABLE)

    def test_make_from_pathlib_path_infers_media(self, tmp_path):
        path = tmp_path / "sample.csv"
        path.write_bytes(b"a,b\n1,x\n2,y\n3,z\n")

        mio = MediaIO.make(path)

        assert isinstance(mio, CsvIO)
        assert mio.media_type.mime_type is MimeTypes.CSV

    def test_make_from_path_with_explicit_media(self, tmp_path):
        path = tmp_path / "no_extension"
        path.write_bytes(_parquet_bytes())

        mio = MediaIO.make(path, MimeTypes.PARQUET)

        assert isinstance(mio, ParquetIO)

    def test_write_table_from_string_path(self, tmp_path):
        src = tmp_path / "src.parquet"
        src.write_bytes(_parquet_bytes())

        out_buf = BytesIO()
        mio = MediaIO.make(out_buf, MimeTypes.PARQUET)
        mio.write_table(str(src))

        result = MediaIO.make(
            BytesIO(out_buf.to_bytes()), MimeTypes.PARQUET
        ).read_arrow_table()
        assert result.equals(SAMPLE_TABLE)

    def test_write_table_from_pathlib_path_across_formats(self, tmp_path):
        src = tmp_path / "src.parquet"
        src.write_bytes(_parquet_bytes())

        out_buf = BytesIO()
        mio = MediaIO.make(out_buf, MimeTypes.JSON)
        mio.write_table(src)

        result = MediaIO.make(
            BytesIO(out_buf.to_bytes()), MimeTypes.JSON
        ).read_arrow_table()
        assert result.to_pylist() == SAMPLE_TABLE.to_pylist()

    def test_write_table_from_directory_path(self, tmp_path):
        dataset = tmp_path / "ds"
        dataset.mkdir()
        (dataset / "part.parquet").write_bytes(_parquet_bytes())

        out_buf = BytesIO()
        mio = MediaIO.make(out_buf, MimeTypes.PARQUET)
        mio.write_table(dataset)

        result = MediaIO.make(
            BytesIO(out_buf.to_bytes()), MimeTypes.PARQUET
        ).read_arrow_table()
        assert result.equals(SAMPLE_TABLE)


# ===================================================================
# Codec helpers
# ===================================================================

class TestCodecHelpers:
    def test_codec_property_none(self):
        buf = BytesIO(_parquet_bytes())
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        assert mio.codec is None

    def test_codec_property_gzip(self):
        mt = MediaType(MimeTypes.JSON, codec=GZIP)
        buf = BytesIO(_gzip_bytes(_json_bytes()))
        mio = MediaIO.make(buf, mt)
        assert mio.codec is GZIP

    def test_decompressed_buffer_no_codec(self):
        buf = BytesIO(_parquet_bytes())
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        result_buf, was_decompressed = mio._decompressed_buffer()
        assert result_buf is buf
        assert was_decompressed is False

    def test_decompressed_buffer_with_codec(self):
        raw = _json_bytes()
        mt = MediaType(MimeTypes.JSON, codec=GZIP)
        buf = BytesIO(_gzip_bytes(raw))
        mio = MediaIO.make(buf, mt)
        result_buf, was_decompressed = mio._decompressed_buffer()
        assert was_decompressed is True
        assert result_buf.to_bytes() == raw

    def test_decompressed_buffer_empty_with_codec(self):
        """Empty buffer should not attempt decompression."""
        mt = MediaType(MimeTypes.JSON, codec=GZIP)
        buf = BytesIO()
        mio = MediaIO.make(buf, mt)
        result_buf, was_decompressed = mio._decompressed_buffer()
        assert was_decompressed is False
        assert result_buf is buf


# ===================================================================
# skip_write guard
# ===================================================================

class TestSkipWrite:
    def test_ignore_mode_on_non_empty(self):
        buf = BytesIO(b"data")
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        assert mio.skip_write(SaveMode.IGNORE) is True

    def test_error_if_exists_on_non_empty(self):
        buf = BytesIO(b"data")
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        with pytest.raises(IOError):
            mio.skip_write(SaveMode.ERROR_IF_EXISTS)

    def test_overwrite_on_non_empty(self):
        buf = BytesIO(b"data")
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        assert mio.skip_write(SaveMode.OVERWRITE) is False

    def test_any_mode_on_empty(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        assert mio.skip_write(SaveMode.IGNORE) is False
        assert mio.skip_write(SaveMode.ERROR_IF_EXISTS) is False
        assert mio.skip_write(SaveMode.OVERWRITE) is False


# ===================================================================
# ParquetIO
# ===================================================================

class TestParquetIO:
    def test_write_read_roundtrip(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(SAMPLE_TABLE)
        result = mio.read_arrow_table()
        assert result.equals(SAMPLE_TABLE)

    def test_read_empty_returns_empty_table(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        result = mio.read_arrow_table()
        assert result.num_rows == 0

    def test_column_projection(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(SAMPLE_TABLE)
        result = mio.read_arrow_table(options=ParquetOptions(columns=["b"]))
        assert result.column_names == ["b"]
        assert result.num_rows == 3

    def test_gzip_compressed_roundtrip(self):
        """Write plain parquet, gzip it, then read via ParquetIO with codec."""
        raw = _parquet_bytes()
        compressed = _gzip_bytes(raw)
        mt = MediaType(MimeTypes.PARQUET, codec=GZIP)
        buf = BytesIO(compressed)
        mio = MediaIO.make(buf, mt)
        result = mio.read_arrow_table()
        assert result.equals(SAMPLE_TABLE)

    def test_write_with_gzip_codec(self):
        """Write through ParquetIO with gzip codec, verify buffer is compressed."""
        mt = MediaType(MimeTypes.PARQUET, codec=GZIP)
        buf = BytesIO()
        mio = MediaIO.make(buf, mt)
        mio.write_arrow_table(SAMPLE_TABLE)
        # Buffer should be gzip-compressed
        raw = buf.to_bytes()
        assert raw[:2] == b"\x1f\x8b"  # gzip magic bytes
        # Should round-trip
        result = mio.read_arrow_table()
        assert result.equals(SAMPLE_TABLE)

    def test_check_options_defaults(self):
        opts = ParquetIO.check_options(None)
        assert isinstance(opts, ParquetOptions)
        assert opts.compression == "zstd"

    def test_check_options_override(self):
        opts = ParquetIO.check_options(None, compression="snappy")
        assert opts.compression == "snappy"


# ===================================================================
# CsvIO
# ===================================================================

class TestCsvIO:
    def test_write_read_roundtrip(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.CSV)
        mio.write_arrow_table(SAMPLE_TABLE)
        result = mio.read_arrow_table()
        assert result.to_pylist() == SAMPLE_TABLE.to_pylist()

    def test_read_empty_returns_empty_table(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.CSV)
        result = mio.read_arrow_table()
        assert result.num_rows == 0

    def test_column_projection(self):
        buf = BytesIO(b"a,b,c\n1,2,3\n4,5,6\n")
        mio = MediaIO.make(buf, MimeTypes.CSV)
        result = mio.read_arrow_table(options=CsvOptions(columns=["a", "c"]))
        assert result.column_names == ["a", "c"]
        assert result.to_pylist() == [{"a": 1, "c": 3}, {"a": 4, "c": 6}]

    def test_gzip_compressed_roundtrip(self):
        mt = MediaType(MimeTypes.CSV, codec=GZIP)
        buf = BytesIO()
        mio = MediaIO.make(buf, mt)
        mio.write_arrow_table(SAMPLE_TABLE)
        result = mio.read_arrow_table()
        assert result.to_pylist() == SAMPLE_TABLE.to_pylist()


# ===================================================================
# JsonIO
# ===================================================================

class TestJsonIO:
    def test_write_read_roundtrip(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.JSON)
        mio.write_arrow_table(SAMPLE_TABLE)
        result = mio.read_arrow_table()
        assert result.to_pylist() == SAMPLE_TABLE.to_pylist()

    def test_read_empty_returns_empty_table(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.JSON)
        result = mio.read_arrow_table()
        assert result.num_rows == 0

    def test_read_single_object_wrapped_in_list(self):
        """A single JSON object (not array) should be treated as [obj]."""
        buf = BytesIO(json.dumps({"a": 1}).encode("utf-8"))
        mio = MediaIO.make(buf, MimeTypes.JSON)
        records = mio.read_pylist()
        assert records == [{"a": 1}]

    def test_read_list_of_dicts(self):
        buf = BytesIO(_json_bytes())
        mio = MediaIO.make(buf, MimeTypes.JSON)
        records = mio.read_pylist()
        assert records == SAMPLE_DICTS

    def test_read_list_of_scalars_expands_to_value_column(self):
        """A JSON list of scalars [1, 2, 3] → rows with a 'value' column."""
        buf = BytesIO(json.dumps([10, 20, 30]).encode("utf-8"))
        mio = MediaIO.make(buf, MimeTypes.JSON)
        result = mio.read_arrow_table()
        assert result.column_names == ["value"]
        assert result.to_pylist() == [{"value": 10}, {"value": 20}, {"value": 30}]

    def test_read_list_of_strings_expands_to_value_column(self):
        buf = BytesIO(json.dumps(["a", "b", "c"]).encode("utf-8"))
        mio = MediaIO.make(buf, MimeTypes.JSON)
        result = mio.read_pylist()
        assert result == [{"value": "a"}, {"value": "b"}, {"value": "c"}]

    def test_read_list_of_mixed_scalars(self):
        buf = BytesIO(json.dumps([1, "two", 3.0, None]).encode("utf-8"))
        mio = MediaIO.make(buf, MimeTypes.JSON)
        result = mio.read_pylist()
        assert len(result) == 4
        assert all("value" in r for r in result)

    def test_read_empty_list_returns_empty_table(self):
        buf = BytesIO(b"[]")
        mio = MediaIO.make(buf, MimeTypes.JSON)
        result = mio.read_arrow_table()
        assert result.num_rows == 0

    def test_write_produces_json_array(self):
        """Write should produce a top-level JSON array."""
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.JSON)
        mio.write_arrow_table(SAMPLE_TABLE)
        raw = buf.to_bytes()
        parsed = json.loads(raw)
        assert isinstance(parsed, list)
        assert parsed == SAMPLE_DICTS

    def test_write_read_pylist(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.JSON)
        mio.write_pylist(SAMPLE_DICTS)
        assert mio.read_pylist() == SAMPLE_DICTS

    def test_gzip_compressed_roundtrip(self):
        raw = _json_bytes()
        mt = MediaType(MimeTypes.JSON, codec=GZIP)
        buf = BytesIO(_gzip_bytes(raw))
        mio = MediaIO.make(buf, mt)
        result = mio.read_arrow_table()
        assert result.to_pylist() == SAMPLE_DICTS

    def test_write_with_gzip_codec(self):
        mt = MediaType(MimeTypes.JSON, codec=GZIP)
        buf = BytesIO()
        mio = MediaIO.make(buf, mt)
        mio.write_arrow_table(SAMPLE_TABLE)
        raw = buf.to_bytes()
        assert raw[:2] == b"\x1f\x8b"  # gzip magic
        # Round-trip
        result = mio.read_arrow_table()
        assert result.to_pylist() == SAMPLE_TABLE.to_pylist()

    def test_read_scalar_toplevel(self):
        """A bare scalar JSON value wraps into [{value: x}]."""
        buf = BytesIO(b"42")
        mio = MediaIO.make(buf, MimeTypes.JSON)
        result = mio.read_pylist()
        assert result == [{"value": 42}]


# ===================================================================
# IPCIO
# ===================================================================

class TestIPCIO:
    def test_write_read_roundtrip(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.ARROW_IPC)
        mio.write_arrow_table(SAMPLE_TABLE)
        result = mio.read_arrow_table()
        assert result.equals(SAMPLE_TABLE)

    def test_read_empty_returns_empty_table(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.ARROW_IPC)
        result = mio.read_arrow_table()
        assert result.num_rows == 0

    def test_column_projection(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.ARROW_IPC)
        mio.write_arrow_table(SAMPLE_TABLE)
        result = mio.read_arrow_table(options=IPCOptions(columns=["a"]))
        assert result.column_names == ["a"]
        assert result.num_rows == 3

    def test_read_from_file_format(self):
        """Verify we can read IPC file format bytes."""
        buf = BytesIO(_ipc_bytes())
        mio = MediaIO.make(buf, MimeTypes.ARROW_IPC)
        result = mio.read_arrow_table()
        assert result.equals(SAMPLE_TABLE)

    def test_read_from_stream_format(self):
        """Verify we can read IPC stream format bytes."""
        stream_buf = io.BytesIO()
        with ipc.new_stream(stream_buf, SAMPLE_TABLE.schema) as w:
            w.write_table(SAMPLE_TABLE)
        buf = BytesIO(stream_buf.getvalue())
        mio = MediaIO.make(buf, MimeTypes.ARROW_IPC)
        result = mio.read_arrow_table()
        assert result.equals(SAMPLE_TABLE)

    def test_gzip_compressed_roundtrip(self):
        raw = _ipc_bytes()
        mt = MediaType(MimeTypes.ARROW_IPC, codec=GZIP)
        buf = BytesIO(_gzip_bytes(raw))
        mio = MediaIO.make(buf, mt)
        result = mio.read_arrow_table()
        assert result.equals(SAMPLE_TABLE)

    def test_write_with_gzip_codec(self):
        mt = MediaType(MimeTypes.ARROW_IPC, codec=GZIP)
        buf = BytesIO()
        mio = MediaIO.make(buf, mt)
        mio.write_arrow_table(SAMPLE_TABLE)
        raw = buf.to_bytes()
        assert raw[:2] == b"\x1f\x8b"  # gzip magic
        result = mio.read_arrow_table()
        assert result.equals(SAMPLE_TABLE)


# ===================================================================
# ZipIO
# ===================================================================

class TestZipIO:
    def test_write_read_roundtrip(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.ZIP)
        mio.write_arrow_table(SAMPLE_TABLE)
        result = mio.read_arrow_table()
        assert result.to_pylist() == SAMPLE_TABLE.to_pylist()

    def test_read_empty_returns_empty(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.ZIP)
        result = mio.read_arrow_table()
        assert result.num_rows == 0

    def test_read_single_parquet_member(self):
        raw_zip = _zip_bytes({"data.parquet": _parquet_bytes()})
        buf = BytesIO(raw_zip)
        mio = MediaIO.make(buf, MimeTypes.ZIP)
        result = mio.read_arrow_table()
        assert result.equals(SAMPLE_TABLE)

    def test_read_specific_member(self):
        t1 = pa.table({"x": [1]})
        t2 = pa.table({"x": [2]})
        raw_zip = _zip_bytes({
            "a.parquet": _parquet_bytes(t1),
            "b.parquet": _parquet_bytes(t2),
        })
        buf = BytesIO(raw_zip)
        mio = MediaIO.make(buf, MimeTypes.ZIP)
        result = mio.read_arrow_table(options=ZipOptions(member="b.parquet"))
        assert result.to_pylist() == [{"x": 2}]

    def test_glob_member_selection(self):
        t1 = pa.table({"x": [1]})
        t2 = pa.table({"x": [2]})
        t3 = pa.table({"y": [3]})
        raw_zip = _zip_bytes({
            "data_a.parquet": _parquet_bytes(t1),
            "data_b.parquet": _parquet_bytes(t2),
            "other.json": _json_bytes([{"y": 3}]),
        })
        buf = BytesIO(raw_zip)
        mio = MediaIO.make(buf, MimeTypes.ZIP)
        result = mio.read_arrow_table(options=ZipOptions(member="data_*.parquet"))
        assert result.num_rows == 2

    def test_missing_member_raises_key_error(self):
        raw_zip = _zip_bytes({"data.parquet": _parquet_bytes()})
        buf = BytesIO(raw_zip)
        mio = MediaIO.make(buf, MimeTypes.ZIP)
        with pytest.raises(KeyError, match="missing.parquet"):
            mio.read_arrow_table(options=ZipOptions(member="missing.parquet"))

    def test_glob_no_match_raises_key_error(self):
        raw_zip = _zip_bytes({"data.parquet": _parquet_bytes()})
        buf = BytesIO(raw_zip)
        mio = MediaIO.make(buf, MimeTypes.ZIP)
        with pytest.raises(KeyError, match="no members matched"):
            mio.read_arrow_table(options=ZipOptions(member="*.csv"))

    def test_write_custom_inner_media(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.ZIP)
        mio.write_arrow_table(SAMPLE_TABLE, options=ZipOptions(inner_media=MimeTypes.ARROW_IPC))
        # Verify inner member is IPC
        with zipfile.ZipFile(io.BytesIO(buf.to_bytes()), "r") as zf:
            names = zf.namelist()
            assert len(names) == 1
            assert names[0].endswith(".ipc") or names[0].endswith(".arrow")

    def test_write_custom_member_name(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.ZIP)
        mio.write_arrow_table(SAMPLE_TABLE, options=ZipOptions(member="custom.parquet"))
        with zipfile.ZipFile(io.BytesIO(buf.to_bytes()), "r") as zf:
            assert "custom.parquet" in zf.namelist()


# ===================================================================
# Convenience wrappers: pylist, pydict
# ===================================================================

class TestConvenienceWrappers:
    def test_write_read_pylist_via_parquet(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_pylist(SAMPLE_DICTS)
        result = mio.read_pylist()
        assert result == SAMPLE_DICTS

    def test_write_read_pydict_via_parquet(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        pydict = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        mio.write_pydict(pydict)
        result = mio.read_pydict()
        assert result == pydict

    def test_write_read_pylist_via_json(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.JSON)
        mio.write_pylist(SAMPLE_DICTS)
        result = mio.read_pylist()
        assert result == SAMPLE_DICTS

    def test_write_read_pylist_via_ipc(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.ARROW_IPC)
        mio.write_pylist(SAMPLE_DICTS)
        result = mio.read_pylist()
        assert result == SAMPLE_DICTS


# ===================================================================
# Polars / pandas wrappers
# ===================================================================

class TestPolarsWrapper:
    def test_read_polars_frame(self):
        buf = BytesIO(_parquet_bytes())
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        df = mio.read_polars_frame()
        assert df.shape == (3, 2)

    def test_read_polars_lazy(self):
        buf = BytesIO(_parquet_bytes())
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        lf = mio.read_polars_frame(options=ParquetOptions(lazy=True))
        from yggdrasil.polars.lib import polars as pl
        assert isinstance(lf, pl.LazyFrame)
        assert lf.collect().shape == (3, 2)

    def test_write_polars_frame(self):
        from yggdrasil.polars.lib import polars as pl
        df = pl.from_arrow(SAMPLE_TABLE)
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_polars_frame(df)
        result = mio.read_arrow_table()
        assert result.to_pylist() == SAMPLE_TABLE.to_pylist()


class TestPandasWrapper:
    def test_read_pandas_frame(self):
        buf = BytesIO(_parquet_bytes())
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        df = mio.read_pandas_frame()
        assert df.shape == (3, 2)

    def test_write_pandas_frame(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_pandas_frame(df)
        result = mio.read_arrow_table()
        assert result.num_rows == 3


# ===================================================================
# Cross-format: transparent codec roundtrip
# ===================================================================

class TestTransparentCodecRoundtrip:
    """Verify that write→read works with gzip codec for every format."""

    @pytest.mark.parametrize("mime", [MimeTypes.PARQUET, MimeTypes.JSON, MimeTypes.ARROW_IPC])
    def test_write_then_read_with_gzip(self, mime):
        mt = MediaType(mime, codec=GZIP)
        buf = BytesIO()
        mio = MediaIO.make(buf, mt)
        mio.write_arrow_table(SAMPLE_TABLE)
        # Buffer must be gzip-compressed
        raw = buf.to_bytes()
        assert raw[:2] == b"\x1f\x8b", f"{mime}: expected gzip magic"
        # Read back
        result = mio.read_arrow_table()
        assert result.to_pylist() == SAMPLE_TABLE.to_pylist()

    @pytest.mark.parametrize("mime", [MimeTypes.PARQUET, MimeTypes.JSON, MimeTypes.ARROW_IPC])
    def test_pre_compressed_read(self, mime):
        """Pre-compress format bytes externally, then read via MediaIO."""
        # Build plain bytes for the format
        if mime is MimeTypes.PARQUET:
            plain = _parquet_bytes()
        elif mime is MimeTypes.JSON:
            plain = _json_bytes()
        else:
            plain = _ipc_bytes()

        mt = MediaType(mime, codec=GZIP)
        buf = BytesIO(_gzip_bytes(plain))
        mio = MediaIO.make(buf, mt)
        result = mio.read_arrow_table()
        assert result.to_pylist() == SAMPLE_TABLE.to_pylist()


# ===================================================================
# Options validation
# ===================================================================

class TestOptionsValidation:
    def test_mode_is_normalised(self):
        opts = ParquetIO.check_options(None, mode="overwrite")
        assert opts.mode == SaveMode.OVERWRITE


# ===================================================================
# Batched iteration (batch_size parameter)
# ===================================================================

# A larger sample for meaningful chunking
BIG_TABLE = pa.table({"i": list(range(10)), "s": [f"v{i}" for i in range(10)]})


class TestBatchedReadArrowTable:
    def test_batch_size_none_returns_table(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        result = mio.read_arrow_table(options=ParquetOptions(batch_size=None))
        assert isinstance(result, pa.Table)
        assert result.num_rows == 10

    def test_batch_size_zero_returns_table(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        result = mio.read_arrow_table(options=ParquetOptions(batch_size=0))
        assert isinstance(result, pa.Table)

    def test_batch_size_negative_returns_table(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        result = mio.read_arrow_table(options=ParquetOptions(batch_size=-5))
        assert isinstance(result, pa.Table)

    def test_batch_size_returns_iterator(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        result = mio.read_arrow_table(options=ParquetOptions(batch_size=3))
        # Should be an iterator, not a table
        assert hasattr(result, '__iter__') and hasattr(result, '__next__')

    def test_batch_size_correct_chunks(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        chunks = list(mio.read_arrow_table(options=ParquetOptions(batch_size=3)))
        # 10 rows / 3 = 4 chunks: 3, 3, 3, 1
        assert len(chunks) == 4
        assert chunks[0].num_rows == 3
        assert chunks[1].num_rows == 3
        assert chunks[2].num_rows == 3
        assert chunks[3].num_rows == 1

    def test_batch_size_equal_to_total(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        chunks = list(mio.read_arrow_table(options=ParquetOptions(batch_size=10)))
        assert len(chunks) == 1
        assert chunks[0].num_rows == 10

    def test_batch_size_larger_than_total(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        chunks = list(mio.read_arrow_table(options=ParquetOptions(batch_size=100)))
        assert len(chunks) == 1
        assert chunks[0].num_rows == 10

    def test_batch_preserves_data(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        chunks = list(mio.read_arrow_table(options=ParquetOptions(batch_size=4)))
        combined = pa.concat_tables(chunks)
        assert combined.to_pylist() == BIG_TABLE.to_pylist()

    def test_batch_with_json(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.JSON)
        mio.write_arrow_table(BIG_TABLE)
        chunks = list(mio.read_arrow_table(options=JsonOptions(batch_size=5)))
        assert len(chunks) == 2
        combined = pa.concat_tables(chunks)
        assert combined.to_pylist() == BIG_TABLE.to_pylist()

    def test_batch_with_ipc(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.ARROW_IPC)
        mio.write_arrow_table(BIG_TABLE)
        chunks = list(mio.read_arrow_table(options=IPCOptions(batch_size=4)))
        assert len(chunks) == 3
        combined = pa.concat_tables(chunks)
        assert combined.equals(BIG_TABLE)

    def test_batch_with_gzip_codec(self):
        """batch_size works correctly even with gzip transparent decompression."""
        mt = MediaType(MimeTypes.PARQUET, codec=GZIP)
        buf = BytesIO()
        mio = MediaIO.make(buf, mt)
        mio.write_arrow_table(BIG_TABLE)
        chunks = list(mio.read_arrow_table(options=ParquetOptions(batch_size=4)))
        combined = pa.concat_tables(chunks)
        assert combined.to_pylist() == BIG_TABLE.to_pylist()


class TestBatchedReadPylist:
    def test_no_batch_returns_list(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        result = mio.read_pylist()
        assert isinstance(result, list)
        assert len(result) == 10

    def test_batch_returns_iterator_of_lists(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        result = mio.read_pylist(options=ParquetOptions(batch_size=4))
        chunks = list(result)
        assert len(chunks) == 3
        assert all(isinstance(c, list) for c in chunks)
        flat = [row for c in chunks for row in c]
        assert flat == BIG_TABLE.to_pylist()


class TestBatchedReadPydict:
    def test_no_batch_returns_dict(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        result = mio.read_pydict()
        assert isinstance(result, dict)

    def test_batch_returns_iterator_of_dicts(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        result = mio.read_pydict(options=ParquetOptions(batch_size=5))
        chunks = list(result)
        assert len(chunks) == 2
        assert all(isinstance(c, dict) for c in chunks)
        # Merge all chunks back
        all_i = [v for c in chunks for v in c["i"]]
        assert all_i == list(range(10))


class TestBatchedReadPolars:
    def test_no_batch_returns_frame(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        from yggdrasil.polars.lib import polars as pl
        result = mio.read_polars_frame()
        assert isinstance(result, pl.DataFrame)

    def test_batch_returns_iterator_of_frames(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        from yggdrasil.polars.lib import polars as pl
        result = mio.read_polars_frame(options=ParquetOptions(batch_size=3))
        chunks = list(result)
        assert len(chunks) == 4
        assert all(isinstance(c, pl.DataFrame) for c in chunks)
        combined = pl.concat(chunks)
        assert combined.shape == (10, 2)

    def test_batch_ignores_lazy(self):
        """When batching, lazy is ignored — each chunk is a DataFrame."""
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        from yggdrasil.polars.lib import polars as pl
        result = mio.read_polars_frame(options=ParquetOptions(batch_size=5, lazy=True))
        chunks = list(result)
        assert all(isinstance(c, pl.DataFrame) for c in chunks)


class TestBatchedReadPandas:
    def test_no_batch_returns_dataframe(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        import pandas as pd
        result = mio.read_pandas_frame()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_batch_returns_iterator_of_dataframes(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE)
        import pandas as pd
        result = mio.read_pandas_frame(options=ParquetOptions(batch_size=4))
        chunks = list(result)
        assert len(chunks) == 3
        assert all(isinstance(c, pd.DataFrame) for c in chunks)
        combined = pd.concat(chunks, ignore_index=True)
        assert len(combined) == 10


class TestBatchedWrite:
    def test_write_arrow_table_batch_size(self):
        """write_arrow_table with batch_size writes last chunk (each overwrites)."""
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(BIG_TABLE, options=ParquetOptions(batch_size=4))
        # The buffer now contains the LAST chunk (rows 8-9, 2 rows)
        result = mio.read_arrow_table()
        assert result.num_rows == 10
        assert result.to_pylist() == BIG_TABLE.to_pylist()

    def test_write_pylist_with_batch_size(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_pylist(BIG_TABLE.to_pylist(), options=ParquetOptions(batch_size=5))
        result = mio.read_arrow_table()
        # Last chunk of 5 rows
        assert result.num_rows == 10

    def test_write_polars_with_batch_size(self):
        from yggdrasil.polars.lib import polars as pl
        df = pl.from_arrow(BIG_TABLE)
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_polars_frame(df, options=ParquetOptions(batch_size=7))
        result = mio.read_arrow_table()
        # Last chunk: 10 - 7 = 3 rows
        assert result.num_rows == 10


# ===================================================================
# Save modes: AUTO, OVERWRITE, APPEND, UPSERT, IGNORE, ERROR_IF_EXISTS
# ===================================================================

class TestSaveModeOverwrite:
    """AUTO and OVERWRITE both replace the buffer entirely."""

    def test_auto_replaces_content(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        t1 = pa.table({"a": [1, 2]})
        t2 = pa.table({"a": [3, 4, 5]})
        mio.write_arrow_table(t1)
        mio.write_arrow_table(t2, options=ParquetOptions(mode="auto"))
        assert mio.read_arrow_table().to_pylist() == [{"a": 3}, {"a": 4}, {"a": 5}]

    def test_overwrite_replaces_content(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        t1 = pa.table({"a": [1, 2]})
        t2 = pa.table({"a": [10]})
        mio.write_arrow_table(t1)
        mio.write_arrow_table(t2, options=ParquetOptions(mode="overwrite"))
        assert mio.read_arrow_table().to_pylist() == [{"a": 10}]

    def test_overwrite_on_empty_buffer(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        t = pa.table({"x": [1]})
        mio.write_arrow_table(t, options=ParquetOptions(mode="overwrite"))
        assert mio.read_arrow_table().to_pylist() == [{"x": 1}]


class TestSaveModeAppend:
    """APPEND concatenates new data after existing data."""

    def test_append_stacks_rows(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        t1 = pa.table({"a": [1, 2]})
        t2 = pa.table({"a": [3, 4]})
        mio.write_arrow_table(t1)
        mio.write_arrow_table(t2, options=ParquetOptions(mode="append"))
        result = mio.read_arrow_table()
        assert result.to_pylist() == [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]

    def test_append_on_empty_buffer(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        t = pa.table({"a": [1, 2]})
        mio.write_arrow_table(t, options=ParquetOptions(mode="append"))
        assert mio.read_arrow_table().to_pylist() == [{"a": 1}, {"a": 2}]

    def test_append_multiple_times(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        for i in range(3):
            mio.write_arrow_table(pa.table({"v": [i]}), options=ParquetOptions(mode="append"))
        assert mio.read_arrow_table().to_pylist() == [{"v": 0}, {"v": 1}, {"v": 2}]

    def test_append_schema_promotion(self):
        """Columns present in only one side are filled with nulls."""
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        t1 = pa.table({"a": [1], "b": ["x"]})
        t2 = pa.table({"a": [2], "c": [3.14]})
        mio.write_arrow_table(t1)
        mio.write_arrow_table(t2, options=ParquetOptions(mode="append"))
        result = mio.read_arrow_table()
        assert result.num_rows == 2
        assert set(result.column_names) == {"a", "b", "c"}

    def test_append_with_json(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.JSON)
        mio.write_arrow_table(pa.table({"x": [1]}))
        mio.write_arrow_table(pa.table({"x": [2]}), options=JsonOptions(mode="append"))
        assert mio.read_arrow_table().to_pylist() == [{"x": 1}, {"x": 2}]

    def test_append_with_ipc(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.ARROW_IPC)
        mio.write_arrow_table(pa.table({"x": [1]}))
        mio.write_arrow_table(pa.table({"x": [2]}), options=IPCOptions(mode="append"))
        assert mio.read_arrow_table().to_pylist() == [{"x": 1}, {"x": 2}]

    def test_append_with_gzip_codec(self):
        mt = MediaType(MimeTypes.PARQUET, codec=GZIP)
        buf = BytesIO()
        mio = MediaIO.make(buf, mt)
        mio.write_arrow_table(pa.table({"x": [1]}))
        mio.write_arrow_table(pa.table({"x": [2]}), options=ParquetOptions(mode="append"))
        result = mio.read_arrow_table()
        assert result.to_pylist() == [{"x": 1}, {"x": 2}]


class TestSaveModeUpsert:
    """UPSERT replaces matching rows and appends new ones."""

    def test_upsert_replaces_matching_rows(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        existing = pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        incoming = pa.table({"id": [2, 3],     "val": ["B", "C"]})
        mio.write_arrow_table(existing)
        mio.write_arrow_table(incoming, options=ParquetOptions(mode="upsert", match_by=["id"]))
        result = mio.read_arrow_table()
        result_dict = {row["id"]: row["val"] for row in result.to_pylist()}
        assert result_dict == {1: "a", 2: "B", 3: "C"}

    def test_upsert_appends_new_rows(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        existing = pa.table({"id": [1], "val": ["a"]})
        incoming = pa.table({"id": [2], "val": ["b"]})
        mio.write_arrow_table(existing)
        mio.write_arrow_table(incoming, options=ParquetOptions(mode="upsert", match_by=["id"]))
        result = mio.read_arrow_table()
        assert result.num_rows == 2
        result_dict = {row["id"]: row["val"] for row in result.to_pylist()}
        assert result_dict == {1: "a", 2: "b"}

    def test_upsert_full_replace_when_all_keys_match(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        existing = pa.table({"id": [1, 2], "val": ["old1", "old2"]})
        incoming = pa.table({"id": [1, 2], "val": ["new1", "new2"]})
        mio.write_arrow_table(existing)
        mio.write_arrow_table(incoming, options=ParquetOptions(mode="upsert", match_by=["id"]))
        result = mio.read_arrow_table()
        result_dict = {row["id"]: row["val"] for row in result.to_pylist()}
        assert result_dict == {1: "new1", 2: "new2"}

    def test_upsert_composite_key(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        existing = pa.table({
            "k1": [1, 1, 2], "k2": ["a", "b", "a"], "val": [10, 20, 30],
        })
        incoming = pa.table({
            "k1": [1, 2], "k2": ["b", "a"], "val": [99, 88],
        })
        mio.write_arrow_table(existing)
        mio.write_arrow_table(incoming, options=ParquetOptions(mode="upsert", match_by=["k1", "k2"]))
        result = mio.read_arrow_table()
        rows = {(r["k1"], r["k2"]): r["val"] for r in result.to_pylist()}
        assert rows == {(1, "a"): 10, (1, "b"): 99, (2, "a"): 88}

    def test_upsert_on_empty_buffer(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        incoming = pa.table({"id": [1], "val": ["x"]})
        mio.write_arrow_table(incoming, options=ParquetOptions(mode="upsert", match_by=["id"]))
        assert mio.read_arrow_table().to_pylist() == [{"id": 1, "val": "x"}]

    def test_upsert_no_match_by_raises(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(pa.table({"id": [1]}))
        with pytest.raises(ValueError, match="match_by"):
            mio.write_arrow_table(
                pa.table({"id": [2]}), options=ParquetOptions(mode="upsert", match_by=None),
            )

    def test_upsert_missing_column_raises(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(pa.table({"id": [1], "val": ["a"]}))
        with pytest.raises(ValueError, match="not in incoming"):
            mio.write_arrow_table(
                pa.table({"other": [2]}), options=ParquetOptions(mode="upsert", match_by=["id"]),
            )

    def test_upsert_with_json(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.JSON)
        mio.write_arrow_table(pa.table({"id": [1, 2], "v": ["a", "b"]}))
        mio.write_arrow_table(
            pa.table({"id": [2], "v": ["B"]}), options=JsonOptions(mode="upsert", match_by=["id"]),
        )
        rows = {r["id"]: r["v"] for r in mio.read_arrow_table().to_pylist()}
        assert rows == {1: "a", 2: "B"}

    def test_upsert_with_gzip(self):
        mt = MediaType(MimeTypes.PARQUET, codec=GZIP)
        buf = BytesIO()
        mio = MediaIO.make(buf, mt)
        mio.write_arrow_table(pa.table({"id": [1, 2], "v": [10, 20]}))
        mio.write_arrow_table(
            pa.table({"id": [2, 3], "v": [99, 30]}),
            options=ParquetOptions(mode="upsert", match_by=["id"]),
        )
        rows = {r["id"]: r["v"] for r in mio.read_arrow_table().to_pylist()}
        assert rows == {1: 10, 2: 99, 3: 30}


class TestSaveModeIgnoreAndError:
    """These modes are handled by skip_write (already tested), but verify
    they still work end-to-end through write_arrow_table."""

    def test_ignore_skips_on_non_empty(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(pa.table({"a": [1]}))
        mio.write_arrow_table(pa.table({"a": [999]}), options=ParquetOptions(mode="ignore"))
        assert mio.read_arrow_table().to_pylist() == [{"a": 1}]

    def test_error_if_exists_raises_on_non_empty(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(pa.table({"a": [1]}))
        with pytest.raises(IOError):
            mio.write_arrow_table(
                pa.table({"a": [999]}), options=ParquetOptions(mode="error_if_exists"),
            )


class TestSaveModeViaConvenienceMethods:
    """Verify that save modes work through pylist/pydict/polars/pandas wrappers."""

    def test_append_via_write_pylist(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_pylist([{"a": 1}])
        mio.write_pylist([{"a": 2}], options=ParquetOptions(mode="append"))
        assert mio.read_pylist() == [{"a": 1}, {"a": 2}]

    def test_upsert_via_write_pydict(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_pydict({"id": [1, 2], "v": [10, 20]})
        mio.write_pydict({"id": [2], "v": [99]}, options=ParquetOptions(mode="upsert", match_by=["id"]))
        rows = {r["id"]: r["v"] for r in mio.read_pylist()}
        assert rows == {1: 10, 2: 99}

    def test_append_via_write_polars_frame(self):
        from yggdrasil.polars.lib import polars as pl
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_polars_frame(pl.DataFrame({"x": [1]}))
        mio.write_polars_frame(pl.DataFrame({"x": [2]}), options=ParquetOptions(mode="append"))
        result = mio.read_polars_frame()
        assert result.to_dicts() == [{"x": 1}, {"x": 2}]

    def test_append_via_write_pandas_frame(self):
        import pandas as pd
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_pandas_frame(pd.DataFrame({"x": [1]}))
        mio.write_pandas_frame(pd.DataFrame({"x": [2]}), options=ParquetOptions(mode="append"))
        result = mio.read_pandas_frame()
        assert list(result["x"]) == [1, 2]


# ===================================================================
# Safe ingestion of unstructured list/generator records
# ===================================================================

class TestWriteTableUnstructured:
    """Cover ``write_table`` with sparse dicts, all-None rows, and generators."""

    def test_list_with_all_none_column(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_table([{"id": None}])
        table = mio.read_arrow_table()
        assert table.num_rows == 1
        assert table.column_names == ["id"]
        assert table.to_pydict() == {"id": [None]}

    def test_list_with_sparse_keys_keeps_all_columns(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_table([{"a": 1}, {"b": "x"}, {"c": True}])
        table = mio.read_arrow_table()
        assert set(table.column_names) == {"a", "b", "c"}
        assert table.num_rows == 3
        assert table.to_pydict() == {
            "a": [1, None, None],
            "b": [None, "x", None],
            "c": [None, None, True],
        }

    def test_list_with_none_row_is_tolerated(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_table([{"id": 1}, None, {"id": 3}])
        rows = mio.read_pylist()
        assert rows == [{"id": 1}, {"id": None}, {"id": 3}]

    def test_list_rejects_non_dict_elements(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        with pytest.raises(TypeError, match="write_table\\(list\\) expects list\\[dict\\]"):
            mio.write_table([{"a": 1}, "not-a-dict"])

    def test_generator_with_sparse_keys(self):
        def gen():
            yield {"id": 1}
            yield {"id": None, "name": "alpha"}
            yield {"extra": True}

        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_table(gen())
        rows = mio.read_pylist()
        assert rows == [
            {"id": 1, "name": None, "extra": None},
            {"id": None, "name": "alpha", "extra": None},
            {"id": None, "name": None, "extra": True},
        ]

    def test_generator_with_all_none_column_roundtrips_via_json(self):
        def gen():
            yield {"id": None}
            yield {"id": None}

        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.JSON)
        mio.write_table(gen())
        rows = mio.read_pylist()
        assert rows == [{"id": None}, {"id": None}]

    def test_empty_list_writes_empty_table(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_table([])
        table = mio.read_arrow_table()
        assert table.num_rows == 0

    def test_empty_generator_writes_empty_table(self):
        def gen():
            if False:
                yield {}

        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_table(gen())
        table = mio.read_arrow_table()
        assert table.num_rows == 0

    def test_write_pylist_accepts_generator(self):
        def gen():
            yield {"id": 1}
            yield {"id": 2, "name": "bob"}

        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_pylist(gen())
        rows = mio.read_pylist()
        assert rows == [
            {"id": 1, "name": None},
            {"id": 2, "name": "bob"},
        ]

    def test_append_via_generator_with_sparse_keys(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_table([{"a": 1}])
        mio.write_table(
            iter([{"a": 2}, {"b": "x"}]),
            options=ParquetOptions(mode="append"),
        )
        rows = mio.read_pylist()
        assert rows == [
            {"a": 1, "b": None},
            {"a": 2, "b": None},
            {"a": None, "b": "x"},
        ]


# ===================================================================
# collect_schema() — cheap schema inspection across formats
# ===================================================================

class TestCollectSchema:
    """Verify that ``collect_schema`` returns the expected yggdrasil Schema
    without collecting all data."""

    @pytest.mark.parametrize(
        "mime",
        [
            MimeTypes.PARQUET,
            MimeTypes.ARROW_IPC,
            MimeTypes.JSON,
            MimeTypes.CSV,
            MimeTypes.XML,
            MimeTypes.ZIP,
        ],
    )
    def test_returns_yggdrasil_schema(self, mime):
        from yggdrasil.data.schema import Schema

        buf = BytesIO()
        mio = MediaIO.make(buf, mime)
        mio.write_arrow_table(SAMPLE_TABLE)

        schema = mio.collect_schema()
        assert isinstance(schema, Schema)
        assert list(schema.keys()) == ["a", "b"]

    @pytest.mark.parametrize(
        "mime",
        [
            MimeTypes.PARQUET,
            MimeTypes.ARROW_IPC,
            MimeTypes.JSON,
            MimeTypes.CSV,
            MimeTypes.XML,
            MimeTypes.ZIP,
        ],
    )
    def test_empty_buffer_returns_empty_schema(self, mime):
        from yggdrasil.data.schema import Schema

        buf = BytesIO()
        mio = MediaIO.make(buf, mime)
        schema = mio.collect_schema()
        assert isinstance(schema, Schema)
        assert list(schema.keys()) == []

    def test_parquet_schema_types_match(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(SAMPLE_TABLE)
        arrow_schema = mio.collect_schema().to_arrow_schema()
        assert arrow_schema.names == ["a", "b"]
        assert arrow_schema.field("a").type == pa.int64()
        assert arrow_schema.field("b").type == pa.string()

    def test_ipc_schema_types_match(self):
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.ARROW_IPC)
        mio.write_arrow_table(SAMPLE_TABLE)
        arrow_schema = mio.collect_schema().to_arrow_schema()
        assert arrow_schema.names == ["a", "b"]
        assert arrow_schema.field("a").type == pa.int64()

    def test_parquet_does_not_decode_row_groups(self, monkeypatch):
        """ParquetIO.collect_schema must read the footer only (no iter_batches)."""
        import pyarrow.parquet as pq

        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(SAMPLE_TABLE)

        real_iter_batches = pq.ParquetFile.iter_batches
        calls = {"count": 0}

        def spy(self, *args, **kwargs):
            calls["count"] += 1
            return real_iter_batches(self, *args, **kwargs)

        monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy)
        schema = mio.collect_schema()
        assert list(schema.keys()) == ["a", "b"]
        assert calls["count"] == 0

    def test_json_large_array_returns_schema(self):
        """JsonIO.collect_schema returns the schema inferred from the first
        record even for large arrays."""
        records = [{"a": i, "b": f"v{i}"} for i in range(10_000)]
        buf = BytesIO(json.dumps(records).encode("utf-8"))
        mio = MediaIO.make(buf, MimeTypes.JSON)
        schema = mio.collect_schema()
        assert list(schema.keys()) == ["a", "b"]

    def test_parquet_with_column_projection_not_used(self):
        """collect_schema returns the full schema regardless of options.columns."""
        buf = BytesIO()
        mio = MediaIO.make(buf, MimeTypes.PARQUET)
        mio.write_arrow_table(SAMPLE_TABLE)
        schema = mio.collect_schema()
        assert list(schema.keys()) == ["a", "b"]

    def test_gzip_parquet_roundtrip(self):
        mt = MediaType(MimeTypes.PARQUET, codec=GZIP)
        buf = BytesIO()
        mio = MediaIO.make(buf, mt)
        mio.write_arrow_table(SAMPLE_TABLE)
        schema = mio.collect_schema()
        assert list(schema.keys()) == ["a", "b"]

    def test_gzip_json_roundtrip(self):
        mt = MediaType(MimeTypes.JSON, codec=GZIP)
        buf = BytesIO()
        mio = MediaIO.make(buf, mt)
        mio.write_arrow_table(SAMPLE_TABLE)
        schema = mio.collect_schema()
        assert list(schema.keys()) == ["a", "b"]

    def test_csv_header_only_path(self):
        buf = BytesIO(b"a,b,c\n1,2,3\n")
        mio = MediaIO.make(buf, MimeTypes.CSV)
        schema = mio.collect_schema()
        assert list(schema.keys()) == ["a", "b", "c"]

    def test_zip_reads_first_member_schema(self):
        raw_zip = _zip_bytes({
            "a.parquet": _parquet_bytes(pa.table({"x": [1]})),
            "b.parquet": _parquet_bytes(pa.table({"y": ["z"]})),
        })
        buf = BytesIO(raw_zip)
        mio = MediaIO.make(buf, MimeTypes.ZIP)
        schema = mio.collect_schema()
        assert list(schema.keys()) == ["x"]

    def test_zip_full_unifies_all_member_schemas(self):
        raw_zip = _zip_bytes({
            "a.parquet": _parquet_bytes(pa.table({"x": [1]})),
            "b.parquet": _parquet_bytes(pa.table({"y": ["z"]})),
        })
        buf = BytesIO(raw_zip)
        mio = MediaIO.make(buf, MimeTypes.ZIP)
        schema = mio.collect_schema(full=True)
        assert set(schema.keys()) == {"x", "y"}

    def test_pathio_full_unifies_all_file_schemas(self, tmp_path):
        from yggdrasil.io.buffer.local_path_io import LocalPathIO

        pq.write_table(pa.table({"x": [1]}), tmp_path / "a.parquet")
        pq.write_table(pa.table({"y": ["z"]}), tmp_path / "b.parquet")

        pio = LocalPathIO.make(tmp_path)
        first_schema = pio.collect_schema()
        assert len(first_schema.keys()) == 1

        full_schema = pio.collect_schema(full=True)
        assert set(full_schema.keys()) == {"x", "y"}
