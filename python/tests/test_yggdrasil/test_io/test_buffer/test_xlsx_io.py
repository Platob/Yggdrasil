from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from yggdrasil.io import MimeTypes
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.buffer.xlsx_io import XlsxIO, XlsxOptions


def _has(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


HAS_POLARS = _has("polars")
HAS_PANDAS = _has("pandas")
HAS_OPENPYXL = _has("openpyxl")
HAS_XLSXWRITER = _has("xlsxwriter")


def _make_table() -> pa.Table:
    return pa.table(
        {
            "id": [1, 2, 3],
            "name": ["alice", "bob", "carol"],
            "price": [3.14, 2.71, 1.41],
        }
    )


# ------------------------------------------------------------------
# Options validation
# ------------------------------------------------------------------


def test_xlsx_options_defaults():
    opt = XlsxOptions()
    assert opt.sheet_name is None
    assert opt.sheet_id is None
    assert opt.has_header is True
    assert opt.include_header is True
    assert opt.skip_rows == 0
    assert opt.write_sheet_name == "Sheet1"
    assert opt.engine == "auto"


def test_xlsx_options_rejects_unknown_engine():
    with pytest.raises(ValueError, match="engine"):
        XlsxOptions(engine="nope")


def test_xlsx_options_rejects_bool_sheet_name():
    with pytest.raises(TypeError, match="sheet_name"):
        XlsxOptions(sheet_name=True)


def test_xlsx_options_normalises_engine_case():
    assert XlsxOptions(engine="POLARS").engine == "polars"


def test_xlsx_options_rejects_negative_sheet_id():
    with pytest.raises(ValueError, match="sheet_id"):
        XlsxOptions(sheet_id=-1)


# ------------------------------------------------------------------
# MediaIO factory
# ------------------------------------------------------------------


def test_mediaio_make_returns_xlsxio_for_xlsx_mimetype():
    io_ = MediaIO.make(BytesIO(), MimeTypes.XLSX)
    assert isinstance(io_, XlsxIO)


# ------------------------------------------------------------------
# Built-in fallback engine (no optional deps required)
# ------------------------------------------------------------------


def test_xlsxio_fallback_write_and_read_roundtrip():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.XLSX)
    table = _make_table()

    io_.write_arrow_table(table, engine="fallback")

    assert buf.size > 0
    assert buf.to_bytes().startswith(b"PK")

    out = io_.read_arrow_table(engine="fallback")
    assert out.column_names == ["id", "name", "price"]
    assert out.to_pylist() == table.to_pylist()


def test_xlsxio_fallback_read_respects_has_header_false():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.XLSX)
    io_.write_arrow_table(_make_table(), engine="fallback", include_header=False)

    out = io_.read_arrow_table(engine="fallback", has_header=False)
    assert out.column_names == ["f0", "f1", "f2"]
    assert out.num_rows == 3


def test_xlsxio_fallback_write_custom_sheet_name():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.XLSX)
    io_.write_arrow_table(_make_table(), engine="fallback", write_sheet_name="Data")

    out = io_.read_arrow_table(engine="fallback", sheet_name="Data")
    assert out.num_rows == 3


def test_xlsxio_fallback_skip_rows():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.XLSX)
    io_.write_arrow_table(_make_table(), engine="fallback")

    # skip_rows=1 drops the header row; remaining 3 rows are data rows,
    # the first of which is promoted to the header when has_header=True.
    out = io_.read_arrow_table(engine="fallback", skip_rows=1, has_header=True)
    assert out.num_rows == 2
    assert out.column_names == ["1", "alice", "3.14"]


def test_xlsxio_fallback_empty_write_produces_valid_xlsx():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.XLSX)
    io_.write_arrow_table(pa.table({"a": pa.array([], type=pa.int64())}), engine="fallback")

    assert buf.to_bytes().startswith(b"PK")
    out = io_.read_arrow_table(engine="fallback")
    assert out.num_rows == 0


def test_xlsxio_fallback_none_values_roundtrip():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.XLSX)
    table = pa.table({"x": [1, None, 3], "y": ["a", "b", None]})

    io_.write_arrow_table(table, engine="fallback")
    out = io_.read_arrow_table(engine="fallback")

    assert out.to_pylist() == [
        {"x": 1, "y": "a"},
        {"x": None, "y": "b"},
        {"x": 3, "y": None},
    ]


# ------------------------------------------------------------------
# Cross-engine interop: write with one engine, read with another
# ------------------------------------------------------------------


@pytest.mark.skipif(
    not (HAS_POLARS and HAS_XLSXWRITER),
    reason="requires polars + xlsxwriter",
)
def test_xlsxio_polars_write_read_with_fallback():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.XLSX)
    io_.write_arrow_table(_make_table(), engine="polars")

    out = io_.read_arrow_table(engine="fallback")
    assert out.column_names == ["id", "name", "price"]
    assert out.num_rows == 3


@pytest.mark.skipif(
    not (HAS_PANDAS and HAS_OPENPYXL),
    reason="requires pandas + openpyxl",
)
def test_xlsxio_pandas_write_read_with_fallback():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.XLSX)
    io_.write_arrow_table(_make_table(), engine="pandas")

    out = io_.read_arrow_table(engine="fallback")
    assert out.column_names == ["id", "name", "price"]
    assert out.num_rows == 3


@pytest.mark.skipif(
    not (HAS_POLARS and HAS_OPENPYXL),
    reason="requires polars + openpyxl",
)
def test_xlsxio_fallback_write_read_with_polars():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.XLSX)
    io_.write_arrow_table(_make_table(), engine="fallback")

    out = io_.read_arrow_table(engine="polars")
    assert out.num_rows == 3
    assert set(out.column_names) == {"id", "name", "price"}


@pytest.mark.skipif(not (HAS_PANDAS and HAS_OPENPYXL), reason="requires pandas + openpyxl")
def test_xlsxio_fallback_write_read_with_pandas():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.XLSX)
    io_.write_arrow_table(_make_table(), engine="fallback")

    out = io_.read_arrow_table(engine="pandas")
    assert out.num_rows == 3
    assert set(out.column_names) == {"id", "name", "price"}


# ------------------------------------------------------------------
# Column projection
# ------------------------------------------------------------------


def test_xlsxio_columns_projection():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.XLSX)
    io_.write_arrow_table(_make_table(), engine="fallback")

    out = io_.read_arrow_table(engine="fallback", columns=["name", "price"])
    assert out.column_names == ["name", "price"]


# ------------------------------------------------------------------
# Empty buffer reads
# ------------------------------------------------------------------


def test_xlsxio_empty_buffer_read_returns_empty_table():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.XLSX)
    out = io_.read_arrow_table()
    assert out.num_rows == 0


# ------------------------------------------------------------------
# Media-type inference from path: BytesIO(path).media_io() should
# pick XlsxIO for *.xlsx files, not the generic ZipIO the ZIP magic
# bytes would otherwise suggest (XLSX is a ZIP container).
# ------------------------------------------------------------------


def test_bytesio_from_xlsx_path_media_io_returns_xlsxio(tmp_path: Path):
    xlsx_path = tmp_path / "forecasts_filtered.xlsx"

    # Build a real .xlsx file using the built-in fallback writer so this
    # test does not depend on any optional writer backend.
    seed = BytesIO()
    MediaIO.make(seed, MimeTypes.XLSX).write_arrow_table(
        _make_table(),
        engine="fallback",
    )
    xlsx_path.write_bytes(seed.to_bytes())

    buf = BytesIO(xlsx_path)

    assert buf.media_type.mime_type is MimeTypes.XLSX
    mio = buf.media_io()
    assert isinstance(mio, XlsxIO)

    out = mio.read_arrow_table(engine="fallback")
    assert out.column_names == ["id", "name", "price"]
    assert out.num_rows == 3


def test_bytesio_from_xlsx_path_explicit_media_override(tmp_path: Path):
    xlsx_path = tmp_path / "data.xlsx"
    seed = BytesIO()
    MediaIO.make(seed, MimeTypes.XLSX).write_arrow_table(
        _make_table(),
        engine="fallback",
    )
    xlsx_path.write_bytes(seed.to_bytes())

    # Passing MimeTypes.XLSX explicitly must also produce an XlsxIO,
    # even if automatic inference were somehow to guess differently.
    mio = BytesIO(xlsx_path).media_io(MimeTypes.XLSX)
    assert isinstance(mio, XlsxIO)
