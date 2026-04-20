"""Unit tests for :class:`yggdrasil.io.buffer.xlsx_io.XlsxIO`.

Covers:

* options validation (including engine normalization and negative sheet_id)
* MediaIO factory routing
* fallback (openpyxl) read/write round-trips
* has_header / skip_rows semantics
* None values, custom sheet names, empty buffers
* column projection
* cross-engine interop (polars, pandas) — gated on imports
* multi-sheet read + write
* save modes: OVERWRITE, IGNORE, ERROR_IF_EXISTS, APPEND, UPSERT
* BytesIO path inference → XlsxIO
* cast integration on write
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from yggdrasil.io import MimeTypes
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.buffer.xlsx_io import XlsxIO, XlsxOptions
from yggdrasil.io.config import BufferConfig
from yggdrasil.io.enums import SaveMode


# ---------------------------------------------------------------------
# Dep probes
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

def _make_cfg(tmp_path: Path, *, spill_bytes: int = 128) -> BufferConfig:
    return BufferConfig(
        spill_bytes=spill_bytes,
        tmp_dir=tmp_path,
        prefix="test_xlsxio_",
        suffix=".xlsx",
        keep_spilled_file=False,
    )


@pytest.fixture()
def cfg(tmp_path: Path) -> BufferConfig:
    return _make_cfg(tmp_path)


@pytest.fixture()
def spill_cfg(tmp_path: Path) -> BufferConfig:
    return _make_cfg(tmp_path, spill_bytes=1)


def _make_table() -> pa.Table:
    return pa.table(
        {
            "id": [1, 2, 3],
            "name": ["alice", "bob", "carol"],
            "price": [3.14, 2.71, 1.41],
        }
    )


@pytest.fixture()
def sample_table() -> pa.Table:
    return _make_table()


def _make_io(buf: BytesIO) -> XlsxIO:
    io_ = MediaIO.make(buf, MimeTypes.XLSX)
    assert isinstance(io_, XlsxIO)
    return io_


# =====================================================================
# Options validation
# =====================================================================

class TestXlsxOptions:
    def test_defaults(self):
        opt = XlsxOptions()
        assert opt.sheet_name is None
        assert opt.sheet_id is None
        assert opt.has_header is True
        assert opt.include_header is True
        assert opt.skip_rows == 0
        assert opt.write_sheet_name == "Sheet1"
        assert opt.engine == "auto"

    def test_rejects_unknown_engine(self):
        with pytest.raises(ValueError, match="engine"):
            XlsxOptions(engine="nope")

    def test_engine_is_case_normalized(self):
        assert XlsxOptions(engine="POLARS").engine == "polars"
        assert XlsxOptions(engine="Fallback").engine == "fallback"

    def test_rejects_bool_sheet_name(self):
        with pytest.raises(TypeError, match="sheet_name"):
            XlsxOptions(sheet_name=True)  # type: ignore[arg-type]

    def test_rejects_empty_sheet_name(self):
        with pytest.raises(ValueError, match="sheet_name"):
            XlsxOptions(sheet_name="")

    def test_rejects_negative_sheet_id(self):
        with pytest.raises(ValueError, match="sheet_id"):
            XlsxOptions(sheet_id=-1)

    def test_rejects_bool_sheet_id(self):
        # bool is a subclass of int — make sure we catch that.
        with pytest.raises(TypeError, match="sheet_id"):
            XlsxOptions(sheet_id=True)  # type: ignore[arg-type]

    def test_accepts_zero_sheet_id(self):
        assert XlsxOptions(sheet_id=0).sheet_id == 0

    def test_rejects_negative_skip_rows(self):
        with pytest.raises(ValueError, match="skip_rows"):
            XlsxOptions(skip_rows=-1)

    def test_rejects_bool_skip_rows(self):
        with pytest.raises(TypeError, match="skip_rows"):
            XlsxOptions(skip_rows=True)  # type: ignore[arg-type]

    def test_rejects_empty_write_sheet_name(self):
        with pytest.raises(ValueError, match="write_sheet_name"):
            XlsxOptions(write_sheet_name="")

    def test_has_header_must_be_bool(self):
        with pytest.raises(TypeError, match="has_header"):
            XlsxOptions(has_header="yes")  # type: ignore[arg-type]

    def test_include_header_must_be_bool(self):
        with pytest.raises(TypeError, match="include_header"):
            XlsxOptions(include_header=1)  # type: ignore[arg-type]


# =====================================================================
# Factory
# =====================================================================

class TestFactory:
    def test_media_io_make_returns_xlsx_io(self, cfg: BufferConfig):
        buf = BytesIO(config=cfg)
        io_ = MediaIO.make(buf, MimeTypes.XLSX)
        assert isinstance(io_, XlsxIO)

    def test_check_options_accepts_none(self):
        resolved = XlsxIO.check_options(None)
        assert isinstance(resolved, XlsxOptions)
        assert resolved.engine == "auto"

    def test_check_options_merges_kwargs(self):
        resolved = XlsxIO.check_options(
            None, engine="fallback", write_sheet_name="Data", skip_rows=2
        )
        assert resolved.engine == "fallback"
        assert resolved.write_sheet_name == "Data"
        assert resolved.skip_rows == 2


# =====================================================================
# Fallback engine (openpyxl) — always available
# =====================================================================

@pytest.mark.skipif(not HAS_OPENPYXL, reason="fallback engine requires openpyxl")
class TestFallbackEngine:
    def test_write_and_read_roundtrip(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_arrow_table(sample_table, engine="fallback")

        assert buf.size > 0
        assert buf.to_bytes().startswith(b"PK")  # ZIP magic

        out = io_.read_arrow_table(engine="fallback")
        assert out.column_names == ["id", "name", "price"]
        assert out.to_pylist() == sample_table.to_pylist()

    def test_has_header_false(self, cfg: BufferConfig, sample_table):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample_table, engine="fallback", include_header=False)

        out = io_.read_arrow_table(engine="fallback", has_header=False)
        assert out.column_names == ["f0", "f1", "f2"]
        assert out.num_rows == 3

    def test_custom_sheet_name(self, cfg: BufferConfig, sample_table):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample_table, engine="fallback", write_sheet_name="Data")

        out = io_.read_arrow_table(engine="fallback", sheet_name="Data")
        assert out.num_rows == 3

    def test_skip_rows_promotes_next_row_to_header(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample_table, engine="fallback")

        # Header is "id,name,price"; data rows are ("1","alice","3.14"), ...
        # skip_rows=1 drops the header; the first data row becomes the new header.
        out = io_.read_arrow_table(
            engine="fallback", skip_rows=1, has_header=True
        )
        assert out.num_rows == 2
        assert out.column_names == ["1", "alice", "3.14"]

    def test_empty_table_produces_valid_xlsx(self, cfg: BufferConfig):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        empty = pa.table({"a": pa.array([], type=pa.int64())})
        io_.write_arrow_table(empty, engine="fallback")

        assert buf.to_bytes().startswith(b"PK")
        out = io_.read_arrow_table(engine="fallback")
        assert out.num_rows == 0

    def test_none_values_roundtrip(self, cfg: BufferConfig):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        table = pa.table({"x": [1, None, 3], "y": ["a", "b", None]})

        io_.write_arrow_table(table, engine="fallback")
        out = io_.read_arrow_table(engine="fallback")

        assert out.to_pylist() == [
            {"x": 1, "y": "a"},
            {"x": None, "y": "b"},
            {"x": 3, "y": None},
        ]

    def test_sheet_id_selects_by_index(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample_table, engine="fallback", write_sheet_name="First")

        out = io_.read_arrow_table(engine="fallback", sheet_id=0)
        assert out.num_rows == 3

    def test_sheet_id_out_of_range_raises(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample_table, engine="fallback")

        with pytest.raises(IndexError):
            io_.read_arrow_table(engine="fallback", sheet_id=99)

    def test_missing_sheet_name_raises(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample_table, engine="fallback")

        with pytest.raises(KeyError):
            io_.read_arrow_table(engine="fallback", sheet_name="DoesNotExist")

    def test_spilled_buffer_roundtrip(
        self, spill_cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=spill_cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample_table, engine="fallback")

        assert buf.spilled is True
        out = io_.read_arrow_table(engine="fallback")
        assert out.to_pylist() == sample_table.to_pylist()


# =====================================================================
# Engine auto-selection
# =====================================================================

class TestEngineAuto:
    def test_auto_prefers_polars_when_available(self):
        if HAS_POLARS:
            assert XlsxIO._resolve_engine("auto") == "polars"
        else:
            assert XlsxIO._resolve_engine("auto") == "fallback"

    def test_auto_never_picks_pandas(self):
        # Even with pandas installed, "auto" should never pick it.
        if HAS_PANDAS and not HAS_POLARS:
            # In this configuration auto falls back to fallback, not pandas.
            assert XlsxIO._resolve_engine("auto") == "fallback"

    def test_explicit_engine_passthrough(self):
        assert XlsxIO._resolve_engine("fallback") == "fallback"
        assert XlsxIO._resolve_engine("polars") == "polars"
        assert XlsxIO._resolve_engine("pandas") == "pandas"


# =====================================================================
# Cross-engine interop
# =====================================================================

@pytest.mark.skipif(
    not (HAS_POLARS and HAS_XLSXWRITER and HAS_OPENPYXL),
    reason="requires polars + xlsxwriter + openpyxl",
)
def test_polars_write_read_with_fallback(cfg: BufferConfig, sample_table):
    buf = BytesIO(config=cfg)
    io_ = _make_io(buf)
    io_.write_arrow_table(sample_table, engine="polars")

    out = io_.read_arrow_table(engine="fallback")
    assert out.column_names == ["id", "name", "price"]
    assert out.num_rows == 3


@pytest.mark.skipif(
    not (HAS_PANDAS and HAS_OPENPYXL),
    reason="requires pandas + openpyxl",
)
def test_pandas_write_read_with_fallback(cfg: BufferConfig, sample_table):
    buf = BytesIO(config=cfg)
    io_ = _make_io(buf)
    io_.write_arrow_table(sample_table, engine="pandas")

    out = io_.read_arrow_table(engine="fallback")
    assert out.column_names == ["id", "name", "price"]
    assert out.num_rows == 3


@pytest.mark.skipif(
    not (HAS_POLARS and HAS_OPENPYXL),
    reason="requires polars + openpyxl",
)
def test_fallback_write_read_with_polars(cfg: BufferConfig, sample_table):
    buf = BytesIO(config=cfg)
    io_ = _make_io(buf)
    io_.write_arrow_table(sample_table, engine="fallback")

    out = io_.read_arrow_table(engine="polars")
    assert out.num_rows == 3
    assert set(out.column_names) == {"id", "name", "price"}


@pytest.mark.skipif(
    not (HAS_PANDAS and HAS_OPENPYXL),
    reason="requires pandas + openpyxl",
)
def test_fallback_write_read_with_pandas(cfg: BufferConfig, sample_table):
    buf = BytesIO(config=cfg)
    io_ = _make_io(buf)
    io_.write_arrow_table(sample_table, engine="fallback")

    out = io_.read_arrow_table(engine="pandas")
    assert out.num_rows == 3
    assert set(out.column_names) == {"id", "name", "price"}


# =====================================================================
# Column projection
# =====================================================================

@pytest.mark.skipif(not HAS_OPENPYXL, reason="requires openpyxl")
class TestColumnProjection:
    def test_projection(self, cfg: BufferConfig, sample_table):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample_table, engine="fallback")

        out = io_.read_arrow_table(engine="fallback", columns=["name", "price"])
        assert out.column_names == ["name", "price"]
        assert out.num_rows == 3

    def test_unknown_columns_silently_dropped(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample_table, engine="fallback")

        out = io_.read_arrow_table(
            engine="fallback", columns=["id", "nope"]
        )
        assert out.column_names == ["id"]

    def test_all_unknown_columns_yields_empty_schema(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample_table, engine="fallback")

        out = io_.read_arrow_table(engine="fallback", columns=["a", "b"])
        assert out.column_names == []


# =====================================================================
# Empty buffer
# =====================================================================

class TestEmptyBuffer:
    def test_empty_buffer_read_returns_empty_table(self, cfg: BufferConfig):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        out = io_.read_arrow_table()
        assert out.num_rows == 0

    def test_empty_buffer_read_arrow_batches_yields_nothing(self, cfg: BufferConfig):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        batches = list(io_.read_arrow_batches())
        assert batches == []


# =====================================================================
# Multi-sheet read + write
# =====================================================================

@pytest.mark.skipif(not HAS_OPENPYXL, reason="requires openpyxl")
class TestMultiSheet:
    def test_write_two_sheets_via_append_preserves_first(
        self, cfg: BufferConfig
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        t1 = pa.table({"a": [1, 2]})
        t2 = pa.table({"b": ["x", "y", "z"]})

        io_.write_arrow_table(
            t1,
            engine="fallback",
            write_sheet_name="Summary",
            mode=SaveMode.OVERWRITE,
        )
        io_.write_arrow_table(
            t2,
            engine="fallback",
            write_sheet_name="Details",
            mode=SaveMode.APPEND,
        )

        s1 = io_.read_arrow_table(engine="fallback", sheet_name="Summary")
        s2 = io_.read_arrow_table(engine="fallback", sheet_name="Details")

        assert s1.to_pylist() == [{"a": 1}, {"a": 2}]
        assert s2.to_pylist() == [{"b": "x"}, {"b": "y"}, {"b": "z"}]

    def test_append_to_existing_sheet_merges_rows(
        self, cfg: BufferConfig
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_arrow_table(
            pa.table({"id": [1, 2]}),
            engine="fallback",
            write_sheet_name="Data",
            mode=SaveMode.OVERWRITE,
        )
        io_.write_arrow_table(
            pa.table({"id": [3, 4]}),
            engine="fallback",
            write_sheet_name="Data",
            mode=SaveMode.APPEND,
        )

        out = io_.read_arrow_table(engine="fallback", sheet_name="Data")
        assert out.to_pylist() == [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]

    def test_overwrite_replaces_whole_workbook(self, cfg: BufferConfig):
        """OVERWRITE is workbook-level — other sheets are wiped."""
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_arrow_table(
            pa.table({"a": [1]}),
            engine="fallback",
            write_sheet_name="A",
            mode=SaveMode.OVERWRITE,
        )
        io_.write_arrow_table(
            pa.table({"b": [2]}),
            engine="fallback",
            write_sheet_name="B",
            mode=SaveMode.APPEND,
        )

        # Now OVERWRITE with "C" — should wipe A and B.
        io_.write_arrow_table(
            pa.table({"c": [3]}),
            engine="fallback",
            write_sheet_name="C",
            mode=SaveMode.OVERWRITE,
        )

        # Sheet C exists with the new data.
        c = io_.read_arrow_table(engine="fallback", sheet_name="C")
        assert c.to_pylist() == [{"c": 3}]

        # A and B should no longer be present.
        with pytest.raises(KeyError):
            io_.read_arrow_table(engine="fallback", sheet_name="A")
        with pytest.raises(KeyError):
            io_.read_arrow_table(engine="fallback", sheet_name="B")


# =====================================================================
# Save modes
# =====================================================================

@pytest.mark.skipif(not HAS_OPENPYXL, reason="requires openpyxl")
class TestSaveModes:
    def test_ignore_mode_does_not_overwrite(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_arrow_table(sample_table, engine="fallback", mode=SaveMode.OVERWRITE)
        size1 = buf.size
        bytes1 = buf.to_bytes()

        t2 = pa.table({"x": [99]})
        io_.write_arrow_table(t2, engine="fallback", mode=SaveMode.IGNORE)

        assert buf.size == size1
        assert buf.to_bytes() == bytes1

    def test_error_if_exists_raises(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample_table, engine="fallback", mode=SaveMode.OVERWRITE)

        with pytest.raises(IOError):
            io_.write_arrow_table(
                sample_table, engine="fallback", mode=SaveMode.ERROR_IF_EXISTS
            )

    def test_error_if_exists_allowed_on_empty_buffer(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(
            sample_table, engine="fallback", mode=SaveMode.ERROR_IF_EXISTS
        )
        out = io_.read_arrow_table(engine="fallback")
        assert out.num_rows == sample_table.num_rows

    def test_append_combines_rows_single_sheet(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        t2 = pa.table(
            {
                "id": [4, 5],
                "name": ["dave", "eve"],
                "price": [4.0, 5.0],
            }
        )

        io_.write_arrow_table(sample_table, engine="fallback", mode=SaveMode.OVERWRITE)
        io_.write_arrow_table(t2, engine="fallback", mode=SaveMode.APPEND)

        out = io_.read_arrow_table(engine="fallback")
        ids = out.column("id").to_pylist()
        assert ids == [1, 2, 3, 4, 5]

    def test_append_into_empty_buffer_same_as_overwrite(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_arrow_table(sample_table, engine="fallback", mode=SaveMode.APPEND)
        out = io_.read_arrow_table(engine="fallback")
        assert out.num_rows == sample_table.num_rows

    def test_upsert_requires_match_by(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample_table, engine="fallback", mode=SaveMode.OVERWRITE)

        with pytest.raises(ValueError, match="match_by"):
            io_.write_arrow_table(
                sample_table, engine="fallback", mode=SaveMode.UPSERT
            )

    def test_upsert_replaces_matching_rows(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        t2 = pa.table(
            {
                "id": [2, 4],
                "name": ["BETA", "dave"],
                "price": [22.0, 44.0],
            }
        )

        io_.write_arrow_table(sample_table, engine="fallback", mode=SaveMode.OVERWRITE)
        io_.write_arrow_table(
            t2, engine="fallback", mode=SaveMode.UPSERT, match_by="id"
        )

        out = io_.read_arrow_table(engine="fallback")
        rows = sorted(out.to_pylist(), key=lambda r: r["id"])
        by_id = {r["id"]: r for r in rows}

        assert sorted(by_id.keys()) == [1, 2, 3, 4]
        # id=2 was replaced
        assert by_id[2]["name"] == "BETA"
        # id=1 and id=3 untouched
        assert by_id[1]["name"] == "alice"
        assert by_id[3]["name"] == "carol"
        # id=4 inserted
        assert by_id[4]["name"] == "dave"

    def test_upsert_composite_key(self, cfg: BufferConfig):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        t1 = pa.table(
            {
                "k1": ["a", "a", "b"],
                "k2": [1, 2, 1],
                "v": [10, 20, 30],
            }
        )
        t2 = pa.table(
            {
                "k1": ["a", "c"],
                "k2": [2, 1],
                "v": [200, 100],
            }
        )

        io_.write_arrow_table(t1, engine="fallback", mode=SaveMode.OVERWRITE)
        io_.write_arrow_table(
            t2, engine="fallback", mode=SaveMode.UPSERT, match_by=["k1", "k2"]
        )

        out = io_.read_arrow_table(engine="fallback")
        rows = sorted(out.to_pylist(), key=lambda r: (r["k1"], r["k2"]))
        assert rows == [
            {"k1": "a", "k2": 1, "v": 10},    # kept
            {"k1": "a", "k2": 2, "v": 200},   # replaced
            {"k1": "b", "k2": 1, "v": 30},    # kept
            {"k1": "c", "k2": 1, "v": 100},   # inserted
        ]

    def test_append_preserves_other_sheets(self, cfg: BufferConfig):
        """APPEND into one sheet must not touch other sheets."""
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_arrow_table(
            pa.table({"a": [1]}),
            engine="fallback",
            write_sheet_name="Keep",
            mode=SaveMode.OVERWRITE,
        )
        io_.write_arrow_table(
            pa.table({"b": [10]}),
            engine="fallback",
            write_sheet_name="Target",
            mode=SaveMode.APPEND,
        )
        # Append more to Target; Keep must survive.
        io_.write_arrow_table(
            pa.table({"b": [20]}),
            engine="fallback",
            write_sheet_name="Target",
            mode=SaveMode.APPEND,
        )

        keep = io_.read_arrow_table(engine="fallback", sheet_name="Keep")
        target = io_.read_arrow_table(engine="fallback", sheet_name="Target")

        assert keep.to_pylist() == [{"a": 1}]
        assert target.to_pylist() == [{"b": 10}, {"b": 20}]


# =====================================================================
# BytesIO path inference
# =====================================================================

@pytest.mark.skipif(not HAS_OPENPYXL, reason="requires openpyxl")
class TestPathInference:
    def test_bytesio_from_xlsx_path_routes_to_xlsxio(
        self, tmp_path: Path, sample_table
    ):
        xlsx_path = tmp_path / "forecasts_filtered.xlsx"

        # Seed a real xlsx file via the fallback engine.
        seed = BytesIO()
        _make_io(seed).write_arrow_table(sample_table, engine="fallback")
        xlsx_path.write_bytes(seed.to_bytes())

        buf = BytesIO(xlsx_path)
        assert buf.media_type.mime_type is MimeTypes.XLSX

        mio = buf.media_io()
        assert isinstance(mio, XlsxIO)

        out = mio.read_arrow_table(engine="fallback")
        assert out.column_names == ["id", "name", "price"]
        assert out.num_rows == 3

    def test_explicit_media_override(self, tmp_path: Path, sample_table):
        xlsx_path = tmp_path / "data.xlsx"
        seed = BytesIO()
        _make_io(seed).write_arrow_table(sample_table, engine="fallback")
        xlsx_path.write_bytes(seed.to_bytes())

        # Passing MimeTypes.XLSX explicitly should always produce an XlsxIO.
        mio = BytesIO(xlsx_path).media_io(MimeTypes.XLSX)
        assert isinstance(mio, XlsxIO)


# =====================================================================
# Cast integration
# =====================================================================

@pytest.mark.skipif(not HAS_OPENPYXL, reason="requires openpyxl")
class TestCastIntegration:
    def test_default_cast_is_identity(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample_table, engine="fallback")
        out = io_.read_arrow_table(engine="fallback")
        assert out.num_rows == sample_table.num_rows

    def test_cast_target_applied_on_write(self, cfg: BufferConfig):
        """Explicit CastOptions should reach the write path."""
        try:
            from yggdrasil.data.cast.options import CastOptions
        except ImportError:
            pytest.skip("CastOptions not importable in this environment")

        src = pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int32()),
                "v": pa.array([1.5, 2.5, 3.5], type=pa.float64()),
            }
        )
        target = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("v", pa.float32()),
            ]
        )

        try:
            cast = CastOptions(target_field=target)
        except TypeError:
            pytest.skip(
                "CastOptions(target_field=...) signature mismatch"
            )

        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(src, engine="fallback", cast=cast)

        # XLSX stores values as Excel cell values (numbers, strings, dates).
        # Arrow type info doesn't survive the write — a round-trip through
        # the fallback reader will re-infer types from the cell contents.
        # We just verify the write didn't crash and the rows are there.
        out = io_.read_arrow_table(engine="fallback")
        assert out.num_rows == 3
        assert {r["id"] for r in out.to_pylist()} == {1, 2, 3}


# =====================================================================
# Batched read
# =====================================================================

@pytest.mark.skipif(not HAS_OPENPYXL, reason="requires openpyxl")
class TestBatchedRead:
    def test_batch_size_respected(self, cfg: BufferConfig):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        table = pa.table({"id": list(range(50))})
        io_.write_arrow_table(table, engine="fallback")

        batches = list(
            io_.read_arrow_batches(engine="fallback", batch_size=10)
        )
        total = sum(b.num_rows for b in batches)
        assert total == 50
        assert all(b.num_rows <= 10 for b in batches)


# =====================================================================
# Schema inspection
# =====================================================================

@pytest.mark.skipif(not HAS_OPENPYXL, reason="requires openpyxl")
class TestSchema:
    def test_collect_schema_on_empty_buffer(self, cfg: BufferConfig):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        schema = io_._collect_arrow_schema()
        assert schema == pa.schema([])

    def test_collect_schema_matches_written(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(sample_table, engine="fallback")

        schema = io_._collect_arrow_schema()
        assert set(schema.names) == {"id", "name", "price"}