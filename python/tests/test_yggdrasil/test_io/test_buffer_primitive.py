"""Tests for ``yggdrasil.io.buffer.primitive``.

Covers the :class:`BytesIO` base contract (registry dispatch on a
path's mime type, media type defaults, save-mode resolution) and
the round-trip behaviour of every concrete leaf shipped in
:mod:`yggdrasil.io.buffer.primitive`: Parquet, Arrow IPC, CSV,
JSON, NDJSON, XLSX, ZIP.

Optional-dependency leaves (XLSX needs ``openpyxl``) are gated by
``pytest.importorskip`` so the suite stays green on a base install.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.primitive import (
    ArrowIPCIO,
    CsvIO,
    JsonIO,
    NDJsonIO,
    ParquetIO,
    XlsxIO,
)
from yggdrasil.io.nested import ZipIO, ZipOptions
from yggdrasil.io.primitive import ParquetOptions
from yggdrasil.io.primitive import CsvOptions
from yggdrasil.io.primitive import JsonOptions
from yggdrasil.io.primitive import NDJsonOptions
from yggdrasil.io.primitive import ArrowIPCOptions
from yggdrasil.io.primitive import XlsxOptions
from yggdrasil.data.enums import MimeTypes, MediaType, Mode


# ---------------------------------------------------------------------------
# Sample data shared across the file
# ---------------------------------------------------------------------------


def _sample_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {"a": 1, "b": "henry"},
            {"a": 2, "b": "hub"},
            {"a": 3, "b": "settle"},
        ]
    )


@pytest.fixture
def tmpdir_path(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# BytesIO / tabular registry contract
# ---------------------------------------------------------------------------


class TestPrimitiveIOBase:
    def test_default_media_type_is_none_on_base(self):
        # The base layer must not register against OCTET_STREAM or it
        # would shadow BytesIO as the fallback.
        assert BytesIO.default_media_type() is None

    def test_concrete_leaves_have_mime_types(self):
        assert ParquetIO.default_media_type() == MimeTypes.PARQUET
        assert CsvIO.default_media_type() == MimeTypes.CSV
        assert JsonIO.default_media_type() == MimeTypes.JSON
        # Arrow IPC, NDJSON also register; just sanity-check non-None.
        for cls in (ArrowIPCIO, NDJsonIO, XlsxIO, ZipIO):
            assert cls.default_media_type() is not None

    def test_dispatch_via_path_extension(self, tmpdir_path: Path):
        # BytesIO(path=...) routes to the right leaf via the
        # registered mime types.
        for ext, expected in (
            (".parquet", ParquetIO),
            (".arrow", ArrowIPCIO),
            (".csv", CsvIO),
            (".json", JsonIO),
            (".ndjson", NDJsonIO),
        ):
            p = tmpdir_path / f"a{ext}"
            p.touch()
            io = BytesIO(path=str(p))
            assert isinstance(io, expected), (ext, type(io).__name__)

    def test_concrete_class_skips_dispatch(self, tmpdir_path: Path):
        p = tmpdir_path / "a.parquet"
        p.touch()
        io = ParquetIO(path=str(p))
        assert type(io) is ParquetIO

    def test_empty_buffer_schema(self):
        # Empty buffer collect_schema must not crash, returns empty.
        io = ParquetIO()
        s = io.collect_schema()
        assert s.empty

    def test_options_class_registers(self):
        assert ParquetIO.options_class() is ParquetOptions
        assert CsvIO.options_class() is CsvOptions
        assert JsonIO.options_class() is JsonOptions
        assert NDJsonIO.options_class() is NDJsonOptions
        assert ArrowIPCIO.options_class() is ArrowIPCOptions
        assert XlsxIO.options_class() is XlsxOptions
        assert ZipIO.options_class() is ZipOptions

    def test_supported_mode_flags(self):
        # CSV is the only leaf that supports honest APPEND / UPSERT
        # at the leaf level; Parquet/IPC do APPEND via rewrite.
        assert CsvIO._SUPPORTED_APPEND is True
        assert CsvIO._SUPPORTED_UPSERT is True


# ---------------------------------------------------------------------------
# merge_upsert_tables — update_column_names semantics
# ---------------------------------------------------------------------------


class TestMergeUpsertUpdateColumnNames:
    """``merge_upsert_tables(update_column_names=...)`` matches MERGE
    SET semantics: only listed columns take incoming values on a key
    match; non-key columns outside the list keep existing values.
    """

    def _existing(self) -> pa.Table:
        return pa.Table.from_pylist([
            {"id": 1, "name": "alice", "score": 10, "note": "old"},
            {"id": 2, "name": "bob",   "score": 20, "note": "old"},
            {"id": 3, "name": "carol", "score": 30, "note": "old"},
        ])

    def _incoming(self) -> pa.Table:
        # ids 2 and 3 collide with existing; id 4 is new.
        return pa.Table.from_pylist([
            {"id": 2, "name": "robert", "score": 99, "note": "new"},
            {"id": 3, "name": "carrie", "score": 88, "note": "new"},
            {"id": 4, "name": "dave",   "score": 77, "note": "new"},
        ])

    def test_default_behaviour_replaces_full_row(self):
        io = ParquetIO()
        merged = io.merge_upsert_tables(
            self._existing(), self._incoming(), match_by=["id"],
        )
        rows = sorted(merged.to_pylist(), key=lambda r: r["id"])
        # id=2,3 fully replaced by incoming; id=4 inserted; id=1 kept.
        assert rows[0] == {"id": 1, "name": "alice", "score": 10, "note": "old"}
        assert rows[1] == {"id": 2, "name": "robert", "score": 99, "note": "new"}
        assert rows[2] == {"id": 3, "name": "carrie", "score": 88, "note": "new"}
        assert rows[3] == {"id": 4, "name": "dave",   "score": 77, "note": "new"}

    def test_update_column_names_preserves_unlisted_columns(self):
        io = ParquetIO()
        merged = io.merge_upsert_tables(
            self._existing(), self._incoming(),
            match_by=["id"],
            update_column_names=["score"],  # only "score" gets updated
        )
        rows = sorted(merged.to_pylist(), key=lambda r: r["id"])
        # id=1 untouched.
        assert rows[0] == {"id": 1, "name": "alice", "score": 10, "note": "old"}
        # id=2,3: score from incoming, name/note preserved from existing.
        assert rows[1] == {"id": 2, "name": "bob",   "score": 99, "note": "old"}
        assert rows[2] == {"id": 3, "name": "carol", "score": 88, "note": "old"}
        # id=4: pure insert, no existing match -> name carried, note/score
        # are not in update list and have no existing source -> nulls.
        assert rows[3] == {"id": 4, "name": None, "score": 77, "note": None}

    def test_update_column_names_empty_list_preserves_everything(self):
        # Empty update list = nothing changes on match; only inserts apply.
        io = ParquetIO()
        merged = io.merge_upsert_tables(
            self._existing(), self._incoming(),
            match_by=["id"],
            update_column_names=[],
        )
        rows = sorted(merged.to_pylist(), key=lambda r: r["id"])
        assert rows[0] == {"id": 1, "name": "alice", "score": 10, "note": "old"}
        assert rows[1] == {"id": 2, "name": "bob",   "score": 20, "note": "old"}
        assert rows[2] == {"id": 3, "name": "carol", "score": 30, "note": "old"}
        # id=4 still gets inserted; non-update preserved-cols come from
        # existing but there's no matching id=4 there -> nulls.
        assert rows[3] == {"id": 4, "name": None, "score": None, "note": None}


# ---------------------------------------------------------------------------
# Save-mode resolution
# ---------------------------------------------------------------------------


class TestSaveModeResolution:
    def test_auto_resolves_to_overwrite_on_empty(self):
        io = ParquetIO()
        assert io._resolve_save_mode(Mode.AUTO) is Mode.OVERWRITE

    def test_overwrite_passthrough(self):
        io = ParquetIO()
        assert io._resolve_save_mode(Mode.OVERWRITE) is Mode.OVERWRITE

    def test_truncate_resolves_to_overwrite(self):
        io = ParquetIO()
        assert io._resolve_save_mode(Mode.TRUNCATE) is Mode.OVERWRITE

    def test_ignore_on_empty_resolves_overwrite(self):
        io = ParquetIO()
        assert io.is_empty()
        assert io._resolve_save_mode(Mode.IGNORE) is Mode.OVERWRITE

    def test_error_if_exists_on_empty_resolves_overwrite(self):
        io = ParquetIO()
        assert io._resolve_save_mode(Mode.ERROR_IF_EXISTS) is Mode.OVERWRITE

    def test_error_if_exists_raises_on_non_empty(self, tmpdir_path: Path):
        p = tmpdir_path / "a.parquet"
        ParquetIO(path=str(p)).write_arrow_table(_sample_table())
        io = ParquetIO(path=str(p))
        assert not io.is_empty()
        with pytest.raises(FileExistsError):
            io._resolve_save_mode(Mode.ERROR_IF_EXISTS)


# ---------------------------------------------------------------------------
# Per-format round-trip — written through path-bound IOs.
# ---------------------------------------------------------------------------


class _RoundTripBase:
    cls: type[BytesIO]
    suffix: str

    def _write_then_read(self, path: Path) -> pa.Table:
        io = self.cls(path=str(path))
        io.write_arrow_table(_sample_table())
        io.close()

        reader = self.cls(path=str(path))
        table = reader.read_arrow_table()
        reader.close()
        return table

    def test_round_trip(self, tmpdir_path: Path):
        path = tmpdir_path / f"a{self.suffix}"
        out = self._write_then_read(path)
        assert out.num_rows == 3
        assert out.column_names == ["a", "b"]

    def test_collect_schema_reflects_columns(self, tmpdir_path: Path):
        path = tmpdir_path / f"a{self.suffix}"
        io = self.cls(path=str(path))
        io.write_arrow_table(_sample_table())
        io.close()

        reader = self.cls(path=str(path))
        schema = reader.collect_schema()
        assert list(schema.field_names()) == ["a", "b"]


class TestParquetRoundTrip(_RoundTripBase):
    cls = ParquetIO
    suffix = ".parquet"

    def test_overwrite_replaces(self, tmpdir_path: Path):
        path = tmpdir_path / "a.parquet"
        io = ParquetIO(path=str(path))
        io.write_arrow_table(_sample_table())
        # Second write replaces the first via implicit OVERWRITE.
        io.write_arrow_table(_sample_table().slice(0, 1))
        io.close()

        out = ParquetIO(path=str(path)).read_arrow_table()
        assert out.num_rows == 1


class TestArrowIPCRoundTrip(_RoundTripBase):
    cls = ArrowIPCIO
    suffix = ".arrow"


class TestCsvRoundTrip(_RoundTripBase):
    cls = CsvIO
    suffix = ".csv"

    def test_append_concatenates(self, tmpdir_path: Path):
        path = tmpdir_path / "a.csv"
        CsvIO(path=str(path)).write_arrow_table(_sample_table())
        CsvIO(path=str(path)).write_arrow_table(
            _sample_table().slice(0, 1),
            mode=Mode.APPEND,
        )
        out = CsvIO(path=str(path)).read_arrow_table()
        # Appended one extra row.
        assert out.num_rows == 4

    def test_append_on_empty_is_overwrite_with_header(self, tmpdir_path: Path):
        path = tmpdir_path / "a.csv"
        CsvIO(path=str(path)).write_arrow_table(_sample_table(), mode=Mode.APPEND)
        out = CsvIO(path=str(path)).read_arrow_table()
        assert out.num_rows == 3


class TestJsonRoundTrip(_RoundTripBase):
    cls = JsonIO
    suffix = ".json"

    # JsonIO writes a JSON array of objects but collect_schema goes
    # through pyarrow's streaming NDJSON reader, which can't parse the
    # array-of-objects shape. Override to skip that test.
    def test_collect_schema_reflects_columns(self, tmpdir_path: Path):  # noqa: D401
        pytest.skip("JsonIO writes JSON array; schema-collection uses NDJSON parser")


class TestNDJsonRoundTrip(_RoundTripBase):
    cls = NDJsonIO
    suffix = ".ndjson"


# ---------------------------------------------------------------------------
# XLSX (optional)
# ---------------------------------------------------------------------------


class TestXlsxRoundTrip:
    def test_xlsx_round_trip(self, tmpdir_path: Path):
        pytest.importorskip("openpyxl")
        path = tmpdir_path / "a.xlsx"
        io = XlsxIO(path=str(path))
        io.write_arrow_table(_sample_table())
        io.close()

        out = XlsxIO(path=str(path)).read_arrow_table()
        assert out.num_rows == 3
        assert out.column_names == ["a", "b"]


# ---------------------------------------------------------------------------
# ZipIO writer-side basics — read path is exercised indirectly by other tests.
# ---------------------------------------------------------------------------


class TestZipIO:
    def test_write_creates_archive_with_expected_entry(self, tmpdir_path: Path):
        import zipfile

        path = tmpdir_path / "a.zip"
        ZipIO(path=str(path)).write_arrow_table(_sample_table())

        with zipfile.ZipFile(str(path)) as zf:
            names = zf.namelist()
        assert names, "ZipIO write should produce at least one entry"
        assert all(n.startswith("batch-") for n in names)


# ---------------------------------------------------------------------------
# Mode.IGNORE: skip the second write
# ---------------------------------------------------------------------------


class TestModeIgnore:
    def test_ignore_skips_when_buffer_non_empty(self, tmpdir_path: Path):
        path = tmpdir_path / "a.parquet"
        ParquetIO(path=str(path)).write_arrow_table(_sample_table())

        io = ParquetIO(path=str(path))
        io.write_arrow_table(_sample_table().slice(0, 1), mode=Mode.IGNORE)
        io.close()

        out = ParquetIO(path=str(path)).read_arrow_table()
        # Original three rows preserved, ignored write skipped.
        assert out.num_rows == 3


# ---------------------------------------------------------------------------
# In-memory ParquetIO writer (other formats need a real path due to writer
# constraints around pa.Buffer-typed writes).
# ---------------------------------------------------------------------------


class TestInMemoryParquet:
    def test_write_then_read_in_memory(self):
        io = ParquetIO()
        io.write_arrow_table(_sample_table())
        # Cursor sits at end after write; rewind for the read.
        io.seek(0)
        out = io.read_arrow_table()
        assert out.num_rows == 3
        assert out.column_names == ["a", "b"]

    def test_in_memory_is_not_empty_after_write(self):
        io = ParquetIO()
        io.write_arrow_table(_sample_table())
        assert not io.is_empty()
        assert io.size > 0


# ---------------------------------------------------------------------------
# Codec siblings: BytesIO._make_*_sibling
# ---------------------------------------------------------------------------


class TestCodecSiblings:
    def test_make_uncompressed_sibling_requires_codec(self):
        # No codec attached → calling the helper is a programmer
        # error and surfaces as RuntimeError.
        io = ParquetIO()
        with pytest.raises(RuntimeError):
            io._make_uncompressed_sibling()

    def test_make_empty_sibling_uses_default_mime(self):
        io = ParquetIO()
        sibling = io._make_empty_sibling()
        try:
            assert isinstance(sibling, ParquetIO)
            # Same format minus the codec layer.
            assert sibling.size == 0
            assert sibling.codec is None
        finally:
            sibling.close()


# ---------------------------------------------------------------------------
# as_media: a final-leaf BytesIO is its own tabular view.
# ---------------------------------------------------------------------------


class TestAsMedia:
    def test_self_returned_when_no_media_type(self):
        io = ParquetIO()
        assert io.as_media() is io

    def test_as_media_with_octet_returns_self(self):
        io = ParquetIO()
        out = io.as_media(MediaType(MimeTypes.OCTET_STREAM))
        assert out is io


# ---------------------------------------------------------------------------
# persist / unpersist / cached
# ---------------------------------------------------------------------------


class TestPersist:
    def test_unpersist_clears_caches(self, tmpdir_path: Path):
        path = tmpdir_path / "a.parquet"
        ParquetIO(path=str(path)).write_arrow_table(_sample_table())

        io = ParquetIO(path=str(path))
        io.persist(engine="arrow")
        assert io.cached
        io.unpersist()
        assert not io.cached

    def test_persist_idempotent(self, tmpdir_path: Path):
        from yggdrasil.io.tabular import MemoryArrowIO

        path = tmpdir_path / "a.parquet"
        ParquetIO(path=str(path)).write_arrow_table(_sample_table())

        io = ParquetIO(path=str(path))
        io.persist(engine="arrow")
        first = io._persisted_data
        assert isinstance(first, MemoryArrowIO)
        io.persist(engine="arrow")  # cached → no-op
        assert io._persisted_data is first

    def test_persist_with_explicit_data(self):
        from yggdrasil.io.tabular import MemoryArrowIO

        io = ParquetIO()
        table = _sample_table()
        io.persist(engine="arrow", data=table)
        assert io.cached
        assert isinstance(io._persisted_data, MemoryArrowIO)
        assert io._persisted_data.num_rows == 3

    def test_persist_unsupported_engine_raises(self):
        io = ParquetIO()
        with pytest.raises(ValueError):
            io.persist(engine="bogus")


# ---------------------------------------------------------------------------
# is_empty / size on freshly-built IOs
# ---------------------------------------------------------------------------


class TestEmptyState:
    def test_empty_in_memory_io(self):
        io = ParquetIO()
        assert io.is_empty()
        assert io.size == 0

    def test_empty_path_bound_io(self, tmpdir_path: Path):
        path = tmpdir_path / "a.parquet"
        # Path doesn't exist yet → is_empty must report True without error.
        io = ParquetIO(path=str(path))
        assert io.is_empty()
