"""Behavior tests for :class:`yggdrasil.io.nested.zip_io.ZipIO`.

Pins the new contract:

* :class:`ZipEntryIO` is lazy — iterating children does NOT
  decompress every entry; only entries actually read get
  materialized.
* :meth:`ZipIO.list_entries` is a directory walk, no decompression.
* :meth:`ZipIO.child` returns a lazy handle.
* Tabular hooks dispatch on the entry name's extension.
* Round-trip through :meth:`write_arrow_table` packs into the entry
  named by ``ZipOptions.entry_name`` using the matching format leaf.
* APPEND keeps non-conflicting survivors; OVERWRITE replaces.
"""
from __future__ import annotations

import zipfile

import pyarrow as pa
import pytest

from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.nested.zip_io import ZipEntryIO, ZipIO, ZipOptions
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.csv_io import CsvFile
from yggdrasil.io.primitive.parquet_io import ParquetFile
from yggdrasil.io.tabular import Tabular


@pytest.fixture
def table() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})


class TestRegistration:

    def test_mime_type_is_zip(self) -> None:
        assert ZipIO.mime_type is MimeTypes.ZIP

    def test_registry(self) -> None:
        assert Tabular.class_for_media_type(MimeTypes.ZIP) is ZipIO


class TestDirectoryWalk:

    def test_list_entries_does_not_materialize(self) -> None:
        z = ZipIO()
        z.write_entries([
            ("a.txt", b"hello"),
            ("b.bin", b"\x00\x01"),
            ("c/", b""),  # directory entry — should be filtered
        ])
        # Directory-style entries are stripped automatically.
        assert z.list_entries() == ["a.txt", "b.bin"]

    def test_iter_children_yields_lazy_handles(self) -> None:
        z = ZipIO()
        z.write_entries([("a.txt", b"hello"), ("b.txt", b"world")])
        children = list(z.iter_children())
        assert all(isinstance(c, ZipEntryIO) for c in children)
        # Entries are lazy at construction time.
        assert all(not c._materialized for c in children)

    def test_child_lookup_returns_lazy(self) -> None:
        z = ZipIO()
        z.write_entries([("a.txt", b"hello")])
        ch = z.child("a.txt")
        assert isinstance(ch, ZipEntryIO)
        assert not ch._materialized
        # Size hint is taken from the directory without reading.
        assert ch.size == 5
        assert not ch._materialized
        # Reading triggers materialization.
        assert ch.to_bytes() == b"hello"
        assert ch._materialized

    def test_child_missing_raises(self) -> None:
        z = ZipIO()
        z.write_entries([("a.txt", b"x")])
        with pytest.raises(KeyError, match="No entry"):
            z.child("nope.txt")


class TestTabularDispatch:

    def test_round_trip_via_write_arrow_table(self, table) -> None:
        z = ZipIO()
        z.write_arrow_table(table)
        assert z.list_entries() == ["data.parquet"]
        assert z.read_arrow_table().equals(table)

    def test_explicit_entry_name_csv(self, table) -> None:
        z = ZipIO()
        z.write_arrow_table(table, options=ZipOptions(entry_name="trades.csv"))
        assert z.list_entries() == ["trades.csv"]
        assert z.read_arrow_table().column("id").to_pylist() == [1, 2, 3]

    def test_unknown_entry_name_extension_raises(self, table) -> None:
        z = ZipIO()
        with pytest.raises(ValueError, match="known tabular MediaType"):
            z.write_arrow_table(table, options=ZipOptions(entry_name="data.qqq"))

    def test_skip_non_tabular_entries_in_aggregate_read(self) -> None:
        # Mix a tabular entry with a non-tabular one; the aggregate
        # read filters out the latter automatically.
        csv = CsvFile()
        csv.write_arrow_table(pa.table({"id": [1, 2]}))
        z = ZipIO()
        z.write_entries([
            ("trades.csv", csv.to_bytes()),
            ("readme.txt", b"hello"),
        ])
        out = z.read_arrow_table()
        assert out.column("id").to_pylist() == [1, 2]


class TestEmpty:

    def test_empty_zip_yields_no_entries(self) -> None:
        z = ZipIO()
        assert z.list_entries() == []
        assert list(z.iter_children()) == []

    def test_empty_zip_read_arrow_returns_empty(self) -> None:
        assert list(ZipIO()._read_arrow_batches(ZipOptions())) == []


class TestModes:

    def test_overwrite_replaces_entries(self, table) -> None:
        z = ZipIO()
        z.write_arrow_table(table)
        smaller = pa.table({"id": [9], "name": ["z"]})
        z.write_arrow_table(smaller, options=ZipOptions(mode=Mode.OVERWRITE))
        assert z.list_entries() == ["data.parquet"]
        assert z.read_arrow_table().equals(smaller)

    def test_append_adds_survivor_entries(self, table) -> None:
        z = ZipIO()
        z.write_arrow_table(
            table, options=ZipOptions(entry_name="first.parquet"),
        )
        z.write_arrow_table(
            table, options=ZipOptions(entry_name="second.parquet", mode=Mode.APPEND),
        )
        assert sorted(z.list_entries()) == ["first.parquet", "second.parquet"]

    def test_append_replaces_same_name(self, table) -> None:
        z = ZipIO()
        z.write_arrow_table(table, options=ZipOptions(entry_name="x.parquet"))
        smaller = pa.table({"id": [9], "name": ["z"]})
        # APPEND with an entry name that already exists: drop the old
        # one, write the new one. (There's no honest "two entries with
        # the same name" zip semantics.)
        z.write_arrow_table(
            smaller, options=ZipOptions(entry_name="x.parquet", mode=Mode.APPEND),
        )
        assert z.list_entries() == ["x.parquet"]
        assert z.read_arrow_table().equals(smaller)

    def test_ignore_skips_when_non_empty(self, table) -> None:
        z = ZipIO()
        z.write_arrow_table(table)
        before = z.size
        z.write_arrow_batches(
            table.to_batches(), options=ZipOptions(mode=Mode.IGNORE),
        )
        assert z.size == before

    def test_error_if_exists(self, table) -> None:
        z = ZipIO()
        z.write_arrow_table(table)
        with pytest.raises(FileExistsError):
            z.write_arrow_batches(
                table.to_batches(), options=ZipOptions(mode=Mode.ERROR_IF_EXISTS),
            )


class TestExternalReader:

    def test_stdlib_zipfile_reads_what_we_wrote(self, tmp_path, table) -> None:
        target = LocalPath(str(tmp_path / "bundle.zip"))
        z = ZipIO(holder=target, owns_holder=False)
        z.write_arrow_table(table, options=ZipOptions(entry_name="data.parquet"))
        # Vanilla zipfile reads the result.
        with zipfile.ZipFile(target.os_path) as zf:
            assert zf.namelist() == ["data.parquet"]
            payload = zf.read("data.parquet")
        # The payload is a real parquet file readable directly.
        leaf = ParquetFile(payload)
        assert leaf.read_arrow_table().equals(table)

    def test_external_writer_into_path_open(self, tmp_path) -> None:
        target = LocalPath(str(tmp_path / "bundle.zip"))
        with target.open("wb") as bio:
            with zipfile.ZipFile(bio, "w") as zf:
                zf.writestr("readme.txt", "hi")

        # ZipIO can read what stdlib zipfile wrote.
        z = ZipIO(holder=target, owns_holder=False)
        assert z.list_entries() == ["readme.txt"]
        ch = z.child("readme.txt")
        assert ch.to_bytes() == b"hi"


class TestLazinessIsHonest:
    """The whole point of ZipEntryIO: don't decompress until asked."""

    def test_iter_children_does_not_decompress(self) -> None:
        z = ZipIO()
        z.write_entries([
            (f"part-{i}.txt", b"x" * 1024) for i in range(8)
        ])
        # Walking the directory is one infolist call. No materialization.
        children = list(z.iter_children())
        assert len(children) == 8
        assert all(not c._materialized for c in children)

    def test_only_touched_children_materialize(self) -> None:
        z = ZipIO()
        z.write_entries([(f"p-{i}.txt", b"abc") for i in range(4)])
        children = list(z.iter_children())
        # Read just one.
        children[2].to_bytes()
        materialized = [c._materialized for c in children]
        assert materialized == [False, False, True, False]
