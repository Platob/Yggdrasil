"""Behavior tests for :class:`yggdrasil.io.nested.zip_file.ZipFile`.

Pins the new contract:

* :class:`ZipEntryFile` is lazy — iterating children does NOT
  decompress every entry; only entries actually read get
  materialized.
* :meth:`ZipFile.list_entries` is a directory walk, no decompression.
* :meth:`ZipFile.child` returns a lazy handle.
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
from yggdrasil.io.nested.zip_file import ZipEntryFile, ZipFile, ZipOptions
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.csv_file import CSVFile
from yggdrasil.io.primitive.parquet_file import ParquetFile
from yggdrasil.io.holder import Holder
from yggdrasil.io.tabular import Tabular


@pytest.fixture
def table() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})


class TestRegistration:

    def test_mime_type_is_zip(self) -> None:
        assert ZipFile.mime_type is MimeTypes.ZIP

    def test_registry(self) -> None:
        assert Holder.class_for_media_type(MimeTypes.ZIP) is ZipFile

    def test_registry_lazy_bootstrap_without_nested_import(self) -> None:
        # Caller never touches ``yggdrasil.io.nested`` — the registry
        # still resolves ``application/zip`` to :class:`ZipFile` because
        # :meth:`Holder.class_for_media_type` self-bootstraps every
        # leaf package on a miss. Regression for the response-body
        # dispatch failure where ``Response.open(mode="rb")`` over an
        # ``application/zip`` body fell back to a plain :class:`IO`
        # and raised ``NotImplementedError`` from ``read_arrow_batches``.
        import subprocess
        import sys
        import textwrap
        script = textwrap.dedent(
            """
            import sys
            mods = [m for m in sys.modules if m.startswith('yggdrasil.io.nested')]
            assert not mods, f'unexpected pre-loaded nested modules: {mods}'
            from yggdrasil.io.holder import Holder
            target = Holder.class_for_media_type('application/zip')
            print(target.__module__ + '.' + target.__name__)
            """
        )
        out = subprocess.check_output(
            [sys.executable, "-c", script], text=True,
        ).strip()
        assert out == "yggdrasil.io.nested.zip_file.ZipFile"


class TestDirectoryWalk:

    def test_list_entries_does_not_materialize(self) -> None:
        z = ZipFile()
        z.write_entries([
            ("a.txt", b"hello"),
            ("b.bin", b"\x00\x01"),
            ("c/", b""),  # directory entry — should be filtered
        ])
        # Directory-style entries are stripped automatically.
        assert z.list_entries() == ["a.txt", "b.bin"]

    def test_iter_children_yields_lazy_handles(self) -> None:
        z = ZipFile()
        z.write_entries([("a.txt", b"hello"), ("b.txt", b"world")])
        children = list(z.iter_children())
        assert all(isinstance(c, ZipEntryFile) for c in children)
        # Entries are lazy at construction time.
        assert all(not c._materialized for c in children)

    def test_child_lookup_returns_lazy(self) -> None:
        z = ZipFile()
        z.write_entries([("a.txt", b"hello")])
        ch = z.child("a.txt")
        assert isinstance(ch, ZipEntryFile)
        assert not ch._materialized
        # Size hint is taken from the directory without reading.
        assert ch.size == 5
        assert not ch._materialized
        # Reading triggers materialization.
        assert ch.to_bytes() == b"hello"
        assert ch._materialized

    def test_child_missing_raises(self) -> None:
        z = ZipFile()
        z.write_entries([("a.txt", b"x")])
        with pytest.raises(KeyError, match="No entry"):
            z.child("nope.txt")


class TestTabularDispatch:

    def test_round_trip_via_write_arrow_table(self, table) -> None:
        z = ZipFile()
        z.write_arrow_table(table)
        assert z.list_entries() == ["data.parquet"]
        assert z.read_arrow_table().equals(table)

    def test_explicit_entry_name_csv(self, table) -> None:
        z = ZipFile()
        z.write_arrow_table(table, options=ZipOptions(entry_name="trades.csv"))
        assert z.list_entries() == ["trades.csv"]
        assert z.read_arrow_table().column("id").to_pylist() == [1, 2, 3]

    def test_unknown_entry_name_extension_raises(self, table) -> None:
        z = ZipFile()
        with pytest.raises(ValueError) as excinfo:
            z.write_arrow_table(table, options=ZipOptions(entry_name="data.qqq"))
        msg = str(excinfo.value)
        assert "data.qqq" in msg
        # Error must name the offending entry, explain why dispatch
        # failed, and surface the registered extensions for recovery.
        assert "Tabular leaf" in msg or "MediaType" in msg
        assert "parquet" in msg and "csv" in msg

    def test_no_extension_entry_name_raises_on_write(self, table) -> None:
        z = ZipFile()
        with pytest.raises(ValueError) as excinfo:
            z.write_arrow_table(table, options=ZipOptions(entry_name="payload"))
        msg = str(excinfo.value)
        assert "payload" in msg
        assert "no recognized extension" in msg or "no MediaType" in msg

    def test_skip_non_tabular_entries_in_aggregate_read(self) -> None:
        # Mix a tabular entry with a non-tabular one; the aggregate
        # read filters out the latter automatically.
        csv = CSVFile()
        csv.write_arrow_table(pa.table({"id": [1, 2]}))
        z = ZipFile()
        z.write_entries([
            ("trades.csv", csv.to_bytes()),
            ("readme.txt", b"hello"),
        ])
        out = z.read_arrow_table()
        assert out.column("id").to_pylist() == [1, 2]

    def test_aggregate_read_raises_when_no_entry_resolves(self) -> None:
        # Zip with entries but NONE of them resolve to a tabular leaf —
        # silently returning zero batches would hide the real problem
        # (entry names missing the format extension, an unknown
        # format, …). Surface the failure with a clear message.
        z = ZipFile()
        z.write_entries([
            ("readme.txt", b"hello"),
            ("notes.bin", b"\x00\x01"),
        ])
        with pytest.raises(ValueError) as excinfo:
            z.read_arrow_table()
        msg = str(excinfo.value)
        assert "readme.txt" in msg and "notes.bin" in msg
        assert "Registered tabular extensions" in msg

    def test_entry_read_raises_on_unrecognized_extension(self) -> None:
        # Reading a single ``ZipEntryFile`` directly against an
        # unrecognized name surfaces the same explicit error rather
        # than yielding an empty iterator.
        z = ZipFile()
        z.write_entries([("opaque.bin", b"\x00\x01")])
        entry = z.child("opaque.bin")
        with pytest.raises(ValueError) as excinfo:
            list(entry._read_arrow_batches(entry.options_class()()))
        msg = str(excinfo.value)
        assert "opaque.bin" in msg
        assert "Registered tabular extensions" in msg


class TestEmpty:

    def test_empty_zip_yields_no_entries(self) -> None:
        z = ZipFile()
        assert z.list_entries() == []
        assert list(z.iter_children()) == []

    def test_empty_zip_read_arrow_returns_empty(self) -> None:
        assert list(ZipFile()._read_arrow_batches(ZipOptions())) == []


class TestModes:

    def test_overwrite_replaces_entries(self, table) -> None:
        z = ZipFile()
        z.write_arrow_table(table)
        smaller = pa.table({"id": [9], "name": ["z"]})
        z.write_arrow_table(smaller, options=ZipOptions(mode=Mode.OVERWRITE))
        assert z.list_entries() == ["data.parquet"]
        assert z.read_arrow_table().equals(smaller)

    def test_append_adds_survivor_entries(self, table) -> None:
        z = ZipFile()
        z.write_arrow_table(
            table, options=ZipOptions(entry_name="first.parquet"),
        )
        z.write_arrow_table(
            table, options=ZipOptions(entry_name="second.parquet", mode=Mode.APPEND),
        )
        assert sorted(z.list_entries()) == ["first.parquet", "second.parquet"]

    def test_append_replaces_same_name(self, table) -> None:
        z = ZipFile()
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
        z = ZipFile()
        z.write_arrow_table(table)
        before = z.size
        z.write_arrow_batches(
            table.to_batches(), options=ZipOptions(mode=Mode.IGNORE),
        )
        assert z.size == before

    def test_error_if_exists(self, table) -> None:
        z = ZipFile()
        z.write_arrow_table(table)
        with pytest.raises(FileExistsError):
            z.write_arrow_batches(
                table.to_batches(), options=ZipOptions(mode=Mode.ERROR_IF_EXISTS),
            )


class TestExternalReader:

    def test_stdlib_zipfile_reads_what_we_wrote(self, tmp_path, table) -> None:
        target = LocalPath(str(tmp_path / "bundle.zip"))
        z = ZipFile(holder=target, owns_holder=False)
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

        # ZipFile can read what stdlib zipfile wrote.
        z = ZipFile(holder=target, owns_holder=False)
        assert z.list_entries() == ["readme.txt"]
        ch = z.child("readme.txt")
        assert ch.to_bytes() == b"hi"


class TestLazinessIsHonest:
    """The whole point of ZipEntryFile: don't decompress until asked."""

    def test_iter_children_does_not_decompress(self) -> None:
        z = ZipFile()
        z.write_entries([
            (f"part-{i}.txt", b"x" * 1024) for i in range(8)
        ])
        # Walking the directory is one infolist call. No materialization.
        children = list(z.iter_children())
        assert len(children) == 8
        assert all(not c._materialized for c in children)

    def test_only_touched_children_materialize(self) -> None:
        z = ZipFile()
        z.write_entries([(f"p-{i}.txt", b"abc") for i in range(4)])
        children = list(z.iter_children())
        # Read just one.
        children[2].to_bytes()
        materialized = [c._materialized for c in children]
        assert materialized == [False, False, True, False]
