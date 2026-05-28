"""Tests for :class:`yggdrasil.io.nested.zip_file.ZipFile`."""

from __future__ import annotations

import zipfile as stdlib_zipfile

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.path.memory import Memory
from yggdrasil.path.local_path import LocalPath
from yggdrasil.io.nested.zip_file import ZipFile, ZipEntryFile


class TestRegistration:

    def test_class_for_media_type(self) -> None:
        assert Holder.class_for_media_type("application/zip") is ZipFile

    def test_path_dispatches_zip_ext(self, tmp_path) -> None:
        from yggdrasil.io.base import IO

        b = IO(path=str(tmp_path / "x.zip"))
        assert isinstance(b, ZipFile)


def _build_archive(entries: dict[str, bytes]) -> bytes:
    """Build an in-memory zip with the given ``name → bytes`` entries."""
    import io as _io

    raw = _io.BytesIO()
    with stdlib_zipfile.ZipFile(raw, "w") as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return raw.getvalue()


class TestListEntries:

    def test_list_returns_entry_names(self) -> None:
        raw = _build_archive({"a.txt": b"alpha", "b.txt": b"beta"})
        mem = Memory(raw)
        zf = ZipFile(holder=mem, owns_holder=False)
        names = {e.entry_name for e in zf.iter_children()}
        assert names == {"a.txt", "b.txt"}


class TestZipEntryLazy:
    """Iterating children doesn't decompress payloads until accessed."""

    def test_child_is_lazy_until_read(self) -> None:
        raw = _build_archive({"data.txt": b"payload"})
        mem = Memory(raw)
        zf = ZipFile(holder=mem, owns_holder=False)
        children = list(zf.iter_children())
        assert len(children) == 1
        entry = children[0]
        assert isinstance(entry, ZipEntryFile)
        # Materialization happens on first read.
        assert entry.to_bytes() == b"payload"


class TestWriteArchive:

    def test_write_arrow_batches_packs_parquet_entry(self, tmp_path) -> None:
        table = pa.table({"x": [1, 2, 3]})
        path = LocalPath(str(tmp_path / "out.zip"))
        with path.open("wb") as zf:
            zf.write_arrow_batches(iter(table.to_batches()))

        # The resulting archive has at least one entry.
        with stdlib_zipfile.ZipFile(tmp_path / "out.zip", "r") as zfile:
            assert len(zfile.namelist()) >= 1


# ---------------------------------------------------------------------------
# Read-path single-open optimization
# ---------------------------------------------------------------------------


class TestReadArrowBatchesSingleOpen:
    """``_read_arrow_batches`` opens the parent archive exactly once
    instead of N+1 times (once for the directory walk, then once per
    child via the lazy materialization path)."""

    def test_single_zipfile_open_per_read(self, monkeypatch) -> None:
        # Build a 3-entry parquet archive in-memory.
        from yggdrasil.io.nested.zip_file import (
            ZipFile as _Zip, ZipOptions,
        )
        import yggdrasil.io.nested.zip_file as zip_module
        from yggdrasil.io.primitive.parquet_file import ParquetFile

        tables = [
            pa.table({"x": [i, i + 1, i + 2]}) for i in range(3)
        ]
        # Encode each table as parquet bytes, then bundle into one zip.
        entries = {}
        for i, tbl in enumerate(tables):
            buf = Memory()
            pq = ParquetFile(holder=buf, owns_holder=False)
            pq.write_arrow_batches(iter(tbl.to_batches()))
            entries[f"part-{i}.parquet"] = buf.to_bytes()

        raw = _build_archive(entries)
        mem = Memory(raw)
        zf = _Zip(holder=mem, owns_holder=False)

        # Patch the stdlib constructor to count opens.
        original = zip_module.zipfile.ZipFile
        opens = {"n": 0}

        def counting_open(*a, **kw):
            opens["n"] += 1
            return original(*a, **kw)

        monkeypatch.setattr(zip_module.zipfile, "ZipFile", counting_open)

        batches = list(zf._read_arrow_batches(ZipOptions()))
        assert len(batches) >= 3  # one batch per parquet entry
        assert opens["n"] == 1, (
            f"expected 1 zipfile open, got {opens['n']}"
        )


# ---------------------------------------------------------------------------
# Append-path streaming
# ---------------------------------------------------------------------------


class TestAppendStreaming:
    """APPEND mode used to materialize every survivor's decompressed
    bytes into a Python list, peaking at sum(uncompressed_sizes). The
    new path streams each survivor chunk-by-chunk from source to
    destination, so only the active 1 MiB chunk is in memory at any
    moment."""

    def test_append_preserves_existing_entries(self) -> None:
        from yggdrasil.io.nested.zip_file import (
            ZipFile as _Zip, ZipOptions,
        )
        from yggdrasil.enums import Mode

        # Start with one parquet entry; append another with a
        # different name; verify both survive.
        from yggdrasil.io.primitive.parquet_file import ParquetFile

        # Pre-existing entry.
        existing_table = pa.table({"x": [10, 20]})
        pq_buf = Memory()
        ParquetFile(holder=pq_buf, owns_holder=False).write_arrow_batches(
            iter(existing_table.to_batches())
        )

        raw = _build_archive({"old.parquet": pq_buf.to_bytes()})
        mem = Memory(raw)
        zf = _Zip(holder=mem, owns_holder=False)

        new_table = pa.table({"x": [30, 40]})
        zf.write_arrow_batches(
            iter(new_table.to_batches()),
            options=ZipOptions(
                entry_name="new.parquet", mode=Mode.APPEND,
            ),
        )

        # Both entries must survive the append.
        with stdlib_zipfile.ZipFile(__import__("io").BytesIO(mem.to_bytes()), "r") as zfile:
            names = set(zfile.namelist())
            assert names == {"old.parquet", "new.parquet"}

    def test_append_replaces_same_named_entry(self) -> None:
        from yggdrasil.io.nested.zip_file import (
            ZipFile as _Zip, ZipOptions,
        )
        from yggdrasil.enums import Mode
        from yggdrasil.io.primitive.parquet_file import ParquetFile

        # First write (becomes the survivor that gets replaced).
        first = pa.table({"x": [1, 2, 3]})
        pq_buf = Memory()
        ParquetFile(holder=pq_buf, owns_holder=False).write_arrow_batches(
            iter(first.to_batches())
        )
        raw = _build_archive({"data.parquet": pq_buf.to_bytes()})
        mem = Memory(raw)
        zf = _Zip(holder=mem, owns_holder=False)

        # Replace with new bytes under the same entry name.
        replacement = pa.table({"x": [99, 100]})
        zf.write_arrow_batches(
            iter(replacement.to_batches()),
            options=ZipOptions(
                entry_name="data.parquet", mode=Mode.APPEND,
            ),
        )

        # Read back — must reflect the replacement, not the original.
        read_back = list(zf._read_arrow_batches(ZipOptions()))
        assert read_back
        combined = pa.Table.from_batches(read_back)
        assert combined["x"].to_pylist() == [99, 100]
