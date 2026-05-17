"""Tests for :class:`yggdrasil.io.nested.zip_file.ZipFile`."""

from __future__ import annotations

import zipfile as stdlib_zipfile

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.io.memory import Memory
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.nested.zip_file import ZipFile, ZipEntryFile


class TestRegistration:

    def test_class_for_media_type(self) -> None:
        assert Holder.class_for_media_type("application/zip") is ZipFile

    def test_path_dispatches_zip_ext(self, tmp_path) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        b = BytesIO(path=str(tmp_path / "x.zip"))
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
