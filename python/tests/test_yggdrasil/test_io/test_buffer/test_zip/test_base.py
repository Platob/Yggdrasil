"""ZipIO + ZipEntryIO core: archive lifecycle, entry metadata,
parent-aware acquire/release.

ZipEntryIO is now a :class:`BytesIO` (not :class:`PrimitiveIO`).
The tests cover:

* construction shapes and parent linkage,
* dirty tracking around ``_acquire`` / ``_release``,
* per-entry metadata (``zip_info``, ``compression``,
  ``compresslevel``),
* the "optimized commit" paths — cheap append vs. rewrite swap.
"""

from __future__ import annotations

import zipfile

import pytest

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.buffer.nested import ZipEntryIO, ZipIO, ZipOptions
from yggdrasil.io.enums import MimeTypes
from .._helpers import sample_table


class TestZipIOBase:
    def test_default_mime_type(self):
        assert ZipIO.default_mime_type() == MimeTypes.ZIP

    def test_options_class(self):
        assert ZipIO.options_class() is ZipOptions

    def test_is_empty_on_missing_path(self, tmp_path):
        assert ZipIO(path=str(tmp_path / "missing.zip")).is_empty()

    def test_path_bound_construction(self, tmp_path):
        io = ZipIO(path=str(tmp_path / "a.zip"))
        assert io.path is not None


class TestZipEntryIsBytesIO:
    def test_is_a_bytes_io(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        entry = zio.make_child("foo.txt")
        assert isinstance(entry, BytesIO)
        assert isinstance(entry, ZipEntryIO)

    def test_carries_parent_link(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        entry = zio.make_child("foo.txt")
        assert entry.parent is zio

    def test_entry_name_property(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        entry = zio.make_child("nested/x.bin")
        assert entry.entry_name == "nested/x.bin"

    def test_zip_info_none_until_committed(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        entry = zio.make_child("x.bin")
        with entry:
            assert entry.zip_info is None
            entry.write(b"hello")
        # After commit the metadata cache is populated.
        info = zio._entry_info("x.bin")
        assert info is not None
        assert info.filename == "x.bin"

    def test_make_child_rejects_backslashes(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        with pytest.raises(ValueError):
            zio.make_child("bad\\name.txt")


class TestEntryRoundTrip:
    def test_write_then_read_via_buffer(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        entry = zio.make_child("hello.txt")
        with entry:
            entry.write(b"hello world")

        # Re-open a fresh handle: the bytes come back from the
        # central directory.
        re = zio.make_child("hello.txt")
        with re:
            re.seek(0)
            assert re.read() == b"hello world"

    def test_dirty_flag_resets_on_acquire(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        entry = zio.make_child("x.txt")
        with entry:
            entry.write(b"abc")
            assert entry._dirty is True
        # Fresh handle after release — should not commit anything
        # extra unless we mutate.
        re = zio.make_child("x.txt")
        with re:
            assert re._dirty is False

    def test_unmodified_open_does_not_dirty(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        entry = zio.make_child("x.txt")
        with entry:
            entry.write(b"v1")
        # Re-open just to read; the buffer is filled by _acquire
        # via replace_with_payload, which must NOT mark dirty.
        re = zio.make_child("x.txt")
        with re:
            re.seek(0)
            re.read()
            assert re._dirty is False


class TestArchiveCommitPaths:
    def test_append_does_not_rewrite_existing_entries(self, tmp_path):
        path = tmp_path / "a.zip"
        zio = ZipIO(path=str(path))
        with zio.make_child("a.txt") as entry:
            entry.write(b"first")
        size_before = path.stat().st_size

        # Adding a brand-new entry should append cheaply (no rewrite
        # of the existing entry).
        with zio.make_child("b.txt") as entry:
            entry.write(b"second")

        with zipfile.ZipFile(str(path)) as zf:
            assert sorted(zf.namelist()) == ["a.txt", "b.txt"]
            assert zf.read("a.txt") == b"first"
            assert zf.read("b.txt") == b"second"
        assert path.stat().st_size > size_before

    def test_existing_entry_rewrite_swaps_payload(self, tmp_path):
        path = tmp_path / "a.zip"
        zio = ZipIO(path=str(path))
        with zio.make_child("a.txt") as entry:
            entry.write(b"v1")
        with zio.make_child("b.txt") as entry:
            entry.write(b"unchanged")

        # Modify a.txt — full rewrite, but b.txt must survive
        # untouched.
        with zio.make_child("a.txt") as entry:
            entry.seek(0)
            entry.truncate(0)
            entry.write(b"v2")

        with zipfile.ZipFile(str(path)) as zf:
            assert zf.read("a.txt") == b"v2"
            assert zf.read("b.txt") == b"unchanged"


class TestEntryMetadataOverrides:
    def test_compression_override(self, tmp_path):
        path = tmp_path / "a.zip"
        zio = ZipIO(path=str(path))
        entry = zio.make_child("a.txt")
        entry.compression = zipfile.ZIP_DEFLATED
        with entry:
            entry.write(b"compressed payload" * 100)

        with zipfile.ZipFile(str(path)) as zf:
            info = zf.getinfo("a.txt")
            assert info.compress_type == zipfile.ZIP_DEFLATED


class TestEntryDelete:
    def test_delete_removes_entry(self, tmp_path):
        path = tmp_path / "a.zip"
        zio = ZipIO(path=str(path))
        with zio.make_child("a.txt") as entry:
            entry.write(b"x")
        with zio.make_child("b.txt") as entry:
            entry.write(b"y")

        with zio.make_child("a.txt") as entry:
            entry.delete()

        assert "a.txt" not in zio.list_entries()
        assert "b.txt" in zio.list_entries()


class TestZipIONesting:
    def test_iter_children_yields_entries(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        with zio.make_child("first.txt") as entry:
            entry.write(b"1")
        with zio.make_child("second.txt") as entry:
            entry.write(b"2")

        names = sorted(c.entry_name for c in zio._iter_children(ZipOptions()))
        assert names == ["first.txt", "second.txt"]

    def test_contains_check(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        with zio.make_child("present.txt") as entry:
            entry.write(b"x")
        assert "present.txt" in zio
        assert "missing.txt" not in zio

    def test_list_entries_sorted(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        with zio.make_child("z.txt") as entry:
            entry.write(b"z")
        with zio.make_child("a.txt") as entry:
            entry.write(b"a")
        assert zio.list_entries() == ["a.txt", "z.txt"]

    def test_for_loop_yields_entries(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        with zio.make_child("a.txt") as entry:
            entry.write(b"1")
        with zio.make_child("b.txt") as entry:
            entry.write(b"2")

        names = sorted(child.entry_name for child in zio)
        assert names == ["a.txt", "b.txt"]

    def test_next_on_zio_rejects_line_iteration(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        with zio.make_child("a.txt") as entry:
            entry.write(b"hello\nworld\n")
        # next(zio) would otherwise fall through to BytesIO's
        # readline-based iterator and silently return raw archive
        # bytes — surface that as a clear error instead.
        with pytest.raises(TypeError, match="not directly iterable"):
            next(zio)

    def test_hidden_entries_filtered_from_listing(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        with zio.make_child("visible.txt") as entry:
            entry.write(b"v")
        with zio.make_child(".hidden") as entry:
            entry.write(b"h")

        assert zio.list_entries() == ["visible.txt"]
        assert ".hidden" not in zio
        assert "visible.txt" in zio
        names = [child.entry_name for child in zio]
        assert names == ["visible.txt"]


class TestZipIOCursorPreservation:
    """Children-surface ops must not move the parent's byte cursor."""

    def _seed(self, tmp_path):
        zio = ZipIO(path=str(tmp_path / "a.zip"))
        with zio.make_child("a.txt") as entry:
            entry.write(b"hello")
        with zio.make_child("b.txt") as entry:
            entry.write(b"world")
        return zio

    def test_iter_children_preserves_cursor(self, tmp_path):
        zio = self._seed(tmp_path)
        zio.open()
        try:
            zio.seek(7)
            pos = zio.tell()
            list(zio.iter_children())
            assert zio.tell() == pos
            for _ in zio:
                pass
            assert zio.tell() == pos
        finally:
            zio.close()

    def test_make_child_preserves_cursor(self, tmp_path):
        zio = self._seed(tmp_path)
        zio.open()
        try:
            zio.seek(5)
            pos = zio.tell()
            zio.make_child("c.txt")
            assert zio.tell() == pos
        finally:
            zio.close()

    def test_list_entries_and_contains_preserve_cursor(self, tmp_path):
        zio = self._seed(tmp_path)
        zio.open()
        try:
            zio.seek(3)
            pos = zio.tell()
            zio.list_entries()
            assert zio.tell() == pos
            assert "a.txt" in zio
            assert zio.tell() == pos
            zio.has_children()
            assert zio.tell() == pos
        finally:
            zio.close()

    def test_commit_entry_preserves_parent_cursor(self, tmp_path):
        zio = self._seed(tmp_path)
        zio.open()
        try:
            zio.seek(4)
            pos = zio.tell()
            # Cheap-append path (new entry).
            with zio.make_child("c.txt") as entry:
                entry.write(b"!")
            assert zio.tell() == pos
            # Rewrite-swap path (existing entry).
            with zio.make_child("a.txt") as entry:
                entry.seek(0)
                entry.truncate(0)
                entry.write(b"HELLO")
            assert zio.tell() == pos
        finally:
            zio.close()

    def test_delete_entry_preserves_parent_cursor(self, tmp_path):
        zio = self._seed(tmp_path)
        zio.open()
        try:
            zio.seek(6)
            pos = zio.tell()
            with zio.make_child("a.txt") as entry:
                entry.delete()
            assert zio.tell() == pos
        finally:
            zio.close()
