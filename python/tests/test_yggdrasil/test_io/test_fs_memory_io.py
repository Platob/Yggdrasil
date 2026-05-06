"""Tests for :class:`yggdrasil.io.fs.memory_io.MemoryPath`.

The path is the buffer: each :class:`MemoryPath` instance owns one
:class:`BytesIO` and the URL is :meth:`URL.from_memory_address` of
that buffer. These tests pin the construction shapes (empty,
seeded, URL round-trip), the I/O surface, and the lack of a
directory model.
"""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.fs.memory_io import MemoryPath
from yggdrasil.io.io_stats import IOKind
from yggdrasil.io.url import URL


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_mints_buffer_and_url(self):
        path = MemoryPath()
        assert isinstance(path.buffer, BytesIO)
        assert path.url.is_memory_address
        assert path.url.memory_address == id(path.buffer)
        assert path.size == 0

    def test_seeded_with_bytes(self):
        path = MemoryPath(b"hello")
        assert path.read_bytes() == b"hello"
        assert path.size == 5

    def test_seeded_with_memoryview(self):
        path = MemoryPath(memoryview(b"world"))
        assert path.read_bytes() == b"world"

    def test_url_round_trip_same_buffer(self):
        path = MemoryPath(b"abc")
        clone = MemoryPath(url=path.url)
        # Both views point at the *same* underlying buffer.
        assert clone.buffer is path.buffer
        assert clone.read_bytes() == b"abc"

    def test_url_string_round_trip_same_buffer(self):
        path = MemoryPath(b"abc")
        clone = MemoryPath(str(path.url))
        assert clone.buffer is path.buffer

    def test_url_resolves_via_from_factory(self):
        path = MemoryPath(b"abc")
        # ``Path.from_`` should dispatch the ``mem:`` scheme to MemoryPath
        # and resolve the address back to the original buffer.
        clone = MemoryPath.from_(path.url)
        assert clone.buffer is path.buffer

    def test_explicit_buffer_passthrough(self):
        buf = BytesIO(b"explicit")
        path = MemoryPath(buf)
        assert path.buffer is buf
        assert path.url.memory_address == id(buf)

    def test_url_pointing_at_non_bytesio_falls_back(self):
        # A live Python object that isn't a BytesIO — the resolver
        # returns it, the path rejects the type and mints fresh
        # bytes rather than viewing something nonsensical.
        sentinel = object()
        url = URL.from_memory_address(sentinel)
        path = MemoryPath(url=url)
        # URL gets re-derived to track the freshly-minted buffer.
        assert path.url != url
        assert path.url.memory_address == id(path.buffer)
        # Hold a reference so ``sentinel`` isn't collected mid-test.
        assert sentinel is not None


# ---------------------------------------------------------------------------
# Stat / size
# ---------------------------------------------------------------------------


class TestStat:
    def test_stat_kind_file(self):
        path = MemoryPath()
        assert path.stat().kind is IOKind.FILE

    def test_stat_size_tracks_writes(self):
        path = MemoryPath()
        path.write_bytes(b"abcdef")
        assert path.stat().size == 6

    def test_stat_alive_even_after_close(self):
        # Closing the path is lifecycle bookkeeping — the bytes are
        # still alive, and any operation on the path reopens the
        # buffer transparently.
        path = MemoryPath(b"abc")
        path.close(force=True)
        assert path.stat().kind is IOKind.FILE
        assert path.stat().size == 3

    def test_full_path_renders_as_mem_url(self):
        path = MemoryPath()
        s = path.full_path()
        assert s.startswith("mem:/")
        assert "0x" in s


# ---------------------------------------------------------------------------
# I/O surface
# ---------------------------------------------------------------------------


class TestIO:
    def test_write_then_read(self):
        path = MemoryPath()
        path.write_bytes(b"hello")
        assert path.read_bytes() == b"hello"

    def test_pread_pwrite(self):
        path = MemoryPath(b"abcdef")
        assert path.pread(3, 1) == b"bcd"
        path.pwrite(b"ZZ", 2)
        assert path.read_bytes() == b"abZZef"

    def test_truncate(self):
        path = MemoryPath(b"abcdef")
        path.truncate(3)
        assert path.read_bytes() == b"abc"

    def test_truncate_grows_with_zero_padding(self):
        path = MemoryPath(b"ab")
        path.truncate(5)
        assert path.read_bytes() == b"ab\x00\x00\x00"

    def test_open_io_returns_bytesio_path_bound(self):
        path = MemoryPath(b"xyz")
        with path.open_io("rb") as fh:
            assert fh.read() == b"xyz"

    def test_open_wb_truncates(self):
        path = MemoryPath(b"original")
        with path.open_io("wb") as fh:
            fh.write(b"new")
        assert path.read_bytes() == b"new"

    def test_open_xb_on_existing_raises(self):
        path = MemoryPath(b"data")
        with pytest.raises(FileExistsError):
            path.open_io("xb")

    def test_open_ab_appends(self):
        path = MemoryPath(b"orig")
        with path.open_io("ab") as fh:
            fh.write(b"-more")
        assert path.read_bytes() == b"orig-more"

    def test_reopen_after_close_revives_buffer(self):
        # Closing the path closes the buffer; reopening brings it
        # back online without minting a new buffer (so the URL's
        # memory address stays stable).
        path = MemoryPath(b"abc")
        original_url = path.url
        path.close(force=True)
        with path.open_io("rb") as fh:
            assert fh.read() == b"abc"
        assert path.url == original_url


# ---------------------------------------------------------------------------
# Shared-buffer semantics
# ---------------------------------------------------------------------------


class TestSharedBuffer:
    def test_two_opens_share_bytes(self):
        path = MemoryPath(b"abc")
        with path.open_io("rb+") as writer, path.open_io("rb") as reader:
            writer.seek(0)
            writer.write(b"XYZ")
            reader.seek(0)
            assert reader.read() == b"XYZ"

    def test_independent_cursors(self):
        path = MemoryPath(b"abcdef")
        with path.open_io("rb") as fh1, path.open_io("rb") as fh2:
            fh1.read(3)
            assert fh2.tell() == 0

    def test_url_clone_sees_writes(self):
        path = MemoryPath(b"abc")
        clone = MemoryPath(url=path.url)
        path.write_bytes(b"REPLACED")
        assert clone.read_bytes() == b"REPLACED"


# ---------------------------------------------------------------------------
# Memoryview
# ---------------------------------------------------------------------------


class TestMemoryview:
    def test_memoryview_returns_bytes(self):
        path = MemoryPath(b"abcdef")
        mv = path.memoryview()
        assert bytes(mv) == b"abcdef"

    def test_memoryview_with_offset_and_size(self):
        path = MemoryPath(b"abcdef")
        mv = path.memoryview(offset=2, size=3)
        assert bytes(mv) == b"cde"


# ---------------------------------------------------------------------------
# No directory semantics
# ---------------------------------------------------------------------------


class TestNoDirectoryModel:
    def test_iterdir_yields_nothing(self):
        path = MemoryPath(b"abc")
        assert list(path.iterdir()) == []

    def test_mkdir_default_is_no_op(self):
        path = MemoryPath()
        # exist_ok=True is the contractual default — should not raise.
        path.mkdir(parents=True, exist_ok=True)

    def test_mkdir_strict_raises(self):
        path = MemoryPath()
        with pytest.raises(NotADirectoryError):
            path.mkdir(parents=True, exist_ok=False)

    def test_unlink_drops_bytes_keeps_buffer(self):
        # ``_remove_file`` truncates rather than closing so the URL
        # stays valid for any other live reference.
        path = MemoryPath(b"abc")
        path.unlink()
        assert path.exists()  # buffer still alive, just empty
        assert path.size == 0
