"""``BytesIO`` core: construction, slots, mode, byte-level ops.

The tabular contract on a raw :class:`BytesIO` redirects through
``as_media`` (covered in ``test_arrow.py`` / ``test_polars.py``).
This file stays at the byte-buffer layer.
"""

from __future__ import annotations

import io as stdio

import pytest

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.enums import MimeTypes


class TestBytesIOConstruction:
    def test_empty_construction(self):
        bio = BytesIO()
        assert bio.size == 0
        assert bio._media_type is None

    def test_from_bytes(self):
        bio = BytesIO(b"hello")
        assert bio.size == 5

    def test_from_bytearray(self):
        bio = BytesIO(bytearray(b"abc"))
        assert bio.size == 3

    def test_from_memoryview(self):
        bio = BytesIO(memoryview(b"xyz"))
        assert bio.size == 3

    def test_from_stdlib_bytesio(self):
        src = stdio.BytesIO(b"data")
        bio = BytesIO(src)
        assert bio.size == 4

    def test_path_bound(self, tmp_path):
        path = tmp_path / "buf.bin"
        path.write_bytes(b"PAYLOAD")
        bio = BytesIO(path=str(path))
        bio.seek(0)
        assert bio.read() == b"PAYLOAD"
        bio.close()

    def test_default_mime_type_is_none(self):
        # BytesIO opts out of the TabularIO registry — the buffer
        # has no single mime to claim.
        assert BytesIO.default_mime_type() is None


class TestBytesIOReadWrite:
    def test_round_trip_bytes(self):
        bio = BytesIO()
        bio.write(b"hello world")
        bio.seek(0)
        assert bio.read() == b"hello world"

    def test_seek_and_partial_read(self):
        bio = BytesIO(b"abcdefgh")
        bio.seek(2)
        assert bio.read(3) == b"cde"
        assert bio.tell() == 5

    def test_truncate_grow(self):
        bio = BytesIO(b"abc")
        bio.truncate(6)
        assert bio.size == 6
        bio.seek(0)
        # Newly grown bytes are zero-padded.
        assert bio.read() == b"abc\x00\x00\x00"

    def test_truncate_shrink(self):
        bio = BytesIO(b"abcdefgh")
        bio.truncate(3)
        assert bio.size == 3
        bio.seek(0)
        assert bio.read() == b"abc"

    def test_write_at_position(self):
        bio = BytesIO(b"abcdef")
        bio.seek(2)
        bio.write(b"XY")
        bio.seek(0)
        assert bio.read() == b"abXYef"


class TestBytesIOMediaType:
    def test_set_media_type_via_kwarg(self):
        # When the media is tabular, the constructor reroutes
        # through the registry to the registered leaf.
        from yggdrasil.io.buffer.primitive import ParquetIO

        io = BytesIO(media_type=MimeTypes.PARQUET)
        assert isinstance(io, ParquetIO)

    def test_octet_media_type_stays_bytes_io(self):
        bio = BytesIO(media_type=MimeTypes.OCTET_STREAM)
        assert type(bio) is BytesIO


class TestBytesIOLifecycle:
    def test_close_releases_buffer(self):
        bio = BytesIO(b"abc")
        bio.close()
        assert bio.closed

    def test_context_manager(self):
        with BytesIO(b"ctx") as bio:
            assert bio.size == 3
        assert bio.closed

    def test_cached_default_false(self):
        # TabularIO contract is implemented on BytesIO; without a
        # tabular view, persist() must fail rather than silently
        # cache nothing.
        bio = BytesIO()
        assert bio.cached is False
