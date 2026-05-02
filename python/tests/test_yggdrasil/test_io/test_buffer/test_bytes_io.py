"""Comprehensive unittest suite for autonomous BytesIO.

Coverage organized into TestCase classes by area:
  TestConstruction          — input shapes + factory
  TestMemoryMode            — write/read/seek/truncate basics
  TestSpillMechanics        — threshold crossing, one-way, cleanup
  TestLifecycle             — open/close/with-block/GC/reopen
  TestReplaceWithPayload    — every replace shape
  TestIOProtocol            — IO[bytes] surface (readline/readinto/etc)
  TestCursorlessIO          — pread/pwrite
  TestStructuredIO          — int/float/bool/u32 helpers
  TestHashing               — xxh3, blake3
  TestMediaType             — inference + with_media_type guards
  TestWriteInto             — path / filelike / overwrite
  TestViewsAndReaders       — view/memoryview/open_reader/open_file
  TestSeekEdgeCases         — SEEK_SET/CUR/END corners
  TestPickling              — getstate/setstate
  TestIdentityDunders       — bool/len/bytes/repr/iter
  TestIntegration           — round-trip scenarios

Run with:
    python3 -m unittest test_bytes_io
or:
    python3 test_bytes_io.py
"""

from __future__ import annotations

import gc
import io as stdio
import os
import pickle
import tempfile
import unittest

from yggdrasil.io import BytesIO
from yggdrasil.io.enums import MediaType


# --------------------------------------------------------------------------
# Helpers shared across test cases
# --------------------------------------------------------------------------


class NonSeekable:
    """Read-only file-like with no seek/tell (forces the non-seekable
    drain branch in BytesIO._init_from_filelike)."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            out = self._data[self._pos:]
            self._pos = len(self._data)
            return out
        out = self._data[self._pos : self._pos + n]
        self._pos += len(out)
        return out


class NotWritableSink:
    """Sink whose ``writable()`` returns False — used to verify
    write_into rejects unwritable destinations."""

    def write(self, x):  # pragma: no cover - never reached
        raise AssertionError("should not be called")

    def writable(self):
        return False


class _MixinTempFiles:
    """Mixin providing a managed temp file factory and per-test cleanup."""

    def setUp(self):
        super().setUp()
        self._created_paths: list[str] = []

    def tearDown(self):
        for p in self._created_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        super().tearDown()

    def make_tempfile(self, content: bytes = b"") -> str:
        fd, name = tempfile.mkstemp(suffix=".bin")
        try:
            if content:
                os.write(fd, content)
        finally:
            os.close(fd)
        self._created_paths.append(name)
        return name


# ==========================================================================
# 1. CONSTRUCTION
# ==========================================================================


class TestConstruction(_MixinTempFiles, unittest.TestCase):
    def test_ctor_none_makes_empty_memory(self):
        with BytesIO() as b:
            self.assertEqual(b.size, 0)
            self.assertFalse(b.spilled)
            self.assertEqual(b.to_bytes(), b"")

    def test_ctor_bytes_input(self):
        with BytesIO(b"hello") as b:
            self.assertEqual(b.size, 5)
            self.assertEqual(b.to_bytes(), b"hello")

    def test_ctor_bytearray_input(self):
        with BytesIO(bytearray(b"world")) as b:
            self.assertEqual(b.to_bytes(), b"world")

    def test_ctor_memoryview_input(self):
        src = bytearray(b"slice me")
        with BytesIO(memoryview(src)[2:7]) as b:
            self.assertEqual(b.to_bytes(), b"ice m")

    def test_ctor_stdlib_io_BytesIO_drains_from_cursor(self):
        src = stdio.BytesIO(b"abcdef")
        src.seek(2)
        with BytesIO(src) as b:
            self.assertEqual(b.to_bytes(), b"cdef")

    def test_ctor_another_BytesIO_deepcopies(self):
        src = BytesIO(b"data")
        b = BytesIO(src)
        try:
            self.assertEqual(b.to_bytes(), b"data")
            src.close()
            # Independent of source — survives src close
            self.assertEqual(b.to_bytes(), b"data")
        finally:
            b.close()

    def test_ctor_BytesIO_copy_true_same_as_default(self):
        """In autonomous mode, copy=True and copy=False both deep-copy."""
        with BytesIO(b"x") as src:
            with BytesIO(src) as b1, BytesIO(src, copy=True) as b2:
                self.assertEqual(b1.to_bytes(), b"x")
                self.assertEqual(b2.to_bytes(), b"x")

    def test_ctor_filelike_seekable(self):
        src = stdio.BytesIO(b"file-like")
        with BytesIO(src) as b:
            self.assertEqual(b.to_bytes(), b"file-like")

    def test_ctor_filelike_nonseekable_pulls_back_when_small(self):
        src = NonSeekable(b"streaming-only")
        with BytesIO(src, spill_bytes=100) as b:
            self.assertEqual(b.to_bytes(), b"streaming-only")
            self.assertFalse(b.spilled, "small drain pulls back to memory")

    def test_ctor_filelike_nonseekable_above_threshold_stays_spilled(self):
        src = NonSeekable(b"x" * 200)
        b = BytesIO(src, spill_bytes=10)
        try:
            self.assertTrue(b.spilled)
            fname = b.path
            self.assertTrue(os.path.exists(fname))
        finally:
            b.close()
        self.assertFalse(os.path.exists(fname))

    def test_ctor_unsupported_type_raises(self):
        self.assertRaises(TypeError, BytesIO, 12345)
        self.assertRaises(TypeError, BytesIO, [1, 2, 3])

    def test_ctor_with_media_type(self):
        mt = MediaType("application/json")
        with BytesIO(b"{}", media_type=mt) as b:
            self.assertEqual(b.media_type, mt)

    def test_factory_from_idempotent_on_BytesIO(self):
        b = BytesIO(b"x")
        try:
            self.assertIs(BytesIO.from_(b), b)
        finally:
            b.close()

    def test_factory_from_wraps_bytes(self):
        with BytesIO.from_(b"y") as b:
            self.assertIsInstance(b, BytesIO)
            self.assertEqual(b.to_bytes(), b"y")


# ==========================================================================
# 2. MEMORY MODE BASICS
# ==========================================================================


class TestMemoryMode(unittest.TestCase):
    def test_write_then_read(self):
        with BytesIO() as b:
            self.assertEqual(b.write(b"first"), 5)
            b.write(b"second")
            b.seek(0)
            self.assertEqual(b.read(), b"firstsecond")

    def test_truncate_shrink(self):
        with BytesIO(b"abcdefgh") as b:
            b.truncate(4)
            self.assertEqual(b.size, 4)
            self.assertEqual(b.to_bytes(), b"abcd")

    def test_truncate_grow_zerofill(self):
        with BytesIO(b"abc") as b:
            b.truncate(6)
            self.assertEqual(b.size, 6)
            self.assertEqual(b.to_bytes(), b"abc\x00\x00\x00")

    def test_truncate_no_arg_uses_pos(self):
        with BytesIO(b"abcdef") as b:
            b.seek(3)
            b.truncate()
            self.assertEqual(b.to_bytes(), b"abc")

    def test_truncate_negative_raises(self):
        with BytesIO(b"abc") as b:
            self.assertRaises(ValueError, b.truncate, -1)

    def test_pos_clamped_on_truncate(self):
        with BytesIO(b"abcdef") as b:
            b.seek(5)
            b.truncate(2)
            self.assertEqual(b.tell(), 2)

    def test_overwrite_at_pos(self):
        with BytesIO(b"hello world") as b:
            b.seek(6)
            b.write(b"WORLD")
            self.assertEqual(b.to_bytes(), b"hello WORLD")

    def test_write_extends_past_end_zerofills(self):
        with BytesIO(b"abc") as b:
            b.seek(10)
            b.write(b"X")
            self.assertEqual(b.size, 11)
            self.assertEqual(b.to_bytes(), b"abc" + b"\x00" * 7 + b"X")


# ==========================================================================
# 3. SPILL MECHANICS
# ==========================================================================


class TestSpillMechanics(unittest.TestCase):
    def test_below_threshold_stays_memory(self):
        with BytesIO(spill_bytes=100) as b:
            b.write(b"x" * 50)
            self.assertFalse(b.spilled)

    def test_crosses_threshold_spills(self):
        b = BytesIO(spill_bytes=10)
        try:
            b.write(b"x" * 5)
            self.assertFalse(b.spilled)
            b.write(b"x" * 20)
            self.assertTrue(b.spilled)
            fname = b.path
            self.assertTrue(os.path.exists(fname))
        finally:
            b.close()
        self.assertFalse(os.path.exists(fname))

    def test_preserves_data_after_spill(self):
        with BytesIO(spill_bytes=10) as b:
            payload = b"abcdef" * 10
            b.write(payload)
            self.assertTrue(b.spilled)
            b.seek(0)
            self.assertEqual(b.read(), payload)

    def test_initial_data_above_threshold_spills(self):
        payload = b"y" * 200
        b = BytesIO(payload, spill_bytes=10)
        try:
            self.assertTrue(b.spilled)
            self.assertEqual(b.to_bytes(), payload)
            fname = b.path
        finally:
            b.close()
        self.assertFalse(os.path.exists(fname))

    def test_spill_is_one_way(self):
        """Truncate below threshold doesn't demote to memory."""
        b = BytesIO(spill_bytes=10)
        try:
            b.write(b"x" * 50)
            self.assertTrue(b.spilled)
            fname = b.path
            b.truncate(5)
            self.assertTrue(b.spilled, "still spilled after small truncate")
            self.assertEqual(b.size, 5)
            self.assertTrue(os.path.exists(fname))
        finally:
            b.close()
        self.assertFalse(os.path.exists(fname))

    def test_truncate_to_zero_keeps_spilled(self):
        b = BytesIO(spill_bytes=10)
        try:
            b.write(b"x" * 50)
            fname = b.path
            b.truncate(0)
            self.assertTrue(b.spilled)
            self.assertEqual(b.size, 0)
            self.assertTrue(os.path.exists(fname))
        finally:
            b.close()
        self.assertFalse(os.path.exists(fname))

    def test_pwrite_extends_spilled_file(self):
        with BytesIO(spill_bytes=5) as b:
            b.write(b"x" * 10)
            b.pwrite(memoryview(b"END"), 100)
            self.assertEqual(b.size, 103)
            self.assertEqual(b.pread(3, 100), b"END")

    def test_need_spill_predicate(self):
        with BytesIO(spill_bytes=100) as b:
            self.assertFalse(b.need_spill(50))
            self.assertTrue(b.need_spill(100))
            self.assertTrue(b.need_spill(200))

    def test_spill_path_extension_uses_media_type(self):
        mt = MediaType("application/json")
        with BytesIO(media_type=mt, spill_bytes=5) as b:
            b.write(b"x" * 10)
            self.assertTrue(b.path.url.endswith(".json"), b.path)

    def test_spill_path_default_extension_is_bin(self):
        with BytesIO(spill_bytes=5) as b:
            b.write(b"x" * 10)
            self.assertTrue(b.path.url.endswith(".bin"), b.path)


# ==========================================================================
# 4. LIFECYCLE
# ==========================================================================


class TestLifecycle(unittest.TestCase):
    def test_with_block_closes(self):
        with BytesIO(b"data", auto_open=False) as b:
            self.assertTrue(b.opened)
        self.assertTrue(b.closed)

    def test_with_block_unlinks_spill(self):
        with BytesIO(spill_bytes=10, auto_open=False) as b:
            b.write(b"x" * 50)
            fname = b.path
        self.assertFalse(os.path.exists(fname))

    def test_with_block_exception_still_unlinks(self):
        fname = None
        with self.assertRaises(ValueError):
            with BytesIO(spill_bytes=10, auto_open=False) as b:
                b.write(b"x" * 50)
                fname = b.path
                raise ValueError("boom")
        self.assertIsNotNone(fname)
        self.assertFalse(os.path.exists(fname))

    def test_nested_with_blocks(self):
        with BytesIO(b"data", auto_open=False) as b:
            with b:
                self.assertTrue(b.opened)
            self.assertTrue(b.opened, "outer still active")
        self.assertTrue(b.closed)

    def test_close_idempotent(self):
        b = BytesIO(b"x")
        b.close()
        b.close()  # should not raise

    def test_reopen_after_close(self):
        b = BytesIO(b"x")
        b.close()
        b.open()
        try:
            self.assertTrue(b.opened)
        finally:
            b.close()

    def test_gc_unlinks_orphan_spill(self):
        fname_holder = []

        def make_orphan():
            b = BytesIO(spill_bytes=10)
            b.write(b"x" * 50)
            fname_holder.append(b.path)

        make_orphan()
        gc.collect()
        fname = fname_holder[0]
        self.assertFalse(os.path.exists(fname))

    def test_close_releases_spill_fd(self):
        b = BytesIO(spill_bytes=5)
        b.write(b"x" * 10)
        fd = b.fileno()
        b.close()
        self.assertRaises(OSError, os.fstat, fd)


# ==========================================================================
# 5. REPLACE_WITH_PAYLOAD
# ==========================================================================


class TestReplaceWithPayload(unittest.TestCase):
    def test_memory_to_memory(self):
        with BytesIO(b"old") as a, BytesIO(b"new content") as payload:
            a.replace_with_payload(payload)
            self.assertEqual(a.to_bytes(), b"new content")

    def test_memory_to_spill(self):
        with BytesIO(b"old", spill_bytes=10) as a:
            big = b"y" * 200
            with BytesIO(big, spill_bytes=10) as payload:
                self.assertTrue(payload.spilled)
                a.replace_with_payload(payload)
            self.assertTrue(a.spilled)
            self.assertEqual(a.size, 200)
            self.assertEqual(a.to_bytes(), big)

    def test_spill_to_memory_unlinks_old(self):
        a = BytesIO(spill_bytes=10)
        try:
            a.write(b"x" * 50)
            old_fname = a.path
            with BytesIO(b"small") as payload:
                a.replace_with_payload(payload)
            self.assertFalse(a.spilled)
            self.assertEqual(a.to_bytes(), b"small")
            self.assertFalse(os.path.exists(old_fname))
        finally:
            a.close()

    def test_spill_to_spill(self):
        a = BytesIO(spill_bytes=10)
        new_fname = None
        try:
            a.write(b"x" * 50)
            old_fname = a.path
            with BytesIO(b"y" * 100, spill_bytes=10) as payload:
                a.replace_with_payload(payload)
            self.assertTrue(a.spilled)
            new_fname = a.path
            self.assertNotEqual(new_fname, old_fname)
            self.assertFalse(os.path.exists(old_fname))
            self.assertTrue(os.path.exists(new_fname))
        finally:
            a.close()
        self.assertFalse(os.path.exists(new_fname))

    def test_replace_with_none_clears(self):
        with BytesIO(b"data") as a:
            a.replace_with_payload(None)
            self.assertEqual(a.size, 0)

    def test_replace_with_bytes(self):
        with BytesIO(b"data") as a:
            a.replace_with_payload(b"different")
            self.assertEqual(a.to_bytes(), b"different")


# ==========================================================================
# 6. IO[bytes] PROTOCOL
# ==========================================================================


class TestIOProtocol(unittest.TestCase):
    def test_predicates(self):
        with BytesIO(b"x") as b:
            self.assertTrue(b.readable())
            self.assertTrue(b.writable())
            self.assertTrue(b.seekable())
            self.assertFalse(b.isatty())

    def test_mode_property(self):
        with BytesIO() as b:
            self.assertEqual(b.mode, "rb+")

    def test_name_memory(self):
        with BytesIO() as b:
            self.assertEqual(b.name, "<memory>")

    def test_name_spilled(self):
        with BytesIO(spill_bytes=5) as b:
            b.write(b"x" * 10)
            self.assertEqual(b.name, b.path)

    def test_fileno_memory_raises(self):
        with BytesIO(b"x") as b:
            self.assertRaises(OSError, b.fileno)

    def test_fileno_spilled_returns_fd(self):
        with BytesIO(spill_bytes=5) as b:
            b.write(b"x" * 10)
            fd = b.fileno()
            self.assertIsInstance(fd, int)
            self.assertEqual(os.fstat(fd).st_size, 10)

    def test_readline_basic(self):
        with BytesIO(b"line1\nline2\nline3") as b:
            self.assertEqual(b.readline(), b"line1\n")
            self.assertEqual(b.readline(), b"line2\n")
            self.assertEqual(b.readline(), b"line3")
            self.assertEqual(b.readline(), b"")

    def test_readline_with_limit(self):
        with BytesIO(b"longline\n") as b:
            self.assertEqual(b.readline(4), b"long")
            self.assertEqual(b.readline(), b"line\n")

    def test_readlines_basic(self):
        with BytesIO(b"a\nb\nc\n") as b:
            self.assertEqual(b.readlines(), [b"a\n", b"b\n", b"c\n"])

    def test_readlines_with_hint(self):
        with BytesIO(b"aaa\nbbb\nccc\n") as b:
            self.assertEqual(b.readlines(hint=4), [b"aaa\n"])

    def test_writelines(self):
        with BytesIO() as b:
            b.writelines([b"x", b"y", b"z"])
            self.assertEqual(b.to_bytes(), b"xyz")

    def test_readinto(self):
        with BytesIO(b"hello world") as b:
            buf = bytearray(5)
            n = b.readinto(buf)
            self.assertEqual(n, 5)
            self.assertEqual(bytes(buf), b"hello")

    def test_readinto_empty_buffer(self):
        with BytesIO(b"data") as b:
            self.assertEqual(b.readinto(bytearray(0)), 0)

    def test_readinto_short_at_eof(self):
        with BytesIO(b"abc") as b:
            buf = bytearray(10)
            n = b.readinto(buf)
            self.assertEqual(n, 3)
            self.assertEqual(bytes(buf[:3]), b"abc")

    def test_readinto1_alias(self):
        with BytesIO(b"abcdef") as b:
            buf = bytearray(3)
            self.assertEqual(b.readinto1(buf), 3)
            self.assertEqual(bytes(buf), b"abc")

    def test_iter_yields_lines(self):
        with BytesIO(b"x\ny\nz\n") as b:
            self.assertEqual(list(b), [b"x\n", b"y\n", b"z\n"])

    def test_write_str_encodes_utf8(self):
        with BytesIO() as b:
            n = b.write_str("héllo")
            encoded = "héllo".encode("utf-8")
            self.assertEqual(b.to_bytes(), encoded)
            self.assertEqual(n, len(encoded))

    def test_write_dispatches_str(self):
        with BytesIO() as b:
            b.write("abc")
            self.assertEqual(b.to_bytes(), b"abc")

    def test_write_filelike_drains(self):
        with BytesIO() as b:
            src = stdio.BytesIO(b"streamed in")
            b.write(src)
            self.assertEqual(b.to_bytes(), b"streamed in")

    def test_write_linebreak_default(self):
        with BytesIO() as b:
            b.write(b"line")
            b.write_linebreak()
            self.assertEqual(b.to_bytes(), b"line\n")

    def test_write_linebreak_custom(self):
        with BytesIO() as b:
            b.write(b"line")
            b.write_linebreak("\r\n")
            self.assertEqual(b.to_bytes(), b"line\r\n")


# ==========================================================================
# 7. CURSORLESS I/O
# ==========================================================================


class TestCursorlessIO(unittest.TestCase):
    def test_pread_pwrite_memory(self):
        with BytesIO() as b:
            b.pwrite(memoryview(b"hello"), 0)
            b.pwrite(memoryview(b"WORLD"), 6)
            b.pwrite(memoryview(b" "), 5)
            self.assertEqual(b.size, 11)
            self.assertEqual(b.pread(11, 0), b"hello WORLD")

    def test_pread_pwrite_spilled(self):
        with BytesIO(spill_bytes=5) as b:
            b.pwrite(memoryview(b"x" * 50), 0)
            self.assertTrue(b.spilled)
            self.assertEqual(b.pread(50, 0), b"x" * 50)

    def test_pread_negative_pos_raises(self):
        with BytesIO(b"x") as b:
            self.assertRaises(ValueError, b.pread, 1, -1)

    def test_pwrite_negative_pos_raises(self):
        with BytesIO() as b:
            self.assertRaises(ValueError, b.pwrite, memoryview(b"x"), -1)

    def test_pread_zero_or_negative_n_returns_empty(self):
        with BytesIO(b"data") as b:
            self.assertEqual(b.pread(0, 0), b"")
            self.assertEqual(b.pread(-1, 0), b"")

    def test_pread_past_end_returns_partial(self):
        with BytesIO(b"abc") as b:
            self.assertEqual(b.pread(10, 0), b"abc")
            self.assertEqual(b.pread(10, 5), b"")

    def test_pwrite_does_not_move_cursor(self):
        with BytesIO(b"abcdef") as b:
            b.seek(3)
            b.pwrite(memoryview(b"XX"), 0)
            self.assertEqual(b.tell(), 3)
            self.assertEqual(b.read(), b"def")

    def test_pread_does_not_move_cursor(self):
        with BytesIO(b"abcdef") as b:
            b.seek(2)
            self.assertEqual(b.pread(2, 0), b"ab")
            self.assertEqual(b.tell(), 2)


# ==========================================================================
# 8. STRUCTURED BINARY I/O
# ==========================================================================


class TestStructuredIO(unittest.TestCase):
    def test_int8_roundtrip(self):
        with BytesIO() as b:
            b.write_int8(-5)
            b.seek(0)
            self.assertEqual(b.read_int8(), -5)

    def test_uint8_roundtrip(self):
        with BytesIO() as b:
            b.write_uint8(200)
            b.seek(0)
            self.assertEqual(b.read_uint8(), 200)

    def test_int16_int32_int64_roundtrip(self):
        with BytesIO() as b:
            b.write_int16(-300)
            b.write_int32(-100000)
            b.write_int64(-1234567890123)
            b.seek(0)
            self.assertEqual(b.read_int16(), -300)
            self.assertEqual(b.read_int32(), -100000)
            self.assertEqual(b.read_int64(), -1234567890123)

    def test_uint_variants_roundtrip(self):
        with BytesIO() as b:
            b.write_uint16(60000)
            b.write_uint32(4_000_000_000)
            b.write_uint64(10**18)
            b.seek(0)
            self.assertEqual(b.read_uint16(), 60000)
            self.assertEqual(b.read_uint32(), 4_000_000_000)
            self.assertEqual(b.read_uint64(), 10**18)

    def test_f32_roundtrip(self):
        with BytesIO() as b:
            b.write_f32(1.5)
            b.seek(0)
            self.assertAlmostEqual(b.read_f32(), 1.5, places=6)

    def test_f64_roundtrip(self):
        with BytesIO() as b:
            b.write_f64(3.14159265358979)
            b.seek(0)
            self.assertAlmostEqual(b.read_f64(), 3.14159265358979, places=15)

    def test_bool_roundtrip(self):
        with BytesIO() as b:
            b.write_bool(True)
            b.write_bool(False)
            b.seek(0)
            self.assertIs(b.read_bool(), True)
            self.assertIs(b.read_bool(), False)

    def test_bytes_u32_roundtrip(self):
        with BytesIO() as b:
            b.write_bytes_u32(b"variable length data")
            b.seek(0)
            self.assertEqual(b.read_bytes_u32(), b"variable length data")

    def test_str_u32_roundtrip(self):
        with BytesIO() as b:
            b.write_str_u32("héllo wörld")
            b.seek(0)
            self.assertEqual(b.read_str_u32(), "héllo wörld")

    def test_read_exact_eof_raises(self):
        with BytesIO(b"\x01") as b:
            b.read_int8()
            self.assertRaises(EOFError, b.read_int8)

    def test_struct_roundtrip_in_spilled(self):
        """Verify struct helpers work over spilled backing."""
        with BytesIO(spill_bytes=10) as b:
            b.write(b"x" * 100)  # force spill
            b.seek(0)
            b.write_int32(42)
            b.write_int64(-99)
            b.seek(0)
            self.assertEqual(b.read_int32(), 42)
            self.assertEqual(b.read_int64(), -99)


# ==========================================================================
# 9. HASHING
# ==========================================================================


class TestHashing(unittest.TestCase):
    def test_xxh3_64_memory(self):
        try:
            import xxhash
        except ImportError:
            self.skipTest("xxhash not installed")
        with BytesIO(b"hash this") as b:
            self.assertEqual(
                b.xxh3_64().intdigest(),
                xxhash.xxh3_64(b"hash this").intdigest(),
            )

    def test_xxh3_int64_in_signed_range(self):
        try:
            import xxhash  # noqa: F401
        except ImportError:
            self.skipTest("xxhash not installed")
        with BytesIO(b"signed-hash") as b:
            sig = b.xxh3_int64()
            self.assertGreaterEqual(sig, -(2**63))
            self.assertLess(sig, 2**63)

    def test_blake3_memory(self):
        try:
            import blake3 as _b3
        except ImportError:
            self.skipTest("blake3 not installed")
        with BytesIO(b"blake3 me") as b:
            self.assertEqual(
                b.blake3().hexdigest(),
                _b3.blake3(b"blake3 me").hexdigest(),
            )

    def test_blake3_spilled_uses_mmap(self):
        try:
            import blake3 as _b3
        except ImportError:
            self.skipTest("blake3 not installed")
        payload = b"x" * 200
        with BytesIO(payload, spill_bytes=10) as b:
            self.assertEqual(
                b.blake3().hexdigest(), _b3.blake3(payload).hexdigest()
            )


# ==========================================================================
# 10. MEDIA TYPE
# ==========================================================================


class TestMediaType(unittest.TestCase):
    def test_explicit_setting(self):
        mt = MediaType("application/json")
        with BytesIO(b"{}", media_type=mt) as b:
            self.assertEqual(b.media_type, mt)

    def test_fallback_when_unknown(self):
        with BytesIO(b"random") as b:
            self.assertIsNotNone(b.media_type)

    def test_with_media_type_empty_buffer_allowed(self):
        with BytesIO() as b:
            mt = MediaType("application/json")
            b.with_media_type(mt)
            self.assertEqual(b._media_type, mt)

    def test_with_media_type_codec_change_allowed(self):
        with BytesIO(b"x") as b:
            mt1 = MediaType("application/json")
            b.with_media_type(mt1)
            mt2 = MediaType("application/json", codec="gzip")
            b.with_media_type(mt2)
            self.assertEqual(b._media_type, mt2)

    def test_with_media_type_mime_change_rejected(self):
        with BytesIO(b"non-empty") as b:
            b.with_media_type(MediaType("application/json"))
            self.assertRaises(
                ValueError, b.with_media_type, MediaType("application/parquet")
            )

    def test_with_media_type_copy_returns_new(self):
        with BytesIO(b"x") as b:
            mt1 = MediaType("application/json")
            b.with_media_type(mt1)
            mt2 = MediaType("application/json", codec="gzip")
            dup = b.with_media_type(mt2, copy=True)
            try:
                self.assertIsNot(dup, b)
                self.assertEqual(dup._media_type, mt2)
                self.assertEqual(b._media_type, mt1)
            finally:
                dup.close()


# ==========================================================================
# 11. WRITE_INTO / TO_PATH
# ==========================================================================


class TestWriteInto(_MixinTempFiles, unittest.TestCase):
    def test_write_into_path_from_memory(self):
        out = self.make_tempfile()
        os.remove(out)  # write_into creates fresh
        with BytesIO(b"contents") as b:
            self.assertEqual(b.write_into(out), 8)
        with open(out, "rb") as fh:
            self.assertEqual(fh.read(), b"contents")

    def test_write_into_path_from_spilled(self):
        out = self.make_tempfile()
        os.remove(out)
        with BytesIO(spill_bytes=10) as b:
            b.write(b"x" * 100)
            b.write_into(out)
        with open(out, "rb") as fh:
            self.assertEqual(fh.read(), b"x" * 100)

    def test_write_into_filelike(self):
        sink = stdio.BytesIO()
        with BytesIO(b"to filelike") as b:
            b.write_into(sink)
        self.assertEqual(sink.getvalue(), b"to filelike")

    def test_write_into_overwrite_false_raises(self):
        out = self.make_tempfile(b"existing")
        with BytesIO(b"x") as b:
            self.assertRaises(
                FileExistsError, b.write_into, out, overwrite=False
            )

    def test_write_into_overwrite_true_replaces(self):
        out = self.make_tempfile(b"old")
        with BytesIO(b"new") as b:
            b.write_into(out, overwrite=True)
        with open(out, "rb") as fh:
            self.assertEqual(fh.read(), b"new")

    def test_write_into_unwritable_sink_raises(self):
        with BytesIO(b"x") as b:
            self.assertRaises(ValueError, b.write_into, NotWritableSink())

    def test_write_into_non_path_non_filelike_raises(self):
        with BytesIO(b"x") as b:
            self.assertRaises(TypeError, b.write_into, 12345)

    def test_to_path_returns_string(self):
        out = self.make_tempfile()
        os.remove(out)
        with BytesIO(b"hello") as b:
            self.assertEqual(b.to_path(out), out)
        with open(out, "rb") as fh:
            self.assertEqual(fh.read(), b"hello")


# ==========================================================================
# 12. VIEWS / READERS / TO_BYTES / TO_BASE64 / DECODE
# ==========================================================================


class TestViewsAndReaders(unittest.TestCase):
    def test_to_bytes_memory(self):
        with BytesIO(b"abc") as b:
            self.assertEqual(b.to_bytes(), b"abc")

    def test_to_bytes_spilled(self):
        with BytesIO(b"y" * 200, spill_bytes=10) as b:
            self.assertEqual(b.to_bytes(), b"y" * 200)

    def test_to_bytes_empty(self):
        with BytesIO() as b:
            self.assertEqual(b.to_bytes(), b"")

    def test_getvalue_alias(self):
        with BytesIO(b"x") as b:
            self.assertEqual(b.getvalue(), b.to_bytes())

    def test_decode(self):
        with BytesIO("héllo".encode("utf-8")) as b:
            self.assertEqual(b.decode(), "héllo")

    def test_decode_empty(self):
        with BytesIO() as b:
            self.assertEqual(b.decode(), "")

    def test_to_base64_urlsafe(self):
        import base64 as _b64

        with BytesIO(b"hello") as b:
            self.assertEqual(b.to_base64(), _b64.urlsafe_b64encode(b"hello").decode())

    def test_to_base64_standard(self):
        import base64 as _b64

        with BytesIO(b"hello") as b:
            self.assertEqual(
                b.to_base64(urlsafe=False), _b64.b64encode(b"hello").decode()
            )

    def test_memoryview_memory(self):
        with BytesIO(b"abc") as b:
            mv = b.memoryview()
            self.assertEqual(bytes(mv), b"abc")

    def test_memoryview_spilled_uses_mmap(self):
        payload = b"x" * 200
        with BytesIO(payload, spill_bytes=10) as b:
            mv = b.memoryview()
            self.assertEqual(bytes(mv), payload)

    def test_memoryview_empty(self):
        with BytesIO() as b:
            self.assertEqual(bytes(b.memoryview()), b"")

    def test_head(self):
        with BytesIO(b"abcdefgh") as b:
            self.assertEqual(b.head(3), b"abc")
            self.assertEqual(b.head(0), b"")
            self.assertEqual(b.head(100), b"abcdefgh")  # capped at size

    def test_exists(self):
        with BytesIO() as b1:
            self.assertFalse(b1.exists())
        with BytesIO(b"x") as b2:
            self.assertTrue(b2.exists())


# ==========================================================================
# 13. SEEK EDGE CASES
# ==========================================================================


class TestSeekEdgeCases(unittest.TestCase):
    def test_seek_set_positive(self):
        with BytesIO(b"abcde") as b:
            b.seek(2)
            self.assertEqual(b.read(), b"cde")

    def test_seek_set_negative_from_end(self):
        with BytesIO(b"abcde") as b:
            b.seek(-1)
            self.assertEqual(b.read(), b"e")
            b.seek(-3)
            self.assertEqual(b.read(), b"cde")

    def test_seek_cur(self):
        with BytesIO(b"abcdef") as b:
            b.seek(2)
            b.seek(2, stdio.SEEK_CUR)
            self.assertEqual(b.read(), b"ef")

    def test_seek_cur_clamp_negative(self):
        with BytesIO(b"abcdef") as b:
            b.seek(2)
            b.seek(-100, stdio.SEEK_CUR)
            self.assertEqual(b.tell(), 0)

    def test_seek_end(self):
        with BytesIO(b"abcdef") as b:
            b.seek(0, stdio.SEEK_END)
            self.assertEqual(b.tell(), 6)
            b.seek(-2, stdio.SEEK_END)
            self.assertEqual(b.read(), b"ef")

    def test_seek_end_zipfile_pattern(self):
        """zipfile probes empty buffers with seek(-22, SEEK_END)."""
        with BytesIO(b"") as b:
            b.seek(-22, stdio.SEEK_END)
            self.assertEqual(b.tell(), 0)

    def test_seek_invalid_whence(self):
        with BytesIO(b"x") as b:
            self.assertRaises(ValueError, b.seek, 0, 99)

    def test_seek_returns_new_pos(self):
        with BytesIO(b"abcdef") as b:
            self.assertEqual(b.seek(3), 3)
            self.assertEqual(b.seek(0, stdio.SEEK_END), 6)


# ==========================================================================
# 14. PICKLING
# ==========================================================================


class TestPickling(unittest.TestCase):
    def test_small_buffer_roundtrip(self):
        with BytesIO(b"small data") as b:
            blob = pickle.dumps(b)
        rb = pickle.loads(blob)
        try:
            self.assertEqual(rb.to_bytes(), b"small data")
        finally:
            rb.close()


# ==========================================================================
# 15. IDENTITY / DUNDERS
# ==========================================================================


class TestIdentityDunders(unittest.TestCase):
    def test_len_matches_size(self):
        with BytesIO(b"abcde") as b:
            self.assertEqual(len(b), 5)

    def test_bool_always_true_empty(self):
        """File-like protocol — empty buffer must be truthy."""
        with BytesIO() as b:
            self.assertIs(bool(b), True)

    def test_bool_always_true_nonempty(self):
        with BytesIO(b"x") as b:
            self.assertIs(bool(b), True)

    def test_bytes_dunder(self):
        with BytesIO(b"abc") as b:
            self.assertEqual(bytes(b), b"abc")

    def test_repr_memory(self):
        with BytesIO(b"x") as b:
            r = repr(b)
            self.assertIn("memory", r)
            self.assertIn("size=1", r)

    def test_repr_empty_constant(self):
        with BytesIO() as b:
            r = repr(b)
            # Empty buffer — _buf is an empty bytearray, not None,
            # so it shows memory mode with size=0.
            self.assertTrue("memory" in r or "empty" in r, r)

    def test_repr_spilled(self):
        with BytesIO(spill_bytes=2) as b:
            b.write(b"x" * 100)
            r = repr(b)
            self.assertIn("spilled", r)
            self.assertIn("path=", r)

    def test_iter_protocol(self):
        with BytesIO(b"a\nb\n") as b:
            self.assertEqual(list(b), [b"a\n", b"b\n"])

    def test_tell_returns_int(self):
        with BytesIO(b"abc") as b:
            b.seek(2)
            self.assertEqual(b.tell(), 2)
            self.assertIsInstance(b.tell(), int)


# ==========================================================================
# 16. INTEGRATION
# ==========================================================================


class TestIntegration(_MixinTempFiles, unittest.TestCase):
    def test_round_trip_memory_to_disk_to_memory(self):
        out = self.make_tempfile()
        os.remove(out)
        with BytesIO(b"alpha beta gamma") as src:
            src.write_into(out)
        with open(out, "rb") as fh:
            self.assertEqual(fh.read(), b"alpha beta gamma")

    def test_write_grow_seek_overwrite_correct(self):
        with BytesIO() as b:
            b.write(b"original content here")  # 21 bytes
            b.seek(9)
            b.write(b"REPLACED")  # 8 bytes at pos 9..16
            self.assertEqual(b.size, 21)
            self.assertEqual(b.to_bytes(), b"original REPLACEDhere")

    def test_spill_then_truncate_then_read(self):
        with BytesIO(spill_bytes=10) as b:
            b.write(b"abcdefghijklmnop" * 4)
            self.assertTrue(b.spilled)
            b.truncate(20)
            b.seek(0)
            self.assertEqual(b.read(), b"abcdefghijklmnopabcd")

    def test_two_independent_buffers_dont_share_spills(self):
        b1 = BytesIO(spill_bytes=10)
        b2 = BytesIO(spill_bytes=10)
        f2 = None
        try:
            b1.write(b"x" * 50)
            b2.write(b"y" * 50)
            f1, f2 = b1.path, b2.path
            self.assertNotEqual(f1, f2)
            self.assertTrue(os.path.exists(f1))
            self.assertTrue(os.path.exists(f2))

            b1.close()
            self.assertFalse(os.path.exists(f1))
            self.assertTrue(os.path.exists(f2), "b1 close doesn't affect b2")
        finally:
            b2.close()
        self.assertFalse(os.path.exists(f2))

    def test_BytesIO_src_independent_after_close(self):
        """Constructing from a BytesIO src deep-copies; src.close shouldn't
        invalidate the new instance."""
        src = BytesIO(spill_bytes=10)
        src.write(b"x" * 50)
        new = BytesIO(src)
        try:
            src.close()
            self.assertEqual(new.to_bytes(), b"x" * 50)
        finally:
            new.close()

    def test_full_lifecycle_use_then_close_spilled(self):
        b = BytesIO(spill_bytes=20)
        try:
            for chunk in [b"a" * 30, b"b" * 30, b"c" * 30]:
                b.write(chunk)
            self.assertTrue(b.spilled)
            fname = b.path
            b.seek(0)
            self.assertEqual(b.read(), b"a" * 30 + b"b" * 30 + b"c" * 30)
        finally:
            b.close()
        self.assertFalse(os.path.exists(fname))

    def test_pwrite_at_arbitrary_position_spilled(self):
        with BytesIO(spill_bytes=5) as b:
            b.write(b"x" * 50)  # spill
            b.pwrite(memoryview(b"INSERTED"), 20)
            self.assertEqual(b.pread(8, 20), b"INSERTED")

    def test_chained_spill_replace_lifecycle(self):
        """Repeatedly replace a spilled buffer; verify each old spill
        unlinks and the final state holds the most-recent payload."""
        b = BytesIO(spill_bytes=10)
        old_paths = []
        try:
            b.write(b"x" * 50)
            old_paths.append(b.path)
            for i in range(5):
                with BytesIO(b"y" * 50, spill_bytes=10) as payload:
                    b.replace_with_payload(payload)
                old_paths.append(b.path)
                # All but the latest path should be gone
                for p in old_paths[:-1]:
                    self.assertFalse(
                        os.path.exists(p),
                        f"old spill {p} should be unlinked after replace #{i}",
                    )
                self.assertTrue(os.path.exists(old_paths[-1]))
            self.assertEqual(b.size, 50)
        finally:
            b.close()
        for p in old_paths:
            self.assertFalse(os.path.exists(p))


# ==========================================================================
# Entry point
# ==========================================================================


if __name__ == "__main__":
    unittest.main(verbosity=2)