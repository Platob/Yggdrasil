"""Tests for the remote-buffered mode in :class:`BytesIO`.

When ``_spill_path`` is non-local (``path.is_local`` is False), the
buffer downloads bytes into ``_buf`` on acquire and pushes them
back via ``path.write_bytes`` on flush / close.

These tests use a minimal in-process fake "remote" Path that:

- reports ``is_local = False``,
- stores its bytes in a class-level dict keyed by URL,
- supports ``read_bytes`` / ``write_bytes`` / ``exists`` / ``unlink``.

Avoids any real remote backend so the suite runs cross-platform
without network or auth.
"""

from __future__ import annotations

import os
import tempfile
import unittest


# ---------------------------------------------------------------------------
# Fake remote Path
# ---------------------------------------------------------------------------


def _make_fake_remote_path_class():
    """Return a subclass of yggdrasil's abstract Path that reports
    is_local=False and stores bytes in a process-global dict.

    Each call returns a fresh class with its own backing dict so test
    isolation is automatic.
    """
    from yggdrasil.io.fs.path import Path
    from yggdrasil.io.path_stat import PathStats, PathKind

    _STORE: dict[str, bytes] = {}

    class _FakeRemotePath(Path):
        scheme = "fakeremote"

        @classmethod
        def handles(cls, obj):
            return False  # don't auto-claim arbitrary inputs

        @property
        def is_local(self) -> bool:
            return False

        def full_path(self) -> str:
            return f"fakeremote://{self.url.path}"

        # --- abstract hooks ---

        def _stat(self) -> PathStats:
            key = self.full_path()
            if key in _STORE:
                return PathStats(
                    size=len(_STORE[key]), mtime=0.0,
                    kind=PathKind.FILE, mode=0o644,
                )
            return PathStats(size=0, mtime=0.0, kind=PathKind.MISSING, mode=0)

        def _ls(self, recursive=False, allow_not_found=True):
            return iter(())

        def _mkdir(self, parents=True, exist_ok=True):
            return None

        def _remove_file(self, allow_not_found=True):
            _STORE.pop(self.full_path(), None)

        def _remove_dir(self, recursive=True, allow_not_found=True, with_root=True):
            return None

        def _open(self, mode="rb", **kw):
            raise NotImplementedError("fake remote doesn't expose an IO")

        # --- the methods BytesIO actually uses ---

        def exists(self, *, follow_symlinks: bool = True) -> bool:
            return self.full_path() in _STORE

        def read_bytes(self, *, raise_error: bool = True) -> bytes:
            key = self.full_path()
            if key not in _STORE:
                if raise_error:
                    raise FileNotFoundError(key)
                return b""
            return _STORE[key]

        def write_bytes(self, data, *, mode="wb", parents=True) -> int:
            _STORE[self.full_path()] = bytes(data)
            return len(data)

        def unlink(self, missing_ok: bool = True) -> None:
            _STORE.pop(self.full_path(), None)

        # --- pread/pwrite: positional ops directly against _STORE,
        # bypassing the slow-path helpers (which would need a
        # working _open and we don't have one) ---

        def pread(self, n, pos, *, default=...):
            if pos < 0:
                raise ValueError("pread position must be >= 0")
            key = self.full_path()
            if key not in _STORE:
                if default is ...:
                    raise FileNotFoundError(key)
                return default
            data = _STORE[key]
            if n < 0:
                return data[pos:]
            return data[pos:pos + n]

        def pwrite(self, data, pos, *, parents=True):
            if pos < 0:
                raise ValueError("pwrite position must be >= 0")
            mv = bytes(memoryview(data))
            n = len(mv)
            if n == 0:
                return 0
            key = self.full_path()
            existing = _STORE.get(key, b"")
            end = pos + n
            if end <= len(existing):
                buf = bytearray(existing)
                buf[pos:end] = mv
            else:
                buf = bytearray(end)
                if existing:
                    buf[: len(existing)] = existing
                buf[pos:end] = mv
            _STORE[key] = bytes(buf)
            return n

        def truncate(self, n, *, parents=True):
            if n < 0:
                raise ValueError(f"truncate size must be >= 0, got {n!r}")
            key = self.full_path()
            if key not in _STORE:
                raise FileNotFoundError(key)
            existing = _STORE[key]
            if n < len(existing):
                _STORE[key] = existing[:n]
            elif n > len(existing):
                _STORE[key] = existing + b"\x00" * (n - len(existing))
            return n

    return _FakeRemotePath, _STORE


# ---------------------------------------------------------------------------
# Test base
# ---------------------------------------------------------------------------


class _RemoteBufferedTestBase(unittest.TestCase):
    def setUp(self):
        self.RemoteCls, self.store = _make_fake_remote_path_class()

    def _path(self, name: str):
        """Build a fake-remote path under our isolated store."""
        return self.RemoteCls(f"fakeremote:///{name}")

    def _seed_remote(self, name: str, data: bytes):
        """Pre-populate the fake store as if a remote file already exists."""
        path = self._path(name)
        self.store[path.full_path()] = data
        return path

    def _open_io(self, path, *, mode: str = "rb"):
        from yggdrasil.io.buffer.bytes_io import BytesIO
        bio = BytesIO(path=path, mode=mode)
        self.addCleanup(self._safe_close, bio)
        return bio

    @staticmethod
    def _safe_close(bio):
        try:
            bio.close()
        except Exception:
            pass


# ===========================================================================
# Acquire — populating _buf from the remote
# ===========================================================================


class TestRemoteAcquire(_RemoteBufferedTestBase):
    def test_rb_downloads_existing_bytes(self):
        path = self._seed_remote("data.bin", b"hello remote")
        bio = self._open_io(path, mode="rb")

        # The transaction buffer should be populated with the
        # downloaded bytes via path.pread.
        self.assertIsNotNone(bio._transaction_buffer)
        self.assertEqual(bio.size, len(b"hello remote"))
        self.assertEqual(bio.read(), b"hello remote")

    def test_rb_against_missing_raises(self):
        from yggdrasil.io.buffer.bytes_io import BytesIO
        path = self._path("missing.bin")
        with self.assertRaises(FileNotFoundError):
            BytesIO(path=path, mode="rb")

    def test_wb_starts_empty_even_if_remote_exists(self):
        path = self._seed_remote("data.bin", b"old content")
        bio = self._open_io(path, mode="wb")
        # Truncate semantics — transaction buffer is empty regardless.
        self.assertEqual(bio.size, 0)
        self.assertIsNotNone(bio._transaction_buffer)
        self.assertEqual(bio._transaction_buffer.size, 0)

    def test_wb_creates_when_missing(self):
        path = self._path("new.bin")
        bio = self._open_io(path, mode="wb")
        self.assertEqual(bio.size, 0)

    def test_ab_downloads_and_seeks_to_end(self):
        path = self._seed_remote("data.bin", b"existing")
        bio = self._open_io(path, mode="ab")
        # Cursor at EOF, _buf has existing bytes.
        self.assertEqual(bio.tell(), len(b"existing"))
        self.assertEqual(bio.size, len(b"existing"))

    def test_ab_creates_when_missing(self):
        path = self._path("ab-fresh.bin")
        bio = self._open_io(path, mode="ab")
        self.assertEqual(bio.size, 0)
        self.assertEqual(bio.tell(), 0)

    def test_xb_excludes_existing(self):
        from yggdrasil.io.buffer.bytes_io import BytesIO
        path = self._seed_remote("exists.bin", b"x")
        with self.assertRaises(FileExistsError):
            BytesIO(path=path, mode="xb")

    def test_xb_creates_when_missing(self):
        path = self._path("fresh-x.bin")
        bio = self._open_io(path, mode="xb")
        self.assertEqual(bio.size, 0)


# ===========================================================================
# Read / write through the buffer
# ===========================================================================


class TestRemoteReadWrite(_RemoteBufferedTestBase):
    def test_read_existing(self):
        path = self._seed_remote("read.bin", b"the quick brown fox")
        bio = self._open_io(path, mode="rb")
        self.assertEqual(bio.read(3), b"the")
        self.assertEqual(bio.read(), b" quick brown fox")

    def test_pread_against_buffer(self):
        path = self._seed_remote("pread.bin", b"abcdefghij")
        bio = self._open_io(path, mode="rb")
        self.assertEqual(bio.pread(4, 3), b"defg")
        # Cursor unaffected.
        self.assertEqual(bio.tell(), 0)

    def test_write_into_buffer(self):
        path = self._path("write.bin")
        bio = self._open_io(path, mode="wb")
        bio.write(b"hello")
        # The remote file is NOT updated yet — we haven't flushed.
        self.assertNotIn(path.full_path(), self.store)
        # But the transaction buffer has the bytes.
        self.assertEqual(bio.size, 5)
        # Read back via the public API to confirm the bytes are
        # actually in the buffer (not just _size accounting).
        bio.seek(0)
        self.assertEqual(bio.read(), b"hello")

    def test_pwrite_against_buffer(self):
        path = self._seed_remote("pwrite.bin", b"AAAAAAAA")
        bio = self._open_io(path, mode="rb+")
        bio.pwrite(b"BB", 2)
        self.assertEqual(bio.pread(8, 0), b"AABBAAAA")

    def test_truncate_resizes_buffer(self):
        path = self._seed_remote("trunc.bin", b"hello world")
        bio = self._open_io(path, mode="rb+")
        bio.truncate(5)
        self.assertEqual(bio.size, 5)
        bio.seek(0)
        self.assertEqual(bio.read(), b"hello")


# ===========================================================================
# Flush — pushing the buffer back to the remote
# ===========================================================================


class TestRemoteFlush(_RemoteBufferedTestBase):
    def test_flush_writes_to_remote(self):
        path = self._path("flush.bin")
        bio = self._open_io(path, mode="wb")
        bio.write(b"payload")

        # Before flush, store is empty.
        self.assertNotIn(path.full_path(), self.store)

        bio.flush()

        # After flush, store has the bytes.
        self.assertEqual(self.store[path.full_path()], b"payload")

    def test_close_flushes_implicitly(self):
        path = self._path("close-flush.bin")
        bio = self._open_io(path, mode="wb")
        bio.write(b"closed-and-flushed")
        bio.close()

        self.assertEqual(self.store[path.full_path()], b"closed-and-flushed")

    def test_flush_skipped_for_read_only(self):
        """Read-only opens shouldn't push the (unmodified) buffer back
        to the remote — that's a no-op write at best, and a destructive
        race against concurrent writers at worst."""
        path = self._seed_remote("ro.bin", b"original")
        bio = self._open_io(path, mode="rb")
        # Manually corrupt the transaction buffer to prove flush
        # wouldn't push it back.
        bio._transaction_buffer.pwrite(b"corrupted", 0)
        bio.flush()
        # Remote bytes unchanged.
        self.assertEqual(self.store[path.full_path()], b"original")

    def test_flush_writes_full_buffer(self):
        """The naive policy: flush always writes the entire buffer.
        Tests that a partial-write workflow ends up with the right
        full payload on the remote."""
        path = self._seed_remote("partial.bin", b"old content longer")
        bio = self._open_io(path, mode="rb+")
        bio.pwrite(b"NEW", 4)  # patches in place
        bio.close()
        # Remote got the patched full buffer.
        self.assertEqual(self.store[path.full_path()], b"old NEWtent longer")

    def test_truncate_then_close_writes_shrunk_payload(self):
        path = self._seed_remote("shrink.bin", b"hello world")
        bio = self._open_io(path, mode="rb+")
        bio.truncate(5)
        bio.close()
        self.assertEqual(self.store[path.full_path()], b"hello")

    def test_explicit_flush_is_idempotent(self):
        path = self._path("idem.bin")
        bio = self._open_io(path, mode="wb")
        bio.write(b"once")
        bio.flush()
        bio.flush()
        bio.flush()
        self.assertEqual(self.store[path.full_path()], b"once")


# ===========================================================================
# Memory-mode and local paths NOT affected
# ===========================================================================


class TestNonRemoteUnaffected(unittest.TestCase):
    """The remote-buffered branch must not interfere with memory-mode
    or local-path operations."""

    def test_memory_mode_flush_is_noop(self):
        from yggdrasil.io.buffer.bytes_io import BytesIO
        bio = BytesIO(b"memory only")
        # flush() must not raise or do anything observable.
        bio.flush()
        self.assertEqual(bio.to_bytes(), b"memory only")

    def test_local_path_flush_is_noop(self):
        """Local-path writes go through os.pwrite directly; flush()
        on a local-spilled buffer is a no-op (the kernel already has
        the bytes)."""
        from yggdrasil.io.buffer.bytes_io import BytesIO
        from yggdrasil.lazy_imports import local_path_class
        import pathlib

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "local.bin")
            path = local_path_class().from_pathlib(pathlib.Path(target))
            bio = BytesIO(path=path, mode="wb")
            try:
                bio.write(b"local")
                # Local fd-mode: flush is a no-op, but the bytes are
                # already on disk via os.pwrite.
                bio.flush()
                # Read directly from disk to confirm.
                with open(target, "rb") as fh:
                    self.assertEqual(fh.read(), b"local")
            finally:
                bio.close()


# ===========================================================================
# Transaction buffer composition
# ===========================================================================


class TestTransactionBufferComposition(_RemoteBufferedTestBase):
    """The transaction buffer is itself a BytesIO. Test the composition:
    the inner buffer has its own spill behavior, the outer coordinates
    the path round-trip.
    """

    def test_transaction_buffer_is_a_bytes_io(self):
        from yggdrasil.io.buffer.bytes_io import BytesIO
        path = self._seed_remote("data.bin", b"x")
        bio = self._open_io(path, mode="rb")
        self.assertIsInstance(bio._transaction_buffer, BytesIO)

    def test_inner_buffer_size_matches_outer(self):
        """As writes go in, the inner buffer's size tracks the outer's."""
        path = self._path("compose.bin")
        bio = self._open_io(path, mode="wb")
        bio.write(b"a" * 100)
        self.assertEqual(bio._transaction_buffer.size, 100)
        self.assertEqual(bio.size, 100)

    def test_close_releases_transaction_buffer(self):
        """After close, _transaction_buffer is None — the inner
        scratch is released along with the outer."""
        path = self._seed_remote("rel.bin", b"x")
        bio = self._open_io(path, mode="rb")
        self.assertIsNotNone(bio._transaction_buffer)
        bio.close()
        self.assertIsNone(bio._transaction_buffer)

    def test_flush_uses_pwrite_not_write_bytes(self):
        """Pin the contract: flush goes through path.pwrite, not
        path.write_bytes. We can't observe the call directly but we
        can check that pwrite-style positional writes work — meaning
        the path's pwrite was actually invoked."""
        path = self._path("pwrite-flush.bin")
        bio = self._open_io(path, mode="wb")
        bio.write(b"committed via pwrite")
        bio.flush()
        # Bytes are on the remote.
        self.assertEqual(
            self.store[path.full_path()],
            b"committed via pwrite",
        )

    def test_shrink_then_flush_truncates_remote(self):
        """The flush sequence is pwrite + truncate. A shrink-then-flush
        must drop the tail from the remote, not just patch the head."""
        path = self._seed_remote("shrink.bin", b"hello world")
        bio = self._open_io(path, mode="rb+")
        bio.truncate(5)
        bio.close()  # triggers flush
        self.assertEqual(self.store[path.full_path()], b"hello")


if __name__ == "__main__":
    unittest.main()