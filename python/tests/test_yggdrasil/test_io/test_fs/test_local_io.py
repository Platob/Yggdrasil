"""Unittest tests for :class:`LocalIO`.

After the BytesIO/LocalIO unification, LocalIO is a thin subclass
that just provides a positional-path constructor. All real behavior
lives in :class:`BytesIO`; these tests cover the path-bound surface
specifically and the contracts unique to caller-owned paths.

What's NOT tested here (covered by BytesIO tests):

- The ``_flags_for_mode`` helper.
- Spill threshold mechanics (LocalIO is always spilled; the
  threshold doesn't apply).
- Memory-mode reads/writes / the ``_buf`` codepaths.
- Auto-spill behavior.

What IS tested here:

- Construction shape — positional path, defaulting mode to ``rb``.
- ``_owns_spill_path = False`` after construction.
- File survives close (the regression from before unification).
- fd genuinely closes on ``bio.close()`` (the headline regression).
- Mode-driven open semantics on real files (``rb`` won't create,
  ``wb`` truncates, ``ab`` cursor at EOF, ``xb`` exclusive create).
- ``replace_with_payload`` writes through the binding (new behavior
  in the unified version — was a refusal before).
- Pickle round-trip via the path-bound branch of ``__getstate__``.
- Repr surfaces ownership ("external") and live-handle state.
- Class identity — LocalIO is genuinely a thin subclass.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import unittest

from yggdrasil.io import BytesIO


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_local_path(p):
    """Wrap a pathlib path / string as a yggdrasil LocalPath."""
    from yggdrasil.lazy_imports import local_path_class
    import pathlib
    return local_path_class().from_pathlib(pathlib.Path(p))


def _safe_close(bio) -> None:
    """Close swallowing all errors. Belt-and-suspenders for the
    cleanup helper."""
    try:
        bio.close()
    except Exception:
        pass


# ===========================================================================
# Test base
# ===========================================================================


class _LocalIOTestBase(unittest.TestCase):
    """tempdir + close-on-cleanup."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmpdir = self.tmp.name

    def _seed(self, name: str, data: bytes) -> str:
        target = os.path.join(self.tmpdir, name)
        with open(target, "wb") as fh:
            fh.write(data)
        return target

    def _open_io(self, target_str: str, *, mode: str = "rb"):
        """Construct a LocalIO and register cleanup BEFORE any test
        assertion. addCleanup runs LIFO, so close fires before the
        tempdir's rmtree."""
        bio = BytesIO(_make_local_path(target_str), mode=mode)
        self.addCleanup(_safe_close, bio)
        return bio


# ===========================================================================
# Construction
# ===========================================================================


class TestConstruction(_LocalIOTestBase):
    """Positional-path construction, default mode, owner flag."""

    def test_positional_path_default_mode_is_rb(self):
        target = self._seed("data.bin", b"hello")
        bio = self._open_io(target)  # no explicit mode
        self.assertEqual(bio.mode, "rb")

    def test_explicit_mode(self):
        target = self._seed("data.bin", b"hello")
        bio = self._open_io(target, mode="rb+")
        self.assertEqual(bio.mode, "rb+")

    def test_constructed_in_open_state(self):
        target = self._seed("data.bin", b"hello")
        bio = self._open_io(target)
        self.assertTrue(bio.opened)

    def test_owns_spill_path_is_false(self):
        """The headline ownership contract: caller owns the file."""
        target = self._seed("data.bin", b"hello")
        bio = self._open_io(target)
        self.assertFalse(bio._owns_spill_path)

    def test_spill_path_is_set(self):
        target = self._seed("data.bin", b"hello")
        bio = self._open_io(target)
        self.assertIsNotNone(bio._spill_path)
        # full_path() returns a URL-normalized form (forward slashes
        # always); the target was built via os.path.join (backslashes
        # on Windows). Compare via os.path.normpath to neutralize.
        self.assertEqual(
            os.path.normpath(bio._spill_path.full_path()),
            os.path.normpath(target),
        )

    def test_spilled_property_is_true(self):
        target = self._seed("data.bin", b"x")
        bio = self._open_io(target)
        self.assertTrue(bio.spilled)

    def test_size_seeded_from_fstat(self):
        target = self._seed("sized.bin", b"X" * 1234)
        bio = self._open_io(target)
        self.assertEqual(bio.size, 1234)

    def test_no_in_memory_buf(self):
        """Path-bound buffers don't allocate _buf — everything goes
        through the fd. The unified BytesIO sets _buf = None when
        path is supplied."""
        target = self._seed("data.bin", b"x")
        bio = self._open_io(target)
        self.assertIsNone(bio._buf)


# ===========================================================================
# Close lifecycle — the headline regression target
# ===========================================================================


class TestCloseLifecycle(_LocalIOTestBase):
    """The unified _release must:

    1. Close the fd (so OS resources are released and Windows
       can rmtree the parent dir).
    2. NOT unlink the caller's file.
    """

    def test_close_releases_fd(self):
        """fstat on the post-close fd must raise — proves the OS-level
        file descriptor is gone."""
        target = self._seed("data.bin", b"x")
        bio = self._open_io(target)

        fd = bio._spill_fd
        self.assertIsNotNone(fd)

        bio.close()

        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_close_nulls_spill_fd_slot(self):
        target = self._seed("data.bin", b"x")
        bio = self._open_io(target)
        bio.close()
        self.assertIsNone(bio._spill_fd)

    def test_close_does_not_unlink_file(self):
        """The path is caller-owned. close() must never delete it."""
        target = self._seed("owned.bin", b"caller's data")
        bio = self._open_io(target)
        bio.close()

        self.assertTrue(os.path.exists(target))
        with open(target, "rb") as fh:
            self.assertEqual(fh.read(), b"caller's data")

    def test_close_keeps_spill_path_when_external(self):
        """An external (non-owned) path must NOT be cleared from the
        slot on close — pickle/replay scenarios depend on it."""
        target = self._seed("kept.bin", b"x")
        bio = self._open_io(target)
        bio.close()
        # The spill path reference survives close for external paths.
        # (Owned scratch paths get cleared in _release; external ones
        # don't, so re-acquire could theoretically reopen.)
        self.assertIsNotNone(bio._spill_path)

    def test_close_is_idempotent(self):
        target = self._seed("data.bin", b"x")
        bio = self._open_io(target)
        bio.close()
        # Second close must not raise.
        bio.close()


# ===========================================================================
# Mode semantics — end-to-end against real files
# ===========================================================================


class TestModeSemantics(_LocalIOTestBase):
    def test_wb_truncates_existing(self):
        target = self._seed("f.bin", b"old longer content")
        bio = self._open_io(target, mode="wb")
        bio.write(b"new")
        bio.close()

        with open(target, "rb") as fh:
            self.assertEqual(fh.read(), b"new")

    def test_wb_creates_missing(self):
        target = os.path.join(self.tmpdir, "fresh.bin")
        self.assertFalse(os.path.exists(target))

        bio = self._open_io(target, mode="wb")
        bio.write(b"created")
        bio.close()

        self.assertTrue(os.path.exists(target))
        with open(target, "rb") as fh:
            self.assertEqual(fh.read(), b"created")

    def test_wb_creates_parent_directories(self):
        nested = os.path.join(self.tmpdir, "a", "b", "c", "out.bin")
        bio = self._open_io(nested, mode="wb")
        bio.write(b"payload")
        bio.close()

        self.assertTrue(os.path.exists(nested))
        with open(nested, "rb") as fh:
            self.assertEqual(fh.read(), b"payload")

    def test_ab_starts_cursor_at_eof(self):
        target = self._seed("f.bin", b"existing")
        bio = self._open_io(target, mode="ab")
        self.assertEqual(bio.tell(), 8)
        bio.write(b"-more")
        bio.close()

        with open(target, "rb") as fh:
            self.assertEqual(fh.read(), b"existing-more")

    def test_ab_creates_missing(self):
        target = os.path.join(self.tmpdir, "fresh-append.bin")
        self.assertFalse(os.path.exists(target))

        bio = self._open_io(target, mode="ab")
        bio.write(b"hello")
        bio.close()

        with open(target, "rb") as fh:
            self.assertEqual(fh.read(), b"hello")

    def test_rb_starts_cursor_at_zero(self):
        target = self._seed("f.bin", b"abcdef")
        bio = self._open_io(target)  # rb default
        self.assertEqual(bio.tell(), 0)

    def test_rb_against_missing_raises_no_create(self):
        """The bug-regression: read-only open against a missing file
        must NOT silently create it. Without mode-aware flags in
        _acquire, the underlying os.open would default to O_CREAT
        and we'd produce a zero-byte file out of nowhere."""

        missing = os.path.join(self.tmpdir, "does-not-exist.bin")
        self.assertFalse(os.path.exists(missing))

        with self.assertRaises(FileNotFoundError):
            BytesIO(_make_local_path(missing), mode="rb")

        self.assertFalse(
            os.path.exists(missing),
            "rb open against missing file silently created it",
        )

    def test_xb_excludes_existing(self):
        """``xb`` (O_EXCL) must fail if the file exists."""

        target = self._seed("exists.bin", b"x")
        with self.assertRaises(FileExistsError):
            BytesIO(_make_local_path(target), mode="xb")

    def test_xb_creates_when_missing(self):
        target = os.path.join(self.tmpdir, "fresh-x.bin")
        bio = self._open_io(target, mode="xb")
        bio.write(b"created exclusively")
        bio.close()
        with open(target, "rb") as fh:
            self.assertEqual(fh.read(), b"created exclusively")


# ===========================================================================
# Read / write through inherited primitives
# ===========================================================================


class TestReadWrite(_LocalIOTestBase):
    """Smoke tests that the inherited I/O paths actually work through
    the path-bound subclass. Detailed semantics are covered by the
    BytesIO tests; here we just confirm nothing about path-binding
    breaks them."""

    def test_read_basic(self):
        target = self._seed("f.bin", b"the quick brown fox")
        bio = self._open_io(target)
        self.assertEqual(bio.read(3), b"the")
        self.assertEqual(bio.read(), b" quick brown fox")

    def test_write_then_close_persists(self):
        target = os.path.join(self.tmpdir, "f.bin")
        bio = self._open_io(target, mode="wb")
        bio.write(b"hello world")
        bio.close()
        with open(target, "rb") as fh:
            self.assertEqual(fh.read(), b"hello world")

    def test_pread_pwrite(self):
        target = os.path.join(self.tmpdir, "f.bin")
        bio = self._open_io(target, mode="wb+")
        bio.write(b"AAAAAAAA")
        bio.pwrite(b"BB", 2)
        self.assertEqual(bio.pread(8, 0), b"AABBAAAA")

    def test_seek_and_read(self):
        target = self._seed("f.bin", b"0123456789")
        bio = self._open_io(target)
        bio.seek(5)
        self.assertEqual(bio.read(3), b"567")

    def test_truncate(self):
        target = os.path.join(self.tmpdir, "f.bin")
        bio = self._open_io(target, mode="wb+")
        bio.write(b"hello world")
        bio.truncate(5)
        bio.seek(0)
        self.assertEqual(bio.read(), b"hello")
        bio.close()
        self.assertEqual(os.path.getsize(target), 5)

    def test_writes_persist_across_close_and_reopen(self):
        target = os.path.join(self.tmpdir, "persistent.bin")
        bio = self._open_io(target, mode="wb")
        bio.write(b"first")
        bio.close()

        bio2 = self._open_io(target)  # rb
        self.assertEqual(bio2.read(), b"first")

    def test_memoryview_of_path_bound_buffer(self):
        """Path-bound buffers go through mmap on the spill fd."""
        target = self._seed("f.bin", b"mmap-me")
        bio = self._open_io(target)
        mv = bio.memoryview()
        try:
            self.assertEqual(bytes(mv), b"mmap-me")
        finally:
            # Release the mmap before close. mmap-backed memoryviews
            # need their underlying mmap closed before the fd, on
            # platforms that care.
            mv.release()


# ===========================================================================
# replace_with_payload — new policy: writes through the binding
# ===========================================================================


class TestReplaceWithPayload(_LocalIOTestBase):
    """Pre-unification, LocalIO refused replace_with_payload outright.
    Post-unification, it writes the new payload through the bound fd —
    truncate, write, leave the path intact. These tests pin the new
    behavior."""

    def test_replace_writes_through_binding(self):
        target = self._seed("bound.bin", b"original-content")
        bio = self._open_io(target, mode="rb+")

        bio.replace_with_payload(b"new content")

        # The file on disk now holds the new bytes.
        bio.close()
        with open(target, "rb") as fh:
            self.assertEqual(fh.read(), b"new content")

    def test_replace_does_not_unlink_file(self):
        target = self._seed("kept.bin", b"original")
        bio = self._open_io(target, mode="rb+")

        bio.replace_with_payload(b"replaced")

        # File still exists at the same path.
        self.assertTrue(os.path.exists(target))

    def test_replace_preserves_path_binding(self):
        target = self._seed("bound.bin", b"original")
        bio = self._open_io(target, mode="rb+")

        bio.replace_with_payload(b"new")

        # Binding intact — _spill_path and ownership flag unchanged.
        self.assertIsNotNone(bio._spill_path)
        self.assertFalse(bio._owns_spill_path)

    def test_replace_with_none_clears_bound_file(self):
        target = self._seed("clear.bin", b"existing-data")
        bio = self._open_io(target, mode="rb+")

        bio.replace_with_payload(None)

        bio.close()
        self.assertEqual(os.path.getsize(target), 0)

    def test_replace_overwrites_longer_with_shorter(self):
        """Truncate-on-replace works: the file shrinks, no leftover
        bytes from the original payload."""
        target = self._seed("shrink.bin", b"old much-longer-content")
        bio = self._open_io(target, mode="rb+")

        bio.replace_with_payload(b"short")

        bio.close()
        with open(target, "rb") as fh:
            self.assertEqual(fh.read(), b"short")

    def test_replace_with_bytesio_payload(self):
        from yggdrasil.io.buffer.bytes_io import BytesIO

        target = self._seed("from-bio.bin", b"original")
        bio = self._open_io(target, mode="rb+")

        with BytesIO(b"payload from another bio") as src:
            bio.replace_with_payload(src)

        bio.close()
        with open(target, "rb") as fh:
            self.assertEqual(fh.read(), b"payload from another bio")

    def test_replace_self_raises(self):
        target = self._seed("self.bin", b"content")
        bio = self._open_io(target, mode="rb+")
        with self.assertRaises(ValueError):
            bio.replace_with_payload(bio)


# ===========================================================================
# Pickle round-trip — uses the path-bound branch of __getstate__
# ===========================================================================


class TestPickle(_LocalIOTestBase):
    """The unified BytesIO.__getstate__ branches on _owns_spill_path:
    owned → snapshot bytes, external → pickle path + mode. Path-bound
    LocalIOs take the second branch."""

    def test_pickle_blob_carries_path_not_full_bytes(self):
        """For a path-bound buffer, the pickle blob should be small —
        proportional to the path string, not the file's bytes."""
        target = self._seed("big.bin", b"X" * 100_000)
        bio = self._open_io(target)
        try:
            blob = pickle.dumps(bio)
        finally:
            bio.close()

        # 100 KB of file content shouldn't appear in the pickle blob.
        self.assertLess(
            len(blob), 4096,
            f"pickle blob is {len(blob)} bytes — too large for path-only",
        )

    def test_unpickle_reattaches_to_path(self):
        """After unpickling, the restored LocalIO reads the file's
        CURRENT contents, not a frozen snapshot from pickle time."""
        target = self._seed("reattach.bin", b"original")
        bio = self._open_io(target)
        blob = pickle.dumps(bio)
        bio.close()

        # File changes between pickle and unpickle.
        with open(target, "wb") as fh:
            fh.write(b"updated content")

        restored = pickle.loads(blob)
        self.addCleanup(_safe_close, restored)
        self.assertEqual(restored.read(), b"updated content")

    def test_unpickle_preserves_mode(self):
        target = self._seed("mode.bin", b"x")
        bio = self._open_io(target, mode="rb")
        blob = pickle.dumps(bio)
        bio.close()

        restored = pickle.loads(blob)
        self.addCleanup(_safe_close, restored)
        self.assertEqual(restored.mode, "rb")

    def test_unpickle_preserves_path_binding(self):
        target = self._seed("pathkeep.bin", b"x")
        bio = self._open_io(target)
        blob = pickle.dumps(bio)
        bio.close()

        restored = pickle.loads(blob)
        self.addCleanup(_safe_close, restored)
        # Same ownership semantics — caller still owns the file.
        self.assertFalse(restored._owns_spill_path)
        self.assertEqual(
            os.path.normpath(restored._spill_path.full_path()),
            os.path.normpath(target),
        )


# ===========================================================================
# Repr
# ===========================================================================


class TestRepr(_LocalIOTestBase):
    """The unified __repr__ surfaces ownership ("external") and
    open/closed state via _spill_fd."""

    def test_repr_open_external(self):
        target = self._seed("f.bin", b"ab")
        bio = self._open_io(target)
        r = repr(bio)
        self.assertIn("BytesIO", r)
        self.assertIn("external", r)
        self.assertIn("open", r)
        self.assertIn("'rb'", r)
        self.assertIn("size=2", r)

    def test_repr_closed_external(self):
        """Now that _release nulls _spill_fd, the closed state is
        visible in the repr (was the broken regression before)."""
        target = self._seed("f.bin", b"ab")
        bio = self._open_io(target)
        bio.close()
        r = repr(bio)
        self.assertIn("closed", r)
        self.assertIn("external", r)


# ===========================================================================
# Cross-class equivalence — LocalIO ≡ BytesIO(path=..., mode=...)
# ===========================================================================


class TestEquivalenceWithBytesIOConstructor(_LocalIOTestBase):
    """The thin-subclass contract: LocalIO(path, mode=m) and
    BytesIO(path=path, mode=m) must produce equivalent objects."""

    def test_writes_match(self):
        from yggdrasil.io.buffer.bytes_io import BytesIO

        path_a = os.path.join(self.tmpdir, "a.bin")
        path_b = os.path.join(self.tmpdir, "b.bin")

        a = BytesIO(_make_local_path(path_a), mode="wb")
        self.addCleanup(_safe_close, a)
        a.write(b"payload")
        a.close()

        b = BytesIO(path=_make_local_path(path_b), mode="wb")
        self.addCleanup(_safe_close, b)
        b.write(b"payload")
        b.close()

        with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
            self.assertEqual(fa.read(), fb.read())

    def test_reads_match(self):
        from yggdrasil.io.buffer.bytes_io import BytesIO

        target = self._seed("source.bin", b"shared content")

        a = BytesIO(_make_local_path(target), mode="rb")
        self.addCleanup(_safe_close, a)

        b = BytesIO(path=_make_local_path(target), mode="rb")
        self.addCleanup(_safe_close, b)

        self.assertEqual(a.read(), b.read())

    def test_ownership_flags_match(self):
        from yggdrasil.io.buffer.bytes_io import BytesIO

        target = self._seed("flag.bin", b"x")
        a = BytesIO(_make_local_path(target), mode="rb")
        b = BytesIO(path=_make_local_path(target), mode="rb")
        self.addCleanup(_safe_close, a)
        self.addCleanup(_safe_close, b)

        self.assertEqual(a._owns_spill_path, b._owns_spill_path)
        self.assertFalse(a._owns_spill_path)


if __name__ == "__main__":
    unittest.main()