"""Tests for ``DataIO._reading_context`` / ``_writing_context``.

These contexts are the central lifecycle helpers — every leaf
funnels through them. Bugs here surface as flaky open/close state,
wrong cursor positions on exit, missed dirty-marks, or mishandled
codec round-trips.

Tests are organized by the contract surface:

- Open/close lifecycle (was_opened propagation).
- Seek positioning (read_seek, write_seek, reset_seek).
- Truncate-before-write.
- Mark-dirty.
- Codec round-trip (sibling lifecycle + bytes replacement).
- Exception safety (sibling discarded on body raise; self untouched).
"""

from __future__ import annotations

from yggdrasil.io.buffer.primitive.csv_io import CsvIO


# ---------------------------------------------------------------------------
# Helper — build a fresh CsvIO with a known byte payload
# ---------------------------------------------------------------------------


def _csv_with_payload(payload: bytes) -> CsvIO:
    """CsvIO with *payload* already written into its buffer."""
    io = CsvIO()
    io.open()
    io.write(payload)
    io.seek(0)
    return io


# ===========================================================================
# Reading context — basic lifecycle
# ===========================================================================


class TestReadingContext:

    def test_opens_buffer_when_caller_left_it_closed(self):
        """If the buffer was closed at entry, the context opens it
        and closes it again on exit."""
        io = CsvIO()
        io.open()
        io.write(b"a,b\n1,2\n")
        io.close()

        assert not io.opened

        options = io.check_options()
        with io._reading_context(options) as target:
            assert target.opened
            assert target is io  # no codec → yields self

        assert not io.opened, "context should restore closed state"

    def test_leaves_buffer_open_when_caller_already_opened_it(self):
        """Opened-by-caller buffer stays opened on exit."""
        io = CsvIO()
        io.open()
        io.write(b"a,b\n1,2\n")
        io.seek(0)

        options = io.check_options()
        with io._reading_context(options) as target:
            assert target.opened

        assert io.opened, "context must not close a buffer it didn't open"
        io.close()

    def test_read_seek_none_leaves_cursor_alone(self):
        """``options.read_seek=None`` skips the seek entirely."""
        io = _csv_with_payload(b"a,b\n1,2\n")
        io.seek(4)

        options = io.check_options(read_seek=None)
        with io._reading_context(options) as target:
            assert target.tell() == 4
        io.close()

    def test_reset_seek_restores_position_when_buffer_was_open(self):
        """With ``reset_seek=True`` and the buffer pre-opened, the
        pre-entry cursor is restored on exit."""
        io = _csv_with_payload(b"a,b\n1,2\n")
        io.seek(4)

        options = io.check_options(reset_seek=True)
        with io._reading_context(options):
            io.seek(0)  # body moved cursor

        assert io.tell() == 4, "reset_seek should restore pre-entry position"
        io.close()

    def test_reset_seek_skipped_when_context_opened_buffer(self):
        """If the context opened the buffer, restoring a position is
        moot — the buffer is closed on exit anyway."""
        io = CsvIO()
        io.open()
        io.write(b"a,b\n1,2\n")
        io.close()

        # Should not raise — if reset_seek tried to seek a closed
        # buffer this test would fail.
        options = io.check_options(reset_seek=True)
        with io._reading_context(options):
            pass

        assert not io.opened


# ===========================================================================
# Writing context — basic lifecycle
# ===========================================================================


class TestWritingContext:

    def test_truncate_before_write_clears_existing_bytes(self):
        io = _csv_with_payload(b"old contents")

        options = io.check_options(truncate_before_write=True)
        with io._writing_context(options) as target:
            target.write(b"new")
        # Buffer state after exit
        io.seek(0)
        assert io.read() == b"new"
        io.close()

    def test_mark_dirty_default_true(self):
        io = _csv_with_payload(b"x")

        options = io.check_options()  # mark_dirty_on_write defaults True
        with io._writing_context(options):
            pass

        assert io.dirty, "context should mark dirty on write by default"
        io.close()

    def test_mark_dirty_disabled(self):
        io = _csv_with_payload(b"x")
        # Clear dirty if the payload-write left it set
        io._dirty = False

        options = io.check_options(mark_dirty_on_write=False)
        with io._writing_context(options):
            pass

        assert not io.dirty
        io.close()
