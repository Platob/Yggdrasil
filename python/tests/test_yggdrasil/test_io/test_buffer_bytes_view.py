"""Tests for ``yggdrasil.io.buffer.bytes_view.BytesIOView``.

The view is a bounded, view-relative window over a parent
:class:`BytesIO`. The tests cover construction validation, the
seek/read/write/truncate primitives, ``pread``/``pwrite``,
``readinto``, the cursor/visibility contract, ``zipfile``-style
clamp-on-negative-seek behavior, and lifecycle interaction with the
parent (in-memory and spilled).
"""

from __future__ import annotations

import io as _stdio

import pytest

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.buffer.bytes_view import BytesIOView


PAYLOAD = b"Spot WTI prompt-month settle " * 8


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_state(self):
        parent = BytesIO(PAYLOAD)
        v = BytesIOView(parent=parent, size=len(PAYLOAD))
        assert v.start == 0
        assert v.size == len(PAYLOAD)
        assert v.pos == 0
        assert v.max_size is None
        assert not v.closed
        assert v.end == len(PAYLOAD)

    def test_offset_window(self):
        parent = BytesIO(PAYLOAD)
        v = BytesIOView(parent=parent, start=4, size=10)
        assert v.start == 4
        assert v.size == 10
        assert v.end == 14
        assert v.to_bytes() == PAYLOAD[4:14]

    def test_negative_start_rejected(self):
        with pytest.raises(ValueError):
            BytesIOView(parent=BytesIO(PAYLOAD), start=-1, size=5)

    def test_negative_size_rejected(self):
        with pytest.raises(ValueError):
            BytesIOView(parent=BytesIO(PAYLOAD), size=-1)

    def test_negative_pos_rejected(self):
        with pytest.raises(ValueError):
            BytesIOView(parent=BytesIO(PAYLOAD), size=4, pos=-1)

    def test_negative_max_size_rejected(self):
        with pytest.raises(ValueError):
            BytesIOView(parent=BytesIO(PAYLOAD), size=0, max_size=-2)

    def test_size_exceeds_max_size_rejected(self):
        with pytest.raises(ValueError):
            BytesIOView(parent=BytesIO(PAYLOAD), size=10, max_size=5)

    def test_via_parent_view(self):
        parent = BytesIO(PAYLOAD)
        v = parent.view(pos=2, size=6)
        assert isinstance(v, BytesIOView)
        assert v.to_bytes() == PAYLOAD[2:8]

    def test_via_parent_view_default_size_runs_to_end(self):
        parent = BytesIO(PAYLOAD)
        v = parent.view(pos=5)
        assert v.size == len(PAYLOAD) - 5

    def test_via_parent_view_negative_pos_rejected(self):
        with pytest.raises(ValueError):
            BytesIO(PAYLOAD).view(pos=-1)

    def test_via_parent_view_negative_size_rejected(self):
        with pytest.raises(ValueError):
            BytesIO(PAYLOAD).view(size=-3)


# ---------------------------------------------------------------------------
# Capability flags
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_capability_flags(self):
        v = BytesIO(PAYLOAD).view(size=10)
        assert v.readable() is True
        assert v.writable() is True
        assert v.seekable() is True

    def test_repr_open(self):
        v = BytesIO(PAYLOAD).view(size=10)
        text = repr(v)
        assert "BytesIOView" in text
        assert "size=10" in text

    def test_repr_closed(self):
        v = BytesIO(PAYLOAD).view(size=10)
        v.close()
        assert "closed" in repr(v)


# ---------------------------------------------------------------------------
# Read primitives
# ---------------------------------------------------------------------------


class TestRead:
    def test_read_advances_cursor(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=12)
        chunk = v.read(5)
        assert chunk == PAYLOAD[:5]
        assert v.pos == 5

    def test_read_clamped_at_end(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=4)
        v.seek(2)
        chunk = v.read(20)
        assert chunk == PAYLOAD[2:4]
        assert v.pos == 4

    def test_read_minus_one_drains(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=8)
        assert v.read(-1) == PAYLOAD[:8]

    def test_read_none_drains(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=8)
        assert v.read(None) == PAYLOAD[:8]

    def test_read_at_end_returns_empty(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=8)
        v.seek(8)
        assert v.read(4) == b""

    def test_to_bytes_does_not_advance(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=8)
        v.seek(3)
        assert v.to_bytes() == PAYLOAD[:8]
        assert v.pos == 3

    def test_to_bytes_zero_view(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=0)
        assert v.to_bytes() == b""

    def test_readall_drains_and_advances(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=8)
        v.seek(2)
        out = v.readall()
        assert out == PAYLOAD[2:8]
        assert v.pos == 8

    def test_readall_at_eof_returns_empty(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=8)
        v.seek(8)
        assert v.readall() == b""

    def test_readinto_partial(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=10)
        target = bytearray(4)
        n = v.readinto(target)
        assert n == 4
        assert bytes(target) == PAYLOAD[:4]
        assert v.pos == 4

    def test_readinto_zero_buffer(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=10)
        n = v.readinto(bytearray(0))
        assert n == 0
        assert v.pos == 0

    def test_readinto1_delegates(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=10)
        target = bytearray(5)
        n = v.readinto1(target)
        assert n == 5
        assert bytes(target) == PAYLOAD[:5]


# ---------------------------------------------------------------------------
# pread (cursorless)
# ---------------------------------------------------------------------------


class TestPread:
    def test_pread_does_not_move_cursor(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=20)
        v.seek(7)
        chunk = v.pread(4, pos=0)
        assert chunk == PAYLOAD[:4]
        assert v.pos == 7

    def test_pread_offset(self):
        v = BytesIO(PAYLOAD).view(pos=3, size=10)
        chunk = v.pread(4, pos=2)
        assert chunk == PAYLOAD[5:9]

    def test_pread_clamped_at_view_end(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=8)
        chunk = v.pread(20, pos=4)
        assert chunk == PAYLOAD[4:8]

    def test_pread_zero_returns_empty(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=20)
        assert v.pread(0, pos=2) == b""

    def test_pread_past_end_returns_empty(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=8)
        assert v.pread(4, pos=20) == b""

    def test_pread_negative_n_rejected(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=8)
        with pytest.raises(ValueError):
            v.pread(-1, pos=0)

    def test_pread_negative_pos_rejected(self):
        v = BytesIO(PAYLOAD).view(pos=0, size=8)
        with pytest.raises(ValueError):
            v.pread(4, pos=-1)


# ---------------------------------------------------------------------------
# Seek
# ---------------------------------------------------------------------------


class TestSeek:
    def test_seek_set_basic(self):
        v = BytesIO(PAYLOAD).view(size=20)
        assert v.seek(7) == 7
        assert v.pos == 7

    def test_seek_set_negative_raises(self):
        v = BytesIO(PAYLOAD).view(size=20)
        with pytest.raises(ValueError):
            v.seek(-1, _stdio.SEEK_SET)

    def test_seek_cur(self):
        v = BytesIO(PAYLOAD).view(size=20)
        v.seek(5)
        v.seek(3, _stdio.SEEK_CUR)
        assert v.pos == 8

    def test_seek_cur_clamps_negative_to_zero(self):
        v = BytesIO(PAYLOAD).view(size=20)
        v.seek(3)
        # io.BytesIO clamps to 0 instead of raising.
        v.seek(-100, _stdio.SEEK_CUR)
        assert v.pos == 0

    def test_seek_end_basic(self):
        v = BytesIO(PAYLOAD).view(size=20)
        v.seek(0, _stdio.SEEK_END)
        assert v.pos == 20

    def test_seek_end_negative_inside_view(self):
        v = BytesIO(PAYLOAD).view(size=20)
        v.seek(-5, _stdio.SEEK_END)
        assert v.pos == 15

    def test_seek_end_negative_clamped(self):
        # The zipfile speculative-probe pattern.
        v = BytesIO(PAYLOAD).view(size=20)
        v.seek(-100, _stdio.SEEK_END)
        assert v.pos == 0

    def test_invalid_whence_raises(self):
        v = BytesIO(PAYLOAD).view(size=20)
        with pytest.raises(ValueError):
            v.seek(0, 99)

    def test_seek_past_end_allowed(self):
        # io.BytesIO allows positions past EOF; reads from there are
        # still empty.
        v = BytesIO(PAYLOAD).view(size=8)
        v.seek(50)
        assert v.pos == 50
        assert v.read() == b""

    def test_tell(self):
        v = BytesIO(PAYLOAD).view(size=20)
        v.seek(11)
        assert v.tell() == 11


# ---------------------------------------------------------------------------
# Write primitives
# ---------------------------------------------------------------------------


class TestWrite:
    def test_write_advances_cursor(self):
        parent = BytesIO(b"")
        v = parent.view(pos=0, size=0)
        n = v.write(b"abc")
        assert n == 3
        assert v.pos == 3
        assert v.size == 3
        assert parent.to_bytes() == b"abc"

    def test_pwrite_does_not_advance_cursor(self):
        parent = BytesIO(b"x" * 20)
        v = parent.view(pos=0, size=20)
        v.seek(7)
        n = v.pwrite(b"abc", pos=2)
        assert n == 3
        assert v.pos == 7
        assert parent.pread(3, 2) == b"abc"

    def test_pwrite_extends_size(self):
        parent = BytesIO(b"")
        v = parent.view(pos=0, size=0)
        n = v.pwrite(b"abcdef", pos=0)
        assert n == 6
        assert v.size == 6

    def test_write_int_rejected(self):
        parent = BytesIO(b"")
        v = parent.view(size=0)
        # Catches bytes(5) → b"\0\0\0\0\0" footgun.
        with pytest.raises(TypeError):
            v.write(5)  # type: ignore[arg-type]

    def test_write_str_rejected(self):
        parent = BytesIO(b"")
        v = parent.view(size=0)
        with pytest.raises(TypeError):
            v.write("foo")  # type: ignore[arg-type]

    def test_write_memoryview_accepted(self):
        parent = BytesIO(b"")
        v = parent.view(pos=0, size=0)
        n = v.write(memoryview(b"abc"))
        assert n == 3
        assert parent.to_bytes() == b"abc"

    def test_write_bytearray_accepted(self):
        parent = BytesIO(b"")
        v = parent.view(pos=0, size=0)
        v.write(bytearray(b"xyz"))
        assert parent.to_bytes() == b"xyz"

    def test_write_zero_bytes_returns_zero(self):
        parent = BytesIO(b"")
        v = parent.view(pos=0, size=0)
        assert v.pwrite(b"", pos=0) == 0
        assert v.size == 0

    def test_pwrite_negative_pos_rejected(self):
        v = BytesIO(b"abc").view(size=3)
        with pytest.raises(ValueError):
            v.pwrite(b"x", pos=-1)


# ---------------------------------------------------------------------------
# max_size cap
# ---------------------------------------------------------------------------


class TestMaxSizeCap:
    def test_pwrite_clipped_when_exceeds_max(self):
        parent = BytesIO(b"")
        v = parent.view(pos=0, size=0, max_size=4)
        n = v.pwrite(b"abcdefgh", pos=0)
        assert n == 4
        assert v.size == 4
        assert parent.to_bytes() == b"abcd"

    def test_pwrite_full_view_returns_zero(self):
        parent = BytesIO(b"")
        v = parent.view(pos=0, size=0, max_size=3)
        v.pwrite(b"abc", pos=0)
        # Now full: another write at the cap returns 0.
        assert v.pwrite(b"X", pos=3) == 0

    def test_write_returns_zero_when_full(self):
        parent = BytesIO(b"")
        v = parent.view(pos=0, size=0, max_size=3)
        v.write(b"abc")
        assert v.write(b"X") == 0


# ---------------------------------------------------------------------------
# Truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_truncate_shrink_in_memory(self):
        parent = BytesIO(b"abcdef")
        v = parent.view(pos=0, size=6)
        ret = v.truncate(3)
        assert v.size == 3
        assert ret == 3
        # Parent's logical size is shrunk.
        assert parent.size == 3
        assert parent.to_bytes() == b"abc"

    def test_truncate_grow_zero_extends(self):
        parent = BytesIO(b"abc")
        v = parent.view(pos=0, size=3)
        ret = v.truncate(6)
        assert ret == 6
        assert v.size == 6
        assert parent.to_bytes() == b"abc\x00\x00\x00"

    def test_truncate_negative_rejected(self):
        v = BytesIO(b"abc").view(size=3)
        with pytest.raises(ValueError):
            v.truncate(-1)

    def test_truncate_caps_at_max_size(self):
        parent = BytesIO(b"abcd")
        v = parent.view(pos=0, size=4, max_size=4)
        # Asking for 8 must clamp to 4.
        ret = v.truncate(8)
        assert v.size == 4
        assert ret == 4

    def test_truncate_to_current_pos_when_size_none(self):
        parent = BytesIO(b"abcdef")
        v = parent.view(pos=0, size=6)
        v.seek(2)
        ret = v.truncate()
        assert v.size == 2
        assert ret == 2

    def test_truncate_clamps_pos(self):
        parent = BytesIO(b"abcdef")
        v = parent.view(pos=0, size=6)
        v.seek(5)
        v.truncate(3)
        assert v.pos == 3


# ---------------------------------------------------------------------------
# Spilled-mode (file-backed parent)
# ---------------------------------------------------------------------------


class TestSpilledParent:
    def _spilled(self, payload: bytes) -> BytesIO:
        # Lowering ``spill_bytes`` so the constructor immediately spills.
        b = BytesIO(payload, spill_bytes=4)
        assert b.spilled
        return b

    def test_pread_through_view_on_spilled(self):
        parent = self._spilled(PAYLOAD)
        v = parent.view(pos=4, size=8)
        assert v.pread(4, pos=0) == PAYLOAD[4:8]

    def test_to_bytes_through_spilled(self):
        parent = self._spilled(PAYLOAD)
        v = parent.view(pos=2, size=10)
        assert v.to_bytes() == PAYLOAD[2:12]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_close_sets_flag(self):
        v = BytesIO(b"abc").view(size=3)
        assert not v.closed
        v.close()
        assert v.closed

    def test_close_idempotent(self):
        v = BytesIO(b"abc").view(size=3)
        v.close()
        v.close()
        assert v.closed

    def test_context_manager_closes(self):
        v = BytesIO(b"abc").view(size=3)
        with v as inner:
            assert inner is v
            assert not inner.closed
        assert v.closed

    def test_flush_does_not_swallow(self):
        # flush propagates parent flush - should not raise on a healthy
        # in-memory parent.
        v = BytesIO(b"abc").view(size=3)
        v.flush()  # no-op


# ---------------------------------------------------------------------------
# remaining property
# ---------------------------------------------------------------------------


class TestRemaining:
    def test_remaining_at_start(self):
        v = BytesIO(PAYLOAD).view(size=20)
        assert v.remaining == 20

    def test_remaining_after_seek(self):
        v = BytesIO(PAYLOAD).view(size=20)
        v.seek(5)
        assert v.remaining == 15

    def test_remaining_at_eof_or_past(self):
        v = BytesIO(PAYLOAD).view(size=20)
        v.seek(50)
        assert v.remaining == 0
