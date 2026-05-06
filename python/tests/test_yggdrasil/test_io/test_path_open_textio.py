"""Regression tests for ``Path.open(mode='wb')`` wrapped in a
:class:`io.TextIOWrapper`.

The screenshot pattern from the bug report::

    with dbx.open_file(file_path, mode="w") as f:   # f: TextIOWrapper
        json.dump(payload, f, indent=4, default=str)
        f.close()

was reported as "looks writing nothing." The shape exercised here is
the local-path equivalent: a ``BytesIO`` opened against a
:class:`LocalPath` in binary mode, then wrapped in
:class:`io.TextIOWrapper` for text I/O. These tests pin the contract
that text writes (``write`` / ``writelines`` / :func:`json.dump`)
land on disk after the wrapper closes — including the variant where
the caller calls ``f.close()`` explicitly inside a ``with`` block.

The path-bound :class:`BytesIO` is constructed without a media-type-
recognised suffix (e.g. ``out.dat`` / extensionless) so we get a
vanilla buffer, not a tabular leaf — the wrapper-close path is the
only thing under test, and tabular dispatch would route through
different write hooks.
"""

from __future__ import annotations

import io as _stdio
import json

import pytest

from yggdrasil.io.fs import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_text(path, mode: str = "wb", *, encoding: str = "utf-8",
               write_through: bool = True) -> _stdio.TextIOWrapper:
    """Mirror the production wrap shape: binary handle + TextIOWrapper.

    ``write_through=True`` matches :meth:`DatabricksPath.open` so
    text writes pass straight through to the underlying buffer
    instead of sitting in the wrapper's own buffer until close.
    """
    binary = path.open(mode=mode)
    return _stdio.TextIOWrapper(
        binary, encoding=encoding, write_through=write_through,
    )


# ---------------------------------------------------------------------------
# Single-shot text writes via the wrapper
# ---------------------------------------------------------------------------


class TestTextIOWrapperWrite:
    """Round-trip plain ``str`` writes through the wrapper."""

    def test_with_block_writes_land(self, tmp_path):
        target = Path.from_(tmp_path / "hello.dat")
        with _open_text(target) as f:
            assert isinstance(f, _stdio.TextIOWrapper)
            f.write("hello world")
        assert target.read_bytes() == b"hello world"

    def test_writelines_lands(self, tmp_path):
        target = Path.from_(tmp_path / "lines.dat")
        with _open_text(target) as f:
            f.writelines(["one\n", "two\n", "three\n"])
        assert target.read_bytes() == b"one\ntwo\nthree\n"

    def test_unicode_encoded_utf8(self, tmp_path):
        target = Path.from_(tmp_path / "uni.dat")
        text = "hello αβγ ✓"
        with _open_text(target) as f:
            f.write(text)
        assert target.read_bytes() == text.encode("utf-8")

    def test_explicit_encoding_round_trip(self, tmp_path):
        target = Path.from_(tmp_path / "uni.dat")
        text = "héllo"
        with _open_text(target, encoding="latin-1") as f:
            f.write(text)
        assert target.read_bytes() == text.encode("latin-1")


# ---------------------------------------------------------------------------
# json.dump — the exact screenshot pattern
# ---------------------------------------------------------------------------


class TestJsonDumpThroughWrapper:
    """The screenshot calls ``json.dump`` through the wrapper. Pin
    that the resulting bytes are a complete, parseable JSON document
    after close — not an empty file."""

    def test_json_dump_in_with_block(self, tmp_path):
        target = Path.from_(tmp_path / "payload.dat")
        payload = {"CCGT": 1.5, "COGEN": 2.5, "TOTAL": 4.0}
        with _open_text(target) as f:
            json.dump(payload, f, indent=4, default=str)
        raw = target.read_bytes()
        assert raw, "wrapper close should flush bytes to disk"
        assert json.loads(raw) == payload

    def test_json_dump_with_explicit_close_inside_with(self, tmp_path):
        """Mirror the screenshot: ``json.dump(...)`` + ``f.close()``
        inside the ``with`` block. The outer ``__exit__`` then runs
        on an already-closed wrapper and must remain a no-op (no
        exception, no truncation of what was written)."""
        target = Path.from_(tmp_path / "payload.dat")
        payload = {"CCGT": 1, "COGEN": 2, "TOTAL": 3}
        with _open_text(target) as f:
            json.dump(payload, f, indent=4, default=str)
            f.close()
        raw = target.read_bytes()
        assert raw, "explicit f.close() inside with-block must persist writes"
        assert json.loads(raw) == payload

    def test_json_dump_without_with_block(self, tmp_path):
        """No ``with``: caller-driven open + close. Same expectation."""
        target = Path.from_(tmp_path / "payload.dat")
        payload = {"a": [1, 2, 3], "b": "x"}
        f = _open_text(target)
        try:
            json.dump(payload, f)
        finally:
            f.close()
        assert json.loads(target.read_bytes()) == payload


# ---------------------------------------------------------------------------
# Flush / close ordering
# ---------------------------------------------------------------------------


class TestFlushAndCloseOrdering:
    def test_flush_persists_without_close(self, tmp_path):
        """``write_through=True`` means the wrapper has nothing of
        its own to flush — bytes are already in the buffer. After
        an explicit ``flush()`` on a *local* path, the file on disk
        must reflect what was written so far. (The wrapper still
        owns the handle, so we do close at the end to release it.)"""
        target = Path.from_(tmp_path / "flush.dat")
        f = _open_text(target)
        try:
            f.write("partial")
            f.flush()
            # On a local path, ``os.pwrite`` has already hit the fd,
            # so the file must be visible at the expected size.
            assert target.size == len(b"partial")
        finally:
            f.close()
        assert target.read_bytes() == b"partial"

    def test_double_close_is_noop(self, tmp_path):
        target = Path.from_(tmp_path / "double.dat")
        f = _open_text(target)
        f.write("x")
        f.close()
        # Second close — must not raise, must not blank the file.
        f.close()
        assert target.read_bytes() == b"x"

    def test_empty_write_creates_empty_file(self, tmp_path):
        """Opening in ``wb`` and closing with no writes leaves an
        empty file (truncate-on-open semantics). This guards the
        opposite failure mode — silently-skipped close that would
        leave the previous content in place."""
        target = Path.from_(tmp_path / "empty.dat")
        target.write_bytes(b"prior content")
        with _open_text(target) as f:
            del f  # noqa: F841 — opened solely for the truncate
        assert target.read_bytes() == b""


# ---------------------------------------------------------------------------
# Larger / multi-write payloads
# ---------------------------------------------------------------------------


class TestMultiWritePayloads:
    def test_many_small_writes(self, tmp_path):
        target = Path.from_(tmp_path / "many.dat")
        with _open_text(target) as f:
            for i in range(1000):
                f.write(f"line-{i:04d}\n")
        raw = target.read_bytes()
        assert raw.count(b"\n") == 1000
        assert raw.startswith(b"line-0000\n")
        assert raw.endswith(b"line-0999\n")

    def test_chunked_json_array(self, tmp_path):
        """Hand-roll a streaming JSON array — exercises mixed
        ``write`` calls of varying sizes through the wrapper."""
        target = Path.from_(tmp_path / "stream.dat")
        with _open_text(target) as f:
            f.write("[")
            for i in range(50):
                if i:
                    f.write(",")
                json.dump({"i": i, "v": i * i}, f)
            f.write("]")
        decoded = json.loads(target.read_bytes())
        assert len(decoded) == 50
        assert decoded[0] == {"i": 0, "v": 0}
        assert decoded[-1] == {"i": 49, "v": 2401}


# ---------------------------------------------------------------------------
# Wrapper hand-rolled exactly as in DatabricksPath.open(mode='w')
# ---------------------------------------------------------------------------


class TestDatabricksOpenShape:
    """:meth:`DatabricksPath.open` with a text mode wraps the binary
    handle in ``TextIOWrapper(write_through=True)`` and returns the
    wrapper. Reproduce that wrap inline — we don't need a real
    Databricks workspace to assert the surface contract."""

    @pytest.mark.parametrize("payload", [
        {"k": "v"},
        {"nested": {"a": [1, 2, 3]}, "n": None},
        [1, 2, 3, "four"],
    ])
    def test_screenshot_pattern(self, tmp_path, payload):
        target = Path.from_(tmp_path / "screenshot.dat")

        binary_handle = target.open(mode="wb")
        f = _stdio.TextIOWrapper(
            binary_handle,
            encoding="utf-8",
            errors="strict",
            write_through=True,
        )
        with f:
            json.dump(payload, f, indent=4, default=str)
            f.close()

        raw = target.read_bytes()
        assert raw, (
            "TextIOWrapper(write_through=True) over a path-bound "
            "BytesIO must persist json.dump output on close."
        )
        assert json.loads(raw) == payload
