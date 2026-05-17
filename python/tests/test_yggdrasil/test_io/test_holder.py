"""Behavior tests for :class:`yggdrasil.io.holder.Holder` + :class:`Memory`.

`Holder` is the position-addressable byte substrate that BytesIO and
Path stack on top of. Tests pin:

* the abstract primitives (`read_mv` / `write_mv` / `reserve` /
  `truncate` / `clear` / `size`) round-trip cleanly on `Memory`;
* dispatch via `Holder(...)` picks the right subclass for url /
  binary / path / data inputs;
* append-at-end (`pos = -1`) and from-end (`pos = -N`) sentinels
  resolve consistently;
* `temporary=True` honors clear-on-close;
* ``stat`` / ``mtime`` / ``media_type`` accessors stay consistent
  with the underlying mutable :class:`IOStats`.
"""
from __future__ import annotations

import os
import time

import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.io.memory import Memory
from yggdrasil.io.path.local_path import LocalPath


class TestSubclassDispatch:

    def test_no_args_picks_memory(self) -> None:
        assert isinstance(Holder(), Memory)

    def test_binary_picks_memory(self) -> None:
        h = Holder(binary=b"hello")
        assert isinstance(h, Memory)
        assert h.read_bytes() == b"hello"

    def test_path_picks_path_subclass(self, tmp_path) -> None:
        target = str(tmp_path / "out.bin")
        h = Holder(path=target)
        assert isinstance(h, LocalPath)

    def test_unknown_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scheme"):
            Holder(scheme="not-a-scheme")


class TestMemoryPrimitives:
    """The five primitives + size."""

    def test_empty_construction(self) -> None:
        m = Memory()
        assert m.size == 0
        assert m.capacity == 0
        assert bytes(m) == b""

    def test_capacity_seed(self) -> None:
        m = Memory(8)  # int → reserve, no payload
        assert m.size == 0
        assert m.capacity == 8

    def test_capacity_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 0"):
            Memory(-1)

    def test_write_at_pos_extends(self) -> None:
        m = Memory()
        n = m.write_bytes(b"abc", 0)
        assert n == 3
        assert m.size == 3
        assert m.read_bytes() == b"abc"

    def test_pwrite_in_place_overwrite(self) -> None:
        m = Memory(b"abcdef")
        m.pwrite(b"ZZ", 2)
        assert m.read_bytes() == b"abZZef"

    def test_pread_subrange(self) -> None:
        m = Memory(b"abcdef")
        assert m.pread(3, 1) == b"bcd"
        assert m.pread(2, 4) == b"ef"

    def test_pwrite_at_minus_one_appends(self) -> None:
        m = Memory(b"hi")
        m.pwrite(b"!", -1)
        assert m.read_bytes() == b"hi!"

    def test_pread_at_minus_one_returns_empty(self) -> None:
        m = Memory(b"hi")
        # pos=-1 → end; reads 0 bytes since nothing follows.
        assert m.pread(0, -1) == b""

    def test_pwrite_at_minus_two_indexes_from_end(self) -> None:
        m = Memory(b"abcd")
        m.pwrite(b"X", -2)
        # -2 → size + (-2) = 2 → write 'X' at index 2
        assert m.read_bytes() == b"abXd"

    def test_truncate_shrinks(self) -> None:
        m = Memory(b"abcdef")
        m.truncate(3)
        assert m.size == 3
        assert m.read_bytes() == b"abc"

    def test_truncate_extends_zero_pads(self) -> None:
        m = Memory(b"ab")
        m.truncate(5)
        assert m.size == 5
        assert m.read_bytes() == b"ab\x00\x00\x00"

    def test_truncate_negative_raises(self) -> None:
        m = Memory(b"ab")
        with pytest.raises(ValueError, match="truncate size must be >= 0"):
            m.truncate(-1)

    def test_reserve_grows_capacity_only(self) -> None:
        m = Memory()
        m.write_bytes(b"abc", 0)
        m.reserve(64)
        assert m.capacity >= 64
        assert m.size == 3  # visible size unchanged

    def test_reserve_smaller_is_noop(self) -> None:
        m = Memory(b"abc")
        m.reserve(0)
        assert m.size == 3
        assert m.capacity >= 3

    def test_clear_resets(self) -> None:
        m = Memory(b"abc")
        m.clear()
        assert m.size == 0
        assert m.capacity == 0
        m.write_bytes(b"new", 0)
        assert m.read_bytes() == b"new"


class TestRangeChecks:
    """`read_mv` and `write_mv` enforce bounds with helpful messages."""

    def test_read_past_end_raises(self) -> None:
        m = Memory(b"abc")
        with pytest.raises(ValueError, match="out of bounds"):
            m.read_mv(10, 0)

    def test_read_pos_past_end_raises(self) -> None:
        m = Memory(b"abc")
        with pytest.raises(ValueError, match="out of bounds"):
            m.read_mv(1, 10)

    def test_negative_n_resolves_to_remaining(self) -> None:
        m = Memory(b"abcdef")
        assert bytes(m.read_mv(-1, 2)) == b"cdef"

    def test_negative_pos_clamped_at_negative_one(self) -> None:
        # pos=-1 is the explicit "at end" sentinel.
        m = Memory(b"abc")
        assert bytes(m.read_mv(0, -1)) == b""


class TestStatAndMtime:

    def test_lazy_stat_present_for_memory(self) -> None:
        m = Memory()
        s = m.stat()
        assert s is not None
        assert s.size == 0

    def test_write_bumps_mtime(self) -> None:
        m = Memory()
        before = m.mtime
        time.sleep(0.005)
        m.write_bytes(b"x", 0)
        assert m.mtime >= before

    def test_stat_reflects_live_size(self) -> None:
        m = Memory()
        m.write_bytes(b"abcd", 0)
        # ``stat()`` snapshots the holder's current state into a
        # fresh :class:`IOStats` on each call.
        assert m.stat().size == 4
        # The holder also exposes ``size`` directly for the hot-path
        # readers that don't need a full stat snapshot.
        assert m.size == 4


class TestLazyMediaType:
    """``Holder._media_type`` resolves lazily on first read.

    The :class:`Holder` constructor parks an ``...`` sentinel on the
    slot when the caller didn't seed an explicit media type; the
    :meth:`media_type` property runs :meth:`URL.infer_media_type`
    once and caches the result. Sibling-construction shapes
    (:meth:`Path.parent`, :meth:`Path.joinpath`) never observe the
    media type, so the eager parse was pure waste in path
    traversal."""

    def test_local_path_media_type_inferred_on_first_read(self, tmp_path) -> None:
        from yggdrasil.data.enums.mime_type import MimeTypes
        lp = LocalPath(str(tmp_path / "data.parquet"))
        # Lazy: slot carries the sentinel until first read.
        assert lp._media_type is ...
        # First read drives ``URL.infer_media_type`` and caches.
        assert lp.media_type is not None
        assert lp.media_type.mime_type is MimeTypes.PARQUET
        assert lp._media_type is lp.media_type

    def test_seeded_via_stat_skips_lazy(self, tmp_path) -> None:
        from yggdrasil.data.enums.media_type import MediaType
        from yggdrasil.data.enums.mime_type import MimeTypes
        from yggdrasil.io.io_stats import IOStats
        seed = MediaType(mime_type=MimeTypes.CSV)
        lp = LocalPath(str(tmp_path / "x"), stat=IOStats(media_type=seed))
        # Eager: stat-seeded media_type lands directly on the slot.
        assert lp._media_type is seed

    def test_parent_inherits_lazy_slot(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "a/b/file.parquet"))
        parent = lp.parent
        # Sibling-constructed paths also start lazy — they don't pay
        # for the URL-mime walk during traversal.
        assert parent._media_type is ...


class TestPredicates:

    def test_memory_predicates(self) -> None:
        m = Memory()
        assert m.is_memory
        assert not m.is_local_path
        assert not m.is_remote_path
        assert m.is_local
        assert not m.is_remote

    def test_local_path_predicates(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "a"))
        assert not lp.is_memory
        assert lp.is_local_path
        assert lp.is_local


class TestTemporaryFlag:

    def test_temporary_clears_on_close(self) -> None:
        m = Memory(b"data", temporary=True)
        assert m.temporary
        m.acquire()
        assert m.size == 4
        m.close()
        # Temporary memory keeps the bytearray reference but clears
        # the visible payload on release.
        assert m.size == 0


class TestEquality:

    def test_equal_to_bytes(self) -> None:
        m = Memory(b"abc")
        assert m == b"abc"
        assert m != b"abz"

    def test_equal_to_other_holder(self) -> None:
        a = Memory(b"abc")
        b = Memory(b"abc")
        assert a == b


class TestWriteLocalPath:

    def test_round_trip_into_memory(self, tmp_path) -> None:
        p = tmp_path / "src.bin"
        p.write_bytes(b"local-bytes")
        m = Memory()
        n = m.write_local_path(os.fspath(p))
        assert n == 11
        assert m.read_bytes() == b"local-bytes"

    def test_partial_n(self, tmp_path) -> None:
        p = tmp_path / "src.bin"
        p.write_bytes(b"123456789")
        m = Memory()
        n = m.write_local_path(os.fspath(p), n=4)
        assert n == 4
        assert m.read_bytes() == b"1234"

    def test_negative_pos_raises(self, tmp_path) -> None:
        p = tmp_path / "src.bin"
        p.write_bytes(b"x")
        m = Memory()
        with pytest.raises(ValueError, match="pos must be >= 0"):
            m.write_local_path(os.fspath(p), pos=-1)


class TestWriteStream:

    def test_drains_bytesio_into_memory(self) -> None:
        import io as _stdio
        m = Memory()
        n = m.write_stream(_stdio.BytesIO(b"stream-bytes"))
        assert n == 12
        assert m.read_bytes() == b"stream-bytes"

    def test_drains_into_local_path(self, tmp_path) -> None:
        import io as _stdio
        target = LocalPath(tmp_path / "out.bin")
        n = target.write_stream(_stdio.BytesIO(b"hello"))
        assert n == 5
        assert (tmp_path / "out.bin").read_bytes() == b"hello"

    def test_default_streams_in_chunks(self) -> None:
        """Default :meth:`Holder._write_stream` is real chunked streaming.

        Multi-MB sources splice into the target through
        :data:`_COPY_CHUNK`-sized (1 MiB) ``write_mv`` calls. Remote
        backends that prefer a single atomic PUT (Volumes,
        Workspace, DBFS) override :meth:`_write_stream` to pass
        the IO straight to their backend uploader; this test pins
        the default.
        """
        import io as _stdio
        from unittest.mock import patch

        m = Memory()
        calls: list[int] = []
        original = type(m).write_mv

        def _spy(self, data, offset, *, update_stat=True):
            if self is m:
                calls.append(len(data))
            return original(self, data, offset, update_stat=update_stat)

        with patch.object(type(m), "write_mv", _spy):
            m.write_stream(_stdio.BytesIO(b"x" * (4 * 1024 * 1024)))

        # 4 MiB source, 1 MiB chunk → 4 round trips to the target's
        # ``write_mv``. (The coercion drain into a wrapper holder
        # may issue additional smaller writes against a DIFFERENT
        # Memory instance — ignored by the ``self is m`` filter.)
        assert calls == [1024 * 1024] * 4

    def test_empty_stream_is_noop(self) -> None:
        import io as _stdio
        m = Memory(b"keep")
        n = m.write_stream(_stdio.BytesIO(b""))
        assert n == 0
        assert m.read_bytes() == b"keep"

    def test_negative_offset_raises(self) -> None:
        import io as _stdio
        m = Memory()
        with pytest.raises(ValueError, match="offset must be >= 0"):
            m.write_stream(_stdio.BytesIO(b"x"), offset=-1)


class TestHolderTabular:
    """Holder satisfies the :class:`Tabular` interface — its default
    Tabular hooks open the holder contextually and delegate to the
    format-dispatched :class:`BytesIO` leaf, so ``LocalPath("x.csv")
    .read_arrow_table()`` works with no per-format ceremony.
    """

    def test_local_path_reads_csv_via_tabular_surface(self, tmp_path) -> None:
        import yggdrasil.io.primitive  # noqa: F401 — register leaves
        from yggdrasil.io.tabular.base import Tabular

        csv_path = tmp_path / "data.csv"
        csv_path.write_bytes(b"id,value\n1,10\n2,20\n3,30\n")

        lp = LocalPath(csv_path)
        assert isinstance(lp, Tabular)
        rows = lp.read_arrow_table().to_pylist()
        assert rows == [
            {"id": 1, "value": 10},
            {"id": 2, "value": 20},
            {"id": 3, "value": 30},
        ]


class TestHolderTransfer:
    """``Holder.upload`` / ``Holder.download`` accept Holder/IO/str/PathLike.

    ``dst.upload(src)`` pulls *src*'s bytes into *dst* (the
    receiver's content becomes the source's). The mirror
    ``src.download(target)`` keeps the old src→target direction.
    """

    def test_path_pulls_bytes_from_memory(self, tmp_path) -> None:
        mem = Memory()
        mem.write_bytes(b"payload")
        dst = LocalPath(str(tmp_path / "dst.bin"))
        out = dst.upload(mem)
        assert out is dst
        assert dst.read_bytes() == b"payload"

    def test_path_pulls_bytes_from_str_source(self, tmp_path) -> None:
        # Source is a str/PathLike → coerced via ``Path.from_``.
        src_path = tmp_path / "src.bin"
        src_path.write_bytes(b"payload")
        dst = LocalPath(str(tmp_path / "dst.bin"))
        out = dst.upload(str(src_path))
        assert out is dst
        assert dst.read_bytes() == b"payload"

    def test_memory_pulls_bytes_from_io_cursor(self) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        with BytesIO() as bio:
            bio.write_bytes(b"payload")
            bio.seek(0)  # Caller positions the cursor; upload reads from there.
            mem = Memory()
            out = mem.upload(bio)
        assert out is mem
        assert mem.read_bytes() == b"payload"

    def test_memory_pulls_bytes_from_holder(self) -> None:
        # Holder→Holder via the abstract path; both ends are in-process
        # Memory, so the generic bytes-copy fallback runs.
        src = Memory()
        src.write_bytes(b"payload")
        dst = Memory()
        out = dst.upload(src)
        assert out is dst
        assert dst.read_bytes() == b"payload"

    def test_trailing_slash_target_appends_filename(self, tmp_path) -> None:
        # The trailing-slash directory hint applies on the target side
        # — the source's filename is joined onto the directory. Memory
        # URLs carry no meaningful name so the fallback is "download".
        mem = Memory()
        mem.write_bytes(b"payload")
        (tmp_path / "sub").mkdir()
        dst_dir = LocalPath(str(tmp_path / "sub") + "/")
        out = dst_dir.upload(mem)
        assert out.name == "download"
        assert out.read_bytes() == b"payload"

    def test_default_download_falls_back_to_generic_name(
        self, tmp_path, monkeypatch,
    ) -> None:
        # Memory holders don't have a useful URL name; the default
        # download filename is "download".
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        mem = Memory()
        mem.write_bytes(b"payload")
        out = mem.download()
        assert out.name == "download"
        assert out.read_bytes() == b"payload"

    def test_upload_rejects_unsupported_source_type(self) -> None:
        mem = Memory()
        with pytest.raises(TypeError, match="Holder, IO, str, or os.PathLike"):
            mem.upload(42)

    def test_upload_slices_source_with_size_and_offset(self) -> None:
        src = Memory()
        src.write_bytes(b"abcdef")
        dst = Memory()
        dst.upload(src, size=3, offset=2)
        # 3 bytes starting at offset 2 → "cde".
        assert dst.read_bytes() == b"cde"

    def test_download_slices_source_with_size_and_offset(self) -> None:
        src = Memory()
        src.write_bytes(b"abcdef")
        dst = Memory()
        src.download(dst, size=2, offset=4)
        assert dst.read_bytes() == b"ef"

    def test_upload_from_io_honors_offset(self) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        with BytesIO() as bio:
            bio.write_bytes(b"abcdef")
            # ``offset`` seeks the cursor; ``size`` caps the read.
            mem = Memory()
            mem.upload(bio, size=2, offset=1)
        assert mem.read_bytes() == b"bc"


class TestIOTransfer:
    """``IO.upload`` / ``IO.download`` are cursor-anchored mirrors of Holder's."""

    def test_io_upload_writes_at_cursor_and_advances(self) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        src = Memory()
        src.write_bytes(b"src-payload")

        with BytesIO() as io:
            io.write_bytes(b"PRE-")
            cursor_before = io.tell()
            io.upload(src)
            # Cursor advanced by the written byte count.
            assert io.tell() == cursor_before + src.size
            io.seek(0)
            assert io.read() == b"PRE-src-payload"

    def test_io_upload_slices_source_by_size_and_offset(self) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        src = Memory()
        src.write_bytes(b"abcdef")

        with BytesIO() as io:
            io.upload(src, size=3, offset=2)
            io.seek(0)
            assert io.read() == b"cde"

    def test_io_download_reads_from_cursor(self) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        with BytesIO() as io:
            io.write_bytes(b"abcdef")
            io.seek(2)  # cursor at 'c'
            dst = Memory()
            io.download(dst, size=3)
            assert dst.read_bytes() == b"cde"
            # Cursor advanced past the read.
            assert io.tell() == 5

    def test_io_download_offset_is_cursor_relative(self) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        with BytesIO() as io:
            io.write_bytes(b"abcdef")
            io.seek(1)
            dst = Memory()
            # cursor=1 + offset=2 → read starts at index 3 → "de"
            io.download(dst, size=2, offset=2)
            assert dst.read_bytes() == b"de"


class TestPathTransferOptimization:
    """``Path._transfer_to`` overrides for local-aware fast paths."""

    def test_local_to_local_uses_shutil_copyfile(
        self, tmp_path, monkeypatch,
    ) -> None:
        # When both ends are local files, the Path override hands off
        # to shutil.copyfile rather than the generic read_bytes →
        # write_bytes round-trip.
        import shutil

        calls: list[tuple[str, str]] = []
        real_copyfile = shutil.copyfile

        def tracer(src, dst, *, follow_symlinks=True):
            calls.append((str(src), str(dst)))
            return real_copyfile(src, dst, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(shutil, "copyfile", tracer)

        src = LocalPath(str(tmp_path / "src.bin"))
        src.write_bytes(b"payload")
        dst = LocalPath(str(tmp_path / "dst.bin"))
        out = dst.upload(src)
        assert out is dst
        assert calls == [(src.os_path, dst.os_path)]
        assert dst.read_bytes() == b"payload"

    def test_local_to_holder_uses_write_local_path(
        self, tmp_path, monkeypatch,
    ) -> None:
        # When the source is local and the destination is a non-IO
        # holder (here Memory), the Path._transfer_to override routes
        # through Holder.write_local_path so multi-GB transfers don't
        # materialise the whole payload.
        seen: list[str] = []
        real = Memory.write_local_path

        def tracer(self, path, **kw):
            seen.append(os.fspath(path))
            return real(self, path, **kw)

        monkeypatch.setattr(Memory, "write_local_path", tracer)

        src = LocalPath(str(tmp_path / "src.bin"))
        src.write_bytes(b"payload")
        mem = Memory()
        out = mem.upload(src)
        assert out is mem
        assert seen == [src.os_path]
        assert mem.read_bytes() == b"payload"


class TestPathDirectoryTransfer:
    """``Path.upload`` / ``Path.download`` recurse on directory sources."""

    def test_directory_upload_creates_target_and_copies_leaves(
        self, tmp_path,
    ) -> None:
        src = LocalPath(str(tmp_path / "src"))
        src.mkdir()
        (src / "a.bin").write_bytes(b"a")
        (src / "b.bin").write_bytes(b"b")

        dst = LocalPath(str(tmp_path / "dst"))
        out = dst.upload(src)

        assert out == dst
        assert dst.is_dir()
        assert (dst / "a.bin").read_bytes() == b"a"
        assert (dst / "b.bin").read_bytes() == b"b"

    def test_directory_upload_recurses_into_subdirectories(
        self, tmp_path,
    ) -> None:
        src = LocalPath(str(tmp_path / "src"))
        (src / "sub").mkdir(parents=True)
        (src / "top.bin").write_bytes(b"top")
        (src / "sub" / "leaf.bin").write_bytes(b"leaf")

        dst = LocalPath(str(tmp_path / "dst"))
        dst.upload(src)

        assert (dst / "top.bin").read_bytes() == b"top"
        assert (dst / "sub").is_dir()
        assert (dst / "sub" / "leaf.bin").read_bytes() == b"leaf"

    def test_trailing_slash_target_nests_under_dst(self, tmp_path) -> None:
        # cp -r src dst/  →  dst/src/...
        src = LocalPath(str(tmp_path / "src"))
        src.mkdir()
        (src / "a.bin").write_bytes(b"a")

        dst_dir = tmp_path / "dst"
        dst_dir.mkdir()
        out = LocalPath(str(dst_dir) + "/").upload(src)

        assert out.name == "src"
        assert (out / "a.bin").read_bytes() == b"a"

    def test_empty_directory_creates_empty_target(self, tmp_path) -> None:
        src = LocalPath(str(tmp_path / "src"))
        src.mkdir()

        dst = LocalPath(str(tmp_path / "dst"))
        out = dst.upload(src)

        assert out.is_dir()
        assert not any(out.iterdir())

    def test_directory_into_memory_raises(self, tmp_path) -> None:
        src = LocalPath(str(tmp_path / "src"))
        src.mkdir()
        (src / "a.bin").write_bytes(b"a")

        with pytest.raises(IsADirectoryError, match="target must be a Path"):
            Memory().upload(src)

    def test_default_download_creates_directory_under_downloads(
        self, tmp_path, monkeypatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        src = LocalPath(str(tmp_path / "src"))
        src.mkdir()
        (src / "leaf.bin").write_bytes(b"leaf")

        out = src.download()

        assert out.is_dir()
        assert out.name == "src"
        assert (out / "leaf.bin").read_bytes() == b"leaf"
