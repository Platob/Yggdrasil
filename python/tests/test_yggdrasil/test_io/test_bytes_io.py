"""Behavior tests for :class:`yggdrasil.io.bytes_io.BytesIO`.

`BytesIO` is the cursor + ``IO[bytes]`` + tabular view that sits on
top of any :class:`Holder`. Tests pin:

* the two operating modes (closed / open with scratch transaction);
* the ``IO[bytes]`` protocol — read / write / readline / readlines /
  writelines / readinto / iter — agrees with stdlib ``io.BytesIO``
  behavior the way external libraries (pandas, polars, json) expect;
* construction routing — bytes / str / file-like / Holder / path /
  another BytesIO all flow through :meth:`from_` correctly;
* the ``with holder.open() as bio: …`` pattern commits exactly the
  scratch buffer onto the durable holder, with both append and
  overwrite modes;
* structured binary primitives (``write_int32`` / ``read_str_u32`` /
  …) round-trip cleanly.
"""
from __future__ import annotations

import io
import json
import struct

import pytest

from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.memory import Memory
from yggdrasil.io.path.local_path import LocalPath


class TestConstruction:

    def test_no_args_yields_empty_memory_bio(self) -> None:
        b = BytesIO()
        assert b.size == 0
        assert b.tell() == 0
        assert b.holder.is_memory
        assert b.owns_holder

    def test_bytes_input(self) -> None:
        b = BytesIO(b"hello")
        assert b.size == 5
        assert b.to_bytes() == b"hello"

    def test_borrows_holder_keeps_it_alive_after_close(self) -> None:
        mem = Memory(b"shared")
        c1 = BytesIO(holder=mem)
        c2 = BytesIO(holder=mem)
        assert not c1.owns_holder
        assert not c2.owns_holder
        c1.close()
        # Closing one cursor must not affect the other or the holder.
        assert c2.size == 6
        assert mem.size == 6

    def test_owns_holder_closes_it(self) -> None:
        mem = Memory(b"owned")
        b = BytesIO(holder=mem, owns_holder=True)
        b.close()
        # Memory survives close (its bytearray lives on); ownership
        # just means BytesIO told the holder to release.
        assert mem.size == 5

    def test_close_preserves_cursor_position(self) -> None:
        # Close keeps the cursor where the last transaction left it
        # — append-mode flows (ArrowIPC composite-key append) rely on
        # reopening at EOF rather than at byte 0.
        mem = Memory(b"shared")
        b = BytesIO(holder=mem)
        b.read(3)
        assert b.tell() == 3
        b.close()
        assert b.tell() == 3

    def test_data_and_holder_both_raise(self) -> None:
        with pytest.raises(TypeError, match="not multiple"):
            BytesIO(b"x", holder=Memory())

    def test_path_and_holder_both_raise(self, tmp_path) -> None:
        with pytest.raises(TypeError, match="not multiple"):
            BytesIO(holder=Memory(), path=str(tmp_path / "x.csv"))

    def test_data_and_path_both_raise(self, tmp_path) -> None:
        with pytest.raises(TypeError, match="not both"):
            BytesIO(b"x", path=str(tmp_path / "x.csv"))

    def test_idempotent_from_(self) -> None:
        a = BytesIO(b"abc")
        b = BytesIO.from_(a)
        assert b is a

    def test_from_file_like_drains(self) -> None:
        from yggdrasil.io.base import IO

        src = io.BytesIO(b"piped-bytes")
        b = IO.from_(src)
        assert b.to_bytes() == b"piped-bytes"

    def test_from_bytes(self) -> None:
        from yggdrasil.io.base import IO

        b = IO.from_(b"raw-bytes")
        assert b.to_bytes() == b"raw-bytes"
        assert b.owns_holder

    def test_from_bytearray(self) -> None:
        from yggdrasil.io.base import IO

        b = IO.from_(bytearray(b"mutable-bytes"))
        assert b.to_bytes() == b"mutable-bytes"

    def test_from_memoryview(self) -> None:
        from yggdrasil.io.base import IO

        b = IO.from_(memoryview(b"view-bytes"))
        assert b.to_bytes() == b"view-bytes"

    def test_from_holder_borrows(self) -> None:
        from yggdrasil.io.base import IO

        mem = Memory(b"holder-bytes")
        b = IO.from_(mem)
        assert b.to_bytes() == b"holder-bytes"
        # Holder borrowed, not owned — caller is responsible for it.
        assert not b.owns_holder
        assert b.holder is mem

    def test_from_other_io_borrows_holder(self) -> None:
        """Different :class:`IO` subclass over the same holder reuses
        the byte substrate — no drain (which would advance the source
        cursor and miss any bytes already consumed)."""
        from yggdrasil.io.base import IO

        a = BytesIO(b"shared-bytes")
        a.read(3)  # advance source cursor to 3
        b = IO.from_(a)
        # Same byte substrate borrowed — the holder is shared.
        assert b.holder is a.holder

    def test_from_pathlib_path(self, tmp_path) -> None:
        import pathlib

        from yggdrasil.io.base import IO
        from yggdrasil.io.path.local_path import LocalPath

        target = tmp_path / "data.bin"
        target.write_bytes(b"on-disk-bytes")
        b = IO.from_(pathlib.Path(target))
        assert isinstance(b.holder, LocalPath)
        assert b.to_bytes() == b"on-disk-bytes"

    def test_from_str_path(self, tmp_path) -> None:
        from yggdrasil.io.base import IO
        from yggdrasil.io.path.local_path import LocalPath

        target = tmp_path / "data.bin"
        target.write_bytes(b"str-path-bytes")
        b = IO.from_(str(target))
        assert isinstance(b.holder, LocalPath)
        assert b.to_bytes() == b"str-path-bytes"

    def test_from_local_file_handle_wraps_as_local_path(self, tmp_path) -> None:
        """``open("path", "rb")`` carries the file path on ``.name``;
        :meth:`IO.from_` recognises it and routes through
        :class:`LocalPath` instead of draining the handle into a
        :class:`MemoryStream`. Result: huge file handles never get
        materialised."""
        from yggdrasil.io.base import IO
        from yggdrasil.io.memory_stream import MemoryStream
        from yggdrasil.io.path.local_path import LocalPath

        target = tmp_path / "huge.bin"
        target.write_bytes(b"file-handle-bytes")

        with open(target, "rb") as fh:
            b = IO.from_(fh)

        # Holder is the local path, not a streaming wrapper.
        assert isinstance(b.holder, LocalPath)
        assert not isinstance(b.holder, MemoryStream)
        assert b.to_bytes() == b"file-handle-bytes"

    def test_from_anonymous_stream_falls_back_to_memory_stream(self) -> None:
        """Stdlib :class:`io.BytesIO` has no ``.name`` → wraps in a
        :class:`MemoryStream` that pulls bytes lazily as they're
        read, so a 10 GB urllib3 response never gets materialised
        into Python bytes up front."""
        from yggdrasil.io.base import IO
        from yggdrasil.io.memory_stream import MemoryStream

        b = IO.from_(io.BytesIO(b"anonymous"))
        assert isinstance(b.holder, MemoryStream)
        # Reading through the cursor drives the lazy pull.
        assert b.to_bytes() == b"anonymous"

    def test_from_stream_with_fake_name_falls_back_to_memory_stream(self) -> None:
        """Sentinel ``.name`` like ``"<fdopen>"`` is not a real path —
        coercion falls through to the :class:`MemoryStream` branch
        instead of crashing on a missing-file check."""
        from yggdrasil.io.base import IO
        from yggdrasil.io.memory_stream import MemoryStream

        src = io.BytesIO(b"sentinel")
        src.name = "<fdopen>"  # type: ignore[attr-defined]
        b = IO.from_(src)
        assert isinstance(b.holder, MemoryStream)
        assert b.to_bytes() == b"sentinel"

    def test_from_unsupported_type_raises(self) -> None:
        from yggdrasil.io.base import IO

        with pytest.raises(TypeError, match="Cannot wrap"):
            IO.from_(42)

    def test_path_kwarg_dispatches_via_extension(self, tmp_path) -> None:
        """``BytesIO(path=...)`` lands on the format leaf for the URL's
        extension. Closes the bug where the ``__new__`` dispatch only
        fired on an explicit ``media_type=``.
        """
        # Side-effect import: register the primitive leaves.
        import yggdrasil.io.primitive  # noqa: F401
        from yggdrasil.io.primitive.parquet_io import ParquetIO
        from yggdrasil.io.primitive.csv_io import CsvIO

        pq_path = str(tmp_path / "x.parquet")
        csv_path = str(tmp_path / "x.csv")
        assert isinstance(BytesIO(path=pq_path), ParquetIO)
        assert isinstance(BytesIO(path=csv_path), CsvIO)

    def test_path_kwarg_owns_path_holder(self, tmp_path) -> None:
        from yggdrasil.io.path.local_path import LocalPath
        b = BytesIO(path=str(tmp_path / "data.csv"))
        assert isinstance(b.holder, LocalPath)
        assert b.owns_holder

    def test_explicit_media_type_still_wins(self, tmp_path) -> None:
        from yggdrasil.data.enums import MediaType, MimeTypes
        from yggdrasil.io.primitive.parquet_io import ParquetIO
        b = BytesIO(
            path=str(tmp_path / "x.csv"),
            media_type=MediaType(MimeTypes.PARQUET),
        )
        assert isinstance(b, ParquetIO)

    def test_holder_media_type_drives_dispatch(self) -> None:
        """A holder pre-tagged with a :class:`MediaType` routes through
        the dispatch table even when no explicit ``media_type=`` /
        ``path=`` is given.
        """
        from yggdrasil.data.enums import MediaType, MimeTypes
        from yggdrasil.io.primitive.parquet_io import ParquetIO
        mem = Memory()
        mem.media_type = MediaType(MimeTypes.PARQUET)
        assert isinstance(BytesIO(holder=mem), ParquetIO)


class TestIOProtocolBasics:

    def test_read_advances_cursor(self) -> None:
        b = BytesIO(b"abcdef")
        assert b.read(3) == b"abc"
        assert b.tell() == 3
        assert b.read() == b"def"
        assert b.tell() == 6

    def test_read_zero_returns_empty(self) -> None:
        b = BytesIO(b"abc")
        assert b.read(0) == b""
        assert b.tell() == 0

    def test_read_more_than_remaining_returns_remaining(self) -> None:
        # Stdlib parity: read(N) where N > remaining returns whatever
        # remains, no exception.
        b = BytesIO(b"abc")
        assert b.read(100) == b"abc"

    def test_readall_alias(self) -> None:
        b = BytesIO(b"abcdef")
        b.read(2)
        assert b.readall() == b"cdef"

    def test_readinto_fills_buffer(self) -> None:
        b = BytesIO(b"abcdef")
        buf = bytearray(4)
        n = b.readinto(buf)
        assert n == 4
        assert buf == b"abcd"
        assert b.tell() == 4

    def test_readline(self) -> None:
        b = BytesIO(b"first\nsecond\nthird")
        assert b.readline() == b"first\n"
        assert b.readline() == b"second\n"
        assert b.readline() == b"third"
        assert b.readline() == b""

    def test_readlines(self) -> None:
        b = BytesIO(b"a\nb\nc\n")
        assert b.readlines() == [b"a\n", b"b\n", b"c\n"]

    def test_iter_lines(self) -> None:
        b = BytesIO(b"a\nb\n")
        assert list(b) == [b"a\n", b"b\n"]

    def test_seek_set_clamped(self) -> None:
        b = BytesIO(b"abc")
        with pytest.raises(ValueError, match="Negative SEEK_SET"):
            b.seek(-3)

    def test_seek_set_minus_one_is_eof_sentinel(self) -> None:
        b = BytesIO(b"abc")
        b.seek(-1)
        assert b.tell() == 3

    def test_seek_end_clamps_at_zero(self) -> None:
        b = BytesIO(b"abc")
        b.seek(-100, io.SEEK_END)
        assert b.tell() == 0

    def test_seek_invalid_whence(self) -> None:
        b = BytesIO(b"abc")
        with pytest.raises(ValueError, match="Invalid whence"):
            b.seek(0, 99)

    def test_predicates(self) -> None:
        b = BytesIO(b"abc")
        assert b.readable()
        assert b.writable()
        assert b.seekable()
        assert not b.isatty()


class TestWritePaths:

    def test_write_bytes_advances_cursor(self) -> None:
        b = BytesIO()
        n = b.write(b"abc")
        assert n == 3
        assert b.tell() == 3

    def test_write_string_encodes_utf8(self) -> None:
        b = BytesIO()
        n = b.write("héllo")
        assert n == len("héllo".encode("utf-8"))
        assert b.to_bytes() == "héllo".encode("utf-8")

    def test_write_file_like(self) -> None:
        src = io.BytesIO(b"piped")
        b = BytesIO()
        n = b.write(src)
        assert n == 5
        assert b.to_bytes() == b"piped"

    def test_writelines_concatenates(self) -> None:
        b = BytesIO()
        b.writelines([b"one\n", b"two\n", b"three"])
        assert b.to_bytes() == b"one\ntwo\nthree"

    def test_truncate_at_pos(self) -> None:
        b = BytesIO(b"abcdef")
        b.seek(3)
        b.truncate()
        assert b.size == 3
        assert b.to_bytes() == b"abc"

    def test_truncate_explicit_size(self) -> None:
        b = BytesIO(b"abcdef")
        b.truncate(2)
        assert b.size == 2

    def test_pwrite_does_not_touch_cursor(self) -> None:
        b = BytesIO(b"abcdef")
        b.seek(2)
        b.pwrite(b"ZZ", 0)
        assert b.to_bytes() == b"ZZcdef"
        assert b.tell() == 2

    def test_pread_does_not_touch_cursor(self) -> None:
        b = BytesIO(b"abcdef")
        b.seek(2)
        assert b.pread(2, 4) == b"ef"
        assert b.tell() == 2

    def test_write_update_stat_false_skips_per_write_dirty(self) -> None:
        b = BytesIO(b"seed")
        holder = b._parent
        # Pre-acquired memory holder seeded above is clean to start.
        holder.clear_dirty()
        # In-place overwrites (no resize) with update_stat=False
        # should not flip the holder's dirty bit on each call.
        for offset in range(4):
            b.pwrite(b"X", offset, update_stat=False)
        assert b.to_bytes() == b"XXXX"
        assert holder.is_dirty() is False
        # Default kwarg restores the per-write dirty mark.
        b.pwrite(b"Y", 0)
        assert holder.is_dirty() is True


class TestStructuredPrimitives:

    def test_int32_round_trip(self) -> None:
        b = BytesIO()
        b.write_int32(-42)
        b.seek(0)
        assert b.read_int32() == -42

    def test_uint64_round_trip(self) -> None:
        b = BytesIO()
        b.write_uint64(2**40 + 7)
        b.seek(0)
        assert b.read_uint64() == 2**40 + 7

    def test_f64_round_trip(self) -> None:
        b = BytesIO()
        b.write_f64(3.14159265358979)
        b.seek(0)
        assert b.read_f64() == pytest.approx(3.14159265358979)

    def test_bool_round_trip(self) -> None:
        b = BytesIO()
        b.write_bool(True)
        b.write_bool(False)
        b.seek(0)
        assert b.read_bool() is True
        assert b.read_bool() is False

    def test_length_prefixed_string(self) -> None:
        b = BytesIO()
        b.write_str_u32("hello")
        b.write_str_u32("héllo")
        b.seek(0)
        assert b.read_str_u32() == "hello"
        assert b.read_str_u32() == "héllo"

    def test_short_read_raises_eof(self) -> None:
        b = BytesIO(struct.pack("<I", 8))  # claims 8 bytes follow
        b.seek(0)
        with pytest.raises(EOFError):
            b.read_bytes_u32()


class TestCursorOverHolder:
    """`with bio: ...` runs the IO as a cursor over the bound holder.

    The IO carries no scratch buffer of its own — every read / write
    flows directly through the holder, so each assertion below is
    really a check that the holder's state matches the caller's view.
    """

    def test_acquire_reads_from_durable(self) -> None:
        mem = Memory(b"durable-bytes")
        b = BytesIO(holder=mem, owns_holder=False, mode="rb+")
        with b:
            assert b.size == len(b"durable-bytes")
            assert b.read() == b"durable-bytes"

    def test_writes_land_directly_on_holder(self) -> None:
        mem = Memory(b"original")
        b = BytesIO(holder=mem, owns_holder=False, mode="rb+")
        with b:
            b.seek(0)
            b.write(b"REPLACED")
            # No scratch — the holder already has the new bytes.
            assert mem.read_bytes() == b"REPLACED"
        assert mem.read_bytes() == b"REPLACED"

    def test_wb_mode_truncates_at_open(self) -> None:
        mem = Memory(b"old-bytes")
        b = BytesIO(holder=mem, owns_holder=False, mode="wb")
        with b:
            assert b.size == 0  # holder truncated at acquire
            b.write(b"new")
        assert mem.read_bytes() == b"new"

    def test_xb_mode_raises_on_non_empty(self) -> None:
        mem = Memory(b"existing")
        b = BytesIO(holder=mem, owns_holder=False, mode="xb")
        with pytest.raises(FileExistsError):
            b.__enter__()

    def test_ab_mode_lands_at_eof(self) -> None:
        mem = Memory(b"head-")
        b = BytesIO(holder=mem, owns_holder=False, mode="ab")
        with b:
            assert b.tell() == 5
            b.write(b"tail")
        assert mem.read_bytes() == b"head-tail"

    def test_flush_is_visible_on_holder(self) -> None:
        mem = Memory()
        b = BytesIO(holder=mem, owns_holder=False, mode="wb+")
        with b:
            b.write(b"chunk1")
            b.flush()
            assert mem.read_bytes() == b"chunk1"
            b.write(b"chunk2")
        assert mem.read_bytes() == b"chunk1chunk2"

    def test_temporary_holder_clears_on_release(self) -> None:
        mem = Memory(temporary=True)
        b = BytesIO(holder=mem, owns_holder=True, mode="wb+")
        with b:
            b.write(b"throwaway")
        # ``Holder._release`` honors :attr:`temporary` and clears the
        # payload when the IO closes.
        assert mem.size == 0


class TestExternalWriters:
    """Standard library writers operate on a raw BytesIO seamlessly.

    These match the integration patterns the user actually hits:

        with holder.open() as b:
            external_lib.dump(b, ...)

    The library expects an object that quacks like ``IO[bytes]``;
    the test asserts the bytes commit to the durable holder once
    the ``with`` block ends.
    """

    def test_json_dump_into_holder_open(self, tmp_path) -> None:
        target = LocalPath(str(tmp_path / "out.json"))
        with target.open("wb") as bio:
            bio.write(json.dumps({"id": 1, "ok": True}).encode("utf-8"))
        assert json.loads(target.read_text()) == {"id": 1, "ok": True}

    def test_pickle_dump_round_trip(self, tmp_path) -> None:
        import pickle

        target = LocalPath(str(tmp_path / "out.pkl"))
        with target.open("wb") as bio:
            pickle.dump({"x": [1, 2, 3]}, bio)
        with target.open("rb") as bio:
            assert pickle.load(bio) == {"x": [1, 2, 3]}

    def test_pandas_to_csv_into_holder_open(self, tmp_path) -> None:
        pd = pytest.importorskip("pandas")
        target = LocalPath(str(tmp_path / "data.csv"))
        with target.open("wb") as bio:
            pd.DataFrame({"id": [1, 2, 3]}).to_csv(bio, index=False)
        text = target.read_text()
        assert text.splitlines() == ["id", "1", "2", "3"]

    def test_polars_write_csv_into_holder_open(self, tmp_path) -> None:
        pl = pytest.importorskip("polars")
        target = LocalPath(str(tmp_path / "data.csv"))
        with target.open("wb") as bio:
            pl.DataFrame({"id": [1, 2, 3]}).write_csv(bio)
        text = target.read_text()
        assert "id" in text and "1" in text and "3" in text

    def test_zip_writer_into_holder_open(self, tmp_path) -> None:
        import zipfile

        target = LocalPath(str(tmp_path / "bundle.zip"))
        with target.open("wb") as bio:
            with zipfile.ZipFile(bio, "w") as zf:
                zf.writestr("hello.txt", "hi")
                zf.writestr("payload.json", '{"ok": true}')
        with zipfile.ZipFile(target.os_path) as zf:
            assert sorted(zf.namelist()) == ["hello.txt", "payload.json"]
            assert zf.read("hello.txt") == b"hi"


class TestConvenienceDrains:

    def test_to_bytes_keeps_cursor(self) -> None:
        b = BytesIO(b"abc")
        b.seek(2)
        assert b.to_bytes() == b"abc"
        assert b.tell() == 2

    def test_decode_text(self) -> None:
        b = BytesIO("héllo".encode("utf-8"))
        assert b.decode() == "héllo"

    def test_to_base64_url_safe(self) -> None:
        b = BytesIO(b"\xff\xfe\xfd")
        assert b.to_base64() == "__79"


class TestAsMedia:
    """`BytesIO.as_media(media_type=...)` returns the registered
    Tabular leaf for that media type, sharing the underlying holder.

    Pinned regression: callers (Response.to_polars, Response.dio,
    DatabricksClient cache) used to crash with AttributeError when
    the buffer was a BytesIO subclass without an as_media hop on the
    base class.
    """

    def test_explicit_media_type_routes_to_leaf(self) -> None:
        from yggdrasil.data.enums import MimeTypes
        from yggdrasil.io.primitive.json_io import JsonIO

        b = BytesIO(b'{"x": 1}')
        mio = b.as_media(MimeTypes.JSON)
        assert isinstance(mio, JsonIO)

    def test_stamped_media_type_resolves_when_no_explicit(self) -> None:
        from yggdrasil.data.enums import MediaTypes
        from yggdrasil.io.primitive.json_io import JsonIO

        b = BytesIO(b'{"x": 1}', media_type=MediaTypes.JSON)
        mio = b.as_media()
        assert isinstance(mio, JsonIO)

    def test_returns_self_when_already_target_class(self) -> None:
        from yggdrasil.data.enums import MediaTypes, MimeTypes

        b = BytesIO(b'{"x": 1}', media_type=MediaTypes.JSON)
        mio = b.as_media(MimeTypes.JSON)
        assert mio is b

    def test_shares_underlying_holder(self) -> None:
        from yggdrasil.data.enums import MimeTypes

        b = BytesIO(b'{"x": 1}')
        mio = b.as_media(MimeTypes.JSON)
        assert mio._parent is b._parent

    def test_no_media_type_raises_keyerror(self) -> None:
        b = BytesIO(b"hello")
        with pytest.raises(KeyError, match="No media_type"):
            b.as_media()

    def test_unregistered_media_type_raises_keyerror(self) -> None:
        b = BytesIO(b"hello")
        with pytest.raises(KeyError):
            b.as_media("application/x-bogus-format")

    def test_round_trip_through_response_to_polars(self) -> None:
        """Pinned regression — Response.to_polars(parse=True) used to
        crash with `'JsonIO' object has no attribute 'as_media'`.
        """
        import datetime as dt

        from yggdrasil.io.request import PreparedRequest
        from yggdrasil.io.response import Response

        req = PreparedRequest.prepare(method="GET", url="https://example.com/x")
        body = b'[{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]'
        r = Response(
            request=req,
            status_code=200,
            headers={"Content-Type": "application/json"},
            tags={},
            buffer=body,  # type: ignore[arg-type]
            received_at=dt.datetime.fromtimestamp(0, tz=dt.timezone.utc),
        )
        df = r.to_polars()
        assert df.shape == (2, 2)
        assert df["x"].to_list() == [1, 2]
        assert df["y"].to_list() == ["a", "b"]

    def test_response_open_dispatches_to_typed_leaf(self) -> None:
        """``Response.open()`` routes through :meth:`Holder.open`, which
        dispatches to the format leaf via the holder's media type.
        """
        import datetime as dt

        from yggdrasil.io.primitive.json_io import JsonIO
        from yggdrasil.io.request import PreparedRequest
        from yggdrasil.io.response import Response

        req = PreparedRequest.prepare(method="GET", url="https://example.com/x")
        r = Response(
            request=req,
            status_code=200,
            headers={"Content-Type": "application/json"},
            tags={},
            buffer=b'{"x": 1}',  # type: ignore[arg-type]
            received_at=dt.datetime.fromtimestamp(0, tz=dt.timezone.utc),
        )
        with r.open(mode="rb") as mio:
            assert isinstance(mio, JsonIO)


class TestArrowStreams:
    """``BytesIO.arrow_input_stream`` / ``arrow_output_stream`` yield
    real :class:`pa.NativeFile` handles, decompress on read when the
    holder carries a codec, and commit (with compression when set) on
    write.
    """

    def test_input_stream_in_memory_is_buffer_reader(self) -> None:
        import pyarrow as pa

        b = BytesIO(b"hello world")
        with b.arrow_input_stream() as src:
            assert isinstance(src, pa.NativeFile)
            assert src.read() == b"hello world"

    def test_input_stream_local_path_uses_memory_map(self, tmp_path) -> None:
        import pyarrow as pa

        p = tmp_path / "blob.bin"
        p.write_bytes(b"on-disk-payload")
        b = BytesIO(path=p)
        with b.arrow_input_stream() as src:
            assert isinstance(src, pa.MemoryMappedFile)
            assert src.read() == b"on-disk-payload"

    def test_input_stream_decompresses_codec(self) -> None:
        import gzip
        import pyarrow as pa

        from yggdrasil.data.enums import MediaType, MimeTypes
        from yggdrasil.data.enums.codec import Codecs

        raw = b"compressed payload" * 32
        b = BytesIO(gzip.compress(raw))
        b.with_media_type(MediaType(mime_type=MimeTypes.JSON, codec=Codecs.GZIP))
        with b.arrow_input_stream() as src:
            assert isinstance(src, pa.NativeFile)
            assert src.read() == raw

    def test_output_stream_overwrite_commits_bytes(self) -> None:
        import pyarrow as pa

        b = BytesIO(b"stale")
        with b.arrow_output_stream() as sink:
            assert isinstance(sink, pa.NativeFile)
            sink.write(b"fresh")
        assert b.to_bytes() == b"fresh"

    def test_output_stream_append_extends_payload(self) -> None:
        b = BytesIO(b"head:")
        with b.arrow_output_stream(append=True) as sink:
            sink.write(b"tail")
        assert b.to_bytes() == b"head:tail"

    def test_output_stream_compresses_when_codec_set(self) -> None:
        import gzip

        from yggdrasil.data.enums import MediaType, MimeTypes
        from yggdrasil.data.enums.codec import Codecs

        b = BytesIO()
        b.with_media_type(MediaType(mime_type=MimeTypes.JSON, codec=Codecs.GZIP))
        with b.arrow_output_stream() as sink:
            sink.write(b"writer payload")
        # On the wire we got gzip; round-trip should yield the source bytes.
        assert b.to_bytes().startswith(b"\x1f\x8b")
        assert gzip.decompress(b.to_bytes()) == b"writer payload"

    def test_arrow_streams_round_trip_ipc_with_codec(self) -> None:
        """End-to-end — write IPC into an arrow_output_stream over a
        gzip-tagged buffer, then read it back through
        arrow_input_stream. Pins the contract the format leaves rely
        on.
        """
        import pyarrow as pa
        import pyarrow.ipc as ipc

        from yggdrasil.data.enums import MediaType, MimeTypes
        from yggdrasil.data.enums.codec import Codecs

        table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})

        b = BytesIO()
        b.with_media_type(MediaType(mime_type=MimeTypes.ARROW_IPC, codec=Codecs.GZIP))

        with b.arrow_output_stream() as sink:
            with ipc.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

        # Bytes on disk are gzipped IPC.
        assert b.to_bytes().startswith(b"\x1f\x8b")

        with b.arrow_input_stream() as src:
            reader = ipc.RecordBatchFileReader(src)
            out = reader.read_all()
        assert out.equals(table)

    def test_output_stream_does_not_commit_on_exception(self) -> None:
        b = BytesIO(b"original")
        with pytest.raises(RuntimeError, match="boom"):
            with b.arrow_output_stream() as sink:
                sink.write(b"ignored")
                raise RuntimeError("boom")
        assert b.to_bytes() == b"original"
