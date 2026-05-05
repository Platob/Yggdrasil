"""External-writer / external-reader compatibility tests for
``yggdrasil.io.buffer.BytesIO``.

The buffer is routinely passed to third-party libraries that expect a
plain binary file-like object — the Databricks ``DBXFileManager``
flow drives it via ``df.to_csv(fh, ...)``, the Power Query connector
hands it to pyarrow's parquet/IPC writers, the FastAPI service
streams it back through gzip, and so on. Any drift from the standard
``io.IOBase`` contract breaks those callers in surprising ways, so we
pin the contract here against real third-party libraries (pandas,
pyarrow, polars, zipfile, gzip, pickle, csv) rather than
hand-rolling stub assertions.

Optional-dependency paths are gated with ``pytest.importorskip`` so
the suite passes on a base install.
"""

from __future__ import annotations

import csv
import gzip
import io as _stdio
import pickle
import struct
import zipfile

import pytest

from yggdrasil.io.buffer import BytesIO


# ---------------------------------------------------------------------------
# stdlib io.IOBase contract — what every third-party writer assumes
# ---------------------------------------------------------------------------


class TestStdlibIOContract:
    """Make sure BytesIO matches the documented ``io.IOBase`` surface
    on the operations external writers rely on. Each assertion below
    pins a real-world expectation we have hit in the wild."""

    def test_advertises_rwseek(self):
        buf = BytesIO()
        assert buf.readable() is True
        assert buf.writable() is True
        assert buf.seekable() is True

    def test_write_returns_bytes_written(self):
        buf = BytesIO()
        # zipfile and pyarrow both use the return value to advance
        # internal offsets — returning None or a wrong count silently
        # corrupts the output.
        assert buf.write(b"abc") == 3
        assert buf.write(b"") == 0

    def test_tell_after_write_matches_size(self):
        buf = BytesIO()
        buf.write(b"abcd")
        assert buf.tell() == 4
        buf.write(b"ef")
        assert buf.tell() == 6

    def test_tell_after_seek_set(self):
        buf = BytesIO(b"abcdef")
        buf.seek(3)
        assert buf.tell() == 3

    def test_seek_returns_new_position(self):
        # io.IOBase.seek must return the new absolute position.
        buf = BytesIO(b"abcdef")
        assert buf.seek(2) == 2
        assert buf.seek(0, _stdio.SEEK_END) == 6
        assert buf.seek(-2, _stdio.SEEK_END) == 4
        assert buf.seek(0, _stdio.SEEK_CUR) == 4

    def test_seek_past_end_is_allowed(self):
        # Writers like zipfile pad with seeks past EOF. The contract
        # is: seek succeeds, subsequent write fills the gap.
        buf = BytesIO(b"abc")
        assert buf.seek(8) == 8
        buf.write(b"Z")
        assert buf.to_bytes() == b"abc\x00\x00\x00\x00\x00Z"

    def test_seek_set_is_absolute(self):
        # Pin: SEEK_SET is *not* relative to the current cursor.
        buf = BytesIO(b"abcdef")
        buf.seek(4)
        buf.seek(2)
        assert buf.tell() == 2
        assert buf.read(2) == b"cd"

    def test_seek_minus_one_is_end_sentinel(self):
        # The only allowed negative SEEK_SET offset is -1, which we
        # treat as a "go to end" shortcut to mirror read(-1).
        buf = BytesIO(b"abcdef")
        assert buf.seek(-1) == 6
        assert buf.read() == b""

    @pytest.mark.parametrize("offset", [-2, -3, -100])
    def test_seek_other_negative_set_raises(self, offset):
        # Stdlib io.BytesIO raises ValueError on negative SEEK_SET;
        # third-party writers that *accidentally* pass a negative
        # offset (e.g. a bug computing a header size) need a loud
        # failure instead of a silent jump into the middle of the
        # buffer.
        buf = BytesIO(b"abcdef")
        with pytest.raises(ValueError):
            buf.seek(offset)
        # Failed seek must not advance the cursor.
        assert buf.tell() == 0

    def test_seek_minus_one_matches_seek_end_zero(self):
        # The end sentinel must agree with the canonical
        # seek(0, SEEK_END) for any consumer that uses tell() to
        # discover the buffer's size after a write.
        buf = BytesIO(b"streamed bytes")
        a = buf.seek(-1)
        b = buf.seek(0, _stdio.SEEK_END)
        assert a == b == buf.size

    def test_overwrite_in_place_via_seek(self):
        # zipfile rewinds to patch the local file header crc/size
        # after streaming the body. Verify a back-seek + write does
        # not truncate the trailing bytes.
        buf = BytesIO()
        buf.write(b"HEADER--BODYBODY")
        buf.seek(0)
        buf.write(b"PATCHED!")
        assert buf.to_bytes() == b"PATCHED!BODYBODY"

    def test_truncate_then_write_extends(self):
        buf = BytesIO(b"abcdef")
        buf.truncate(3)
        buf.seek(0, _stdio.SEEK_END)
        buf.write(b"XYZ")
        assert buf.to_bytes() == b"abcXYZ"

    def test_write_with_memoryview_slice(self):
        # pyarrow hands us memoryview slices off a larger buffer.
        # Slicing must be honored — we should not write the parent
        # buffer's full contents.
        payload = bytearray(b"0123456789")
        mv = memoryview(payload)[2:6]
        buf = BytesIO()
        assert buf.write(mv) == 4
        assert buf.to_bytes() == b"2345"

    def test_flush_is_noop_and_returns_none(self):
        buf = BytesIO()
        buf.write(b"x")
        assert buf.flush() is None  # idiomatic stdlib return


# ---------------------------------------------------------------------------
# zipfile — exercises seek-back-to-patch-header
# ---------------------------------------------------------------------------


class TestZipfileWriter:
    def test_zip_roundtrip(self):
        # zipfile.ZipFile streams entries, then back-seeks to patch
        # each local file header's crc/sizes. This is the most
        # demanding use of seek/write/tell in the stdlib.
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("a.txt", b"alpha contents")
            zf.writestr("nested/b.bin", b"\x00\x01\x02" * 64)

        # Reset cursor; zipfile.ZipFile in read mode seeks to end to
        # find the central directory.
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            assert sorted(zf.namelist()) == ["a.txt", "nested/b.bin"]
            assert zf.read("a.txt") == b"alpha contents"
            assert zf.read("nested/b.bin") == b"\x00\x01\x02" * 64

    def test_zip_via_path_bound_buffer(self, tmp_path):
        # The Databricks flow opens a path-bound BytesIO("wb") and
        # hands it to a writer. Make sure path-bound buffers behave
        # the same.
        target = tmp_path / "out.zip"
        with BytesIO(path=str(target), mode="wb") as buf:
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("hello.txt", b"world")
        # File flushed to disk on close.
        with zipfile.ZipFile(str(target), "r") as zf:
            assert zf.read("hello.txt") == b"world"


# ---------------------------------------------------------------------------
# gzip — pure write/seek-to-end
# ---------------------------------------------------------------------------


class TestGzipWriter:
    def test_gzip_roundtrip(self):
        payload = b"Brent ICE front-month settle\n" * 200
        buf = BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(payload)
        buf.seek(0)
        with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
            assert gz.read() == payload

    def test_gzip_streamed_writes_match(self):
        # gzip writes in chunks; verify partial writes accumulate.
        buf = BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            for i in range(50):
                gz.write(f"line {i}\n".encode())
        buf.seek(0)
        decoded = gzip.GzipFile(fileobj=buf, mode="rb").read()
        assert decoded == b"".join(f"line {i}\n".encode() for i in range(50))


# ---------------------------------------------------------------------------
# pickle — uses write() and the buffer protocol
# ---------------------------------------------------------------------------


class TestPickleStream:
    def test_pickle_dump_load(self):
        buf = BytesIO()
        obj = {"k": [1, 2, 3], "s": "héllo"}
        pickle.dump(obj, buf)
        buf.seek(0)
        assert pickle.load(buf) == obj

    def test_multiple_pickles_streamed(self):
        # tell()/seek() must let consumers checkpoint.
        buf = BytesIO()
        offsets = []
        for v in (1, "two", [3, 3, 3]):
            offsets.append(buf.tell())
            pickle.dump(v, buf)

        out = []
        for off in offsets:
            buf.seek(off)
            out.append(pickle.load(buf))
        assert out == [1, "two", [3, 3, 3]]


# ---------------------------------------------------------------------------
# csv — pure text writer, exercised through io.TextIOWrapper
# ---------------------------------------------------------------------------


class TestCSVStdlib:
    def test_csv_writer_via_textwrapper(self):
        # The Databricks flow uses df.to_csv(fh) where fh is opened
        # in "wb" mode. pandas wraps it in a TextIOWrapper. Verify
        # that wrapping works.
        buf = BytesIO()
        wrapper = _stdio.TextIOWrapper(
            buf, encoding="utf-8", newline="", write_through=True
        )
        writer = csv.writer(wrapper)
        writer.writerow(["a", "b", "c"])
        writer.writerow([1, 2, 3])
        wrapper.flush()
        wrapper.detach()  # don't close the underlying buffer
        assert buf.to_bytes() == b"a,b,c\r\n1,2,3\r\n"


# ---------------------------------------------------------------------------
# pandas — direct-handle writers as used in yggdrasil.databricks.fs
# ---------------------------------------------------------------------------


@pytest.fixture
def pd():
    return pytest.importorskip("pandas")


class TestPandasWriter:
    def test_to_csv_roundtrip(self, pd):
        # Mirrors yggdrasil.databricks.fs.to_csv: open("wb") then
        # df.to_csv(fh, ...). pandas writes utf-8 bytes through a
        # TextIOWrapper internally.
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        buf = BytesIO()
        df.to_csv(buf, index=False)

        buf.seek(0)
        roundtrip = pd.read_csv(buf)
        pd.testing.assert_frame_equal(roundtrip, df)

    def test_to_csv_compressed_gzip(self, pd):
        # pandas' compression=`gzip` opens a GzipFile around the
        # handle, which means seek(0, SEEK_END) and tell() in tight
        # loops. Make sure that still works.
        df = pd.DataFrame({"x": range(64), "y": [str(i) for i in range(64)]})
        buf = BytesIO()
        df.to_csv(buf, index=False, compression="gzip")
        # Decompress manually to prove it's a real gzip stream.
        buf.seek(0)
        decompressed = gzip.decompress(buf.to_bytes())
        assert decompressed.startswith(b"x,y\n0,0\n")

    def test_to_json_roundtrip(self, pd):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        buf = BytesIO()
        df.to_json(buf, orient="records", lines=True)
        # to_json should leave a tell() consistent with size.
        assert buf.tell() == buf.size
        buf.seek(0)
        roundtrip = pd.read_json(buf, orient="records", lines=True)
        pd.testing.assert_frame_equal(roundtrip, df)


# ---------------------------------------------------------------------------
# pyarrow Parquet / IPC — heavy seek/tell users
# ---------------------------------------------------------------------------


@pytest.fixture
def pa():
    return pytest.importorskip("pyarrow")


class TestPyArrowWriter:
    def test_parquet_write_and_read(self, pa):
        pq = pytest.importorskip("pyarrow.parquet")
        table = pa.table({"a": pa.array([1, 2, 3]), "b": pa.array(["x", "y", "z"])})

        buf = BytesIO()
        pq.write_table(table, buf)

        # Parquet appends a magic footer; the reader seeks to end to
        # find it, then jumps backwards. Pin that this works against
        # a fresh BytesIO.
        buf.seek(0)
        out = pq.read_table(buf)
        assert out.equals(table)

    def test_parquet_via_native_file(self, pa):
        pq = pytest.importorskip("pyarrow.parquet")
        # arrow_io() exposes a NativeFile around the buffer — that's
        # the path the FastAPI service uses for streaming. The
        # NativeFile wrapper closes the underlying object on exit,
        # so we read back from a fresh handle over the same bytes.
        table = pa.table({"a": [10, 20, 30]})
        buf = BytesIO()
        with buf.arrow_io("wb") as nf:
            pq.write_table(table, nf)
        payload = buf.to_bytes()
        assert payload[:4] == b"PAR1"
        out = pq.read_table(_stdio.BytesIO(payload))
        assert out.equals(table)

    def test_arrow_ipc_stream(self, pa):
        table = pa.table({"a": [1, 2, 3]})
        buf = BytesIO()
        with pa.ipc.new_stream(buf, table.schema) as writer:
            writer.write_table(table)
        buf.seek(0)
        with pa.ipc.open_stream(buf) as reader:
            out = reader.read_all()
        assert out.equals(table)

    def test_arrow_ipc_file_seekable(self, pa):
        # ``new_file`` writes a footer and is the seek-heavy variant.
        table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        buf = BytesIO()
        with pa.ipc.new_file(buf, table.schema) as writer:
            writer.write_table(table)
        # IPC file format requires reader to seek backwards from end.
        buf.seek(0)
        with pa.ipc.open_file(buf) as reader:
            out = reader.read_all()
        assert out.equals(table)


# ---------------------------------------------------------------------------
# polars — Arrow-backed writer that also relies on the io contract
# ---------------------------------------------------------------------------


@pytest.fixture
def pl():
    return pytest.importorskip("polars")


class TestPolarsWriter:
    def test_write_csv(self, pl):
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        buf = BytesIO()
        df.write_csv(buf)
        buf.seek(0)
        out = pl.read_csv(buf)
        assert out.equals(df)

    def test_write_parquet(self, pl):
        pytest.importorskip("pyarrow")
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        buf = BytesIO()
        df.write_parquet(buf)
        buf.seek(0)
        out = pl.read_parquet(buf)
        assert out.equals(df)


# ---------------------------------------------------------------------------
# struct — exercises the int read/write helpers in lockstep with stdlib
# ---------------------------------------------------------------------------


class TestStructInterop:
    def test_struct_pack_then_typed_read(self):
        # External code may pack via struct and read via our helpers.
        # Pin the wire format (little-endian) by mixing both sides.
        payload = struct.pack("<iqf", -7, 1 << 33, 1.5)
        buf = BytesIO(payload)
        assert buf.read_int32() == -7
        assert buf.read_int64() == 1 << 33
        assert abs(buf.read_f32() - 1.5) < 1e-6

    def test_typed_write_then_struct_unpack(self):
        buf = BytesIO()
        buf.write_int32(-7)
        buf.write_int64(1 << 33)
        buf.write_f32(1.5)
        a, b, c = struct.unpack("<iqf", buf.to_bytes())
        assert (a, b) == (-7, 1 << 33)
        assert abs(c - 1.5) < 1e-6


# ---------------------------------------------------------------------------
# Spilled-buffer behavior under external writers
# ---------------------------------------------------------------------------


class TestSpilledExternalWriters:
    def test_zip_roundtrip_after_spill(self):
        # Force-spill via small spill threshold; writers must keep
        # working when the backing flips from memory to a real fd.
        buf = BytesIO(spill_bytes=64)
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("big.bin", b"X" * 4096)
            zf.writestr("small.bin", b"hi")
        assert buf.spilled  # sanity: we actually exercised the spill path
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            assert zf.read("big.bin") == b"X" * 4096
            assert zf.read("small.bin") == b"hi"

    def test_pickle_roundtrip_after_spill(self):
        buf = BytesIO(spill_bytes=64)
        obj = {"chunk": "x" * 1024, "n": 7}
        pickle.dump(obj, buf)
        assert buf.spilled
        buf.seek(0)
        assert pickle.load(buf) == obj
