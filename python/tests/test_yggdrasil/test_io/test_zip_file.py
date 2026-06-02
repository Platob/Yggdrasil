"""Tests for :class:`yggdrasil.io.zip_file.ZipFile`."""

from __future__ import annotations

import zipfile as stdlib_zipfile

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.path.memory import Memory
from yggdrasil.path.local_path import LocalPath
from yggdrasil.io.zip_file import ZipFile, ZipEntryFile


class TestRegistration:

    def test_class_for_media_type(self) -> None:
        assert Holder.class_for_media_type("application/zip") is ZipFile

    def test_path_dispatches_zip_ext(self, tmp_path) -> None:
        from yggdrasil.io.base import IO

        b = IO(path=str(tmp_path / "x.zip"))
        assert isinstance(b, ZipFile)


def _build_archive(entries: dict[str, bytes]) -> bytes:
    """Build an in-memory zip with the given ``name → bytes`` entries."""
    import io as _io

    raw = _io.BytesIO()
    with stdlib_zipfile.ZipFile(raw, "w") as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return raw.getvalue()


class _CountingRangedHolder:
    """Minimal ranged holder that records every ``_read_mv(n, pos)``."""

    SUPPORTS_RANGED_RANDOM_ACCESS = True

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.reads: list[tuple[int, int]] = []

    @property
    def size(self) -> int:
        return len(self._data)

    def _read_mv(self, n: int, pos: int):
        self.reads.append((n, pos))
        end = len(self._data) if n < 0 else pos + n
        return memoryview(self._data[pos:end])


class TestRangedBlockReader:
    """The block-caching reader behind zip's ranged metadata / entry reads."""

    def test_small_reads_in_one_block_are_one_fetch(self) -> None:
        from yggdrasil.io.zip_file import _RangedBlockReader

        data = bytes((i % 256) for i in range(3_000_000))  # > 2 blocks
        h = _CountingRangedHolder(data)
        r = _RangedBlockReader(h, len(data))
        r.seek(10)
        assert bytes(r.read(5)) == data[10:15]
        r.seek(2000)
        assert bytes(r.read(100)) == data[2000:2100]
        # Both windows live in block 0 — a single backend fetch.
        assert len(h.reads) == 1

    def test_selective_read_fetches_only_touched_blocks(self) -> None:
        from yggdrasil.io.zip_file import _RangedBlockReader

        block = _RangedBlockReader.BLOCK
        data = bytes((i % 256) for i in range(5 * block))
        h = _CountingRangedHolder(data)
        r = _RangedBlockReader(h, len(data))
        # Read a window near the end (e.g. a central directory) ...
        r.seek(len(data) - 50)
        assert bytes(r.read(50)) == data[-50:]
        # ... and a window in block 1 — never the whole object.
        r.seek(block + 7)
        assert bytes(r.read(20)) == data[block + 7:block + 27]
        assert len(h.reads) == 2  # last block + block 1 only

    def test_seek_whence_and_eof(self) -> None:
        from yggdrasil.io.zip_file import _RangedBlockReader

        data = b"abcdefghij"
        r = _RangedBlockReader(_CountingRangedHolder(data), len(data))
        assert r.seek(-3, 2) == 7 and bytes(r.read()) == b"hij"
        r.seek(0)
        assert bytes(r.read(4)) == b"abcd" and r.tell() == 4
        r.seek(2, 1)
        assert bytes(r.read()) == b"ghij"


class TestListEntries:

    def test_list_returns_entry_names(self) -> None:
        raw = _build_archive({"a.txt": b"alpha", "b.txt": b"beta"})
        mem = Memory(raw)
        zf = ZipFile(holder=mem, owns_holder=False)
        names = {e.entry_name for e in zf.iter_children()}
        assert names == {"a.txt", "b.txt"}


class TestZipEntryLazy:
    """Iterating children doesn't decompress payloads until accessed."""

    def test_child_is_lazy_until_read(self) -> None:
        raw = _build_archive({"data.txt": b"payload"})
        mem = Memory(raw)
        zf = ZipFile(holder=mem, owns_holder=False)
        children = list(zf.iter_children())
        assert len(children) == 1
        entry = children[0]
        assert isinstance(entry, ZipEntryFile)
        # Materialization happens on first read.
        assert entry.to_bytes() == b"payload"


class TestWriteArchive:

    def test_write_arrow_batches_packs_parquet_entry(self, tmp_path) -> None:
        table = pa.table({"x": [1, 2, 3]})
        path = LocalPath(str(tmp_path / "out.zip"))
        with path.open("wb") as zf:
            zf.write_arrow_batches(iter(table.to_batches()))

        # The resulting archive has at least one entry.
        with stdlib_zipfile.ZipFile(tmp_path / "out.zip", "r") as zfile:
            assert len(zfile.namelist()) >= 1


# ---------------------------------------------------------------------------
# Read-path single-open optimization
# ---------------------------------------------------------------------------


class TestReadArrowBatchesSingleOpen:
    """``_read_arrow_batches`` opens the parent archive exactly once
    instead of N+1 times (once for the directory walk, then once per
    child via the lazy materialization path)."""

    def test_single_zipfile_open_per_read(self, monkeypatch) -> None:
        # Build a 3-entry parquet archive in-memory.
        from yggdrasil.io.zip_file import (
            ZipFile as _Zip, ZipOptions,
        )
        import yggdrasil.io.zip_file as zip_module
        from yggdrasil.io.parquet_file import ParquetFile

        tables = [
            pa.table({"x": [i, i + 1, i + 2]}) for i in range(3)
        ]
        # Encode each table as parquet bytes, then bundle into one zip.
        entries = {}
        for i, tbl in enumerate(tables):
            buf = Memory()
            pq = ParquetFile(holder=buf, owns_holder=False)
            pq.write_arrow_batches(iter(tbl.to_batches()))
            entries[f"part-{i}.parquet"] = buf.to_bytes()

        raw = _build_archive(entries)
        mem = Memory(raw)
        zf = _Zip(holder=mem, owns_holder=False)

        # Patch the stdlib constructor to count opens.
        original = zip_module.zipfile.ZipFile
        opens = {"n": 0}

        def counting_open(*a, **kw):
            opens["n"] += 1
            return original(*a, **kw)

        monkeypatch.setattr(zip_module.zipfile, "ZipFile", counting_open)

        batches = list(zf._read_arrow_batches(ZipOptions()))
        assert len(batches) >= 3  # one batch per parquet entry
        assert opens["n"] == 1, (
            f"expected 1 zipfile open, got {opens['n']}"
        )


# ---------------------------------------------------------------------------
# Append-path streaming
# ---------------------------------------------------------------------------


class TestAppendStreaming:
    """APPEND mode used to materialize every survivor's decompressed
    bytes into a Python list, peaking at sum(uncompressed_sizes). The
    new path streams each survivor chunk-by-chunk from source to
    destination, so only the active 1 MiB chunk is in memory at any
    moment."""

    def test_append_preserves_existing_entries(self) -> None:
        from yggdrasil.io.zip_file import (
            ZipFile as _Zip, ZipOptions,
        )
        from yggdrasil.enums import Mode

        # Start with one parquet entry; append another with a
        # different name; verify both survive.
        from yggdrasil.io.parquet_file import ParquetFile

        # Pre-existing entry.
        existing_table = pa.table({"x": [10, 20]})
        pq_buf = Memory()
        ParquetFile(holder=pq_buf, owns_holder=False).write_arrow_batches(
            iter(existing_table.to_batches())
        )

        raw = _build_archive({"old.parquet": pq_buf.to_bytes()})
        mem = Memory(raw)
        zf = _Zip(holder=mem, owns_holder=False)

        new_table = pa.table({"x": [30, 40]})
        zf.write_arrow_batches(
            iter(new_table.to_batches()),
            options=ZipOptions(
                entry_name="new.parquet", mode=Mode.APPEND,
            ),
        )

        # Both entries must survive the append.
        with stdlib_zipfile.ZipFile(__import__("io").BytesIO(mem.to_bytes()), "r") as zfile:
            names = set(zfile.namelist())
            assert names == {"old.parquet", "new.parquet"}

    def test_append_replaces_same_named_entry(self) -> None:
        from yggdrasil.io.zip_file import (
            ZipFile as _Zip, ZipOptions,
        )
        from yggdrasil.enums import Mode
        from yggdrasil.io.parquet_file import ParquetFile

        # First write (becomes the survivor that gets replaced).
        first = pa.table({"x": [1, 2, 3]})
        pq_buf = Memory()
        ParquetFile(holder=pq_buf, owns_holder=False).write_arrow_batches(
            iter(first.to_batches())
        )
        raw = _build_archive({"data.parquet": pq_buf.to_bytes()})
        mem = Memory(raw)
        zf = _Zip(holder=mem, owns_holder=False)

        # Replace with new bytes under the same entry name.
        replacement = pa.table({"x": [99, 100]})
        zf.write_arrow_batches(
            iter(replacement.to_batches()),
            options=ZipOptions(
                entry_name="data.parquet", mode=Mode.APPEND,
            ),
        )

        # Read back — must reflect the replacement, not the original.
        read_back = list(zf._read_arrow_batches(ZipOptions()))
        assert read_back
        combined = pa.Table.from_batches(read_back)
        assert combined["x"].to_pylist() == [99, 100]


# ---------------------------------------------------------------------------
# Per-entry write surface
# ---------------------------------------------------------------------------


class TestEntryReadWrite:
    """`z.entry(name, *, mode)` returns a :class:`ZipEntryFile` that's
    both readable and writable — single handle for the per-entry
    surface."""

    def test_entry_open_writes_raw_bytes(self) -> None:
        from yggdrasil.io.zip_file import ZipFile as _Zip

        zf = _Zip(holder=Memory(), owns_holder=False)
        with zf.entry("notes.txt").open("wb") as f:
            f.write(b"hello entry")

        # Reread off the underlying archive bytes.
        with stdlib_zipfile.ZipFile(
            __import__("io").BytesIO(zf.to_bytes()), "r",
        ) as zfile:
            assert zfile.namelist() == ["notes.txt"]
            assert zfile.read("notes.txt") == b"hello entry"

    def test_entry_open_committed_payload_replaces_prior_entry(self) -> None:
        from yggdrasil.io.zip_file import ZipFile as _Zip

        # Pre-populate with one entry under the same name.
        raw = _build_archive({"x.bin": b"old"})
        zf = _Zip(holder=Memory(raw), owns_holder=False)

        with zf.entry("x.bin").open("wb") as f:
            f.write(b"new")

        with stdlib_zipfile.ZipFile(
            __import__("io").BytesIO(zf.to_bytes()), "r",
        ) as zfile:
            assert zfile.read("x.bin") == b"new"

    def test_entry_open_preserves_other_entries(self) -> None:
        from yggdrasil.io.zip_file import ZipFile as _Zip

        raw = _build_archive({"a.bin": b"alpha", "b.bin": b"beta"})
        zf = _Zip(holder=Memory(raw), owns_holder=False)

        with zf.entry("c.bin").open("wb") as f:
            f.write(b"gamma")

        with stdlib_zipfile.ZipFile(
            __import__("io").BytesIO(zf.to_bytes()), "r",
        ) as zfile:
            assert set(zfile.namelist()) == {"a.bin", "b.bin", "c.bin"}
            assert zfile.read("a.bin") == b"alpha"
            assert zfile.read("b.bin") == b"beta"
            assert zfile.read("c.bin") == b"gamma"

    def test_entry_open_exception_drops_staged_bytes(self) -> None:
        from yggdrasil.io.zip_file import ZipFile as _Zip

        zf = _Zip(holder=Memory(), owns_holder=False)
        try:
            with zf.entry("aborted.bin").open("wb") as f:
                f.write(b"partial")
                raise RuntimeError("user code raised")
        except RuntimeError:
            pass

        # Archive remains empty — the staged bytes never reached the
        # central directory.
        assert zf.size == 0

    def test_entry_write_arrow_batches_packs_parquet(self) -> None:
        from yggdrasil.io.zip_file import ZipFile as _Zip

        table = pa.table({"x": [1, 2, 3]})
        zf = _Zip(holder=Memory(), owns_holder=False)
        zf.entry("data.parquet").write_arrow_batches(iter(table.to_batches()))

        with stdlib_zipfile.ZipFile(
            __import__("io").BytesIO(zf.to_bytes()), "r",
        ) as zfile:
            assert zfile.namelist() == ["data.parquet"]

    def test_entry_returns_zip_entry_file(self) -> None:
        from yggdrasil.io.zip_file import ZipFile as _Zip

        zf = _Zip(holder=Memory(), owns_holder=False)
        entry = zf.entry("x.bin")
        assert isinstance(entry, ZipEntryFile)

    def test_entry_read_existing(self) -> None:
        from yggdrasil.io.zip_file import ZipFile as _Zip

        raw = _build_archive({"x.bin": b"alpha"})
        zf = _Zip(holder=Memory(raw), owns_holder=False)
        entry = zf.entry("x.bin")
        assert entry.to_bytes() == b"alpha"

    def test_entry_mode_error_if_exists(self) -> None:
        from yggdrasil.io.zip_file import ZipFile as _Zip
        from yggdrasil.enums import Mode

        raw = _build_archive({"x.bin": b"old"})
        zf = _Zip(holder=Memory(raw), owns_holder=False)
        with pytest.raises(FileExistsError):
            with zf.entry("x.bin", mode=Mode.ERROR_IF_EXISTS).open("wb") as f:
                f.write(b"new")

    def test_entry_mode_ignore_skips_existing(self) -> None:
        from yggdrasil.io.zip_file import ZipFile as _Zip
        from yggdrasil.enums import Mode

        raw = _build_archive({"x.bin": b"keep-me"})
        zf = _Zip(holder=Memory(raw), owns_holder=False)
        with zf.entry("x.bin", mode=Mode.IGNORE).open("wb") as f:
            f.write(b"ignored")

        with stdlib_zipfile.ZipFile(
            __import__("io").BytesIO(zf.to_bytes()), "r",
        ) as zfile:
            assert zfile.read("x.bin") == b"keep-me"

    def test_entry_mode_append_concatenates(self) -> None:
        from yggdrasil.io.zip_file import ZipFile as _Zip
        from yggdrasil.enums import Mode

        raw = _build_archive({"log.txt": b"first\n"})
        zf = _Zip(holder=Memory(raw), owns_holder=False)
        with zf.entry("log.txt", mode=Mode.APPEND).open("wb") as f:
            f.write(b"second\n")

        with stdlib_zipfile.ZipFile(
            __import__("io").BytesIO(zf.to_bytes()), "r",
        ) as zfile:
            assert zfile.read("log.txt") == b"first\nsecond\n"


class TestParallelEntryWrites:
    """Concurrent ``z.entry("a") / z.entry("b") / ...`` writes must
    end up with every committed entry in the archive — the per-entry
    commit holds a per-archive lock for the central-directory rewrite
    so the directory never tears."""

    def test_concurrent_distinct_entries_all_persist(self) -> None:
        import concurrent.futures as cf
        from yggdrasil.io.zip_file import ZipFile as _Zip

        zf = _Zip(holder=Memory(), owns_holder=False)
        n = 16

        def write_one(i: int) -> None:
            with zf.entry(f"part-{i:02d}.bin").open("wb") as f:
                # Variable-size payloads so the test exercises
                # different commit timings.
                f.write(f"payload-{i}".encode() * (i + 1))

        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write_one, range(n)))

        with stdlib_zipfile.ZipFile(
            __import__("io").BytesIO(zf.to_bytes()), "r",
        ) as zfile:
            names = sorted(zfile.namelist())
            assert names == [f"part-{i:02d}.bin" for i in range(n)]
            # Every payload must be intact.
            for i in range(n):
                expected = f"payload-{i}".encode() * (i + 1)
                assert zfile.read(f"part-{i:02d}.bin") == expected
