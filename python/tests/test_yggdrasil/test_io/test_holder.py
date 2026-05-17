"""Tests for :class:`yggdrasil.io.holder.Holder`.

Covers the four layers that share the class:

1. **Construction dispatch** — :meth:`Holder.__new__` routes by
   scheme (``Holder(path=...)`` → :class:`LocalPath`,
   ``Holder(binary=...)`` → :class:`Memory`) and by media type
   (``IO(path="x.parquet")`` → :class:`ParquetFile`).
2. **Byte primitives** — :meth:`read_mv` / :meth:`write_mv` /
   :meth:`reserve` / :meth:`truncate` / :meth:`clear` /
   :attr:`size` round-trip on :class:`Memory` and
   :class:`LocalPath`.
3. **Cursor + parent chain** — :meth:`Holder.open` returns a
   parent-bound cursor; :attr:`parent` and :attr:`parents` walk the
   chain.
4. **Format registry** — :attr:`Holder.mime_type` registers
   subclasses; :meth:`Holder.class_for_media_type` /
   :meth:`Holder.for_holder` look them up.
"""

from __future__ import annotations

import pathlib

import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.io.memory import Memory
from yggdrasil.io.path.local_path import LocalPath


# ---------------------------------------------------------------------------
# Construction dispatch
# ---------------------------------------------------------------------------


class TestSchemeDispatch:
    """``Holder(...)`` picks the right concrete storage subclass."""

    def test_no_args_picks_memory(self) -> None:
        assert isinstance(Holder(), Memory)

    def test_binary_picks_memory(self) -> None:
        h = Holder(binary=b"hello")
        assert isinstance(h, Memory)
        assert h.read_bytes() == b"hello"

    def test_path_picks_local_path(self, tmp_path) -> None:
        h = Holder(path=str(tmp_path / "out.bin"))
        assert isinstance(h, LocalPath)

    def test_url_picks_local_path_via_scheme(self, tmp_path) -> None:
        h = Holder(url=f"file://{tmp_path / 'x.bin'}")
        assert isinstance(h, LocalPath)

    def test_unknown_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scheme"):
            Holder(scheme="not-a-scheme")

    def test_pathlib_path_picks_local_path(self, tmp_path) -> None:
        h = Holder(path=pathlib.Path(tmp_path) / "x.bin")
        assert isinstance(h, LocalPath)


class TestFormatDispatch:
    """``Holder.__new__`` redirects IO-hierarchy inputs to the format leaf."""

    def test_bytes_io_with_parquet_path_dispatches(self, tmp_path) -> None:
        from yggdrasil.io.bytes_io import BytesIO
        from yggdrasil.io.primitive.parquet_file import ParquetFile

        b = BytesIO(path=str(tmp_path / "x.parquet"))
        assert isinstance(b, ParquetFile)

    def test_bytes_io_with_csv_path_dispatches(self, tmp_path) -> None:
        from yggdrasil.io.bytes_io import BytesIO
        from yggdrasil.io.primitive.csv_file import CSVFile

        b = BytesIO(path=str(tmp_path / "x.csv"))
        assert isinstance(b, CSVFile)

    def test_explicit_media_type_wins_over_extension(self, tmp_path) -> None:
        from yggdrasil.data.enums import MediaType, MimeTypes
        from yggdrasil.io.bytes_io import BytesIO
        from yggdrasil.io.primitive.parquet_file import ParquetFile

        b = BytesIO(
            path=str(tmp_path / "x.csv"),
            media_type=MediaType(MimeTypes.PARQUET),
        )
        assert isinstance(b, ParquetFile)

    def test_storage_holder_media_type_drives_dispatch(self) -> None:
        from yggdrasil.data.enums import MediaType, MimeTypes
        from yggdrasil.io.bytes_io import BytesIO
        from yggdrasil.io.primitive.parquet_file import ParquetFile

        mem = Memory()
        mem.media_type = MediaType(MimeTypes.PARQUET)
        assert isinstance(BytesIO(holder=mem), ParquetFile)

    def test_no_media_type_falls_back_to_bytes_io(self) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        b = BytesIO(b"plain bytes")
        assert type(b) is BytesIO

    def test_data_string_path_dispatches(self, tmp_path) -> None:
        from yggdrasil.io.base import IO
        from yggdrasil.io.primitive.csv_file import CSVFile

        b = IO(data=str(tmp_path / "x.csv"))
        assert isinstance(b, CSVFile)


class TestConflictingArgs:

    def test_holder_and_data_raise(self) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        mem = Memory()
        with pytest.raises(TypeError, match="holder= OR data OR path="):
            BytesIO(b"hi", holder=mem)

    def test_data_and_path_raise(self, tmp_path) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        with pytest.raises(TypeError, match="data= OR path="):
            BytesIO(b"hi", path=str(tmp_path / "x.bin"))


# ---------------------------------------------------------------------------
# Memory byte primitives
# ---------------------------------------------------------------------------


class TestMemoryPrimitives:

    def test_empty(self) -> None:
        m = Memory()
        assert m.size == 0
        assert bytes(m) == b""

    def test_initial_payload(self) -> None:
        m = Memory(b"abcdef")
        assert m.size == 6
        assert m.read_bytes() == b"abcdef"

    def test_reserve_grows_capacity_not_size(self) -> None:
        m = Memory()
        m.reserve(64)
        assert m.size == 0
        assert m.capacity >= 64

    def test_capacity_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 0"):
            Memory(-1)

    def test_pread_subrange(self) -> None:
        m = Memory(b"abcdef")
        assert m.pread(3, 1) == b"bcd"

    def test_pwrite_in_place(self) -> None:
        m = Memory(b"abcdef")
        m.pwrite(b"ZZ", 2)
        assert m.read_bytes() == b"abZZef"

    def test_pwrite_at_end_appends(self) -> None:
        m = Memory(b"hi")
        m.pwrite(b"!", -1)
        assert m.read_bytes() == b"hi!"

    def test_pwrite_negative_indexes_from_end(self) -> None:
        m = Memory(b"abcd")
        m.pwrite(b"X", -2)
        assert m.read_bytes() == b"abXd"

    def test_truncate_shrinks_visible_size(self) -> None:
        m = Memory(b"abcdef")
        m.truncate(3)
        assert m.read_bytes() == b"abc"
        assert m.size == 3

    def test_clear_empties(self) -> None:
        m = Memory(b"data")
        m.clear()
        assert m.size == 0
        assert m.read_bytes() == b""


# ---------------------------------------------------------------------------
# LocalPath byte primitives
# ---------------------------------------------------------------------------


class TestLocalPathPrimitives:

    def test_write_creates_file(self, tmp_path) -> None:
        target = LocalPath(str(tmp_path / "out.bin"))
        target.write_bytes(b"payload")
        assert (tmp_path / "out.bin").read_bytes() == b"payload"
        assert target.size == 7

    def test_read_after_write_round_trip(self, tmp_path) -> None:
        target = LocalPath(str(tmp_path / "x.bin"))
        target.write_bytes(b"round trip")
        assert target.read_bytes() == b"round trip"

    def test_pwrite_in_place(self, tmp_path) -> None:
        target = LocalPath(str(tmp_path / "x.bin"))
        target.write_bytes(b"abcdef")
        target.pwrite(b"ZZ", 2)
        assert target.read_bytes() == b"abZZef"

    def test_truncate_resizes(self, tmp_path) -> None:
        target = LocalPath(str(tmp_path / "x.bin"))
        target.write_bytes(b"abcdef")
        target.truncate(3)
        assert target.read_bytes() == b"abc"

    def test_clear_unlinks_or_empties(self, tmp_path) -> None:
        target = LocalPath(str(tmp_path / "x.bin"))
        target.write_bytes(b"data")
        target.clear()
        assert target.size == 0


# ---------------------------------------------------------------------------
# Cursor + parent chain
# ---------------------------------------------------------------------------


class TestParentChain:
    """:attr:`parent` exposes the underlying byte holder; :attr:`parents`
    iterates the cursor chain out to the root storage."""

    def test_top_level_storage_has_no_parent(self) -> None:
        assert Memory(b"x").parent is None
        assert list(Memory(b"x").parents) == []

    def test_open_cursor_has_parent_set_to_storage(self) -> None:
        mem = Memory(b"hello")
        with mem.open("rb") as cursor:
            assert cursor.parent is mem
            assert list(cursor.parents) == [mem]

    def test_open_cursor_reads_from_parent(self) -> None:
        mem = Memory(b"hello world")
        with mem.open("rb") as cursor:
            assert cursor.read() == b"hello world"

    def test_open_cursor_writes_land_on_parent(self) -> None:
        mem = Memory(b"original")
        with mem.open("wb") as cursor:
            cursor.write(b"REPLACED")
        assert mem.read_bytes() == b"REPLACED"

    def test_explicit_parent_kwarg(self) -> None:
        from yggdrasil.io.base import IO

        mem = Memory(b"borrow")
        cursor = IO(parent=mem, mode="rb")
        assert cursor.parent is mem
        assert cursor.read() == b"borrow"

    def test_holder_kwarg_legacy_alias(self) -> None:
        from yggdrasil.io.base import IO

        mem = Memory(b"legacy")
        cursor = IO(holder=mem, mode="rb")
        assert cursor.parent is mem


class TestOpenFormatDispatch:
    """``LocalPath('x.parquet').open()`` lands on :class:`ParquetFile`."""

    def test_local_path_parquet_opens_as_parquet_file(self, tmp_path) -> None:
        from yggdrasil.io.primitive.parquet_file import ParquetFile

        lp = LocalPath(str(tmp_path / "x.parquet"))
        cursor = lp.open(mode="rb", auto_open=False)
        assert isinstance(cursor, ParquetFile)
        assert cursor.parent is lp

    def test_local_path_csv_opens_as_csv_file(self, tmp_path) -> None:
        from yggdrasil.io.primitive.csv_file import CSVFile

        lp = LocalPath(str(tmp_path / "x.csv"))
        cursor = lp.open(mode="rb", auto_open=False)
        assert isinstance(cursor, CSVFile)

    def test_unknown_extension_falls_back_to_plain_io(self, tmp_path) -> None:
        from yggdrasil.io.base import IO

        lp = LocalPath(str(tmp_path / "x.bin"))
        cursor = lp.open(mode="rb+", auto_open=False)
        assert type(cursor) is IO

    def test_explicit_media_type_overrides_extension(self, tmp_path) -> None:
        from yggdrasil.data.enums import MediaType, MimeTypes
        from yggdrasil.io.primitive.parquet_file import ParquetFile

        lp = LocalPath(str(tmp_path / "x.csv"))
        cursor = lp.open(
            mode="rb", media_type=MediaType(MimeTypes.PARQUET),
            auto_open=False,
        )
        assert isinstance(cursor, ParquetFile)


# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------


class TestFormatRegistry:

    def test_registered_classes_lists_primitives(self) -> None:
        names = set(Holder.registered_classes())
        for expected in ("PARQUET", "CSV", "JSON", "NDJSON", "ARROW_IPC", "XLSX"):
            assert expected in names

    def test_class_for_media_type_parquet(self) -> None:
        from yggdrasil.io.primitive.parquet_file import ParquetFile

        assert Holder.class_for_media_type("parquet") is ParquetFile

    def test_class_for_media_type_csv(self) -> None:
        from yggdrasil.io.primitive.csv_file import CSVFile

        assert Holder.class_for_media_type("text/csv") is CSVFile

    def test_class_for_media_type_default_on_miss(self) -> None:
        sentinel = object()
        assert Holder.class_for_media_type("xx-unknown", default=sentinel) is sentinel

    def test_class_for_media_type_raises_on_miss(self) -> None:
        with pytest.raises(KeyError, match="Cannot coerce"):
            Holder.class_for_media_type("xx-unknown")

    def test_for_holder_dispatches_via_stamped_media(self) -> None:
        from yggdrasil.data.enums import MediaType, MimeTypes
        from yggdrasil.io.primitive.parquet_file import ParquetFile

        mem = Memory()
        mem.media_type = MediaType(MimeTypes.PARQUET)
        leaf = Holder.for_holder(mem)
        assert isinstance(leaf, ParquetFile)

    def test_for_holder_explicit_media_type(self) -> None:
        from yggdrasil.data.enums import MediaType, MimeTypes
        from yggdrasil.io.primitive.csv_file import CSVFile

        mem = Memory()
        leaf = Holder.for_holder(mem, media_type=MediaType(MimeTypes.CSV))
        assert isinstance(leaf, CSVFile)

    def test_for_holder_missing_media_raises(self) -> None:
        with pytest.raises(KeyError, match="No media_type"):
            Holder.for_holder(Memory())

    def test_lazy_bootstrap_resolves_nested_leaves(self) -> None:
        assert Holder.class_for_media_type("application/zip") is not None


# ---------------------------------------------------------------------------
# Lazy media-type slot
# ---------------------------------------------------------------------------


class TestLazyMediaType:

    def test_unseeded_holder_infers_from_url(self, tmp_path) -> None:
        from yggdrasil.data.enums import MimeTypes

        lp = LocalPath(str(tmp_path / "data.csv"))
        assert lp.media_type is not None
        assert lp.media_type.mime_type == MimeTypes.CSV

    def test_explicit_stat_seed_wins(self, tmp_path) -> None:
        from yggdrasil.data.enums import MediaType, MimeTypes
        from yggdrasil.io.io_stats import IOStats

        seed = MediaType(MimeTypes.PARQUET)
        lp = LocalPath(str(tmp_path / "data.csv"), stat=IOStats(media_type=seed))
        assert lp.media_type == seed

    def test_setter_overrides(self) -> None:
        from yggdrasil.data.enums import MediaType, MimeTypes

        mem = Memory()
        mem.media_type = MediaType(MimeTypes.PARQUET)
        assert mem.media_type.mime_type == MimeTypes.PARQUET

    def test_setter_accepts_string(self) -> None:
        from yggdrasil.data.enums import MimeTypes

        mem = Memory()
        mem.media_type = "application/json"
        assert mem.media_type.mime_type == MimeTypes.JSON

    def test_setter_none_clears(self) -> None:
        from yggdrasil.data.enums import MediaType, MimeTypes

        mem = Memory()
        mem.media_type = MediaType(MimeTypes.PARQUET)
        mem.media_type = None
        assert mem.media_type is None


# ---------------------------------------------------------------------------
# Temporary / disposable
# ---------------------------------------------------------------------------


class TestTemporary:

    def test_temporary_clears_on_release(self) -> None:
        m = Memory(b"data", temporary=True)
        m.acquire()
        assert m.size == 4
        m.close()
        assert m.size == 0

    def test_non_temporary_can_be_reopened(self) -> None:
        m = Memory(b"data")
        with m.open("rb") as cursor:
            assert cursor.read() == b"data"


# ---------------------------------------------------------------------------
# Acquire / open lifecycle
# ---------------------------------------------------------------------------


class TestAcquireLifecycle:

    def test_open_returns_cursor_bound_to_parent(self) -> None:
        mem = Memory(b"hello")
        cursor = mem.open("rb")
        assert cursor.parent is mem
        cursor.close()

    def test_with_block_acquires_and_releases(self) -> None:
        mem = Memory()
        with mem:
            assert mem.opened

    def test_open_with_appendable_mode_positions_at_end(self) -> None:
        mem = Memory(b"abc")
        with mem.open("ab+") as cursor:
            cursor.write(b"DEF")
        assert mem.read_bytes() == b"abcDEF"

    def test_open_with_wb_mode_truncates(self) -> None:
        mem = Memory(b"old")
        with mem.open("wb") as cursor:
            cursor.write(b"new")
        assert mem.read_bytes() == b"new"


# ---------------------------------------------------------------------------
# _from_url — cursor vs storage sibling construction
# ---------------------------------------------------------------------------


class TestFromUrl:
    """:meth:`Holder._from_url` builds a sibling Holder for *url*.

    Cursor case: build at *url* directly. Top-level storage case:
    build at ``url.parent`` (self is a container). :class:`Path`
    overrides this — every Path addresses a specific URL.
    """

    def test_cursor_builds_at_url(self) -> None:
        from yggdrasil.io.bytes_io import BytesIO
        from yggdrasil.io.url import URL

        cursor = BytesIO(holder=Memory(b"x"), owns_holder=False, mode="rb")
        sibling = cursor._from_url(URL.from_("/foo/bar/data.parquet"))
        # Cursor branch → URL passes through unchanged.
        assert str(sibling.url).endswith("/foo/bar/data.parquet")

    def test_storage_builds_at_url_parent(self) -> None:
        from yggdrasil.io.url import URL

        mem = Memory()
        sibling = mem._from_url(URL.from_("/foo/bar/data.parquet"))
        # Top-level storage branch → URL.parent.
        assert str(sibling.url).endswith("/foo/bar")

    def test_local_path_override_keeps_url(self, tmp_path) -> None:
        # Path overrides _from_url; LocalPath sibling at the requested URL.
        from yggdrasil.io.url import URL

        lp = LocalPath(str(tmp_path / "a.bin"))
        target = URL.from_(f"file://{tmp_path / 'b.bin'}")
        sibling = lp._from_url(target)
        assert isinstance(sibling, LocalPath)
        assert sibling.url == target


# ---------------------------------------------------------------------------
# Pos / append-at-end sentinel
# ---------------------------------------------------------------------------


class TestPositionSemantics:

    def test_pos_minus_one_is_append_at_end(self) -> None:
        m = Memory(b"hi")
        m.pwrite(b"!", -1)
        assert m.read_bytes() == b"hi!"

    def test_pread_at_minus_one_returns_empty(self) -> None:
        assert Memory(b"hi").pread(0, -1) == b""

    def test_pwrite_minus_two_indexes_from_end(self) -> None:
        m = Memory(b"abcd")
        m.pwrite(b"X", -2)
        assert m.read_bytes() == b"abXd"
