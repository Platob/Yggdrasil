"""Integration tests for :class:`ZipEntryIO` and the
:class:`ZipIO` Fragment surface.

These tests exercise real :class:`ZipIO` + real :class:`ZipEntryIO`
end-to-end — no mocks. The full round-trip is:

    holder ZipIO  →  open_entry_io / read_fragments  →  ZipEntryIO
                  ←  _commit_entry_payload (on _release)

So every assertion about persisted state implicitly verifies
``_acquire`` pulled the right bytes in, the buffer surface mutated
them as expected, and ``_release`` wrote them back into the holder's
archive bytes.

Coverage:

* ZipEntryIO lifecycle — acquire, write, release, dirty/clean
  branches, delete, multi-handle sharing.
* Fragment surface — read_fragments shape, URL fragment selector,
  parent linkage, key filter, open_io toggle, fragment_for both
  for existing and synthetic entries, ZipEntryIO.as_fragment().
* Holder rewrite — modifying one entry preserves siblings;
  delete removes only the named entry.
* Tabular per-entry I/O — Arrow IPC stream OVERWRITE + APPEND
  inside one entry.
"""

from __future__ import annotations

import io as _io
import zipfile

import pytest

from yggdrasil.io.buffer.primitive.zip_io import (
    ZipEntryIO,
    ZipIO,
)
from yggdrasil.io.enums import Mode
from yggdrasil.io.fragment import Fragment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _holder_with_entries(names_to_payload: dict[str, bytes]) -> ZipIO:
    """Build an opened :class:`ZipIO` containing the given named entries.

    Uses zipfile directly rather than the holder's batch-* writer so
    we can stage arbitrary entry names — the per-entry surface is
    name-agnostic, the batch-* prefix is only meaningful for the
    archive-level Arrow IPC reader.
    """
    raw = _io.BytesIO()
    with zipfile.ZipFile(raw, mode="w") as zf:
        for name, payload in names_to_payload.items():
            zf.writestr(name, payload)

    io = ZipIO()
    io.open()
    io.write(raw.getvalue())
    io.seek(0)
    return io


def _archive_names(io: ZipIO) -> list[str]:
    io.seek(0)
    with zipfile.ZipFile(io, mode="r") as zf:
        return sorted(zf.namelist())


def _archive_payload(io: ZipIO, name: str) -> bytes:
    io.seek(0)
    with zipfile.ZipFile(io, mode="r") as zf:
        return zf.read(name)


# ===========================================================================
# ZipEntryIO lifecycle
# ===========================================================================


class TestZipEntryIOLifecycle:
    """The acquire / mutate / release round-trip through buffer commit."""

    def test_acquire_pulls_existing_payload(self):
        io = _holder_with_entries({"hello.txt": b"hi"})
        try:
            entry = io.open_entry("hello.txt") if hasattr(io, "open_entry") else io._open_entry_io("hello.txt", auto_open=True)
            assert isinstance(entry, ZipEntryIO)
            entry.seek(0)
            assert entry.read() == b"hi"
            entry.close()
        finally:
            io.close()

    def test_acquire_on_missing_entry_starts_empty(self):
        io = _holder_with_entries({"other.txt": b"x"})
        try:
            entry = io._open_entry_io("missing.txt", auto_open=True)
            assert entry.is_empty()
            entry.close()
        finally:
            io.close()

    def test_clean_release_does_not_modify_holder(self):
        """Reading without writing leaves archive bytes unchanged."""
        io = _holder_with_entries({"a.txt": b"A", "b.txt": b"B"})
        try:
            before = bytes(io.read())
            io.seek(0)

            entry = io._open_entry_io("a.txt", auto_open=True)
            entry.seek(0)
            _ = entry.read()
            entry.close()

            io.seek(0)
            after = bytes(io.read())
            assert before == after
        finally:
            io.close()

    def test_dirty_release_writes_back(self):
        """Mutating the buffer + closing rewrites the entry's bytes."""
        io = _holder_with_entries({"a.txt": b"A"})
        try:
            entry = io._open_entry_io("a.txt", auto_open=True)
            entry.seek(0)
            entry.truncate(0)
            entry.write(b"AAA")
            entry.close()

            assert _archive_payload(io, "a.txt") == b"AAA"
        finally:
            io.close()

    def test_release_preserves_sibling_entries(self):
        """The slow-path archive rewrite must keep other entries."""
        io = _holder_with_entries({"a.txt": b"A", "b.txt": b"B", "c.txt": b"C"})
        try:
            entry = io._open_entry_io("b.txt", auto_open=True)
            entry.seek(0)
            entry.truncate(0)
            entry.write(b"BBBB")
            entry.close()

            assert _archive_payload(io, "a.txt") == b"A"
            assert _archive_payload(io, "b.txt") == b"BBBB"
            assert _archive_payload(io, "c.txt") == b"C"
        finally:
            io.close()

    def test_write_to_new_entry_appends_via_fast_path(self):
        """Writing to a not-yet-existing entry uses zipfile's APPEND mode."""
        io = _holder_with_entries({"existing.txt": b"keep"})
        try:
            entry = io._open_entry_io("fresh.txt", auto_open=True)
            entry.write(b"new bytes")
            entry.close()

            assert _archive_payload(io, "existing.txt") == b"keep"
            assert _archive_payload(io, "fresh.txt") == b"new bytes"
        finally:
            io.close()

    def test_write_to_empty_holder_creates_archive(self):
        """First-ever entry write into an empty holder builds the zip."""
        io = ZipIO()
        io.open()
        try:
            assert io.is_empty()
            entry = io._open_entry_io("first.txt", auto_open=True)
            entry.write(b"hello")
            entry.close()

            assert not io.is_empty()
            assert _archive_payload(io, "first.txt") == b"hello"
        finally:
            io.close()

    def test_delete_removes_entry_from_archive(self):
        io = _holder_with_entries({"a.txt": b"A", "b.txt": b"B"})
        try:
            entry = io._open_entry_io("a.txt", auto_open=True)
            entry.delete()
            entry.close()

            assert "a.txt" not in _archive_names(io)
            assert _archive_payload(io, "b.txt") == b"B"
        finally:
            io.close()

    def test_delete_then_rewrite_creates_fresh_entry(self):
        io = _holder_with_entries({"a.txt": b"old"})
        try:
            entry = io._open_entry_io("a.txt", auto_open=True)
            entry.delete()
            entry.write(b"new")
            entry.close()

            assert _archive_payload(io, "a.txt") == b"new"
        finally:
            io.close()


# ===========================================================================
# Multi-handle sharing
# ===========================================================================


class TestSharedHandles:
    """Two opens of the same name share the SAME ZipEntryIO."""

    def test_same_name_returns_same_instance(self):
        io = _holder_with_entries({"shared.txt": b"hi"})
        try:
            a = io._open_entry_io("shared.txt", auto_open=True)
            b = io._open_entry_io("shared.txt", auto_open=True)
            assert a is b
            a.close()
        finally:
            io.close()

    def test_different_names_return_different_instances(self):
        io = _holder_with_entries({"a.txt": b"A", "b.txt": b"B"})
        try:
            a = io._open_entry_io("a.txt", auto_open=True)
            b = io._open_entry_io("b.txt", auto_open=True)
            assert a is not b
            assert a.entry_name == "a.txt"
            assert b.entry_name == "b.txt"
            a.close()
            b.close()
        finally:
            io.close()

    def test_release_drops_from_live_map(self):
        io = _holder_with_entries({"a.txt": b"A"})
        try:
            a = io._open_entry_io("a.txt", auto_open=True)
            assert "a.txt" in io._live_entries
            a.close()
            # Closed handle should not pin the live-map entry.
            assert io._live_entries.get("a.txt") is None
        finally:
            io.close()


# ===========================================================================
# Tabular per-entry I/O — Arrow IPC stream
# ===========================================================================


class TestEntryArrowIPC:
    """One Arrow IPC stream per entry, with OVERWRITE + APPEND."""

    def test_arrow_round_trip_through_one_entry(self, arrow_table):
        io = ZipIO()
        io.open()
        try:
            entry = io._open_entry_io("data.arrow", auto_open=True)
            entry.write_arrow_table(arrow_table, mode=Mode.OVERWRITE)
            entry.seek(0)
            result = entry.read_arrow_table()
            entry.close()

            assert result.equals(arrow_table)
        finally:
            io.close()

    def test_arrow_append_concatenates_streams(self, arrow_table):
        """APPEND into a non-empty entry concatenates IPC streams."""
        io = ZipIO()
        io.open()
        try:
            entry = io._open_entry_io("data.arrow", auto_open=True)
            entry.write_arrow_table(arrow_table, mode=Mode.OVERWRITE)
            entry.write_arrow_table(arrow_table, mode=Mode.APPEND)
            entry.seek(0)
            result = entry.read_arrow_table()
            entry.close()

            assert result.num_rows == 2 * arrow_table.num_rows
        finally:
            io.close()

    def test_arrow_persists_across_handles(self, arrow_table):
        """Close + reopen the same entry — bytes survive in the archive."""
        io = ZipIO()
        io.open()
        try:
            first = io._open_entry_io("data.arrow", auto_open=True)
            first.write_arrow_table(arrow_table)
            first.close()

            second = io._open_entry_io("data.arrow", auto_open=True)
            second.seek(0)
            result = second.read_arrow_table()
            second.close()

            assert result.equals(arrow_table)
        finally:
            io.close()

    def test_arrow_entry_options_used_when_no_options_passed(self, arrow_table):
        """Default options on a fresh entry behave as OVERWRITE."""
        io = ZipIO()
        io.open()
        try:
            entry = io._open_entry_io("data.arrow", auto_open=True)
            entry.write_arrow_table(arrow_table)  # no options
            entry.seek(0)
            result = entry.read_arrow_table()
            entry.close()

            assert result.equals(arrow_table)
        finally:
            io.close()


# ===========================================================================
# Fragment surface — read_fragments
# ===========================================================================


class TestReadFragments:
    """ZipIO.read_fragments yields one Fragment per entry."""

    def test_yields_one_fragment_per_entry(self):
        io = _holder_with_entries({"a.txt": b"A", "b.txt": b"B", "c.txt": b"C"})
        try:
            frags = list(io.read_fragments())
            assert len(frags) == 3
            names = {f.io.entry_name for f in frags}
            assert names == {"a.txt", "b.txt", "c.txt"}
        finally:
            io.close()

    def test_fragments_carry_attached_zip_entry_io(self):
        io = _holder_with_entries({"a.txt": b"A"})
        try:
            (frag,) = list(io.read_fragments())
            assert isinstance(frag.io, ZipEntryIO)
            assert frag.io.holder is io
        finally:
            io.close()

    def test_fragment_io_is_readable(self):
        """Eager-attach means the IO is ready to read on yield."""
        io = _holder_with_entries({"a.txt": b"hello"})
        try:
            (frag,) = list(io.read_fragments())
            frag.io.seek(0)
            assert frag.io.read() == b"hello"
        finally:
            io.close()

    def test_parent_is_holder_root_fragment(self):
        io = _holder_with_entries({"a.txt": b"A"})
        try:
            (frag,) = list(io.read_fragments())
            assert frag.parent is not None
            assert frag.parent.parent is None  # root
            assert frag.parent.io is io
            assert frag.depth == 1
        finally:
            io.close()

    def test_root_walks_back_to_holder(self):
        io = _holder_with_entries({"a.txt": b"A"})
        try:
            (frag,) = list(io.read_fragments())
            assert frag.root is frag.parent
            assert frag.root.io is io
        finally:
            io.close()

    def test_all_fragments_share_parent_root(self):
        """One root fragment per read_fragments call, shared across yields."""
        io = _holder_with_entries({"a.txt": b"A", "b.txt": b"B"})
        try:
            frags = list(io.read_fragments())
            roots = {id(f.parent) for f in frags}
            assert len(roots) == 1  # all share the same root instance
        finally:
            io.close()

    def test_empty_holder_yields_nothing(self):
        io = ZipIO()
        io.open()
        try:
            assert list(io.read_fragments()) == []
        finally:
            io.close()

    def test_key_filter_with_glob(self):
        io = _holder_with_entries(
            {"data.json": b"{}", "data.csv": b"x,y", "readme.txt": b"hi"}
        )
        try:
            frags = list(io.read_fragments(key="*.json"))
            assert len(frags) == 1
            assert frags[0].io.entry_name == "data.json"
        finally:
            io.close()

    def test_key_filter_no_match_yields_nothing(self):
        io = _holder_with_entries({"a.txt": b"A"})
        try:
            assert list(io.read_fragments(key="*.parquet")) == []
        finally:
            io.close()

    def test_open_io_false_detaches_io(self):
        """open_io=False yields fragments with io=None."""
        io = _holder_with_entries({"a.txt": b"A"})
        try:
            (frag,) = list(io.read_fragments(open_io=False))
            assert frag.io is None
        finally:
            io.close()

    def test_open_io_true_is_default(self):
        io = _holder_with_entries({"a.txt": b"A"})
        try:
            (frag,) = list(io.read_fragments())
            assert frag.io is not None
        finally:
            io.close()


# ===========================================================================
# Fragment URL composition
# ===========================================================================


class TestFragmentURL:
    """Each fragment's URL is the holder URL with the entry name as
    the URL fragment selector."""

    def test_url_carries_entry_name_as_fragment(self):
        """Skip when the holder has no URL — URL composition only
        kicks in for path-backed holders."""
        io = _holder_with_entries({"data.json": b"{}"})
        try:
            (frag,) = list(io.read_fragments())
            holder_url = getattr(io, "url", None)
            if holder_url is None:
                pytest.skip("Holder has no URL (in-memory ZipIO)")

            url = frag.infos.url
            assert url is not None
            # The entry name should appear as the URL fragment.
            assert getattr(url, "fragment", None) == "data.json"
        finally:
            io.close()

    def test_url_is_none_for_in_memory_holder(self):
        """In-memory ZipIO with no path → entry URL stays None."""
        io = _holder_with_entries({"data.json": b"{}"})
        try:
            (frag,) = list(io.read_fragments())
            if getattr(io, "url", None) is None:
                assert frag.infos.url is None
        finally:
            io.close()


# ===========================================================================
# fragment_for(name)
# ===========================================================================


class TestFragmentForLookup:
    """ZipIO.fragment_for builds a Fragment for a name (existing or not)."""

    def test_fragment_for_existing_entry(self):
        io = _holder_with_entries({"a.txt": b"A"})
        try:
            frag = io.fragment_for("a.txt")
            assert isinstance(frag, Fragment)
            assert frag.io.entry_name == "a.txt"
            frag.io.seek(0)
            assert frag.io.read() == b"A"
        finally:
            io.close()

    def test_fragment_for_synthetic_entry_is_writable(self):
        """A Fragment for a not-yet-existing entry can be written into."""
        io = _holder_with_entries({"existing.txt": b"keep"})
        try:
            frag = io.fragment_for("new.txt")
            assert frag.io.is_empty()
            frag.io.write(b"created")
            frag.io.close()

            assert _archive_payload(io, "new.txt") == b"created"
            assert _archive_payload(io, "existing.txt") == b"keep"
        finally:
            io.close()


# ===========================================================================
# ZipEntryIO.as_fragment
# ===========================================================================


class TestAsFragment:
    """Promote a ZipEntryIO directly into a Fragment."""

    def test_as_fragment_attaches_self(self):
        io = _holder_with_entries({"a.txt": b"A"})
        try:
            entry = io._open_entry_io("a.txt", auto_open=True)
            frag = entry.as_fragment()
            assert frag.io is entry
            entry.close()
        finally:
            io.close()

    def test_as_fragment_parent_is_holder_root(self):
        io = _holder_with_entries({"a.txt": b"A"})
        try:
            entry = io._open_entry_io("a.txt", auto_open=True)
            frag = entry.as_fragment()
            assert frag.parent is not None
            assert frag.parent.io is io
            assert frag.parent.parent is None
            entry.close()
        finally:
            io.close()


# ===========================================================================
# Holder containment / iteration
# ===========================================================================


class TestHolderContainment:
    """ZipIO.__contains__ and __iter__ surface."""

    def test_contains_returns_true_for_existing(self):
        io = _holder_with_entries({"a.txt": b"A"})
        try:
            assert "a.txt" in io
        finally:
            io.close()

    def test_contains_returns_false_for_missing(self):
        io = _holder_with_entries({"a.txt": b"A"})
        try:
            assert "missing.txt" not in io
        finally:
            io.close()

    def test_contains_rejects_non_str(self):
        io = _holder_with_entries({"a.txt": b"A"})
        try:
            assert (123 in io) is False
        finally:
            io.close()

    def test_iter_yields_fragments(self):
        io = _holder_with_entries({"a.txt": b"A", "b.txt": b"B"})
        try:
            frags = list(io)
            assert len(frags) == 2
            assert all(isinstance(f, Fragment) for f in frags)
        finally:
            io.close()

    def test_list_entries_sorted(self):
        io = _holder_with_entries({"b.txt": b"B", "a.txt": b"A", "c.txt": b"C"})
        try:
            assert io.list_entries() == ["a.txt", "b.txt", "c.txt"]
        finally:
            io.close()


# ===========================================================================
# End-to-end: write through fragment, read back through fragment
# ===========================================================================


class TestEndToEndArrow:
    """Full Fragment-shaped pipeline: build target, write Arrow, read Arrow."""

    def test_write_arrow_through_synthetic_fragment_then_read(self, arrow_table):
        io = ZipIO()
        io.open()
        try:
            # Producer side: build a write target via fragment_for.
            target = io.fragment_for("payload.arrow")
            target.io.write_arrow_table(arrow_table)
            target.io.close()

            # Consumer side: enumerate fragments, find the one we wrote.
            frags = list(io.read_fragments())
            (frag,) = [f for f in frags if f.io.entry_name == "payload.arrow"]
            frag.io.seek(0)
            result = frag.io.read_arrow_table()
            frag.io.close()

            assert result.equals(arrow_table)
        finally:
            io.close()

    def test_multiple_entries_each_with_its_own_arrow_stream(self, arrow_table):
        """Two entries, each holding a distinct Arrow IPC stream."""
        io = ZipIO()
        io.open()
        try:
            # Slice arrow_table into two halves so we can tell them apart.
            half = arrow_table.num_rows // 2
            t1 = arrow_table.slice(0, half)
            t2 = arrow_table.slice(half)

            io.fragment_for("first.arrow").io.write_arrow_table(t1)
            io.fragment_for("second.arrow").io.write_arrow_table(t2)
            # Close the live handles via the live-map.
            for entry in list(io._live_entries.values()):
                entry.close()

            f1 = io.fragment_for("first.arrow")
            f1.io.seek(0)
            r1 = f1.io.read_arrow_table()
            f1.io.close()

            f2 = io.fragment_for("second.arrow")
            f2.io.seek(0)
            r2 = f2.io.read_arrow_table()
            f2.io.close()

            assert r1.num_rows == half
            assert r2.num_rows == arrow_table.num_rows - half
            assert r1.equals(t1)
            assert r2.equals(t2)
        finally:
            io.close()