"""Tests for ``yggdrasil.io.buffer.nested`` (excluding the Delta sub-package).

Covers:

- :class:`NestedIO` / :class:`NestedOptions` base contract.
- :class:`FolderIO` flat folders, recursive sub-folders, partition
  columns, save modes, child minting, name validation, and the
  module-private helpers (``_parse_kv_segment``,
  ``_partition_path_segment``, ``_coerce_partition_column``).
- :class:`PartitionedFolderIO` thin wrapper.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pytest

from yggdrasil.data.schema import Field
from yggdrasil.io.buffer.nested import (
    FolderIO,
    FolderOptions,
    NestedIO,
    NestedOptions,
)
from yggdrasil.io.buffer.nested.folder_io import (
    _coerce_partition_column,
    _parse_kv_segment,
    _partition_path_segment,
)
from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.buffer.primitive import ParquetIO, CsvIO
from yggdrasil.io.enums import MimeTypes, Mode


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


def _flat_table() -> pa.Table:
    return pa.Table.from_pylist(
        [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]
    )


def _partitioned_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {"a": 1, "year": "2024"},
            {"a": 2, "year": "2024"},
            {"a": 3, "year": "2025"},
        ]
    )


# ---------------------------------------------------------------------------
# NestedIO base
# ---------------------------------------------------------------------------


class TestNestedIOBase:
    def test_path_required(self):
        with pytest.raises(ValueError):
            FolderIO()

    def test_path_positional_or_keyword(self, tmp_path: Path):
        io_a = FolderIO(str(tmp_path))
        io_b = FolderIO(path=str(tmp_path))
        assert str(io_a.path) == str(io_b.path)

    def test_path_keyword_wins_over_positional(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        io = FolderIO(str(tmp_path), path=str(sub))
        assert str(io.path).endswith("sub")

    def test_default_mime_type_base_is_none(self):
        assert NestedIO.default_mime_type() is None

    def test_folder_default_mime_type(self):
        assert FolderIO.default_mime_type() == MimeTypes.FOLDER

    def test_options_class_default(self):
        assert NestedIO.options_class() is NestedOptions
        assert FolderIO.options_class() is FolderOptions

# ---------------------------------------------------------------------------
# FolderIO — flat folder behavior
# ---------------------------------------------------------------------------


class TestFolderIOFlat:
    def test_round_trip(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        io.write_arrow_table(_flat_table())
        out = io.read_arrow_table()
        assert out.num_rows == 3
        assert set(out.column_names) == {"a", "b"}

    def test_default_child_format_is_parquet(self, tmp_path: Path):
        FolderIO(path=str(tmp_path)).write_arrow_table(_flat_table())
        files = sorted(os.listdir(str(tmp_path)))
        assert files
        assert all(f.endswith(".parquet") for f in files)

    def test_child_media_type_override(self, tmp_path: Path):
        FolderIO(path=str(tmp_path)).write_arrow_table(
            _flat_table(),
            child_media_type=MimeTypes.CSV,
        )
        files = sorted(os.listdir(str(tmp_path)))
        assert files
        assert all(f.endswith(".csv") for f in files)

    def test_is_empty_on_missing_folder(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path / "missing"))
        assert io.is_empty()

    def test_is_empty_on_empty_folder(self, tmp_path: Path):
        assert FolderIO(path=str(tmp_path)).is_empty()

    def test_is_empty_after_write(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        io.write_arrow_table(_flat_table())
        assert not io.is_empty()

    def test_collect_schema_merges_columns(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        io.write_arrow_table(_flat_table())
        names = list(io.collect_schema().field_names())
        assert set(names) == {"a", "b"}

    def test_iter_children_returns_primitive_io(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        io.write_arrow_table(_flat_table())

        children = list(io.iter_children())
        assert children
        # Concrete tabular leaves are BytesIO subclasses with a
        # format-specific class (not raw BytesIO).
        assert all(isinstance(c, BytesIO) and type(c) is not BytesIO for c in children)
        assert all(c.parent is io for c in children)

    def test_iter_children_empty_folder(self, tmp_path: Path):
        assert list(FolderIO(path=str(tmp_path)).iter_children()) == []

    def test_iter_children_missing_folder(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path / "missing"))
        assert list(io.iter_children()) == []

    def test_iter_children_skips_hidden_entries(self, tmp_path: Path):
        # Hidden files (leading dot) are filtered out.
        (tmp_path / ".hidden.parquet").touch()
        ParquetIO(path=str(tmp_path / "visible.parquet")).write_arrow_table(
            _flat_table()
        )

        names = sorted(c.path.name for c in FolderIO(path=str(tmp_path)).iter_children())
        assert ".hidden.parquet" not in names
        assert "visible.parquet" in names

    def test_recursive_descent_through_subfolder(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        ParquetIO(path=str(sub / "a.parquet")).write_arrow_table(_flat_table())

        io = FolderIO(path=str(tmp_path))
        children = list(io.iter_children())
        # First-level enumeration yields a sub-FolderIO.
        assert any(isinstance(c, FolderIO) for c in children)

        # But the read path recurses transparently.
        out = io.read_arrow_table()
        assert out.num_rows == 3

    def test_recursive_false_clips_at_top_level(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        ParquetIO(path=str(sub / "a.parquet")).write_arrow_table(_flat_table())

        io = FolderIO(path=str(tmp_path), recursive=False)
        children = list(io.iter_children())
        assert children == []


# ---------------------------------------------------------------------------
# Save mode behavior on folders
# ---------------------------------------------------------------------------


class TestFolderSaveModes:
    def test_overwrite_clears_then_writes(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        io.write_arrow_table(_flat_table())
        # Second write replaces.
        io.write_arrow_table(_flat_table().slice(0, 1), mode=Mode.OVERWRITE)
        out = io.read_arrow_table()
        assert out.num_rows == 1

    def test_append_adds_sibling_child(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        io.write_arrow_table(_flat_table())
        before = set(os.listdir(str(tmp_path)))
        io.write_arrow_table(_flat_table().slice(0, 1), mode=Mode.APPEND)
        after = set(os.listdir(str(tmp_path)))
        assert before < after, "APPEND must add a new child file"
        assert io.read_arrow_table().num_rows == 4

    def test_ignore_skips_when_folder_non_empty(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        io.write_arrow_table(_flat_table())
        before = set(os.listdir(str(tmp_path)))
        io.write_arrow_table(_flat_table().slice(0, 1), mode=Mode.IGNORE)
        after = set(os.listdir(str(tmp_path)))
        assert before == after

    def test_ignore_writes_when_folder_empty(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        io.write_arrow_table(_flat_table(), mode=Mode.IGNORE)
        assert io.read_arrow_table().num_rows == 3

    def test_error_if_exists_raises_on_non_empty(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        io.write_arrow_table(_flat_table())
        with pytest.raises(FileExistsError):
            io.write_arrow_table(_flat_table(), mode=Mode.ERROR_IF_EXISTS)

    def test_error_if_exists_writes_on_empty(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        io.write_arrow_table(_flat_table(), mode=Mode.ERROR_IF_EXISTS)
        assert io.read_arrow_table().num_rows == 3

    def test_resolve_save_mode_auto_to_overwrite(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        assert io._resolve_save_mode(Mode.AUTO) is Mode.OVERWRITE

    def test_resolve_save_mode_truncate_to_overwrite(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        assert io._resolve_save_mode(Mode.TRUNCATE) is Mode.OVERWRITE


# ---------------------------------------------------------------------------
# make_child name validation
# ---------------------------------------------------------------------------


class TestMakeChild:
    def test_make_child_returns_primitive_io(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        child = io.make_child("a.parquet")
        try:
            assert isinstance(child, ParquetIO)
            assert child.parent is io
            assert child.path.name == "a.parquet"
        finally:
            child.close()

    def test_make_child_creates_intermediate_dirs(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        child = io.make_child("a/b/c.parquet")
        try:
            assert (tmp_path / "a" / "b").is_dir()
        finally:
            child.close()

    def test_backslash_rejected(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        with pytest.raises(ValueError, match="backslash"):
            io.make_child("bad\\name.parquet")

    def test_absolute_rejected(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        with pytest.raises(ValueError, match="relative"):
            io.make_child("/abs.parquet")

    def test_dotdot_rejected(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        with pytest.raises(ValueError, match="\\.\\."):
            io.make_child("../escape.parquet")

    def test_explicit_media_type(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        # Drive media type via the parameter; CSV writes through CsvIO.
        child = io.make_child("a.csv", media_type=MimeTypes.CSV)
        try:
            assert isinstance(child, CsvIO)
        finally:
            child.close()


# ---------------------------------------------------------------------------
# Partitioned writes / reads
# ---------------------------------------------------------------------------


class TestPartitionedFolder:
    def test_partition_layout_on_write(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path), partition_columns=["year"])
        io.write_arrow_table(_partitioned_table())

        names = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
        assert names == ["year=2024", "year=2025"]

    def test_partition_round_trip(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path), partition_columns=["year"])
        io.write_arrow_table(_partitioned_table())

        out = io.read_arrow_table()
        assert out.num_rows == 3
        assert "year" in out.column_names

    def test_partition_columns_stripped_from_child_files(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path), partition_columns=["year"])
        io.write_arrow_table(_partitioned_table())

        # Read a child file directly: its schema should NOT include year.
        first_file = next(
            p for p in (tmp_path / "year=2024").iterdir() if p.is_file()
        )
        child_table = ParquetIO(path=str(first_file)).read_arrow_table()
        assert "year" not in child_table.column_names

    def test_partition_columns_via_options(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path))
        io.write_arrow_table(
            _partitioned_table(),
            partition_columns=["year"],
        )

        names = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
        assert names == ["year=2024", "year=2025"]

    def test_partition_strict_off_reads_mismatch(self, tmp_path: Path):
        (tmp_path / "year=2024").mkdir()
        ParquetIO(
            path=str(tmp_path / "year=2024" / "data.parquet")
        ).write_arrow_table(pa.Table.from_pylist([{"a": 1}]))

        io = FolderIO(
            path=str(tmp_path),
            partition_columns=["year", "month"],
        )
        # Non-strict read should not raise even though the tree only
        # has one k=v segment.
        out = io.read_arrow_table(partition_strict=False)
        assert out.num_rows == 1

    def test_missing_partition_column_in_input_raises(self, tmp_path: Path):
        io = FolderIO(path=str(tmp_path), partition_columns=["year"])
        # Input lacks the declared partition column.
        with pytest.raises(ValueError):
            io.write_arrow_table(_flat_table())

    def test_url_unquote_on_read(self, tmp_path: Path):
        # Partition values may contain path-unsafe characters that
        # are percent-encoded on disk; the read path should unquote.
        io = FolderIO(path=str(tmp_path), partition_columns=["v"])
        io.write_arrow_table(
            pa.Table.from_pylist([{"a": 1, "v": "a/b"}]),
        )
        # On disk: v=a%2Fb/
        names = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
        assert names == ["v=a%2Fb"]
        out = io.read_arrow_table()
        v_vals = out.column("v").to_pylist()
        assert v_vals == ["a/b"]


# ---------------------------------------------------------------------------
# Schema-driven partitioning + .schema sidecar
# ---------------------------------------------------------------------------


def _partitioned_schema_with_tag():
    """Build a Schema where ``year`` is tagged as a partition column."""
    from yggdrasil.data.schema import Schema

    base = Schema.from_arrow(_partitioned_table().schema)
    base["year"] = base["year"].with_partition_by(True, inplace=False)
    return base


class TestFolderSchemaDrivenPartition:
    def test_partitions_derive_from_schema_partition_by_tag(self, tmp_path: Path):
        # ``schema=`` carries a Schema whose ``year`` field is
        # tagged ``partition_by=True``. FolderIO should pick that
        # up without an explicit ``partition_columns`` arg and lay
        # the data down under a Hive-style ``year=*`` subtree.
        io = FolderIO(path=str(tmp_path), schema=_partitioned_schema_with_tag())
        io.write_arrow_table(_partitioned_table())

        names = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
        assert names == ["year=2024", "year=2025"]

    def test_explicit_partition_columns_override_schema(self, tmp_path: Path):
        # When a caller passes both ``partition_columns`` and a
        # tagged ``schema``, the explicit list wins — same priority
        # the documented resolution order promises.
        io = FolderIO(
            path=str(tmp_path),
            schema=_partitioned_schema_with_tag(),
            partition_columns=["a"],
        )
        io.write_arrow_table(_partitioned_table())

        names = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
        # ``a`` becomes the partition column, not ``year``.
        assert all(name.startswith("a=") for name in names)

    def test_schema_sidecar_persisted_on_first_write(self, tmp_path: Path):
        # On first write FolderIO drops a ``.schema`` sidecar so
        # subsequent readers pick up the schema (and the tagged
        # partition layout) without inferring from leaves.
        schema = _partitioned_schema_with_tag()
        io = FolderIO(path=str(tmp_path), schema=schema)
        io.write_arrow_table(_partitioned_table())

        sidecar = tmp_path / FolderIO.SCHEMA_FILE_NAME
        assert sidecar.exists()
        # Sidecar is hidden — the default ``_is_ignored_path`` skips
        # dot-prefixed entries, so iter_children doesn't surface it.
        child_names = [p.name for p in tmp_path.iterdir()]
        assert sidecar.name in child_names  # on disk
        assert not any(  # not yielded
            getattr(c, "path", None) == sidecar
            for c in io._iter_children(io.options_class()())
        )

    def test_sidecar_round_trips_partition_columns(self, tmp_path: Path):
        # A reader that didn't pass a schema should still get the
        # partitioned layout via the persisted sidecar.
        writer = FolderIO(
            path=str(tmp_path),
            schema=_partitioned_schema_with_tag(),
        )
        writer.write_arrow_table(_partitioned_table())

        reader = FolderIO(path=str(tmp_path))
        loaded = reader.declared_schema()
        assert loaded is not None
        partition_names = [f.name for f in loaded.partition_fields]
        assert partition_names == ["year"]
        # And reads inject the partition column back into the rows.
        out = reader.read_arrow_table()
        assert "year" in out.column_names


# ---------------------------------------------------------------------------
# Auto-prune UPSERT
# ---------------------------------------------------------------------------


class TestFolderUpsertAutoPrune:
    def test_upsert_only_rewrites_partitions_touched_by_incoming(
        self, tmp_path: Path
    ):
        # Seed the cache with two partitions: year=2024 and year=2025.
        # Then UPSERT with rows targeting only year=2025 — the 2024
        # partition's leaf set must remain untouched (auto-pruned),
        # while 2025 gets the merged row.
        io = FolderIO(path=str(tmp_path), partition_columns=["year"])
        io.write_arrow_table(_partitioned_table())

        before_2024 = sorted(
            p.name for p in (tmp_path / "year=2024").iterdir()
        )

        # Build an incoming row whose match-by key (``a``) collides
        # with one of the seeded year=2025 rows so we exercise the
        # actual merge, not just an append. Schema matches
        # ``_partitioned_table`` (a: int64, year: string).
        incoming = pa.Table.from_pylist([{"a": 3, "year": "2025"}])
        io.write_arrow_table(
            incoming,
            mode=Mode.UPSERT,
            match_by_names=["a"],
        )

        after_2024 = sorted(
            p.name for p in (tmp_path / "year=2024").iterdir()
        )
        # Auto-prune: year=2024's leaves are unchanged.
        assert before_2024 == after_2024

        out = io.read_arrow_table().sort_by([("a", "ascending")])
        rows = out.to_pylist()
        # ``a==3`` collided so the existing row was replaced rather
        # than duplicated — exactly one survives.
        a3_rows = [r for r in rows if r["a"] == 3]
        assert len(a3_rows) == 1
        # And the 2024 partition still has its untouched rows.
        a_2024 = sorted(r["a"] for r in rows if r["year"] == "2024")
        assert a_2024 == [1, 2]

    def test_upsert_passes_through_to_base_for_non_partitioned(
        self, tmp_path: Path
    ):
        # Without partition columns there's no tree to prune by —
        # FolderIO should fall through to the base
        # read-all/merge/overwrite pattern. Asserting via the public
        # outcome: an upsert merges as expected.
        io = FolderIO(path=str(tmp_path))
        io.write_arrow_table(_flat_table())
        existing_count = io.read_arrow_table().num_rows

        # Use the same dtypes as ``_flat_table`` (a: int64, b: string).
        incoming = pa.Table.from_pylist([{"a": 1, "b": "updated"}])
        io.write_arrow_table(
            incoming, mode=Mode.UPSERT, match_by_names=["a"],
        )
        out = io.read_arrow_table()
        # The ``a==1`` collision merged one incoming row over an
        # existing one — row count stays stable, only the content
        # of the matched row changed.
        assert out.num_rows == existing_count


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


class TestParseKvSegment:
    def test_simple_kv(self):
        assert _parse_kv_segment("year=2024") == ("year", "2024")

    def test_no_equals_returns_none(self):
        assert _parse_kv_segment("plain") is None

    def test_empty_key_returns_none(self):
        assert _parse_kv_segment("=val") is None

    def test_empty_value_allowed(self):
        # Hive's default null-partition convention.
        assert _parse_kv_segment("year=") == ("year", "")

    def test_url_unquoted(self):
        assert _parse_kv_segment("v=a%2Fb") == ("v", "a/b")


class TestPartitionPathSegment:
    def test_single_key(self):
        assert _partition_path_segment({"year": "2024"}) == "year=2024"

    def test_multiple_keys_joined(self):
        out = _partition_path_segment({"year": "2024", "month": "01"})
        assert out == "year=2024/month=01"

    def test_path_unsafe_value_quoted(self):
        out = _partition_path_segment({"v": "a/b"})
        assert out == "v=a%2Fb"

    def test_none_value_rejected(self):
        with pytest.raises(ValueError, match="None partition value"):
            _partition_path_segment({"year": None})


class TestCoercePartitionColumn:
    def test_string_name(self):
        f = _coerce_partition_column("year")
        assert isinstance(f, Field)
        assert f.name == "year"

    def test_passthrough_field(self):
        existing = Field(name="month", dtype="int32")
        out = _coerce_partition_column(existing)
        assert out is existing


# ---------------------------------------------------------------------------
# NestedOptions: defaults
# ---------------------------------------------------------------------------


class TestNestedOptions:
    def test_default_values(self):
        opts = NestedOptions()
        assert opts.child_media_type is None
        assert opts.child_row_size == 0
        assert opts.child_byte_size == 0

    def test_folder_options_partition_defaults(self):
        opts = FolderOptions()
        assert opts.partition_columns is None
        assert opts.sort_partitions is True
        assert opts.partition_strict is True


# ---------------------------------------------------------------------------
# Mixed children (registered tabular leaves + opaque BytesIO)
# ---------------------------------------------------------------------------


class TestMixedChildren:
    def test_opaque_bytes_io_yielded_for_unknown_extension(self, tmp_path: Path):
        # Write a parquet file plus an opaque .bin file. The folder
        # iteration yields a BytesIO for the unknown blob and a
        # ParquetIO for the tabular leaf.
        ParquetIO(path=str(tmp_path / "a.parquet")).write_arrow_table(
            _flat_table(),
        )
        (tmp_path / "blob.bin").write_bytes(b"opaque payload")

        from yggdrasil.io.buffer import BytesIO

        children = sorted(
            FolderIO(path=str(tmp_path)).iter_children(),
            key=lambda c: c.path.name,
        )
        names = [c.path.name for c in children]
        assert "blob.bin" in names

        opaque = next(c for c in children if c.path.name == "blob.bin")
        assert isinstance(opaque, BytesIO)
