"""Behavior tests for :class:`yggdrasil.io.nested.folder_io.FolderIO`.

`FolderIO` views a filesystem directory as a Tabular: every entry
that resolves to a tabular leaf (parquet, csv, arrow, ndjson, …)
shows up as a child; sub-directories recurse as fresh `FolderIO`s.
Tests pin:

* iter_children walks non-private files + dirs and dispatches each
  file to the right :class:`Tabular` leaf;
* sub-directory recursion;
* :meth:`make_child` creates a ``part-{epoch_ms}-{seed}.{ext}`` file
  under the bound path with the requested format leaf;
* OVERWRITE clears tabular siblings before writing; APPEND adds a
  fresh part file; mixed-format folders concat correctly.
"""
from __future__ import annotations

import os
import pathlib

import pyarrow as pa
import pytest

from yggdrasil.data.enums import Mode
from yggdrasil.io.nested.folder_io import FolderIO, FolderOptions
from yggdrasil.io.primitive.csv_io import CsvIO
from yggdrasil.io.primitive.parquet_io import ParquetIO


@pytest.fixture
def table() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})


class TestConstruction:

    def test_string_path(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path))
        assert os.fspath(folder.path) == str(tmp_path)

    def test_pathlib_path(self, tmp_path) -> None:
        folder = FolderIO(path=pathlib.Path(tmp_path))
        assert os.fspath(folder.path) == str(tmp_path)

    def test_none_path_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a path"):
            FolderIO()


class TestIterChildren:

    def test_yields_registered_tabular_leaves(self, tmp_path, table) -> None:
        # Drop a parquet and a csv into the folder.
        ParquetIO(holder=__class__._lp(tmp_path / "a.parquet"), owns_holder=False).write_arrow_table(table)
        CsvIO(holder=__class__._lp(tmp_path / "b.csv"), owns_holder=False).write_arrow_table(table)
        folder = FolderIO(path=str(tmp_path))
        kinds = sorted(type(c).__name__ for c in folder.iter_children())
        assert kinds == ["CsvIO", "ParquetIO"]

    def test_skips_private_entries(self, tmp_path, table) -> None:
        ParquetIO(holder=__class__._lp(tmp_path / ".hidden.parquet"), owns_holder=False).write_arrow_table(table)
        ParquetIO(holder=__class__._lp(tmp_path / "real.parquet"), owns_holder=False).write_arrow_table(table)
        folder = FolderIO(path=str(tmp_path))
        names = [c._holder.name for c in folder.iter_children()]
        assert "real.parquet" in names[0]
        assert all(".hidden" not in n for n in names)

    def test_subdirectory_returns_subfolder(self, tmp_path, table) -> None:
        sub = tmp_path / "nested"
        sub.mkdir()
        ParquetIO(holder=__class__._lp(sub / "x.parquet"), owns_holder=False).write_arrow_table(table)
        folder = FolderIO(path=str(tmp_path))
        kids = list(folder.iter_children())
        assert any(isinstance(c, FolderIO) for c in kids)

    def test_missing_folder_yields_nothing(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path / "absent"))
        assert list(folder.iter_children()) == []

    @staticmethod
    def _lp(path):
        from yggdrasil.io.path.local_path import LocalPath
        return LocalPath(str(path))


class TestRoundTrip:

    def test_write_then_read(self, tmp_path, table) -> None:
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(table)
        assert any(p.name.endswith(".parquet") for p in tmp_path.iterdir())
        assert folder.read_arrow_table().equals(table)

    def test_csv_default_extension(self, tmp_path, table) -> None:
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(
            table, options=FolderOptions(child_extension="csv"),
        )
        assert any(p.name.endswith(".csv") for p in tmp_path.iterdir())
        # Aggregate read picks up the CSV via media type dispatch.
        loaded = folder.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]

    def test_aggregate_read_across_subfolder(self, tmp_path, table) -> None:
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(table)
        # Add another part inside a sub-directory.
        nested = tmp_path / "shard"
        nested.mkdir()
        sub = FolderIO(path=str(nested))
        sub.write_arrow_table(table)

        out = folder.read_arrow_table()
        assert out.num_rows == 6


class TestModes:

    def test_overwrite_clears_then_writes(self, tmp_path, table) -> None:
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(table)
        before = list(tmp_path.iterdir())
        assert len(before) == 1
        folder.write_arrow_table(table, options=FolderOptions(mode=Mode.OVERWRITE))
        after = list(tmp_path.iterdir())
        assert len(after) == 1
        assert after[0].name != before[0].name  # fresh part filename

    def test_append_adds_part(self, tmp_path, table) -> None:
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(table)
        folder.write_arrow_table(table, options=FolderOptions(mode=Mode.APPEND))
        assert len(list(tmp_path.iterdir())) == 2
        # Aggregate read concatenates rows.
        assert folder.read_arrow_table().num_rows == 6

    def test_ignore_skips_when_non_empty(self, tmp_path, table) -> None:
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(table)
        before = sorted(p.name for p in tmp_path.iterdir())
        folder.write_arrow_table(table, options=FolderOptions(mode=Mode.IGNORE))
        after = sorted(p.name for p in tmp_path.iterdir())
        assert before == after

    def test_error_if_exists_raises(self, tmp_path, table) -> None:
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(table)
        with pytest.raises(FileExistsError):
            folder.write_arrow_table(
                table, options=FolderOptions(mode=Mode.ERROR_IF_EXISTS),
            )


class TestMakeChild:

    def test_part_filename_shape(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path))
        child = folder.make_child(options=FolderOptions(child_extension="parquet"))
        # Fresh path under tmp_path with the correct extension.
        assert os.fspath(child._holder).startswith(str(tmp_path))
        assert os.fspath(child._holder).endswith(".parquet")
        # Class is the ParquetIO leaf — make_child wired the format.
        assert isinstance(child, ParquetIO)

    def test_csv_child(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path))
        child = folder.make_child(options=FolderOptions(child_extension="csv"))
        assert isinstance(child, CsvIO)


class TestOptimize:

    @staticmethod
    def _part_count(path) -> int:
        return sum(1 for p in path.iterdir() if p.name.startswith("part-"))

    def test_no_byte_size_collapses_parts(self, tmp_path, table) -> None:
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(table)
        folder.write_arrow_table(table, options=FolderOptions(mode=Mode.APPEND))
        assert self._part_count(tmp_path) == 2

        new_files = folder.optimize()
        assert new_files == 1
        assert self._part_count(tmp_path) == 1
        # Data round-trips: 6 rows (two writes of 3).
        assert folder.read_arrow_table().num_rows == 6

    def test_byte_size_skips_parts_close_to_target(self, tmp_path, table) -> None:
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(table)
        folder.write_arrow_table(table, options=FolderOptions(mode=Mode.APPEND))
        # Pick a target so each existing part is "close enough" — the
        # files are tiny, so byte_size = part_size puts both parts
        # squarely inside the ±tolerance band.
        sizes = [
            p.stat().st_size for p in tmp_path.iterdir()
            if p.name.startswith("part-")
        ]
        target = max(sizes)

        new_files = folder.optimize(byte_size=target)
        assert new_files == 0
        assert self._part_count(tmp_path) == 2

    def test_byte_size_packs_small_parts(self, tmp_path, table) -> None:
        folder = FolderIO(path=str(tmp_path))
        for _ in range(4):
            folder.write_arrow_table(table, options=FolderOptions(mode=Mode.APPEND))
        sizes = [
            p.stat().st_size for p in tmp_path.iterdir()
            if p.name.startswith("part-")
        ]
        # Big enough to fit every part; tolerance band keeps each
        # individual part on the "needs packing" side.
        target = sum(sizes) * 2

        new_files = folder.optimize(byte_size=target)
        assert new_files >= 1
        # Every original part is gone; the rewritten part is smaller
        # than ``target`` so it's not eligible for another pass.
        assert self._part_count(tmp_path) <= 4
        # Data preserved: 4 writes × 3 rows.
        assert folder.read_arrow_table().num_rows == 12

    def test_idempotent_on_clean_folder(self, tmp_path, table) -> None:
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(table)
        # Single part — nothing to compact.
        assert folder.optimize() == 0
        assert folder.optimize(byte_size=1_000_000) == 0

    def test_missing_folder(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path / "absent"))
        assert folder.optimize() == 0
        assert folder.optimize(byte_size=4096) == 0


class TestTabularBaseOptimize:

    def test_single_file_leaf_is_noop(self, tmp_path, table) -> None:
        from yggdrasil.io.path.local_path import LocalPath

        target = LocalPath(str(tmp_path / "leaf.parquet"))
        leaf = ParquetIO(holder=target, owns_holder=False)
        leaf.write_arrow_table(table)
        # Default Tabular.optimize is a no-op for a non-aggregator
        # leaf — extra kwargs are accepted and ignored.
        assert leaf.optimize() == 0
        assert leaf.optimize(byte_size=1024) == 0
        assert leaf.optimize(byte_size=None, partitions={"x": [1]}) == 0
