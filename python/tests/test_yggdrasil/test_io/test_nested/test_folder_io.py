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
from yggdrasil.io.primitive.csv_file import CSVFile
from yggdrasil.io.primitive.parquet_file import ParquetFile


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
        ParquetFile(holder=__class__._lp(tmp_path / "a.parquet"), owns_holder=False).write_arrow_table(table)
        CSVFile(holder=__class__._lp(tmp_path / "b.csv"), owns_holder=False).write_arrow_table(table)
        folder = FolderIO(path=str(tmp_path))
        kinds = sorted(type(c).__name__ for c in folder.iter_children())
        assert kinds == ["CSVFile", "ParquetFile"]

    def test_skips_private_entries(self, tmp_path, table) -> None:
        ParquetFile(holder=__class__._lp(tmp_path / ".hidden.parquet"), owns_holder=False).write_arrow_table(table)
        ParquetFile(holder=__class__._lp(tmp_path / "real.parquet"), owns_holder=False).write_arrow_table(table)
        folder = FolderIO(path=str(tmp_path))
        names = [c._parent.name for c in folder.iter_children()]
        assert "real.parquet" in names[0]
        assert all(".hidden" not in n for n in names)

    def test_subdirectory_returns_subfolder(self, tmp_path, table) -> None:
        sub = tmp_path / "nested"
        sub.mkdir()
        ParquetFile(holder=__class__._lp(sub / "x.parquet"), owns_holder=False).write_arrow_table(table)
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
        # Default child media type is Arrow IPC (extension ``.ipc``).
        assert any(p.name.endswith(".ipc") for p in tmp_path.iterdir())
        assert folder.read_arrow_table().equals(table)

    def test_csv_default_extension(self, tmp_path, table) -> None:
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(
            table, options=FolderOptions(child_media_type="csv"),
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
        child = folder.make_child(
            options=FolderOptions(child_media_type="parquet"),
        )
        # Fresh path under tmp_path with the correct extension.
        assert os.fspath(child._parent).startswith(str(tmp_path))
        assert os.fspath(child._parent).endswith(".parquet")
        # Class is the ParquetFile leaf — make_child wired the format.
        assert isinstance(child, ParquetFile)

    def test_csv_child(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path))
        child = folder.make_child(options=FolderOptions(child_media_type="csv"))
        assert isinstance(child, CSVFile)


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


class TestWriteRechunking:

    def test_byte_size_splits_stream_into_multiple_parts(self, tmp_path) -> None:
        # Wide-ish batches so the byte_size threshold is reachable
        # without spilling tens of MiB through the test.
        rng = list(range(2048))
        batches = [
            pa.record_batch({
                "id": rng,
                "blob": ["x" * 64] * len(rng),
            })
            for _ in range(8)
        ]
        single_nbytes = batches[0].nbytes
        # Pick byte_size so the rechunker emits at least 3 parts.
        target = max(1, single_nbytes * 2)

        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_batches(
            batches,
            options=FolderOptions(byte_size=target, mode=Mode.APPEND),
        )

        parts = [p for p in tmp_path.iterdir() if p.name.startswith("part-")]
        assert len(parts) >= 3
        # All rows survive the split.
        loaded = folder.read_arrow_table()
        assert loaded.num_rows == sum(b.num_rows for b in batches)

    def test_no_byte_size_keeps_single_part(self, tmp_path, table) -> None:
        folder = FolderIO(path=str(tmp_path))
        # Many small batches with no sizing knob → one part file.
        folder.write_arrow_batches(
            [table.to_batches()[0]] * 5,
            options=FolderOptions(mode=Mode.APPEND),
        )
        parts = [p for p in tmp_path.iterdir() if p.name.startswith("part-")]
        assert len(parts) == 1


class TestMergeByName:

    @staticmethod
    def _ids(folder) -> list[int]:
        return sorted(folder.read_arrow_table().column("id").to_pylist())

    def test_append_drops_already_present_keys(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(
            pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]}),
        )
        # Same id=2 collides with existing; id=4 is fresh.
        folder.write_arrow_table(
            pa.table({"id": [2, 4], "v": ["B", "D"]}),
            options=FolderOptions(
                mode=Mode.APPEND, match_by=["id"],
            ),
        )
        # Existing rows untouched (id=2 keeps "b"); only id=4 added.
        out = folder.read_arrow_table().sort_by("id")
        assert out.column("id").to_pylist() == [1, 2, 3, 4]
        assert out.column("v").to_pylist() == ["a", "b", "c", "D"]

    def test_upsert_rewrites_matching_keys(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(
            pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]}),
        )
        folder.write_arrow_table(
            pa.table({"id": [2, 4], "v": ["B", "D"]}),
            options=FolderOptions(
                mode=Mode.UPSERT, match_by=["id"],
            ),
        )
        out = folder.read_arrow_table().sort_by("id")
        # id=2 was overwritten ("b" → "B"); id=4 added; rest untouched.
        assert out.column("id").to_pylist() == [1, 2, 3, 4]
        assert out.column("v").to_pylist() == ["a", "B", "c", "D"]

    def test_append_without_match_by_keeps_duplicates(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(pa.table({"id": [1, 2]}))
        folder.write_arrow_table(
            pa.table({"id": [2, 3]}),
            options=FolderOptions(mode=Mode.APPEND),
        )
        out = folder.read_arrow_table().sort_by("id")
        # No match_by → no dedup; id=2 appears twice.
        assert out.column("id").to_pylist() == [1, 2, 2, 3]


class TestDelete:

    def test_delete_drops_matching_rows_and_leaves_others(self, tmp_path) -> None:
        from yggdrasil.io.tabular.execution.expr import col

        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(
            pa.table({"id": [1, 2, 3, 4, 5], "v": ["a", "b", "c", "d", "e"]}),
        )
        deleted = folder.delete(col("id") >= 3)
        assert deleted == 3
        out = folder.read_arrow_table().sort_by("id")
        assert out.column("id").to_pylist() == [1, 2]

    def test_delete_returns_zero_when_nothing_matches(self, tmp_path) -> None:
        from yggdrasil.io.tabular.execution.expr import col

        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(pa.table({"id": [1, 2, 3]}))
        before = sorted(p.name for p in tmp_path.iterdir())
        deleted = folder.delete(col("id") > 100)
        assert deleted == 0
        # No rewrite happened — the part file is the same.
        after = sorted(p.name for p in tmp_path.iterdir())
        assert before == after

    def test_delete_full_part_unlinks_file(self, tmp_path) -> None:
        from yggdrasil.io.tabular.execution.expr import col

        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(pa.table({"id": [1, 2, 3]}))
        # Predicate matches every row → file deleted, no replacement.
        deleted = folder.delete(col("id") >= 1)
        assert deleted == 3
        assert list(tmp_path.iterdir()) == []
        assert folder.read_arrow_table().num_rows == 0

    def test_delete_per_leaf_isolation(self, tmp_path) -> None:
        """Only the leaf containing matching rows is rewritten."""
        from yggdrasil.io.tabular.execution.expr import col

        folder = FolderIO(path=str(tmp_path))
        # Two appended parts. The predicate only matches rows in the
        # second part — the first part stays on disk untouched.
        folder.write_arrow_table(pa.table({"id": [1, 2]}))
        folder.write_arrow_table(
            pa.table({"id": [3, 4]}), options=FolderOptions(mode=Mode.APPEND),
        )
        before = {p.name for p in tmp_path.iterdir()}
        deleted = folder.delete(col("id") == 4)
        assert deleted == 1
        after = {p.name for p in tmp_path.iterdir()}
        # The id=[1,2] leaf is bit-identical (same name); the id=[3,4]
        # leaf has been replaced by a fresh-named part holding id=[3].
        unchanged = before & after
        assert len(unchanged) == 1
        assert sorted(folder.read_arrow_table().column("id").to_pylist()) == [1, 2, 3]

    def test_delete_recurses_into_subfolders(self, tmp_path) -> None:
        from yggdrasil.io.tabular.execution.expr import col

        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(pa.table({"id": [1, 2]}))
        sub_path = tmp_path / "shard"
        sub_path.mkdir()
        sub = FolderIO(path=str(sub_path))
        sub.write_arrow_table(pa.table({"id": [3, 4]}))

        deleted = folder.delete(col("id") > 2)
        assert deleted == 2
        assert sorted(folder.read_arrow_table().column("id").to_pylist()) == [1, 2]

    def test_delete_accepts_sql_string(self, tmp_path) -> None:
        # SQL-string predicates round-trip through the SQL lifter,
        # which depends on the optional :mod:`sqlglot` extra.
        pytest.importorskip("sqlglot")
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(pa.table({"id": [1, 2, 3, 4]}))
        deleted = folder.delete("id IN (2, 4)")
        assert deleted == 2
        assert sorted(folder.read_arrow_table().column("id").to_pylist()) == [1, 3]

    def test_delete_rejects_non_predicate(self, tmp_path) -> None:
        from yggdrasil.io.tabular.execution.expr import col

        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(pa.table({"id": [1]}))
        with pytest.raises(TypeError, match="Predicate"):
            folder.delete(col("id"))  # bare column is not a predicate


class TestTabularBaseOptimize:

    def test_single_file_leaf_is_noop(self, tmp_path, table) -> None:
        from yggdrasil.io.path.local_path import LocalPath

        target = LocalPath(str(tmp_path / "leaf.parquet"))
        leaf = ParquetFile(holder=target, owns_holder=False)
        leaf.write_arrow_table(table)
        # Default Tabular.optimize is a no-op for a non-aggregator
        # leaf — extra kwargs are accepted and ignored.
        assert leaf.optimize() == 0
        assert leaf.optimize(byte_size=1024) == 0
        assert leaf.optimize(byte_size=None, partitions={"x": [1]}) == 0


class TestPredicatePrune:
    """``FolderIO._read_arrow_batches`` short-circuits via
    :meth:`Tabular._should_prune_by_predicate`: a folder whose
    :attr:`static_values` (own seed or inherited from
    :attr:`tabular_parent`) make ``options.predicate`` provably false
    contributes zero rows; per-child the same check skips sub-folders /
    leaves whose static surface decides the predicate negatively."""

    def test_self_prune_skips_directory_walk(
        self, tmp_path, table, monkeypatch,
    ) -> None:
        """A predicate provably false against the folder's own
        static_values must not list the directory at all."""
        from yggdrasil.io.tabular.execution.expr import col

        ParquetFile(
            holder=__class__._lp(tmp_path / "p.parquet"), owns_holder=False,
        ).write_arrow_table(table)

        # Folder claims region=us across every row it yields. A predicate
        # that excludes us is provably false — no iterdir, no read.
        folder = FolderIO(path=str(tmp_path), static_values={"region": "us"})

        original = os.scandir
        visited: list[str] = []

        def tracking_scandir(path):
            visited.append(str(path))
            return original(path)

        monkeypatch.setattr(os, "scandir", tracking_scandir)
        out = folder.read_arrow_table(
            options=FolderOptions(predicate=col("region") == "eu"),
        )
        assert out.num_rows == 0
        # Self-prune fired before iter_children → no scandir of the folder.
        assert not any(str(tmp_path) in p for p in visited)

    def test_self_prune_admits_when_predicate_matches_static(
        self, tmp_path, table,
    ) -> None:
        from yggdrasil.io.tabular.execution.expr import col

        ParquetFile(
            holder=__class__._lp(tmp_path / "p.parquet"), owns_holder=False,
        ).write_arrow_table(table)
        folder = FolderIO(path=str(tmp_path), static_values={"region": "us"})
        out = folder.read_arrow_table(
            options=FolderOptions(predicate=col("region") == "us"),
        )
        assert out.num_rows == table.num_rows

    def test_undecidable_predicate_falls_through(self, tmp_path, table) -> None:
        """A predicate over a column outside static_values is
        conservatively treated as 'could match' — the read still
        runs, leaving row-level filtering to downstream consumers."""
        from yggdrasil.io.tabular.execution.expr import col

        ParquetFile(
            holder=__class__._lp(tmp_path / "p.parquet"), owns_holder=False,
        ).write_arrow_table(table)
        folder = FolderIO(path=str(tmp_path), static_values={"region": "us"})
        out = folder.read_arrow_table(
            options=FolderOptions(predicate=col("id") > 0),
        )
        assert out.num_rows == table.num_rows

    def test_per_child_prune_skips_unmatched_subfolder(
        self, tmp_path, table,
    ) -> None:
        """Sub-folders inherit static_values via the tabular_parent
        chain; the per-child prune skips those whose KV makes the
        predicate provably false without recursing into them."""
        from yggdrasil.io.tabular.execution.expr import col

        # Hand-roll a partition-shaped layout (no YGGFolderIO) so we
        # exercise the FolderIO path directly: two sub-folders, each
        # carrying its own static_values seed inherited from a parent
        # FolderIO that mints them.
        us_dir = tmp_path / "us"
        eu_dir = tmp_path / "eu"
        us_dir.mkdir()
        eu_dir.mkdir()
        ParquetFile(
            holder=__class__._lp(us_dir / "p.parquet"), owns_holder=False,
        ).write_arrow_table(table)
        ParquetFile(
            holder=__class__._lp(eu_dir / "p.parquet"), owns_holder=False,
        ).write_arrow_table(table)

        class _StaticFolderIO(FolderIO):
            """FolderIO that stamps its sub-folder children with a
            ``region`` seed derived from the directory name. Same
            shape :class:`YGGFolderIO` uses to prune partitions, but
            without the partition-schema ceremony — keeps this test
            focused on the FolderIO-level prune."""

            # ``None`` opts out of mime-type registration so the test
            # subclass doesn't collide with :class:`FolderIO` for the
            # ``inode/directory`` slot.
            mime_type = None

            def iter_children(self):
                for child in super().iter_children():
                    if isinstance(child, FolderIO):
                        child._static_value_seed.setdefault(
                            "region", child.path.name,
                        )
                    yield child

        folder = _StaticFolderIO(path=str(tmp_path))
        out = folder.read_arrow_table(
            options=FolderOptions(predicate=col("region") == "us"),
        )
        # Only the us sub-folder's rows survive — the eu sub-folder
        # was rejected at the per-child prune gate.
        assert out.num_rows == table.num_rows

    @staticmethod
    def _lp(path):
        from yggdrasil.io.path.local_path import LocalPath
        return LocalPath(str(path))
