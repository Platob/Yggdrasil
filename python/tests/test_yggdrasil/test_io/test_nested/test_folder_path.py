"""Tests for :class:`yggdrasil.io.nested.folder_path.Folder`."""

from __future__ import annotations

import os
import pathlib

import pyarrow as pa
import pytest

from yggdrasil.enums import Mode
from yggdrasil.execution.expr import col
from yggdrasil.path.folder import Folder, FolderOptions
from yggdrasil.path.local_path import LocalPath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_table(n: int = 3) -> pa.Table:
    return pa.table({
        "id": pa.array(list(range(n)), pa.int64()),
        "name": pa.array([f"row_{i}" for i in range(n)], pa.utf8()),
    })


def _partitioned_schema() -> pa.Schema:
    """Schema with a partition_by-tagged string column ``pk``."""
    return pa.schema([
        pa.field("pk", pa.utf8(), metadata={b"t:partition_by": b"True"}),
        pa.field("val", pa.int64()),
    ])


def _partitioned_batch(
    pks: list[str],
    vals: list[int],
) -> pa.RecordBatch:
    return pa.record_batch(
        [pa.array(pks, pa.utf8()), pa.array(vals, pa.int64())],
        schema=_partitioned_schema(),
    )


@pytest.fixture(autouse=True)
def _clear_singletons():
    """Reset singleton + partition-data caches between tests.

    Folder is Singleton-cached by URL — stale entries from prior
    tests sharing the same (or overlapping) tmp_path URL would hand
    back a cached instance with warm caches, poisoning assertions.
    """
    Folder._INSTANCES.clear()
    Folder._PARTITION_DATA_CACHE.clear()
    yield
    Folder._INSTANCES.clear()
    Folder._PARTITION_DATA_CACHE.clear()


# ===================================================================
# TestConstruction
# ===================================================================


class TestConstruction:

    def test_from_string_path(self, tmp_path: pathlib.Path) -> None:
        fp = Folder(path=str(tmp_path))
        assert fp.path is not None
        assert "Folder" in repr(fp)

    def test_from_pathlib(self, tmp_path: pathlib.Path) -> None:
        fp = Folder(path=tmp_path)
        assert fp.path is not None

    def test_from_url(self, tmp_path: pathlib.Path) -> None:
        url = f"file://{tmp_path}"
        fp = Folder(path=url)
        assert fp.path is not None

    def test_missing_path_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a path"):
            Folder(path=None)

    def test_repr_contains_class_and_path(self, tmp_path: pathlib.Path) -> None:
        fp = Folder(path=str(tmp_path))
        r = repr(fp)
        assert r.startswith("Folder(")
        assert "path=" in r


# ===================================================================
# TestWriteRead
# ===================================================================


class TestWriteRead:

    def test_write_table_read_arrow_table_round_trip(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        src = _simple_table()
        fp.write_table(src)
        out = fp.read_arrow_table()
        assert out.num_rows == src.num_rows
        assert out.column("id").to_pylist() == [0, 1, 2]
        assert out.column("name").to_pylist() == ["row_0", "row_1", "row_2"]

    def test_append_mode_accumulates_rows(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        t1 = _simple_table(2)
        t2 = _simple_table(3)
        fp.write_table(t1)
        fp.write_table(t2)  # default mode is APPEND
        out = fp.read_arrow_table()
        assert out.num_rows == 5

    def test_overwrite_clears_previous(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        fp.write_table(_simple_table(5))
        assert fp.read_arrow_table().num_rows == 5

        replacement = _simple_table(2)
        fp.write_table(replacement, options=FolderOptions(mode=Mode.OVERWRITE))
        out = fp.read_arrow_table()
        assert out.num_rows == 2
        assert out.column("id").to_pylist() == [0, 1]

    def test_ignore_skips_when_data_exists(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        fp.write_table(_simple_table(3))
        fp.write_table(_simple_table(10), options=FolderOptions(mode=Mode.IGNORE))
        out = fp.read_arrow_table()
        # IGNORE should have left the original 3 rows untouched.
        assert out.num_rows == 3

    def test_error_if_exists_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        fp.write_table(_simple_table(3))
        with pytest.raises(FileExistsError):
            fp.write_table(
                _simple_table(1),
                options=FolderOptions(mode=Mode.ERROR_IF_EXISTS),
            )

    def test_write_record_batch(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        batch = pa.record_batch(
            {"x": pa.array([10, 20]), "y": pa.array(["a", "b"])},
        )
        fp.write_table(batch)
        out = fp.read_arrow_table()
        assert out.num_rows == 2
        assert out.column("x").to_pylist() == [10, 20]

    def test_write_list_of_dicts(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        fp.write_table([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        out = fp.read_arrow_table()
        assert out.num_rows == 2
        assert sorted(out.column("a").to_pylist()) == [1, 2]

    def test_empty_folder_yields_empty_table(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "empty"))
        out = fp.read_arrow_table()
        assert out.num_rows == 0


# ===================================================================
# TestPartitionedWrite
# ===================================================================


class TestPartitionedWrite:

    def test_write_produces_hive_dirs(
        self, tmp_path: pathlib.Path,
    ) -> None:
        root = tmp_path / "partitioned"
        fp = Folder(path=str(root))
        batch = _partitioned_batch(["a", "a", "b"], [1, 2, 3])
        fp.write_arrow_batches([batch])

        dirs = sorted(
            e for e in os.listdir(str(root))
            if os.path.isdir(root / e) and not e.startswith(".")
        )
        assert "pk=a" in dirs
        assert "pk=b" in dirs

    def test_read_back_preserves_partition_values(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "partitioned"))
        batch = _partitioned_batch(["x", "y", "x"], [10, 20, 30])
        fp.write_arrow_batches([batch])

        out = fp.read_arrow_table()
        rows = out.to_pydict()
        # pk=x should have vals 10, 30; pk=y should have val 20
        x_vals = [v for pk, v in zip(rows["pk"], rows["val"]) if pk == "x"]
        y_vals = [v for pk, v in zip(rows["pk"], rows["val"]) if pk == "y"]
        assert sorted(x_vals) == [10, 30]
        assert y_vals == [20]

    def test_overwrite_clears_stale_partitions(
        self, tmp_path: pathlib.Path,
    ) -> None:
        root = tmp_path / "partitioned"
        fp = Folder(path=str(root))
        fp.write_arrow_batches([_partitioned_batch(["a", "b", "c"], [1, 2, 3])])
        # Verify partition c exists
        assert (root / "pk=c").exists()

        # Overwrite with only a, b — c should be gone
        fp.write_arrow_batches(
            [_partitioned_batch(["a", "b"], [10, 20])],
            options=FolderOptions(mode=Mode.OVERWRITE),
        )
        assert not (root / "pk=c").exists()
        out = fp.read_arrow_table()
        assert sorted(out.column("pk").to_pylist()) == ["a", "b"]

    def test_predicate_prunes_partitions_on_read(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "partitioned"))
        batch = _partitioned_batch(
            ["a", "a", "b", "c"], [1, 2, 3, 4],
        )
        fp.write_arrow_batches([batch])

        pred = col("pk") == "a"
        opts = FolderOptions(predicate=pred)
        out = fp.read_arrow_table(options=opts)
        assert all(pk == "a" for pk in out.column("pk").to_pylist())
        assert sorted(out.column("val").to_pylist()) == [1, 2]

    def test_multiple_partitions_round_trip(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """Two-level partition: pk (outer) and region (inner)."""
        schema = pa.schema([
            pa.field("pk", pa.utf8(), metadata={b"t:partition_by": b"True"}),
            pa.field("region", pa.utf8(), metadata={b"t:partition_by": b"True"}),
            pa.field("val", pa.int64()),
        ])
        batch = pa.record_batch(
            [
                pa.array(["a", "a", "b"], pa.utf8()),
                pa.array(["us", "eu", "us"], pa.utf8()),
                pa.array([1, 2, 3], pa.int64()),
            ],
            schema=schema,
        )
        root = tmp_path / "multi_part"
        fp = Folder(path=str(root))
        fp.write_arrow_batches([batch])

        out = fp.read_arrow_table()
        assert out.num_rows == 3
        # Hive tree should have pk=a/region=us, pk=a/region=eu, pk=b/region=us
        assert (root / "pk=a" / "region=us").exists()
        assert (root / "pk=a" / "region=eu").exists()
        assert (root / "pk=b" / "region=us").exists()

    def test_inlist_predicate_on_partition_column(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "partitioned"))
        batch = _partitioned_batch(
            ["a", "b", "c", "d"], [1, 2, 3, 4],
        )
        fp.write_arrow_batches([batch])

        pred = col("pk").is_in(["a", "c"])
        opts = FolderOptions(predicate=pred)
        out = fp.read_arrow_table(options=opts)
        assert sorted(out.column("pk").to_pylist()) == ["a", "c"]
        assert sorted(out.column("val").to_pylist()) == [1, 3]


# ===================================================================
# TestIterChildren
# ===================================================================


class TestIterChildren:

    def test_yields_leaf_files(self, tmp_path: pathlib.Path) -> None:
        fp = Folder(path=str(tmp_path))
        fp.write_table(_simple_table(2))
        children = list(fp.iter_children())
        # At least one leaf child (part-*.ipc)
        assert len(children) >= 1
        assert all(not isinstance(c, Folder) for c in children)

    def test_skips_dotfiles_and_ygg_metadata(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path))
        fp.write_table(_simple_table(1))
        # The .ygg sidecar exists but should not appear in children
        ygg_dir = tmp_path / ".ygg"
        assert ygg_dir.exists()
        # Create an extra dotfile
        (tmp_path / ".hidden_file").write_text("hidden")
        children = list(fp.iter_children())
        child_names = [
            getattr(getattr(c, "path", None), "name", None)
            or getattr(getattr(c, "_parent", None), "name", None)
            or ""
            for c in children
        ]
        assert not any(n.startswith(".") for n in child_names if n)

    def test_recurses_into_subdirectories(
        self, tmp_path: pathlib.Path,
    ) -> None:
        # Create a subdirectory with data
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        sub_fp = Folder(path=str(subdir))
        sub_fp.write_table(_simple_table(1))

        # Also write data in root
        root_fp = Folder(path=str(tmp_path))
        root_fp.write_table(_simple_table(1))

        children = list(root_fp.iter_children())
        types = [type(c).__name__ for c in children]
        # Should have at least one Folder child (the subdirectory)
        assert "Folder" in types

    def test_missing_folder_yields_nothing(self) -> None:
        fp = Folder(path="/tmp/_ygg_test_nonexistent_xyz_98765")
        children = list(fp.iter_children())
        assert children == []


# ===================================================================
# TestSchema
# ===================================================================


class TestSchema:

    def test_collect_schema_from_written_data(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        fp.write_table(_simple_table())
        schema = fp.collect_schema()
        names = [c.name for c in schema.children]
        assert "id" in names
        assert "name" in names

    def test_schema_persists_in_sidecar(
        self, tmp_path: pathlib.Path,
    ) -> None:
        root = tmp_path / "data"
        fp = Folder(path=str(root))
        fp.write_table(_simple_table())
        sidecar = root / ".ygg" / "schema.arrow"
        assert sidecar.exists()
        assert sidecar.stat().st_size > 0

    def test_column_projection_on_read(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        fp.write_table(_simple_table())
        out = fp.read_arrow_table(columns=["id"])
        assert out.column_names == ["id"]
        assert out.column("id").to_pylist() == [0, 1, 2]

    def test_schema_with_metadata_round_trips(
        self, tmp_path: pathlib.Path,
    ) -> None:
        from yggdrasil.data import field, schema

        s = schema(
            fields=[field("x", pa.int64()), field("y", pa.utf8())],
            metadata={"custom_tag": "test_value"},
        )
        arrow_s = s.to_arrow_schema()
        table = pa.table(
            {"x": pa.array([1, 2]), "y": pa.array(["a", "b"])},
            schema=arrow_s,
        )
        root = tmp_path / "meta"
        fp = Folder(path=str(root))
        fp.write_table(table)

        # Fresh Folder reads the sidecar
        Folder._INSTANCES.clear()
        fp2 = Folder(path=str(root))
        recovered = fp2.collect_schema()
        assert recovered is not None
        recovered_arrow = recovered.to_arrow_schema()
        assert recovered_arrow.metadata is not None
        assert b"custom_tag" in recovered_arrow.metadata


# ===================================================================
# TestPredicateFiltering
# ===================================================================


class TestPredicateFiltering:

    def test_predicate_filters_rows_on_read(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        fp.write_table(_simple_table(5))
        pred = col("id") > 2
        opts = FolderOptions(predicate=pred)
        out = fp.read_arrow_table(options=opts)
        assert all(v > 2 for v in out.column("id").to_pylist())
        assert out.num_rows == 2

    def test_partition_pruning_skips_directories(
        self, tmp_path: pathlib.Path,
    ) -> None:
        root = tmp_path / "partitioned"
        fp = Folder(path=str(root))
        batch = _partitioned_batch(["a", "b", "c"], [1, 2, 3])
        fp.write_arrow_batches([batch])

        pred = col("pk") == "b"
        opts = FolderOptions(predicate=pred)
        out = fp.read_arrow_table(options=opts)
        assert out.column("pk").to_pylist() == ["b"]
        assert out.column("val").to_pylist() == [2]

    def test_combined_partition_prune_and_row_filter(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "partitioned"))
        batch = _partitioned_batch(
            ["a", "a", "a", "b", "b"], [10, 20, 30, 40, 50],
        )
        fp.write_arrow_batches([batch])

        # Partition prune to pk=a, then row filter val > 15
        pred = (col("pk") == "a") & (col("val") > 15)
        opts = FolderOptions(predicate=pred)
        out = fp.read_arrow_table(options=opts)
        assert all(pk == "a" for pk in out.column("pk").to_pylist())
        assert all(v > 15 for v in out.column("val").to_pylist())
        assert sorted(out.column("val").to_pylist()) == [20, 30]

    def test_empty_predicate_reads_all(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        fp.write_table(_simple_table(4))
        opts = FolderOptions(predicate=None)
        out = fp.read_arrow_table(options=opts)
        assert out.num_rows == 4

    def test_inlist_predicate_on_partition_column(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "partitioned"))
        batch = _partitioned_batch(
            ["alpha", "beta", "gamma", "delta"],
            [1, 2, 3, 4],
        )
        fp.write_arrow_batches([batch])

        pred = col("pk").is_in(["beta", "delta"])
        opts = FolderOptions(predicate=pred)
        out = fp.read_arrow_table(options=opts)
        assert sorted(out.column("pk").to_pylist()) == ["beta", "delta"]
        assert sorted(out.column("val").to_pylist()) == [2, 4]


# ===================================================================
# TestModes
# ===================================================================


class TestModes:

    def test_append_mode_appends(self, tmp_path: pathlib.Path) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        fp.write_table(_simple_table(2), options=FolderOptions(mode=Mode.APPEND))
        fp.write_table(_simple_table(3), options=FolderOptions(mode=Mode.APPEND))
        out = fp.read_arrow_table()
        assert out.num_rows == 5

    def test_overwrite_mode_replaces(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        fp.write_table(_simple_table(10))
        fp.write_table(
            _simple_table(2),
            options=FolderOptions(mode=Mode.OVERWRITE),
        )
        out = fp.read_arrow_table()
        assert out.num_rows == 2

    def test_ignore_mode_no_ops(self, tmp_path: pathlib.Path) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        fp.write_table(_simple_table(3))
        fp.write_table(
            _simple_table(100),
            options=FolderOptions(mode=Mode.IGNORE),
        )
        out = fp.read_arrow_table()
        assert out.num_rows == 3

    def test_error_if_exists_mode_raises(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "data"))
        fp.write_table(_simple_table(1))
        with pytest.raises(FileExistsError, match="already contains"):
            fp.write_table(
                _simple_table(1),
                options=FolderOptions(mode=Mode.ERROR_IF_EXISTS),
            )


# ===================================================================
# TestOptimize
# ===================================================================


class TestOptimize:

    def test_optimize_compacts_files(
        self, tmp_path: pathlib.Path,
    ) -> None:
        root = tmp_path / "data"
        fp = Folder(path=str(root))
        # Write three separate part files
        for _ in range(3):
            fp.write_table(_simple_table(2))

        parts_before = [
            f for f in os.listdir(str(root)) if f.startswith("part-")
        ]
        assert len(parts_before) == 3

        compacted = fp.optimize()
        assert compacted >= 1

        parts_after = [
            f for f in os.listdir(str(root)) if f.startswith("part-")
        ]
        assert len(parts_after) < len(parts_before)

    def test_optimize_preserves_data(
        self, tmp_path: pathlib.Path,
    ) -> None:
        root = tmp_path / "data"
        fp = Folder(path=str(root))
        fp.write_table(pa.table({"v": [1, 2]}))
        fp.write_table(pa.table({"v": [3, 4]}))

        before = sorted(fp.read_arrow_table().column("v").to_pylist())
        fp.optimize()
        after = sorted(fp.read_arrow_table().column("v").to_pylist())
        assert before == after == [1, 2, 3, 4]
