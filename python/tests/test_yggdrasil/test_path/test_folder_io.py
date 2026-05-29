"""Tests for opening a Path on a folder to get Folder behavior."""
from __future__ import annotations

import os
import pathlib

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from yggdrasil.enums import Mode
from yggdrasil.enums.media_type import MediaTypes
from yggdrasil.execution.expr import col
from yggdrasil.path.folder import Folder, FolderOptions
from yggdrasil.path.local_path import LocalPath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_table(n: int = 5) -> pa.Table:
    return pa.table({
        "id": pa.array(list(range(n)), pa.int64()),
        "label": pa.array([f"row_{i}" for i in range(n)], pa.utf8()),
    })


def _partitioned_schema() -> pa.Schema:
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
    Folder._INSTANCES.clear()
    Folder._PARTITION_DATA_CACHE.clear()
    yield
    Folder._INSTANCES.clear()
    Folder._PARTITION_DATA_CACHE.clear()


# ===================================================================
# TestPathOpenFolder
# ===================================================================


class TestPathOpenFolder:

    def test_local_path_as_media_folder_returns_folder_path(
        self, tmp_path: pathlib.Path,
    ) -> None:
        lp = LocalPath(str(tmp_path), singleton_ttl=False)
        fp = lp.as_media("folder")
        assert isinstance(fp, Folder)

    def test_folder_path_write_table_read_arrow_table_roundtrip(
        self, tmp_path: pathlib.Path,
    ) -> None:
        lp = LocalPath(str(tmp_path / "data"), singleton_ttl=False)
        fp = Folder(path=lp)
        src = _simple_table()
        fp.write_table(src)
        out = fp.read_arrow_table()
        assert out.num_rows == src.num_rows
        assert out.column("id").to_pylist() == list(range(5))
        assert out.column("label").to_pylist() == [f"row_{i}" for i in range(5)]

    def test_folder_path_write_creates_files_inside_directory(
        self, tmp_path: pathlib.Path,
    ) -> None:
        root = tmp_path / "parts"
        fp = Folder(path=str(root))
        fp.write_table(_simple_table(3))
        entries = [
            e for e in os.listdir(str(root))
            if not e.startswith(".") and os.path.isfile(root / e)
        ]
        assert len(entries) >= 1
        assert all(e.startswith("part-") for e in entries)

    def test_folder_path_iter_children_lists_child_files(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "children"))
        fp.write_table(_simple_table(2))
        fp.write_table(_simple_table(3))
        children = list(fp.iter_children())
        assert len(children) >= 2
        # None of the children should be a Folder (flat folder)
        assert all(not isinstance(c, Folder) for c in children)

    def test_folder_path_partition_by_creates_hive_directories(
        self, tmp_path: pathlib.Path,
    ) -> None:
        root = tmp_path / "hive"
        fp = Folder(path=str(root))
        batch = _partitioned_batch(["alpha", "beta", "alpha"], [1, 2, 3])
        fp.write_arrow_batches([batch])
        dirs = sorted(
            e for e in os.listdir(str(root))
            if os.path.isdir(root / e) and not e.startswith(".")
        )
        assert "pk=alpha" in dirs
        assert "pk=beta" in dirs


# ===================================================================
# TestFolderModes
# ===================================================================


class TestFolderModes:

    def test_overwrite_clears_old_files(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "ow"))
        fp.write_table(_simple_table(10))
        assert fp.read_arrow_table().num_rows == 10
        fp.write_table(
            _simple_table(2),
            options=FolderOptions(mode=Mode.OVERWRITE),
        )
        out = fp.read_arrow_table()
        assert out.num_rows == 2
        assert out.column("id").to_pylist() == [0, 1]

    def test_append_adds_new_files(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "ap"))
        fp.write_table(_simple_table(3), options=FolderOptions(mode=Mode.APPEND))
        fp.write_table(_simple_table(4), options=FolderOptions(mode=Mode.APPEND))
        out = fp.read_arrow_table()
        assert out.num_rows == 7

    def test_ignore_noops_when_files_exist(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "ig"))
        fp.write_table(_simple_table(3))
        fp.write_table(
            _simple_table(100),
            options=FolderOptions(mode=Mode.IGNORE),
        )
        out = fp.read_arrow_table()
        assert out.num_rows == 3

    def test_multiple_appends_accumulate_rows(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "multi"))
        opts = FolderOptions(mode=Mode.APPEND)
        fp.write_table(_simple_table(2), options=opts)
        fp.write_table(_simple_table(3), options=opts)
        fp.write_table(_simple_table(5), options=opts)
        out = fp.read_arrow_table()
        assert out.num_rows == 10


# ===================================================================
# TestFolderMultiFormat
# ===================================================================


class TestFolderMultiFormat:

    def test_folder_path_with_parquet_leaf_format(
        self, tmp_path: pathlib.Path,
    ) -> None:
        root = tmp_path / "pq"
        fp = Folder(path=str(root))
        opts = FolderOptions(child_media_type=MediaTypes.PARQUET)
        fp.write_table(_simple_table(4), options=opts)
        # Verify parquet files were created
        files = [
            e for e in os.listdir(str(root))
            if not e.startswith(".") and os.path.isfile(root / e)
        ]
        assert len(files) >= 1
        assert any(f.endswith(".parquet") for f in files)
        out = fp.read_arrow_table()
        assert out.num_rows == 4

    def test_folder_path_with_ipc_leaf_format(
        self, tmp_path: pathlib.Path,
    ) -> None:
        root = tmp_path / "ipc"
        fp = Folder(path=str(root))
        opts = FolderOptions(child_media_type=MediaTypes.ARROW_IPC)
        fp.write_table(_simple_table(3), options=opts)
        files = [
            e for e in os.listdir(str(root))
            if not e.startswith(".") and os.path.isfile(root / e)
        ]
        assert len(files) >= 1
        assert any(f.endswith(".ipc") for f in files)
        out = fp.read_arrow_table()
        assert out.num_rows == 3

    def test_read_arrow_table_merges_all_children(
        self, tmp_path: pathlib.Path,
    ) -> None:
        root = tmp_path / "mixed"
        fp = Folder(path=str(root))
        # Write two batches as separate part files via two write calls
        fp.write_table(pa.table({"v": [1, 2]}))
        fp.write_table(pa.table({"v": [3, 4, 5]}))
        out = fp.read_arrow_table()
        assert out.num_rows == 5
        assert sorted(out.column("v").to_pylist()) == [1, 2, 3, 4, 5]


# ===================================================================
# TestFolderSchemaDrift — part files whose schema drifted across writes
# ===================================================================


def _drift_a() -> pa.RecordBatch:
    # Older layout: binary body, proper timestamp.
    return pa.record_batch(
        [
            pa.array([1, 2], pa.int64()),
            pa.array([b"a", b"b"], pa.binary()),
            pa.array([1_000, 2_000], pa.timestamp("us", "UTC")),
        ],
        names=["id", "body", "ts"],
    )


def _drift_b() -> pa.RecordBatch:
    # Newer layout: large_binary body, legacy int64 timestamp.
    return pa.record_batch(
        [
            pa.array([3, 4], pa.int64()),
            pa.array([b"c", b"d"], pa.large_binary()),
            pa.array([3_000, 4_000], pa.int64()),
        ],
        names=["id", "body", "ts"],
    )


class TestFolderSchemaDrift:
    """Heterogeneous part files (binary↔large_binary, int64↔timestamp).

    Cached folders can end up with part files written under different
    schema revisions. ``pa.Table.from_batches`` and the IPC writer both
    demand byte-identical schemas; reads and merge-upserts must survive
    the drift rather than crash the whole batch.
    """

    def test_read_across_drifted_part_files(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "drift"))
        opts = FolderOptions(mode=Mode.APPEND)
        fp.write_arrow_batches([_drift_a()], options=opts)
        fp.write_arrow_batches([_drift_b()], options=opts)

        # Sanity: the two parts really did land with distinct schemas.
        raw = list(fp._read_arrow_batches(fp.options_class()()))
        body_types = {str(b.schema.field("body").type) for b in raw}
        assert body_types == {"binary", "large_binary"}

        out = fp.read_arrow_table()
        assert out.num_rows == 4
        assert sorted(out.column("id").to_pylist()) == [1, 2, 3, 4]

    def test_upsert_against_drifted_existing_part(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "merge"))
        opts = FolderOptions(mode=Mode.UPSERT, match_by=["id"])
        fp.write_arrow_batches([_drift_a()], options=opts)
        # Second part has a different schema; the merge rewrite chains
        # both into one part file and must not blow up the IPC writer.
        fp.write_arrow_batches([_drift_b()], options=opts)

        update = pa.record_batch(
            [
                pa.array([1], pa.int64()),
                pa.array([b"A"], pa.large_binary()),
                pa.array([9_000], pa.int64()),
            ],
            names=["id", "body", "ts"],
        )
        fp.write_arrow_batches([update], options=opts)

        out = fp.read_arrow_table().sort_by("id")
        assert out.column("id").to_pylist() == [1, 2, 3, 4]
        row = out.filter(pc.equal(out.column("id"), 1))
        assert row.column("body").to_pylist() == [b"A"]


# ===================================================================
# TestFolderPredicate
# ===================================================================


class TestFolderPredicate:

    def test_hive_partitioned_read_with_predicate_prunes(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "part"))
        batch = _partitioned_batch(
            ["a", "a", "b", "c", "c"], [10, 20, 30, 40, 50],
        )
        fp.write_arrow_batches([batch])
        pred = col("pk") == "a"
        opts = FolderOptions(predicate=pred)
        out = fp.read_arrow_table(options=opts)
        assert all(pk == "a" for pk in out.column("pk").to_pylist())
        assert sorted(out.column("val").to_pylist()) == [10, 20]

    def test_read_with_inlist_predicate(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "part"))
        batch = _partitioned_batch(
            ["x", "y", "z", "w"], [1, 2, 3, 4],
        )
        fp.write_arrow_batches([batch])
        pred = col("pk").is_in(["x", "z"])
        opts = FolderOptions(predicate=pred)
        out = fp.read_arrow_table(options=opts)
        assert sorted(out.column("pk").to_pylist()) == ["x", "z"]
        assert sorted(out.column("val").to_pylist()) == [1, 3]

    def test_read_all_returns_all_rows(
        self, tmp_path: pathlib.Path,
    ) -> None:
        fp = Folder(path=str(tmp_path / "part"))
        batch = _partitioned_batch(
            ["a", "b", "c"], [10, 20, 30],
        )
        fp.write_arrow_batches([batch])
        # No predicate: read everything back
        out = fp.read_arrow_table()
        assert out.num_rows == 3
        assert sorted(out.column("pk").to_pylist()) == ["a", "b", "c"]
        assert sorted(out.column("val").to_pylist()) == [10, 20, 30]
