"""Tests for :class:`yggdrasil.io.nested.ygg_folder_io.YGGFolderIO`.

Coverage:

* Hive-partitioned writes (``<root>/<col>=<val>/...``).
* Partition pruning via ``options.prune_values``.
* Read-after-write round-trip with partition columns reconstructed
  from directory names.
* :meth:`optimize` compacts small parts per partition.
* The listing cache short-circuits repeated walks (no second
  scandir on a cache hit).
* Multi-column partitioning.
"""
from __future__ import annotations

import os

import pyarrow as pa
import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.enums import Mode
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.primitive import Int64Type, StringType
from yggdrasil.io.nested.folder_io import FolderOptions
from yggdrasil.io.nested.ygg_folder_io import YGGFolderIO


# ---------------------------------------------------------------------------
# Schema fixtures
# ---------------------------------------------------------------------------


def _single_partition_schema() -> Schema:
    s = Schema()
    s.with_field(Field(name="id", dtype=Int64Type()))
    s.with_field(
        Field(name="region", dtype=StringType()).with_partition_by(True)
    )
    s.with_field(Field(name="value", dtype=StringType()))
    return s


def _multi_partition_schema() -> Schema:
    s = Schema()
    s.with_field(Field(name="id", dtype=Int64Type()))
    s.with_field(
        Field(name="year", dtype=Int64Type()).with_partition_by(True)
    )
    s.with_field(
        Field(name="region", dtype=StringType()).with_partition_by(True)
    )
    s.with_field(Field(name="value", dtype=StringType()))
    return s


@pytest.fixture
def table() -> pa.Table:
    return pa.table({
        "id": [1, 2, 3, 4],
        "region": ["us", "us", "eu", "eu"],
        "value": ["a", "b", "c", "d"],
    })


# ---------------------------------------------------------------------------
# Hive layout
# ---------------------------------------------------------------------------


class TestHiveLayout:

    def test_partition_columns_from_schema(self, tmp_path) -> None:
        y = YGGFolderIO(path=str(tmp_path), schema=_single_partition_schema())
        assert y.partition_columns == ["region"]

    def test_no_schema_means_no_partitions(self, tmp_path) -> None:
        y = YGGFolderIO(path=str(tmp_path))
        assert y.partition_columns == []

    def test_write_creates_col_eq_val_dirs(self, tmp_path, table) -> None:
        y = YGGFolderIO(
            path=str(tmp_path), schema=_single_partition_schema(),
        )
        y.write_arrow_table(table)
        dirs = sorted(os.listdir(tmp_path))
        assert dirs == ["region=eu", "region=us"]

    def test_partition_columns_dropped_from_payload(
        self, tmp_path, table,
    ) -> None:
        """Partition column lives in the dir name, not the part file."""
        y = YGGFolderIO(
            path=str(tmp_path), schema=_single_partition_schema(),
        )
        y.write_arrow_table(
            table, options=FolderOptions(child_extension="parquet"),
        )
        # Inspect the parquet footer schema directly — pyarrow's
        # ``read_table`` would synthesize the partition column back
        # from the directory name (Hive-style auto-inference), so
        # we open the file via ``ParquetFile`` and look at the
        # actual stored schema.
        import pyarrow.parquet as pq
        us_dir = tmp_path / "region=us"
        part = next(p for p in us_dir.iterdir() if p.name.startswith("part-"))
        stored_cols = pq.ParquetFile(str(part)).schema_arrow.names
        assert "region" not in stored_cols
        assert {"id", "value"} <= set(stored_cols)

    def test_multi_partition_layout(self, tmp_path) -> None:
        y = YGGFolderIO(
            path=str(tmp_path), schema=_multi_partition_schema(),
        )
        t = pa.table({
            "id": [1, 2, 3],
            "year": [2024, 2024, 2025],
            "region": ["us", "eu", "us"],
            "value": ["a", "b", "c"],
        })
        y.write_arrow_table(t)
        # Year is the outer partition.
        assert sorted(os.listdir(tmp_path)) == ["year=2024", "year=2025"]
        assert sorted(os.listdir(tmp_path / "year=2024")) == [
            "region=eu", "region=us",
        ]


# ---------------------------------------------------------------------------
# Read-after-write
# ---------------------------------------------------------------------------


class TestReadAfterWrite:

    def test_partition_column_reattached(self, tmp_path, table) -> None:
        y = YGGFolderIO(
            path=str(tmp_path), schema=_single_partition_schema(),
        )
        y.write_arrow_table(table)
        out = y.read_arrow_table()
        assert out.num_rows == 4
        assert "region" in out.column_names
        assert sorted(out.to_pylist(), key=lambda r: r["id"]) == sorted(
            table.to_pylist(), key=lambda r: r["id"],
        )

    def test_multi_partition_round_trip(self, tmp_path) -> None:
        y = YGGFolderIO(
            path=str(tmp_path), schema=_multi_partition_schema(),
        )
        t = pa.table({
            "id": [1, 2, 3, 4],
            "year": [2024, 2024, 2025, 2025],
            "region": ["us", "eu", "us", "eu"],
            "value": ["a", "b", "c", "d"],
        })
        y.write_arrow_table(t)
        out = y.read_arrow_table()
        assert out.num_rows == 4
        assert sorted(out.column("year").to_pylist()) == [2024, 2024, 2025, 2025]


# ---------------------------------------------------------------------------
# Partition pruning
# ---------------------------------------------------------------------------


class TestPartitionPruning:

    def test_prune_to_single_partition(self, tmp_path, table) -> None:
        y = YGGFolderIO(
            path=str(tmp_path), schema=_single_partition_schema(),
        )
        y.write_arrow_table(table)
        out = y.read_arrow_table(
            options=FolderOptions(prune_values={"region": ("us",)}),
        )
        assert out.num_rows == 2
        assert set(out.column("region").to_pylist()) == {"us"}

    def test_prune_to_multiple_partitions(self, tmp_path) -> None:
        y = YGGFolderIO(
            path=str(tmp_path), schema=_single_partition_schema(),
        )
        y.write_arrow_table(pa.table({
            "id": [1, 2, 3, 4],
            "region": ["us", "eu", "ap", "us"],
            "value": ["a", "b", "c", "d"],
        }))
        out = y.read_arrow_table(
            options=FolderOptions(prune_values={"region": ("us", "ap")}),
        )
        assert out.num_rows == 3
        assert set(out.column("region").to_pylist()) == {"us", "ap"}

    def test_prune_unknown_value_returns_empty(self, tmp_path, table) -> None:
        y = YGGFolderIO(
            path=str(tmp_path), schema=_single_partition_schema(),
        )
        y.write_arrow_table(table)
        out = y.read_arrow_table(
            options=FolderOptions(prune_values={"region": ("antarctica",)}),
        )
        assert out.num_rows == 0

    def test_prune_int_partition_value(self, tmp_path) -> None:
        y = YGGFolderIO(
            path=str(tmp_path), schema=_multi_partition_schema(),
        )
        t = pa.table({
            "id": [1, 2, 3],
            "year": [2024, 2025, 2026],
            "region": ["us", "us", "us"],
            "value": ["a", "b", "c"],
        })
        y.write_arrow_table(t)
        out = y.read_arrow_table(
            options=FolderOptions(prune_values={"year": (2025,)}),
        )
        assert out.num_rows == 1
        assert out.column("year").to_pylist() == [2025]

    def test_prune_skips_directory_walk(
        self, tmp_path, table, monkeypatch,
    ) -> None:
        """Pruned partitions don't scandir partitions outside the IN set."""
        y = YGGFolderIO(
            path=str(tmp_path), schema=_single_partition_schema(),
        )
        y.write_arrow_table(table)
        # Warm cache once with the full walk so the partition-tree
        # scandir is paid for once; the pruned read should reuse it.
        y.read_arrow_table()

        original = os.scandir
        calls = {"n": 0}

        def counting_scandir(path):
            calls["n"] += 1
            return original(path)

        monkeypatch.setattr(os, "scandir", counting_scandir)
        y.read_arrow_table(
            options=FolderOptions(prune_values={"region": ("us",)}),
        )
        # Cache hit on the partition-tree walk → 0 partition scandirs;
        # only the leaf directory's scandir for part files remains.
        assert calls["n"] <= 1


# ---------------------------------------------------------------------------
# Listing cache
# ---------------------------------------------------------------------------


class TestListingCache:

    def test_partition_tree_walk_cached(
        self, tmp_path, table, monkeypatch,
    ) -> None:
        """The partition-tree walk hits the cache on the second read.

        Note that the leaf-directory listing (the per-partition scan
        for part files) goes through :class:`LocalPath`'s
        :meth:`iterdir` and isn't part of this cache; so we count
        only the tree walks (scandir on a path equal to the root,
        not a leaf).
        """
        y = YGGFolderIO(
            path=str(tmp_path), schema=_single_partition_schema(),
        )
        y.write_arrow_table(table)

        original = os.scandir
        root_calls = {"n": 0}
        root_str = str(tmp_path)

        def counting_scandir(path):
            if str(path) == root_str:
                root_calls["n"] += 1
            return original(path)

        monkeypatch.setattr(os, "scandir", counting_scandir)
        y.read_arrow_table()
        first = root_calls["n"]
        y.read_arrow_table()
        # Second read MUST reuse the cached root listing.
        assert root_calls["n"] == first

    def test_invalidate_after_write(self, tmp_path, table) -> None:
        y = YGGFolderIO(
            path=str(tmp_path), schema=_single_partition_schema(),
        )
        y.write_arrow_table(table)
        y.read_arrow_table()  # warm cache

        # Append rows in a new partition. The write path invalidates
        # the partition's listing entry, so the next read picks them up.
        y.write_arrow_batches(
            pa.table({
                "id": [99],
                "region": ["ap"],
                "value": ["new"],
            }).to_batches(),
            options=FolderOptions(mode=Mode.APPEND),
        )
        out = y.read_arrow_table()
        assert out.num_rows == 5
        assert "ap" in set(out.column("region").to_pylist())


# ---------------------------------------------------------------------------
# Optimize
# ---------------------------------------------------------------------------


class TestOptimize:

    def test_compacts_multiple_parts(self, tmp_path) -> None:
        y = YGGFolderIO(
            path=str(tmp_path), schema=_single_partition_schema(),
        )
        y.write_arrow_table(pa.table({
            "id": [1], "region": ["us"], "value": ["a"],
        }))
        y.write_arrow_batches(
            pa.table({"id": [2], "region": ["us"], "value": ["b"]}).to_batches(),
            options=FolderOptions(mode=Mode.APPEND),
        )
        us_dir = tmp_path / "region=us"
        before = [p for p in us_dir.iterdir() if p.name.startswith("part-")]
        assert len(before) == 2
        compacted = y.optimize()
        assert compacted == 1
        after = [p for p in us_dir.iterdir() if p.name.startswith("part-")]
        assert len(after) == 1
        # Data still intact post-compaction.
        out = y.read_arrow_table()
        assert sorted(out.column("id").to_pylist()) == [1, 2]

    def test_optimize_is_idempotent(self, tmp_path, table) -> None:
        y = YGGFolderIO(
            path=str(tmp_path), schema=_single_partition_schema(),
        )
        y.write_arrow_table(table)
        # Single write → one part per partition; optimize is a no-op.
        assert y.optimize() == 0

    def test_byte_size_skips_close_to_target(self, tmp_path) -> None:
        y = YGGFolderIO(
            path=str(tmp_path), schema=_single_partition_schema(),
        )
        y.write_arrow_table(pa.table({
            "id": [1], "region": ["us"], "value": ["a"],
        }))
        y.write_arrow_batches(
            pa.table({"id": [2], "region": ["us"], "value": ["b"]}).to_batches(),
            options=FolderOptions(mode=Mode.APPEND),
        )
        us_dir = tmp_path / "region=us"
        sizes = [
            p.stat().st_size for p in us_dir.iterdir()
            if p.name.startswith("part-")
        ]
        # Pick byte_size matching the existing parts — both should be
        # left alone (they're already "at target") so optimize is a no-op.
        assert y.optimize(byte_size=max(sizes)) == 0
        after = [p for p in us_dir.iterdir() if p.name.startswith("part-")]
        assert len(after) == 2

    def test_no_schema_falls_back_to_folderio_walk(self, tmp_path, table) -> None:
        # No schema → no partition columns → optimize delegates to the
        # plain :meth:`FolderIO.optimize` walk.
        y = YGGFolderIO(path=str(tmp_path))
        # Drop two parquet parts directly under the root so there's
        # something to compact.
        from yggdrasil.io.primitive.parquet_io import ParquetIO
        from yggdrasil.io.path.local_path import LocalPath
        for name in ("part-1.parquet", "part-2.parquet"):
            ParquetIO(
                holder=LocalPath(str(tmp_path / name)), owns_holder=False,
            ).write_arrow_table(table)
        assert y.optimize() == 1


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


class TestModes:

    def test_overwrite_clears_partition_tree(self, tmp_path) -> None:
        y = YGGFolderIO(
            path=str(tmp_path), schema=_single_partition_schema(),
        )
        y.write_arrow_table(pa.table({
            "id": [1, 2], "region": ["us", "eu"], "value": ["a", "b"],
        }))
        y.write_arrow_table(
            pa.table({"id": [9], "region": ["ap"], "value": ["z"]}),
            options=FolderOptions(mode=Mode.OVERWRITE),
        )
        # Old partitions wiped; only the new one survives.
        assert os.listdir(tmp_path) == ["region=ap"]

    def test_append_keeps_existing_partitions(self, tmp_path, table) -> None:
        y = YGGFolderIO(
            path=str(tmp_path), schema=_single_partition_schema(),
        )
        y.write_arrow_table(table)
        y.write_arrow_table(
            pa.table({"id": [9], "region": ["ap"], "value": ["z"]}),
            options=FolderOptions(mode=Mode.APPEND),
        )
        assert sorted(os.listdir(tmp_path)) == [
            "region=ap", "region=eu", "region=us",
        ]
