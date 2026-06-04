"""Tests for :class:`yggdrasil.parquet.ParquetFolder` — the optimized Folder
over a directory of parquet part files — and the ``yggdrasil.io.parquet_file``
back-compat shim left behind by the move to :mod:`yggdrasil.parquet`."""
from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.enums import Mode, MimeTypes
from yggdrasil.execution.expr import col
from yggdrasil.parquet import (
    ParquetFile, ParquetFolder, ParquetFolderOptions, ParquetOptions,
)


def test_shim_re_exports_the_canonical_class():
    from yggdrasil.io.parquet_file import ParquetFile as Shim
    from yggdrasil.parquet.parquet_file import ParquetFile as Canon
    assert Shim is Canon is ParquetFile
    assert ParquetFile.__module__ == "yggdrasil.parquet.parquet_file"


def test_options_and_mime():
    from yggdrasil.enums.media_type import MediaTypes
    assert ParquetFolder.options_class() is ParquetFolderOptions
    assert ParquetFolder.mime_type is MimeTypes.PARQUET_FOLDER
    # children default to parquet
    assert ParquetFolderOptions().child_media_type is MediaTypes.PARQUET


def _table():
    return pa.table({"id": [1, 2, 3, 4], "region": ["us", "us", "eu", "eu"],
                     "v": [10, 20, 30, 40]})


class TestParquetFolderRead:
    def test_write_read_round_trip(self, tmp_path):
        f = ParquetFolder(path=str(tmp_path))
        f.write_arrow_table(_table(), options=ParquetFolderOptions(mode=Mode.OVERWRITE))
        # part file(s) landed
        assert any(p.endswith(".parquet") for p in os.listdir(tmp_path))
        out = f.read_arrow_table()
        assert out.num_rows == 4
        assert sorted(out.column_names) == ["id", "region", "v"]
        assert sorted(out.column("id").to_pylist()) == [1, 2, 3, 4]

    def test_predicate_pushdown_filters_rows(self, tmp_path):
        f = ParquetFolder(path=str(tmp_path))
        f.write_arrow_table(_table(), options=ParquetFolderOptions(mode=Mode.OVERWRITE))
        got = f.read_arrow_table(predicate=(col("region") == "eu") & (col("v") > 30))
        assert sorted(got.column("id").to_pylist()) == [4]

    def test_hive_partitioned_directory_is_discovered_and_pruned(self, tmp_path):
        # Lay out a Hive-partitioned tree by hand; the dataset scan must surface
        # the ``region`` partition column and prune on a predicate over it.
        for r, ids, vs in (("us", [1, 2], [10, 20]), ("eu", [3, 4], [30, 40])):
            d = tmp_path / f"region={r}"
            d.mkdir()
            pq.write_table(pa.table({"id": ids, "v": vs}), str(d / "part.parquet"))
        f = ParquetFolder(path=str(tmp_path))

        allrows = f.read_arrow_table()
        assert allrows.num_rows == 4
        assert "region" in allrows.column_names                     # partition col surfaced
        assert set(allrows.column("region").to_pylist()) == {"us", "eu"}

        pruned = f.read_arrow_table(predicate=col("region") == "eu")
        assert sorted(pruned.column("id").to_pylist()) == [3, 4]

    def test_empty_directory_reads_empty(self, tmp_path):
        f = ParquetFolder(path=str(tmp_path))
        out = f.read_arrow_table()
        assert out.num_rows == 0


class TestParquetFolderPolars:
    def test_scan_and_read_polars(self, tmp_path):
        f = ParquetFolder(path=str(tmp_path))
        f.write_arrow_table(_table(), options=ParquetFolderOptions(mode=Mode.OVERWRITE))
        lf = f.scan_polars_frame()
        assert lf.collect().height == 4
        df = f.read_polars_frame()
        assert df.height == 4

    def test_polars_predicate_pushdown(self, tmp_path):
        f = ParquetFolder(path=str(tmp_path))
        f.write_arrow_table(_table(), options=ParquetFolderOptions(mode=Mode.OVERWRITE))
        got = f.read_arrow_table(predicate=col("v") >= 30)
        assert sorted(got.column("id").to_pylist()) == [3, 4]


class TestParquetFolderArrowDataset:
    def test_read_arrow_dataset_is_hive_partitioned(self, tmp_path):
        for r in ("us", "eu"):
            d = tmp_path / f"region={r}"
            d.mkdir()
            pq.write_table(pa.table({"id": [1], "v": [1]}), str(d / "p.parquet"))
        ds = ParquetFolder(path=str(tmp_path))._read_arrow_dataset(ParquetOptions())
        assert "region" in ds.schema.names   # partition column discovered
