"""FolderIO: Arrow round-trip across multiple child files."""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.nested import FolderIO, FolderOptions
from .._helpers import sample_table


class TestFolderArrow:
    def test_round_trip_via_parquet(self, tmp_path):
        FolderIO(path=str(tmp_path)).write_arrow_table(sample_table())
        out = FolderIO(path=str(tmp_path)).read_arrow_table()
        assert out.num_rows == 3
        assert out.column_names == ["a", "b"]

    def test_child_row_size_splits(self, tmp_path):
        FolderIO(path=str(tmp_path)).write_arrow_table(
            sample_table(),
            options=FolderOptions(child_row_size=1),
        )
        # Three rows, one per file → at least three children.
        files = list(tmp_path.iterdir())
        data_files = [f for f in files if f.is_file() and not f.name.startswith(".")]
        assert len(data_files) >= 3

    def test_partitioned_round_trip(self, tmp_path):
        import pyarrow as pa

        table = pa.Table.from_pylist(
            [
                {"year": "2024", "value": 1},
                {"year": "2024", "value": 2},
                {"year": "2025", "value": 3},
            ]
        )
        folder = FolderIO(path=str(tmp_path), partition_columns=["year"])
        folder.write_arrow_table(table)
        out = FolderIO(
            path=str(tmp_path),
            partition_columns=["year"],
        ).read_arrow_table()
        assert out.num_rows == 3
        assert "year" in out.column_names
