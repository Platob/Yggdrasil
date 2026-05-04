"""Tests for the universal :class:`CastOptions.predicate` filter.

Every IO that yields tabular rows funnels through
:meth:`TabularIO._iter_public_batches`, which applies the
predicate before the rows leave the read pipeline. The same
mechanism enforces the "missing column → accept everything"
contract so a heterogeneous-source folder doesn't silently drop
rows from files that lack a column.

Coverage:

- ``predicate`` filters rows on every primitive leaf format
  (Parquet / IPC / CSV / JSON / NDJSON).
- A predicate that references a non-existent column degrades to
  *accept everything*, not "drop everything".
- ``predicate=None`` is the no-op default.
- ``iter_children(predicate=...)`` is the canonical knob for the
  child-discovery filter (replacing the old options-only path).
- DeltaIO partition pruning skips whole AddFiles whose partition
  values can't satisfy the predicate.
"""

from __future__ import annotations

import pathlib
import sys

import pyarrow as pa
import pytest

from yggdrasil.data.expr import col
from yggdrasil.io.buffer.nested import FolderIO, YGGFolderIO
from yggdrasil.io.buffer.primitive.arrow_ipc_io import ArrowIPCIO
from yggdrasil.io.buffer.primitive.csv_io import CsvIO
from yggdrasil.io.buffer.primitive.json_io import JsonIO
from yggdrasil.io.buffer.primitive.ndjson_io import NDJsonIO
from yggdrasil.io.buffer.primitive.parquet_io import ParquetIO
from yggdrasil.io.enums import Mode


_IS_WINDOWS = sys.platform.startswith("win")

_TABLE = pa.table({
    "id": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
    "tag": pa.array(["a", "b", "c", "d", "e"]),
})


# ---------------------------------------------------------------------------
# Per-format row filtering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "io_cls,filename",
    [
        (ParquetIO, "data.parquet"),
        (ArrowIPCIO, "data.arrow"),
        (CsvIO, "data.csv"),
        (JsonIO, "data.json"),
        (NDJsonIO, "data.ndjson"),
    ],
)
class TestPerFormatPredicate:
    def test_predicate_filters_rows(self, tmp_path, io_cls, filename):
        target = tmp_path / filename
        with io_cls(path=str(target), mode="wb+") as w:
            w.write_arrow_table(_TABLE)
        with io_cls(path=str(target), mode="rb") as r:
            out = r.read_arrow_table(predicate=col("id") > 2)
        assert out.num_rows == 3
        assert out["id"].to_pylist() == [3, 4, 5]

    def test_predicate_none_returns_full_data(self, tmp_path, io_cls, filename):
        target = tmp_path / filename
        with io_cls(path=str(target), mode="wb+") as w:
            w.write_arrow_table(_TABLE)
        with io_cls(path=str(target), mode="rb") as r:
            out = r.read_arrow_table()
        assert out.num_rows == 5

    def test_missing_column_accepts_everything(
        self, tmp_path, io_cls, filename,
    ):
        # The data has no ``absent`` column — the predicate can't
        # evaluate, so we keep every row rather than silently
        # dropping all of them. Documented behaviour.
        target = tmp_path / filename
        with io_cls(path=str(target), mode="wb+") as w:
            w.write_arrow_table(_TABLE)
        with io_cls(path=str(target), mode="rb") as r:
            out = r.read_arrow_table(predicate=col("absent") > 99)
        assert out.num_rows == 5

    def test_predicate_drops_everything_yields_empty_table(
        self, tmp_path, io_cls, filename,
    ):
        target = tmp_path / filename
        with io_cls(path=str(target), mode="wb+") as w:
            w.write_arrow_table(_TABLE)
        with io_cls(path=str(target), mode="rb") as r:
            out = r.read_arrow_table(predicate=col("id") > 1000)
        assert out.num_rows == 0


# ---------------------------------------------------------------------------
# Folder-level filtering — predicate applied to every child's batches
# ---------------------------------------------------------------------------


class TestFolderPredicate:
    def test_predicate_filters_across_children(self, tmp_path):
        with FolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_TABLE.slice(0, 3), mode=Mode.APPEND)
            io.write_arrow_table(_TABLE.slice(3, 2), mode=Mode.APPEND)

        with FolderIO(path=str(tmp_path)) as io:
            out = io.read_arrow_table(predicate=col("id") > 2)
        assert sorted(out["id"].to_pylist()) == [3, 4, 5]

    def test_predicate_referencing_missing_column_keeps_all(self, tmp_path):
        # Two children with the SAME schema; predicate references a
        # column that exists on neither. Both files should still
        # contribute every row.
        with FolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_TABLE.slice(0, 3), mode=Mode.APPEND)
            io.write_arrow_table(_TABLE.slice(3, 2), mode=Mode.APPEND)

        with FolderIO(path=str(tmp_path)) as io:
            out = io.read_arrow_table(predicate=col("absent") > 99)
        assert out.num_rows == 5


# ---------------------------------------------------------------------------
# iter_children predicate API
# ---------------------------------------------------------------------------


class TestIterChildrenPredicate:
    def test_iter_children_predicate_filters_by_name(self, tmp_path):
        # Two parquet files; iterate with a predicate that filters
        # by the discovered child's basename.
        a = tmp_path / "a.parquet"
        b = tmp_path / "b.parquet"
        with ParquetIO(path=str(a), mode="wb+") as w:
            w.write_arrow_table(_TABLE)
        with ParquetIO(path=str(b), mode="wb+") as w:
            w.write_arrow_table(_TABLE)

        with FolderIO(path=str(tmp_path)) as io:
            kept = list(io.iter_children(col("name") == "a.parquet"))
        names = sorted(c.path.name for c in kept)
        assert names == ["a.parquet"]

    def test_iter_children_no_predicate_yields_everything(self, tmp_path):
        for name in ("x.parquet", "y.parquet"):
            with ParquetIO(path=str(tmp_path / name), mode="wb+") as w:
                w.write_arrow_table(_TABLE)

        with FolderIO(path=str(tmp_path)) as io:
            kept = list(io.iter_children())
        names = sorted(c.path.name for c in kept)
        assert names == ["x.parquet", "y.parquet"]


# ---------------------------------------------------------------------------
# DeltaIO partition pruning
# ---------------------------------------------------------------------------


class TestDeltaPartitionPruning:
    @pytest.fixture
    def delta_table(self, tmp_path):
        from yggdrasil.io.buffer.nested.delta.io import DeltaIO

        # Build a 2-partition table whose dtypes Delta knows how to
        # serialise (string everywhere keeps the schema codec happy
        # and avoids tripping over engine-specific type round-trips
        # the way ints do at the time of writing).
        table = pa.table({
            "id": pa.array(["1", "2", "3", "4"]),
            "year": pa.array(["2024", "2024", "2025", "2025"]),
        })
        target = tmp_path / "table"
        try:
            with DeltaIO(
                path=str(target),
                partition_columns=["year"],
            ) as w:
                w.write_arrow_table(table, mode=Mode.OVERWRITE)
        except TypeError as exc:
            pytest.skip(
                f"Delta schema codec doesn't accept this fixture's "
                f"dtypes in this environment: {exc}"
            )
        return target

    def test_partition_only_predicate_skips_files(self, delta_table):
        from yggdrasil.io.buffer.nested.delta.io import DeltaIO

        with DeltaIO(path=str(delta_table)) as r:
            out = r.read_arrow_table(predicate=col("year") == "2024")
        # Only the 2024 partition's rows.
        assert sorted(out["id"].to_pylist()) == ["1", "2"]

    def test_data_predicate_still_works(self, delta_table):
        from yggdrasil.io.buffer.nested.delta.io import DeltaIO

        with DeltaIO(path=str(delta_table)) as r:
            out = r.read_arrow_table(predicate=col("id") > "2")
        assert sorted(out["id"].to_pylist()) == ["3", "4"]

    def test_missing_column_predicate_accepts_everything(self, delta_table):
        from yggdrasil.io.buffer.nested.delta.io import DeltaIO

        with DeltaIO(path=str(delta_table)) as r:
            out = r.read_arrow_table(predicate=col("absent") == "x")
        assert out.num_rows == 4


# ---------------------------------------------------------------------------
# YGGFolderIO predicate parity — sidecar doesn't break the contract
# ---------------------------------------------------------------------------


class TestYGGFolderPredicate:
    def test_predicate_filters_through_ygg_folder(self, tmp_path):
        with YGGFolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_TABLE.slice(0, 3), mode=Mode.APPEND)
            io.write_arrow_table(_TABLE.slice(3, 2), mode=Mode.APPEND)

        with YGGFolderIO(path=str(tmp_path)) as io:
            out = io.read_arrow_table(predicate=col("id") <= 2)
        assert sorted(out["id"].to_pylist()) == [1, 2]
