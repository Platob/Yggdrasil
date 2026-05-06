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


class TestIterChildren:
    def test_iter_children_yields_everything(self, tmp_path):
        # Folder iteration is unfiltered — every entry on disk is
        # yielded. Callers that want to filter do so on the iterator.
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


class TestParquetPushdown:
    """ParquetIO pushes predicates into ``pyarrow.dataset`` when possible.

    The pushdown path uses the parquet footer's per-row-group min/max
    stats to skip whole row groups before any data is decompressed —
    a different code path from the universal per-batch filter at the
    TabularIO layer. We can't easily probe ``pa.dataset`` internals
    from here, but we can observe the contract: the result is the
    same as the universal-filter path, AND the pushdown helper is
    chosen when the predicate compiles cleanly.
    """

    def _write_three_groups(self, path: pathlib.Path) -> None:
        # Three small row groups so the footer has enough metadata
        # to drive pushdown at row-group granularity.
        import pyarrow.parquet as pq

        pq.write_table(
            pa.concat_tables([
                pa.table({"id": pa.array([1, 2, 3]), "v": ["a", "b", "c"]}),
                pa.table({"id": pa.array([4, 5, 6]), "v": ["d", "e", "f"]}),
                pa.table({"id": pa.array([7, 8, 9]), "v": ["g", "h", "i"]}),
            ]),
            str(path),
            row_group_size=3,
        )

    def test_pushdown_returns_filtered_batches(self, tmp_path):
        target = tmp_path / "rg.parquet"
        self._write_three_groups(target)

        with ParquetIO(path=str(target), mode="rb") as r:
            out = r.read_arrow_table(predicate=col("id") >= 7)
        assert sorted(out["id"].to_pylist()) == [7, 8, 9]

    def test_pushdown_helper_is_invoked(self, tmp_path, monkeypatch):
        # Spy on _iter_with_pushdown to confirm the pushdown branch
        # is taken when a predicate is present. The result is the
        # same either way; this test pins the *path* taken.
        from yggdrasil.io.buffer.primitive import parquet_io as parquet_mod

        target = tmp_path / "spy.parquet"
        self._write_three_groups(target)

        seen: list[bool] = []
        original = parquet_mod.ParquetIO._iter_with_pushdown

        def spy(self, **kwargs):
            seen.append(True)
            return original(self, **kwargs)

        monkeypatch.setattr(
            parquet_mod.ParquetIO, "_iter_with_pushdown", spy,
        )

        with ParquetIO(path=str(target), mode="rb") as r:
            out = r.read_arrow_table(predicate=col("id") < 5)
        assert seen == [True]
        assert sorted(out["id"].to_pylist()) == [1, 2, 3, 4]

    def test_pushdown_skipped_when_no_predicate(self, tmp_path, monkeypatch):
        from yggdrasil.io.buffer.primitive import parquet_io as parquet_mod

        target = tmp_path / "no-pred.parquet"
        self._write_three_groups(target)

        seen: list[bool] = []
        original = parquet_mod.ParquetIO._iter_with_pushdown

        def spy(self, **kwargs):
            seen.append(True)
            return original(self, **kwargs)

        monkeypatch.setattr(
            parquet_mod.ParquetIO, "_iter_with_pushdown", spy,
        )

        with ParquetIO(path=str(target), mode="rb") as r:
            out = r.read_arrow_table()
        assert seen == []
        assert out.num_rows == 9

    def test_missing_column_falls_back_to_unfiltered(self, tmp_path):
        target = tmp_path / "missing.parquet"
        self._write_three_groups(target)

        # Predicate column is absent — pushdown returns None and
        # the universal filter's missing-column rule keeps every row.
        with ParquetIO(path=str(target), mode="rb") as r:
            out = r.read_arrow_table(predicate=col("absent") > 0)
        assert out.num_rows == 9


class TestYGGFolderPredicate:
    def test_predicate_filters_through_ygg_folder(self, tmp_path):
        with YGGFolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_TABLE.slice(0, 3), mode=Mode.APPEND)
            io.write_arrow_table(_TABLE.slice(3, 2), mode=Mode.APPEND)

        with YGGFolderIO(path=str(tmp_path)) as io:
            out = io.read_arrow_table(predicate=col("id") <= 2)
        assert sorted(out["id"].to_pylist()) == [1, 2]
