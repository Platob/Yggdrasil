"""Advanced Polars tests — lazy operations, predicate pushdown, and streaming writes."""
from __future__ import annotations

import pathlib

import polars as pl
import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.enums import Mode
from yggdrasil.execution.expr import col
from yggdrasil.io.nested.folder_path import FolderPath, FolderOptions
from yggdrasil.path.local_path import LocalPath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ipc_leaf(tmp_path: pathlib.Path, name: str = "data.ipc"):
    p = LocalPath(str(tmp_path / name), singleton_ttl=False)
    return p.as_media("arrow")


def _parquet_leaf(tmp_path: pathlib.Path, name: str = "data.parquet"):
    p = LocalPath(str(tmp_path / name), singleton_ttl=False)
    return p.as_media("parquet")


# ---------------------------------------------------------------------------
# TestPolarsLazy
# ---------------------------------------------------------------------------


class TestPolarsLazy:

    def test_write_lazyframe_read_back_as_arrow(self, tmp_path: pathlib.Path) -> None:
        lf = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        leaf = _parquet_leaf(tmp_path)
        leaf.write_table(lf)

        result = leaf.read_arrow_table()
        assert result.num_rows == 3
        assert result.column("a").to_pylist() == [1, 2, 3]
        assert result.column("b").to_pylist() == ["x", "y", "z"]

    def test_lazyframe_filter_before_write(self, tmp_path: pathlib.Path) -> None:
        lf = pl.LazyFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
        filtered = lf.filter(pl.col("x") > 2)
        leaf = _parquet_leaf(tmp_path)
        leaf.write_table(filtered)

        result = leaf.read_arrow_table()
        assert result.num_rows == 3
        assert result.column("x").to_pylist() == [3, 4, 5]
        assert result.column("y").to_pylist() == [30, 40, 50]

    def test_lazyframe_select_before_write(self, tmp_path: pathlib.Path) -> None:
        lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        projected = lf.select("a", "c")
        leaf = _ipc_leaf(tmp_path)
        leaf.write_table(projected)

        result = leaf.read_arrow_table()
        assert result.num_rows == 3
        assert set(result.column_names) == {"a", "c"}
        assert result.column("a").to_pylist() == [1, 2, 3]
        assert result.column("c").to_pylist() == [7, 8, 9]

    def test_lazyframe_group_by_agg_before_write(self, tmp_path: pathlib.Path) -> None:
        lf = pl.LazyFrame({
            "category": ["a", "a", "b", "b", "b"],
            "value": [10, 20, 30, 40, 50],
        })
        aggregated = lf.group_by("category").agg(
            pl.col("value").sum().alias("total"),
            pl.col("value").count().alias("count"),
        )
        leaf = _parquet_leaf(tmp_path)
        leaf.write_table(aggregated)

        result = pl.from_arrow(leaf.read_arrow_table())
        result = result.sort("category")
        assert result["category"].to_list() == ["a", "b"]
        assert result["total"].to_list() == [30, 120]
        assert result["count"].to_list() == [2, 3]

    def test_lazyframe_join_before_write(self, tmp_path: pathlib.Path) -> None:
        left = pl.LazyFrame({"id": [1, 2, 3], "name": ["alice", "bob", "carol"]})
        right = pl.LazyFrame({"id": [1, 2, 3], "score": [90, 80, 95]})
        joined = left.join(right, on="id")
        leaf = _ipc_leaf(tmp_path)
        leaf.write_table(joined)

        result = leaf.read_arrow_table()
        assert result.num_rows == 3
        assert set(result.column_names) == {"id", "name", "score"}
        assert result.column("id").to_pylist() == [1, 2, 3]
        assert result.column("score").to_pylist() == [90, 80, 95]

    def test_lazyframe_sort_before_write(self, tmp_path: pathlib.Path) -> None:
        lf = pl.LazyFrame({"x": [3, 1, 2], "y": ["c", "a", "b"]})
        sorted_lf = lf.sort("x")
        leaf = _parquet_leaf(tmp_path)
        leaf.write_table(sorted_lf)

        result = leaf.read_arrow_table()
        assert result.num_rows == 3
        assert result.column("x").to_pylist() == [1, 2, 3]
        assert result.column("y").to_pylist() == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# TestPolarsLazyPredicate
# ---------------------------------------------------------------------------


class TestPolarsLazyPredicate:

    def test_write_data_read_with_predicate(self, tmp_path: pathlib.Path) -> None:
        df = pl.DataFrame({"x": [1, 3, 5, 7, 9, 11], "label": ["a", "b", "c", "d", "e", "f"]})
        leaf = _ipc_leaf(tmp_path)
        leaf.write_table(df)

        opts = CastOptions(predicate=(col("x") > 5))
        result = leaf.read_arrow_table(options=opts)
        assert result.column("x").to_pylist() == [7, 9, 11]
        assert result.column("label").to_pylist() == ["d", "e", "f"]

    def test_partitioned_folder_read_with_predicate(self, tmp_path: pathlib.Path) -> None:
        schema = pa.schema([
            pa.field("pk", pa.utf8(), metadata={b"t:partition_by": b"True"}),
            pa.field("val", pa.int64()),
        ])
        batch = pa.record_batch(
            [
                pa.array(["a", "a", "b", "b", "c"], pa.utf8()),
                pa.array([1, 2, 3, 4, 5], pa.int64()),
            ],
            schema=schema,
        )
        root = tmp_path / "partitioned"
        fp = FolderPath(path=str(root))
        fp.write_arrow_batches([batch])

        pred = col("pk") == "b"
        opts = FolderOptions(predicate=pred)
        result = fp.read_arrow_table(options=opts)
        assert all(pk == "b" for pk in result.column("pk").to_pylist())
        assert sorted(result.column("val").to_pylist()) == [3, 4]

    def test_chained_lazy_ops_write_read_match(self, tmp_path: pathlib.Path) -> None:
        lf = pl.LazyFrame({
            "id": list(range(20)),
            "group": ["x", "y"] * 10,
            "value": list(range(100, 120)),
        })
        # Chain: filter -> select -> sort
        transformed = (
            lf.filter(pl.col("group") == "x")
            .select("id", "value")
            .sort("id")
        )
        leaf = _parquet_leaf(tmp_path)
        leaf.write_table(transformed)

        result = leaf.read_arrow_table()
        expected_ids = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        expected_vals = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
        assert result.column("id").to_pylist() == expected_ids
        assert result.column("value").to_pylist() == expected_vals


# ---------------------------------------------------------------------------
# TestPolarsStreaming
# ---------------------------------------------------------------------------


class TestPolarsStreaming:

    def test_write_large_lazyframe(self, tmp_path: pathlib.Path) -> None:
        n = 100_000
        lf = pl.LazyFrame({
            "idx": list(range(n)),
            "value": [float(i) * 0.5 for i in range(n)],
        })
        leaf = _parquet_leaf(tmp_path, name="large.parquet")
        leaf.write_table(lf)

        result = leaf.read_arrow_table()
        assert result.num_rows == n
        assert result.column("idx").to_pylist()[0] == 0
        assert result.column("idx").to_pylist()[-1] == n - 1

    def test_read_back_large_lazyframe_as_polars(self, tmp_path: pathlib.Path) -> None:
        n = 100_000
        lf = pl.LazyFrame({
            "idx": list(range(n)),
            "value": [float(i) * 0.5 for i in range(n)],
        })
        leaf = _parquet_leaf(tmp_path, name="large.parquet")
        leaf.write_table(lf)

        arrow_table = leaf.read_arrow_table()
        result = pl.from_arrow(arrow_table)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (n, 2)
        assert result["idx"][0] == 0
        assert result["idx"][-1] == n - 1
        assert result["value"][0] == 0.0
        assert result["value"][-1] == (n - 1) * 0.5
